import gc
import tempfile
import unittest

import torch
import torch.nn as nn

from diffusers.quantizers.auto import DiffusersAutoQuantizer
from diffusers.quantizers.gemlite import GemLiteQuantizer
from diffusers.quantizers.gemlite.gemlite_quantizer import _replace_with_gemlite_linear
from diffusers.quantizers.quantization_config import GemLiteConfig, QuantizationMethod
from diffusers.utils import is_gemlite_available, is_torch_available

from ...testing_utils import (
    backend_empty_cache,
    backend_reset_peak_memory_stats,
    enable_full_determinism,
    nightly,
    require_accelerate,
    require_accelerator,
    require_gemlite,
    require_gemlite_version_greater_or_equal,
    torch_device,
)


# TODO(gemlite): bump to the next released version once it ships, so the version gate
# actually protects against installs older than the API these tests rely on.
_GEMLITE_MIN_VERSION = "0.5.1"


enable_full_determinism()


if is_torch_available():
    from ..utils import get_memory_consumption_stat


if is_gemlite_available():
    from gemlite.core import DType, GemLiteLinearTriton


def _create_packed_gemlite_state_dict(in_features=64, out_features=32, w_nbits=4, group_size=64):
    source_layer = GemLiteLinearTriton(
        W_nbits=w_nbits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        input_dtype=DType.FP16,
        output_dtype=DType.FP16,
    )
    weight = torch.arange(out_features * in_features, dtype=torch.int32).remainder(2**w_nbits).to(torch.uint8)
    scales = torch.ones((weight.numel() // group_size, 1), dtype=torch.float16)
    zeros = torch.full((weight.numel() // group_size, 1), (2**w_nbits - 1) // 2, dtype=torch.float16)

    source_layer.pack(weight, scales, zeros, bias=None)

    return source_layer.state_dict()


class GemLiteConfigTest(unittest.TestCase):
    def test_config_defaults(self):
        config = GemLiteConfig()

        self.assertEqual(config.quant_method, QuantizationMethod.GEMLITE)
        self.assertEqual(config.compute_dtype, torch.float16)
        self.assertTrue(config.pre_quantized)
        self.assertEqual(config.to_diff_dict()["quant_method"], QuantizationMethod.GEMLITE)

    def test_config_from_dict(self):
        config = DiffusersAutoQuantizer.from_dict(
            {"quant_method": "gemlite", "compute_dtype": "bfloat16", "modules_to_not_convert": ["proj_out"]}
        )

        self.assertIsInstance(config, GemLiteConfig)
        self.assertEqual(config.compute_dtype, torch.bfloat16)
        self.assertEqual(config.modules_to_not_convert, ["proj_out"])

    def test_config_round_trip(self):
        config = GemLiteConfig(compute_dtype=torch.bfloat16, modules_to_not_convert=["proj_out"])

        restored = DiffusersAutoQuantizer.from_dict(config.to_dict())

        self.assertIsInstance(restored, GemLiteConfig)
        self.assertEqual(restored.compute_dtype, torch.bfloat16)
        self.assertEqual(restored.modules_to_not_convert, ["proj_out"])


class GemLiteQuantizerTest(unittest.TestCase):
    def test_rejects_on_the_fly_quantization(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=False)

        with self.assertRaisesRegex(ValueError, "only supports loading already-quantized checkpoints"):
            quantizer.validate_environment()

    def test_update_torch_dtype_coerces_to_compute_dtype(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(compute_dtype=torch.float16), pre_quantized=True)

        self.assertEqual(quantizer.update_torch_dtype(None), torch.float16)
        self.assertEqual(quantizer.update_torch_dtype(torch.float16), torch.float16)
        self.assertEqual(quantizer.update_torch_dtype(torch.bfloat16), torch.float16)

    def test_update_missing_keys_filters_meta_scale(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=True)

        missing_keys = ["0.weight", "0.meta_scale", "1.bias", "2.meta_scale"]
        filtered = quantizer.update_missing_keys(model=None, missing_keys=missing_keys, prefix="")

        self.assertEqual(filtered, ["0.weight", "1.bias"])


@require_gemlite
@require_gemlite_version_greater_or_equal(_GEMLITE_MIN_VERSION)
class GemLiteQuantizerBackendTest(unittest.TestCase):
    def test_auto_quantizer_mapping(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=True)

        self.assertIsInstance(quantizer, GemLiteQuantizer)
        self.assertTrue(quantizer.pre_quantized)

        # Should not raise
        quantizer.validate_environment()

    def test_check_if_quantized_param_rejects_non_gemlite_modules(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=True)

        plain_model = nn.Sequential(nn.Linear(32, 32, bias=False))
        self.assertFalse(
            quantizer.check_if_quantized_param(plain_model, torch.ones(4, 4, dtype=torch.uint8), "0.W_q", {})
        )

        gemlite_model = nn.Sequential(GemLiteLinearTriton())
        self.assertFalse(quantizer.check_if_quantized_param(gemlite_model, torch.ones(32, 32), "0.weight", {}))
        self.assertTrue(
            quantizer.check_if_quantized_param(gemlite_model, torch.ones(4, 4, dtype=torch.uint8), "0.W_q", {})
        )

    def test_replace_skips_dotted_nested_modules(self):
        inner = nn.Sequential(nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False))
        model = nn.Sequential(inner, nn.Linear(32, 32, bias=False))

        replaced = _replace_with_gemlite_linear(model, ["0"])

        self.assertEqual(replaced, 1)
        self.assertIs(type(model[0][0]), nn.Linear)
        self.assertIs(type(model[0][1]), nn.Linear)
        self.assertIsInstance(model[1], GemLiteLinearTriton)

    def test_process_before_weight_loading_writes_back_config(self):
        config = GemLiteConfig()
        quantizer = DiffusersAutoQuantizer.from_config(config, pre_quantized=True)

        model = nn.Sequential(nn.Linear(32, 32, bias=False))

        class FakeConfig:
            quantization_config = None

        model.config = FakeConfig()

        quantizer._process_model_before_weight_loading(model, device_map=None)

        self.assertIs(model.config.quantization_config, config)

    def test_uses_upstream_gemlite_module_and_finalizes_loaded_state(self):
        in_features = 64
        out_features = 32
        gemlite_state_dict = _create_packed_gemlite_state_dict(in_features, out_features)

        model = nn.Sequential(nn.Linear(in_features, out_features, bias=False))
        replaced = _replace_with_gemlite_linear(model, [])
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=True)

        self.assertEqual(replaced, 1)
        self.assertIsInstance(model[0], GemLiteLinearTriton)

        for name, value in gemlite_state_dict.items():
            quantizer.create_quantized_param(model, value, f"0.{name}", "cpu")

        self.assertEqual(model[0]._gemlite_loaded_param_names, set(gemlite_state_dict))

        quantizer._process_model_after_weight_loading(model)
        loaded_state_dict = model[0].state_dict()
        for name, expected_value in gemlite_state_dict.items():
            self.assertTrue(torch.equal(loaded_state_dict[name], expected_value), name)


@nightly
@require_gemlite
@require_gemlite_version_greater_or_equal(_GEMLITE_MIN_VERSION)
@require_accelerator
@require_accelerate
class GemLiteBaseTesterMixin:
    model_id = "hf-internal-testing/tiny-flux-transformer"
    pipeline_model_id = "hf-internal-testing/tiny-flux-pipe"
    torch_dtype = torch.float16
    expected_memory_reduction = 1.1
    modules_to_not_convert = ["proj_out"]

    @classmethod
    def setUpClass(cls):
        from gemlite.helper import A16W8_INT8

        from diffusers import FluxTransformer2DModel
        from diffusers.configuration_utils import FrozenDict

        cls._tmpdir = tempfile.TemporaryDirectory()

        model = FluxTransformer2DModel.from_pretrained(cls.model_id, torch_dtype=cls.torch_dtype).to(torch_device)

        quantizer_helper = A16W8_INT8(device=str(torch_device), dtype=cls.torch_dtype)
        skip_set = set(cls.modules_to_not_convert)
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any((key + "." in name) or (key == name) for key in skip_set):
                continue
            parent, child_name = _get_parent_and_child(model, name)
            setattr(parent, child_name, quantizer_helper.from_linear(module))

        config_dict = dict(model._internal_dict)
        config_dict["quantization_config"] = GemLiteConfig(
            compute_dtype=cls.torch_dtype, modules_to_not_convert=cls.modules_to_not_convert
        ).to_dict()
        model._internal_dict = FrozenDict(config_dict)

        model.save_pretrained(cls._tmpdir.name)

        model.cpu()
        del model
        gc.collect()
        backend_empty_cache(torch_device)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()
        gc.collect()
        backend_empty_cache(torch_device)

    def setUp(self):
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        gc.collect()

    def tearDown(self):
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        gc.collect()

    def _load_quantized_model(self):
        from diffusers import FluxTransformer2DModel

        return FluxTransformer2DModel.from_pretrained(self._tmpdir.name, torch_dtype=self.torch_dtype)

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 4096, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "pooled_projections": torch.randn(
                (1, 768),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
            "img_ids": torch.randn((4096, 3), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "txt_ids": torch.randn((512, 3), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "guidance": torch.tensor([3.5]).to(torch_device, self.torch_dtype),
        }

    def test_gemlite_layers(self):
        model = self._load_quantized_model()
        model.to(torch_device)
        skip = set(self.modules_to_not_convert)

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                assert name in skip, f"nn.Linear {name} was not replaced with GemLiteLinearTriton"
            if isinstance(module, GemLiteLinearTriton):
                assert name not in skip, f"skipped module {name} was unexpectedly replaced"

    def test_gemlite_memory_usage(self):
        from diffusers import FluxTransformer2DModel

        inputs = self.get_dummy_inputs()
        inputs = {
            k: v.to(device=torch_device, dtype=self.torch_dtype) for k, v in inputs.items() if not isinstance(v, bool)
        }

        unquantized_model = FluxTransformer2DModel.from_pretrained(self.model_id, torch_dtype=self.torch_dtype)
        unquantized_model.to(torch_device)
        unquantized_memory = get_memory_consumption_stat(unquantized_model, inputs)

        quantized_model = self._load_quantized_model()
        quantized_model.to(torch_device)
        quantized_memory = get_memory_consumption_stat(quantized_model, inputs)

        assert unquantized_memory / quantized_memory >= self.expected_memory_reduction

    def test_modules_to_not_convert(self):
        model = self._load_quantized_model()
        model.to(torch_device)
        skip = set(self.modules_to_not_convert)

        matched = {name: module for name, module in model.named_modules() if name in skip}
        assert set(matched) == skip, f"skip entries unmatched: {skip - set(matched)}"
        for name, module in matched.items():
            assert isinstance(module, nn.Linear), f"skipped module {name} is not nn.Linear"
            assert not isinstance(module, GemLiteLinearTriton), f"skipped module {name} was replaced"

    def test_dtype_assignment(self):
        model = self._load_quantized_model()
        with self.assertRaises(ValueError):
            model.to(torch.float16)
        with self.assertRaises(ValueError):
            model.float()
        with self.assertRaises(ValueError):
            model.half()
        model.to(torch_device)

    def test_serialization(self):
        model = self._load_quantized_model()
        inputs = self.get_dummy_inputs()
        model.to(torch_device)
        with torch.no_grad():
            model_output = model(**inputs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            from diffusers import FluxTransformer2DModel

            saved_model = FluxTransformer2DModel.from_pretrained(tmp_dir, torch_dtype=self.torch_dtype)
            saved_model.to(torch_device)
            with torch.no_grad():
                saved_model_output = saved_model(**inputs)

        assert torch.allclose(model_output.sample, saved_model_output.sample, rtol=1e-5, atol=1e-5)

    def test_model_cpu_offload(self):
        from diffusers import FluxPipeline

        transformer = self._load_quantized_model()
        pipe = FluxPipeline.from_pretrained(
            self.pipeline_model_id, transformer=transformer, torch_dtype=self.torch_dtype
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        _ = pipe("a cat holding a sign that says hello", num_inference_steps=2)


class FluxTransformerGemLiteINT8Test(GemLiteBaseTesterMixin, unittest.TestCase):
    expected_memory_reduction = 1.2


def _get_parent_and_child(model, full_name):
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


if __name__ == "__main__":
    unittest.main()
