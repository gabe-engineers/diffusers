import unittest

import torch

from diffusers.quantizers.auto import DiffusersAutoQuantizer
from diffusers.quantizers.gemlite import GemLiteQuantizer
from diffusers.quantizers.quantization_config import GemLiteConfig, QuantizationMethod


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


class GemLiteQuantizerTest(unittest.TestCase):
    def test_auto_quantizer_mapping(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=True)

        self.assertIsInstance(quantizer, GemLiteQuantizer)
        self.assertTrue(quantizer.pre_quantized)

    def test_rejects_on_the_fly_quantization(self):
        quantizer = DiffusersAutoQuantizer.from_config(GemLiteConfig(), pre_quantized=False)

        with self.assertRaisesRegex(ValueError, "only supports loading already-quantized checkpoints"):
            quantizer.validate_environment()


if __name__ == "__main__":
    unittest.main()
