from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...utils import get_module_from_name, is_gemlite_available, is_torch_available, logging
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin

if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)


GEMLITE_STATE_NAMES = ("W_q", "bias", "scales", "zeros", "metadata", "orig_shape", "meta_scale")


def _is_in_skip_modules(name: str, modules_to_not_convert: list[str]) -> bool:
    return any((key + "." in name) or (key == name) for key in modules_to_not_convert)


def _replace_with_gemlite_linear(model: "ModelMixin", modules_to_not_convert: list[str]) -> int:
    from gemlite.core import GemLiteLinearTriton

    def replace(module: "nn.Module", prefix: str = "") -> int:
        replaced = 0
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not _is_in_skip_modules(child_name, modules_to_not_convert):
                gemlite_linear = GemLiteLinearTriton().to(child.weight.device)
                if child.bias is None:
                    gemlite_linear.bias = None
                gemlite_linear._gemlite_loaded_param_names = set()
                setattr(module, name, gemlite_linear)
                replaced += 1
            else:
                replaced += replace(child, child_name)
        return replaced

    return replace(model)


class GemLiteQuantizer(DiffusersQuantizer):
    """
    Diffusers quantizer for GemLite checkpoints serialized from GemLiteLinear modules.

    This integration only supports loading already-quantized checkpoints. It replaces `torch.nn.Linear` modules with
    `GemLiteLinearTriton` modules before weight loading so GemLite state-dict keys such as `W_q`, `scales`, `zeros`,
    `metadata`, `orig_shape`, and `meta_scale` are recognized by Diffusers' low-memory loader.
    """

    requires_calibration = False
    required_packages = ["gemlite"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        self.compute_dtype = quantization_config.compute_dtype
        self.modules_to_not_convert = quantization_config.modules_to_not_convert or []
        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

    def validate_environment(self, *args, **kwargs):
        if not self.pre_quantized:
            raise ValueError(
                "GemLite quantization in Diffusers only supports loading already-quantized checkpoints. "
                "Please load a checkpoint whose config contains a GemLite `quantization_config`."
            )
        if not is_gemlite_available():
            raise ImportError(
                "Loading a GemLite quantized model requires the gemlite library. Please install it with "
                "`pip install gemlite`."
            )
        try:
            from gemlite.core import GemLiteLinearTriton  # noqa: F401
        except Exception as error:
            raise ImportError("GemLite is installed but its core linear module could not be imported.") from error

    def update_torch_dtype(self, torch_dtype: "torch.dtype" = None) -> "torch.dtype":
        if torch_dtype is None:
            return self.compute_dtype
        if torch_dtype != self.compute_dtype:
            logger.info(
                "Overriding torch_dtype=%s with `torch_dtype=%s` to match GemLite compute dtype.",
                torch_dtype,
                self.compute_dtype,
            )
            return self.compute_dtype
        return torch_dtype

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        from gemlite.core import GemLiteLinearTriton

        module, tensor_name = get_module_from_name(model, param_name)
        return tensor_name in GEMLITE_STATE_NAMES and isinstance(module, GemLiteLinearTriton)

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: dict[str, Any] | None = None,
        unexpected_keys: list[str] | None = None,
        **kwargs,
    ):
        module, tensor_name = get_module_from_name(model, param_name)
        if tensor_name not in GEMLITE_STATE_NAMES:
            raise ValueError(f"`{param_name}` is not a GemLite serialized tensor.")

        setattr(module, tensor_name, param_value.to(target_device))
        module._gemlite_loaded_param_names.add(tensor_name)

    def check_quantized_param_shape(self, *args, **kwargs):
        return True

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        return [key for key in missing_keys if not key.endswith(".meta_scale")]

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: list[str] = [],
        **kwargs,
    ):
        self.modules_to_not_convert.extend(keep_in_fp32_modules)
        self.modules_to_not_convert = [module for module in self.modules_to_not_convert if module is not None]

        replaced = _replace_with_gemlite_linear(model, self.modules_to_not_convert)
        if replaced == 0:
            logger.warning("No linear modules were replaced with GemLite linear layers.")

        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "ModelMixin", **kwargs):
        from gemlite.core import GemLiteLinearTriton

        for module in model.modules():
            if isinstance(module, GemLiteLinearTriton):
                state_dict = {
                    name: getattr(module, name)
                    for name in GEMLITE_STATE_NAMES
                    if name in module._gemlite_loaded_param_names
                    if getattr(module, name) is not None
                }
                module.load_state_dict(state_dict)
        return model

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True
