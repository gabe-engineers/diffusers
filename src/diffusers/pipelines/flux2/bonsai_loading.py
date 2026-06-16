from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from ...models import AutoencoderKLFlux2, Flux2Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler


GEMLITE_STATE_NAMES = ("W_q", "bias", "scales", "zeros", "metadata", "orig_shape", "meta_scale")
DEFAULT_GEMLITE_SKIP_MODULES = [
    "proj_out",
    "x_embedder",
    "context_embedder",
    "time_guidance_embed",
    "norm_out",
    "double_stream_modulation_img",
    "double_stream_modulation_txt",
    "single_stream_modulation",
]


def resolve_bonsai_snapshot_path(
    pretrained_model_name_or_path: str | Path,
    *,
    cache_dir: str | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    revision: str | None = None,
    token: str | bool | None = None,
) -> Path:
    model_path = Path(pretrained_model_name_or_path)
    if model_path.exists():
        return model_path.resolve()

    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=str(pretrained_model_name_or_path),
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        revision=revision,
        token=token,
    )
    return Path(snapshot_path)


def _ensure_hqq_checkpoint_dir(text_encoder_dir: Path) -> tuple[Path, Path | None]:
    qmodel_path = text_encoder_dir / "qmodel.pt"
    if qmodel_path.exists():
        return text_encoder_dir, None

    pytorch_bin_path = text_encoder_dir / "pytorch_model.bin"
    if not pytorch_bin_path.exists():
        raise FileNotFoundError(
            f"Expected either `{qmodel_path.name}` or `{pytorch_bin_path.name}` in `{text_encoder_dir}`."
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="bonsai-hqq-"))
    shutil.copy2(text_encoder_dir / "config.json", temp_dir / "config.json")
    try:
        (temp_dir / "qmodel.pt").symlink_to(pytorch_bin_path)
    except OSError:
        shutil.copy2(pytorch_bin_path, temp_dir / "qmodel.pt")
    return temp_dir, temp_dir


def load_bonsai_text_encoder(
    snapshot_path: Path,
    *,
    compute_dtype,
    device: str,
    backend: str = "gemlite",
):
    try:
        if backend == "gemlite" and str(device).startswith("cuda"):
            from gemlite.core import set_packing_bitwidth

            set_packing_bitwidth(8)
        from hqq.models.hf.base import AutoHQQHFModel
        from hqq.utils.patching import prepare_for_inference
    except ImportError as error:
        raise ImportError(
            "Flux2BonsaiPipeline requires the `hqq` package to load the Bonsai text encoder."
        ) from error

    load_dir, temp_dir = _ensure_hqq_checkpoint_dir(snapshot_path / "text_encoder")
    try:
        model = AutoHQQHFModel.from_quantized(
            str(load_dir),
            compute_dtype=compute_dtype,
            device=device,
        )
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    inference_backend = backend if backend and str(device).startswith("cuda") else "default"
    prepare_for_inference(model, backend=inference_backend)
    model.eval()
    return model


def _is_in_skip_modules(name: str, modules_to_not_convert: list[str]) -> bool:
    if not modules_to_not_convert:
        return False
    components = name.split(".")
    return any(module_name in components or module_name in name for module_name in modules_to_not_convert)


def _replace_with_gemlite_linear(model, modules_to_not_convert: list[str]) -> int:
    import torch
    import torch.nn as nn
    from gemlite.core import GemLiteLinearTriton

    class BonsaiGemLiteLinear(GemLiteLinearTriton):
        def __init__(self, source_linear: nn.Linear):
            super().__init__()
            self._bonsai_is_placeholder = True
            self._bonsai_has_bias = source_linear.bias is not None
            device = source_linear.weight.device

            for param_name in ("W_q", "scales", "zeros", "metadata", "orig_shape", "meta_scale"):
                if hasattr(self, param_name) and param_name not in self._parameters:
                    delattr(self, param_name)
                self.register_parameter(param_name, nn.Parameter(torch.empty(0, device=device), requires_grad=False))

            if self._bonsai_has_bias:
                if hasattr(self, "bias") and "bias" not in self._parameters:
                    delattr(self, "bias")
                self.register_parameter("bias", nn.Parameter(torch.empty(0, device=device), requires_grad=False))
            else:
                self.bias = None

        def _bonsai_register_runtime_parameters(self):
            for name in ("W_q", "scales", "zeros", "bias"):
                value = getattr(self, name, None)
                if value is None or not isinstance(value, torch.Tensor):
                    continue
                if name in self._parameters:
                    del self._parameters[name]
                setattr(self, name, nn.Parameter(value.detach(), requires_grad=False))

        def _bonsai_finalize_from_loaded_state_dict(self):
            if not getattr(self, "_bonsai_is_placeholder", False):
                return

            state_dict = {
                name: param.detach()
                for name, param in self._parameters.items()
                if param is not None and param.numel() > 0
            }

            missing = [
                name for name in ("W_q", "scales", "zeros", "metadata", "orig_shape") if name not in state_dict
            ]
            if missing:
                raise ValueError(
                    "Cannot finalize a Bonsai GemLite layer because these tensors are missing: "
                    f"{missing}."
                )

            for name in GEMLITE_STATE_NAMES:
                if name in self._parameters:
                    del self._parameters[name]

            self.load_state_dict(state_dict)
            self._bonsai_register_runtime_parameters()
            self._bonsai_is_placeholder = False

    def replace(module, prefix: str = "") -> int:
        replaced = 0
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not _is_in_skip_modules(child_name, modules_to_not_convert):
                setattr(module, name, BonsaiGemLiteLinear(child))
                replaced += 1
            else:
                replaced += replace(child, child_name)
        return replaced

    return replace(model)


def _get_transformer_checkpoint_path(transformer_dir: Path) -> Path:
    candidates = (
        transformer_dir / "diffusion_pytorch_model.bin",
        transformer_dir / "state_dict.pt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find a Bonsai transformer checkpoint in `{transformer_dir}`.")


def _load_transformer_state_dict(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def load_bonsai_transformer(
    snapshot_path: Path,
    *,
    compute_dtype,
    device: str,
):
    import torch
    from accelerate import init_empty_weights

    if not str(device).startswith("cuda"):
        raise ValueError("Flux2BonsaiPipeline requires a CUDA device to load the GemLite Bonsai transformer.")

    try:
        import gemlite  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Flux2BonsaiPipeline requires the `gemlite` package to load the Bonsai transformer."
        ) from error

    transformer_dir = snapshot_path / "transformer"
    transformer_config = Flux2Transformer2DModel.load_config(transformer_dir)
    quantization_config = transformer_config.get("quantization_config", {})
    modules_to_not_convert = quantization_config.get("modules_to_not_convert") or DEFAULT_GEMLITE_SKIP_MODULES

    with init_empty_weights():
        transformer = Flux2Transformer2DModel.from_config(transformer_config)

    replaced = _replace_with_gemlite_linear(transformer, modules_to_not_convert)
    if replaced == 0:
        raise ValueError("No linear layers were replaced while preparing the Bonsai GemLite transformer.")

    state_dict = _load_transformer_state_dict(_get_transformer_checkpoint_path(transformer_dir))
    load_result = transformer.load_state_dict(state_dict, strict=False, assign=True)

    missing_keys = [key for key in load_result.missing_keys if not key.endswith(".meta_scale")]
    unexpected_keys = [key for key in load_result.unexpected_keys if not key.endswith(".meta_scale")]
    if missing_keys or unexpected_keys:
        raise ValueError(
            "Failed to load the Bonsai transformer checkpoint cleanly. "
            f"missing_keys={missing_keys[:10]}, unexpected_keys={unexpected_keys[:10]}"
        )

    for module in transformer.modules():
        finalize = getattr(module, "_bonsai_finalize_from_loaded_state_dict", None)
        if finalize is not None:
            finalize()

    transformer = transformer.to(device=device, dtype=compute_dtype)
    transformer._inference_dtype = compute_dtype
    transformer.eval()
    return transformer


def load_bonsai_scheduler(snapshot_path: Path):
    return FlowMatchEulerDiscreteScheduler.from_pretrained(str(snapshot_path), subfolder="scheduler")


def load_bonsai_vae(snapshot_path: Path, *, torch_dtype):
    return AutoencoderKLFlux2.from_pretrained(str(snapshot_path), subfolder="vae", torch_dtype=torch_dtype)
