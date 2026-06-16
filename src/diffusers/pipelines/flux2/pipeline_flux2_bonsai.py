from __future__ import annotations

from pathlib import Path

import torch
from transformers import Qwen2TokenizerFast

from ...utils import logging, replace_example_docstring
from .bonsai_loading import (
    load_bonsai_scheduler,
    load_bonsai_text_encoder,
    load_bonsai_transformer,
    load_bonsai_vae,
    resolve_bonsai_snapshot_path,
)
from .pipeline_flux2_klein import EXAMPLE_DOC_STRING, Flux2KleinPipeline


logger = logging.get_logger(__name__)


class Flux2BonsaiPipeline(Flux2KleinPipeline):
    r"""
    A Bonsai-specific FLUX.2 Klein pipeline that manually loads the HQQ text encoder and GemLite transformer.

    This pipeline is an alternate load path for Bonsai repositories such as
    `gabe-engineers/bonsai-image-ternary-4B-gemlite-2bit`. It reuses [`Flux2KleinPipeline`] for inference while
    bypassing the generic quantizer stack during component construction.
    """

    @classmethod
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        r"""
        Instantiate a Bonsai FLUX.2 Klein pipeline from a local path or Hub repo.

        Unlike the generic [`DiffusionPipeline.from_pretrained`] path, this loader manually constructs the HQQ text
        encoder and GemLite transformer so Bonsai checkpoints can be loaded without the global quantizer stack.

        Examples:
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        token = kwargs.pop("token", None)

        torch_dtype = kwargs.pop("torch_dtype", None)
        device = kwargs.pop("device", None)
        text_encoder_backend = kwargs.pop("text_encoder_backend", "gemlite")
        transformer_dtype = kwargs.pop("transformer_dtype", torch_dtype or torch.float16)
        text_encoder_dtype = kwargs.pop("text_encoder_dtype", torch_dtype or torch.float16)
        vae_dtype = kwargs.pop("vae_dtype", torch_dtype or torch.bfloat16)
        is_distilled = kwargs.pop("is_distilled", False)

        ignored_keys = (
            "custom_pipeline",
            "device_map",
            "low_cpu_mem_usage",
            "quantization_config",
            "subfolder",
            "use_safetensors",
            "variant",
        )
        ignored_kwargs = {key: kwargs.pop(key) for key in ignored_keys if key in kwargs}
        if ignored_kwargs:
            logger.info(
                "Ignoring unsupported kwargs for Flux2BonsaiPipeline manual loading: %s",
                ", ".join(sorted(ignored_kwargs)),
            )
        if kwargs:
            logger.warning("Ignoring additional kwargs for Flux2BonsaiPipeline: %s", ", ".join(sorted(kwargs)))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        snapshot_path = resolve_bonsai_snapshot_path(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            revision=revision,
            token=token,
        )

        scheduler = load_bonsai_scheduler(snapshot_path)
        tokenizer = Qwen2TokenizerFast.from_pretrained(str(snapshot_path), subfolder="tokenizer")
        vae = load_bonsai_vae(snapshot_path, torch_dtype=vae_dtype)
        text_encoder = load_bonsai_text_encoder(
            snapshot_path,
            compute_dtype=text_encoder_dtype,
            device=device,
            backend=text_encoder_backend,
        )
        transformer = load_bonsai_transformer(
            snapshot_path,
            compute_dtype=transformer_dtype,
            device=device,
        )

        pipe = cls(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            is_distilled=is_distilled,
        )
        return pipe.to(device)
