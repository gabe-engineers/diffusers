# Bonsai Omni Fix Log

This note captures the fixes needed to serve `gabe-engineers/bonsai-image-ternary-4B-gemlite-2bit` through:

- `diffusers`
- `transformers`
- `vllm-omni --diffusion-load-format diffusers`

## 1. Transformers: nested HQQ checkpoints

The Bonsai text encoder stores HQQ weights in the HQQ-native nested format:

```python
{
  "model.layers.0.self_attn.q_proj": {
    "W_q": ...,
    "scale": ...,
    "zero": ...,
    ...
  }
}
```

Transformers' HQQ loader expected flattened keys like:

```python
{
  "model.layers.0.self_attn.q_proj.W_q": ...,
  "model.layers.0.self_attn.q_proj.scale": ...,
  "model.layers.0.self_attn.q_proj.zero": ...,
}
```

Result before the fix:

- `q_proj` stayed `Linear`
- HQQ module blobs were `UNEXPECTED`
- flat `*.weight` params were `MISSING`

Fix:

- normalize nested HQQ checkpoints before the standard HF load path
- keep dtype inference working on nested checkpoint payloads

This fix lives in the Transformers fork on branch:

- `fix/hqq-nested-checkpoint-load`

## 2. Diffusers: GemLite tensors were not moved with `.to(device)`

GemLite `load_state_dict()` populates runtime tensors such as:

- `W_q`
- `scales`
- `zeros`
- `bias`

as plain attributes rather than registered parameters. After diffusers finalized the placeholder GemLite layer, later `.to("cuda")` calls could leave those tensors on CPU.

Result before the fix:

- Triton/GemLite would fail with pointer/device errors during execution

Fix in `src/diffusers/quantizers/gemlite/gemlite_quantizer.py`:

- re-register runtime tensors as frozen `nn.Parameter`s after finalize
- this makes subsequent `.to(device)` move the packed state correctly

## 3. Diffusers: GemLite compute dtype must stay fp16

The model's GemLite config uses fp16 compute. Some runtime paths still requested bf16, which caused Triton/GemLite dtype mismatches.

Result before the fix:

- `Both operands must be same dtype. Got bf16 and fp16`

Fix in `src/diffusers/quantizers/gemlite/gemlite_quantizer.py`:

- coerce the GemLite component dtype to `compute_dtype`
- for Bonsai, that means the GemLite transformer stays fp16

Recommended Omni flag:

```bash
--dtype half
```

so non-GemLite components also load as fp16.

## 4. Triton autotune compatibility for GemLite config pruners

GemLite's Triton config pruning path can yield a generator. Triton's autotuner later assumes the pruned config set is reusable.

Result before the fix:

- warmup failed with `ValueError: min() iterable argument is empty`

Fix:

- patch Triton's `Autotuner.prune_configs()` at interpreter startup
- if the result is an iterator, materialize it into a list

Repo files:

- `src/diffusers/utils/triton_compat.py`
- `src/sitecustomize.py`

This keeps the workaround narrow and automatic for the container runtime.

## 5. Container temp directory

Transformers' nested-HQQ normalization writes a temporary flattened checkpoint. On Runpod, `/tmp` may be on a small overlay filesystem and can fill up.

Result before the fix:

- temp save failures during text encoder load

Fix in the image:

- set `TMPDIR`, `TMP`, and `TEMP` to `/workspace/tmp`
- create that directory in the entrypoint

Repo files:

- `tmp/Dockerfile.runpod-bonsai`
- `scripts/runpod_bonsai_entrypoint.sh`

## Working serve command

After these fixes, the working Omni path is:

```bash
vllm serve gabe-engineers/bonsai-image-ternary-4B-gemlite-2bit \
  --omni \
  --diffusion-load-format diffusers \
  --dtype half \
  --port 8097
```

## Focused validations

Local:

```bash
PYTHONPATH=src tmp/.venv/bin/python -m unittest tests.others.test_triton_compat
PYTHONPATH=src tmp/.venv/bin/python -m unittest tests.quantization.gemlite.test_gemlite
```

Expected:

- Triton compat test passes
- GemLite quantizer tests pass

Runtime:

- text encoder loads as `HQQLinear`
- transformer loads as `GemLiteDiffusersLinear`
- Omni warmup completes
- API server starts successfully
