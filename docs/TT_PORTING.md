# Tenstorrent handoff

## Requested target

Port the pinned MiniMax-H3 FL2VA + LightX2V FL2V Turbo 4-step v1.2 model to TT
and provide an endpoint for comparative generation. The target is the selected
model and its inference semantics, not FastH3 and not a custom fine-tuned checkpoint.
768p real-time latency is not required for this initial port.

The repository includes a complete NVIDIA reference implementation. It does not
contain TT kernels, TT compiler configuration or a verified TT deployment.

## Preserve before optimizing

1. Download the pinned base and LoRA from `models/manifest.json`; verify hashes.
2. Preserve LoRA rank=128, alpha=8, request scale=1.0. Apply alpha/rank exactly
   once. Preserve projection name mappings and FFN gate/up packing in `lora.py`.
3. Reproduce the scheduler: 5 sigma points / 4 forward evaluations, video shift=6,
   audio shift=3. Read the pinned scheduler implementation; do not translate the
   string "4-step" into `num_inference_steps=4` in this vLLM API.
4. Preserve preprocessing, tokenization, text encoder, video/audio VAE, dimensions,
   frame alignment, audio generation and explicit seed. Equal seeds across
   platforms alone do not imply identical initial noise or bit-identical output.
5. Record any numeric precision, attention, caching or sampling changes. The
   reference uses BF16 model/adapter weights and cuDNN attention; it does not
   request an extra quantization mode. Inspect individual kernels' accumulation
   precision rather than assuming every operation is BF16.

## Useful source entry points

All paths below are relative to `vendor/vllm-omni/`:

- `vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py`
- `vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py`
- `vllm_omni/diffusion/models/minimax_h3/lora.py`
- `vllm_omni/diffusion/models/minimax_h3/scheduling_minimax_h3_euler_ancestral.py`
- `vllm_omni/diffusion/models/minimax_h3/` for supporting model components.
- `tests/diffusion/models/minimax_h3/test_minimax_h3_lora.py`

CUDA/cuDNN/Triton/NCCL-specific code will require an equivalent TT implementation
or substitution with validation. TP=4 / USP=2 / ring=1 is the NVIDIA reference
topology, not a mandatory TT chip allocation. Please record the actual TT device
type, chip count per request, parallel layout and concurrency.

## Initial comparison protocol

- First reproduce 5-second text-to-video, then a supplied shareable first-frame
  input. Extend to 10/15 seconds after these work.
- Use identical prompts/input bytes, dimensions, duration, adapter and sampling
  parameters. Keep reference assets and generated clips alongside their parameter
  records for manual quality assessment; no private user assets are supplied here.
- Report warm pipeline and end-to-end times separately; include device allocation,
  model/LoRA hashes, precision and actual output frame count/audio properties.
- Assess prompt following, first-frame consistency, motion, temporal artifacts
  and audio/visual synchronization. Do not claim parity from latency or one seed.
- Provide endpoint documentation, access method and an example request. Matching
  the native sync API is convenient; a documented adapter is also acceptable.

First/last-frame pairs and Ref2VA are not acceptance claims in this package.
The selected LoRA loader rejects Ref2VA; adding it requires a separate model and
adapter decision. No production gateway credentials or billing integration are
needed to implement this reference endpoint.
