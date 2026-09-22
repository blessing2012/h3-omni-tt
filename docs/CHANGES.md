# Source changes and attribution

Baseline: https://github.com/vllm-project/vllm-omni/commit/039808e0d97d7969a2cb102d074c1e45b8ceef0b

The supplied source checkout had two modified tracked files and two added
Python clients. The inference loader and its tests are captured byte-for-byte.
The two clients have only their source-machine model path replaced by a portable
`/models/Minimax-h3-Turbo/` default. Original and delivered SHA-256 values are
recorded separately in `provenance.json`. The original Git history, editor state,
credentials, model binaries and machine configuration are not part of this repo.

## Inference change

`vllm_omni/diffusion/models/minimax_h3/lora.py` supports the published LightX2V
4-step v1.2 artifact in addition to v1.0. It selects and validates alpha by exact
filename: v1.0 uses 128; **v1.2 uses 8**. Rank stays 128. It passes the correct
alpha into PEFT and the existing LoRA loader. Thus v1.2's intrinsic alpha/rank is
1/16 before the request scale; accidentally reusing alpha=128 would multiply
the adapter update by 16. This must survive any TT conversion or weight merging.

The existing target validation, attention projection mapping and gated-FFN
packing remain in place. The loader deliberately rejects this adapter for Ref2VA.
Pass the exact v1.2 file path: legacy directory selection still picks v1.0.

`tests/diffusion/models/minimax_h3/test_minimax_h3_lora.py` extends coverage to
both alpha values, effective matrix updates, packed FFN projections, rejection
cases and exactly-once scaling through the manager.

## Added clients

- `examples/online_serving/text_to_video/run_minimax_h3_fl2va.py`: batch first-frame
  generation, explicit Turbo selection and media/timing validation.
- `examples/online_serving/text_to_video/benchmark_minimax_h3_fl2va_5s.py`: varied
  prompts/input sizes. Its historical filename/docstring says 5s/480p, but its
  captured constants select **10s/1344×768**. It also defaults to an image that
  is not supplied in this handoff. Treat actual arguments and recorded request
  parameters as authoritative; use the root example for the initial comparison.

The delivered clients use the portable `/models/Minimax-h3-Turbo/` default;
override `--lora-path` and `--input-image` when using them elsewhere. The patch
also uses portable paths; machine-specific directories are not needed to run it.

## Packaging additions

The root documentation, model manifest/downloader, source checks, request example
and launch/test wrappers were added for this handoff. They do not modify model
weights or the vendored inference implementation. The launcher retains the
provided eight-GPU inference flags while explicitly selecting the local source.

The visible delta does **not** establish that all optimizations in the upstream
tree were authored by LinearGameAI. Existing TP/Ulysses, cuDNN attention, tiled
VAE and other kernels retain their upstream authorship. No additional private
kernel patch was found among the checkout's reported changes.
