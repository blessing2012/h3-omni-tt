# Validation scope — 2026-09-22

## Passed

- Compared all **3,452 unmodified upstream files** against the archive for commit
  `039808e0d97d7969a2cb102d074c1e45b8ceef0b`. The inference loader and its tests match
  the source checkout byte-for-byte. The two added clients differ only in the
  portable default model directory documented in `CHANGES.md`. No other local
  implementation delta is bundled.
- Parsed the captured/new Python files and checked shell syntax.
- Checked the actual import path and dependency versions in the existing Linux
  `vllm/vllm-omni:v0.28.0` environment with GPUs hidden. The patched v1.2 loader
  was imported from the source checkout and reported alpha=8 / rank=128.
- Ran the captured LoRA test module: **30 passed**. This covers v1.0/v1.2 loading,
  effective alpha/rank updates, packed projections, manager scaling and invalid
  artifact rejection using tiny CPU tensors.
- The source-host v1.2 LoRA's full SHA-256 equals the published Hugging Face LFS
  SHA-256 in the pinned revision. All 81 base-partition files match the public
  names and sizes; small files additionally match Git blob hashes.
- The request example dry-run emits explicit Turbo/seed/duration settings.

## CPU harness details

The CUDA wheel's LoRA loader requests pinned host memory even with all GPUs hidden.
An initial direct CPU run therefore failed at `.pin_memory()`. The supplied CPU
harness disables only `vllm.lora.lora_model.PIN_MEMORY` in that test process;
LoRA arithmetic, target conversion and assertions remain unchanged. This is not
a serving/inference modification. Pytest's inherited distributed-run options
are cleared because the runtime image does not include the xdist test plugin.

Warnings about the source snapshot's ungenerated `_version.py` (fallback `dev`),
pytest plugin rewrite order and unused `asyncio_mode` are non-failing. The pinned
commit, source hashes, image digest and installed distribution versions establish
the actual version; the fallback `dev` string is not used as provenance.

## Not established by this handoff

- No model inference, GPU kernels, quality comparison or latency benchmark was
  rerun. The GPUs remained unused by this work and the video maintenance hold
  was preserved. Existing containers/services were not restarted.
- The newly packaged Docker `serve` command was reviewed and its Python source
  selection was checked, but a fresh full model startup was not performed.
- Large base-model weight shards were **not** rehashed on the source machine.
  Their expected content hashes come from the pinned public Hub metadata.
  `download_models.py --full-hash` verifies all downloaded shards on the receiver.
- No TT-specific implementation or endpoint has been validated. Colleague reports
  of working quality/performance are not a new benchmark result in this repository.
- The second captured benchmark client's name/docstring and constants disagree;
  see `CHANGES.md`. No timing result is inferred from its filename.

Secret scanning is performed with Gitleaks v8.30.1. Eight narrowly fingerprinted
upstream false positives (Python symbols/function parameters and documentation)
are listed in `.gitleaksignore`; no changed file is exempted. They match the
unmodified public upstream snapshot. Production credentials and host operational
records are not included.
