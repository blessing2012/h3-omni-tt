# MiniMax-H3 / LightX2V Turbo — vLLM-Omni handoff

Reproducible NVIDIA reference source for Tenstorrent porting and evaluation.
This snapshot captures LinearGameAI's selected vLLM-Omni implementation and
LightX2V FL2V Turbo v1.2 configuration. It is **not FastH3 or Sol-H3**, and does
not contain an independently fine-tuned LinearGameAI checkpoint.

The immediate objective is to reproduce this model/configuration on TT and
expose a test endpoint. Real-time 768p generation is **not** an acceptance
requirement for this handoff. No new video-generation benchmark was run while
packaging it; see [validation scope](docs/VALIDATION.md).

## Exact components

| Component | Pinned selection |
|---|---|
| Base | `MiniMaxAI/MiniMax-H3`, native `FL2VA` partition |
| LoRA | `lightx2v/Minimax-h3-Turbo` / `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` |
| LoRA parameters | rank 128, alpha **8**, request scale 1.0 |
| Schedule | **5 sigma points = 4 denoiser evaluations**, video flow shift 6, audio shift 3 |
| Reference runtime | `vllm/vllm-omni:v0.28.0`, immutable digest in `provenance.json` |
| Source baseline | vLLM-Omni `039808e0d97d7969a2cb102d074c1e45b8ceef0b` plus captured changes |
| Parallel configuration | 8 NVIDIA GPUs; DiT TP=4 / Ulysses=2 / ring=1; text encoder TP=8; tiled VAE parallel=8 |
| Attention | `CUDNN_ATTN` |
| Example output | 1344×768, 24 FPS, 5/10/15 seconds, synchronized 32 kHz stereo audio |

The base model is about 144 GB, plus a 1.38 GB LoRA. Weights are downloaded
from pinned public revisions, not stored in Git. Their separate licenses apply.
See [model manifest](models/manifest.json) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Repository map

- `vendor/vllm-omni/`: complete upstream source snapshot with the four captured files applied; local model paths are sanitized.
- `patches/leo-h3-turbo.patch`: the delivered changes against the upstream commit, including portable model paths.
- `provenance.json`: source identity, changed-file hashes, image digest and observed dependency versions.
- `models/manifest.json`: immutable model revisions, file sizes and content hashes.
- `scripts/`: model retrieval, source/version checks, NVIDIA launcher and CPU tests.
- `examples/request.py`: explicit text-to-video or first-frame request and timing capture.
- [Changes](docs/CHANGES.md), [API](docs/API.md), [TT porting notes](docs/TT_PORTING.md).

## NVIDIA reference quick start

Use Linux x86-64 with Docker, the NVIDIA container runtime and eight suitable
GPUs for the supplied serving profile. Reserve at least 200 GB for weights,
plus space for the container image, caches and outputs.

```bash

# CPU tools only; no GPU inference.
python3 -m venv .venv
. .venv/bin/activate
pip install huggingface_hub requests
python scripts/download_models.py --full-hash

# These modes do not expose GPUs to the container.
bash scripts/run_container.sh check
bash scripts/run_container.sh test

# Starts the reference server and uses all eight GPUs. Run when they are available.
bash scripts/run_container.sh serve
```

`WEIGHTS_DIR=/absolute/path` can select another model storage directory. Keep
the directory layout produced by the downloader. The server mounts only this
repository and the model directory; it does not require a privileged container
or a whole-home-directory mount. It serves on host loopback port 26000 by default.
Set `BIND_ADDRESS` deliberately if a different network binding is needed.

The launcher explicitly sets `PYTHONPATH` to the vendored source and checks the
actual imported LoRA loader. **Starting the stock image alone is insufficient:**
its installed loader does not include the captured v1.2 alpha mapping.
Both the server's allowed LoRA and the request-time LoRA selection are explicit.

## Make one request

After the server reports healthy, in another terminal:

```bash
curl --fail http://127.0.0.1:26000/health

# Text-to-video, fixed seed and Turbo settings.
python examples/request.py \
  --prompt 'A red paper boat drifts across a shallow pond, gentle ripples and water sounds.' \
  --duration 5 --seed 2101 --output outputs/t2v.mp4

# First-frame image-to-video. Supply your own shareable image.
python examples/request.py --image /path/to/first-frame.png \
  --prompt 'Animate the scene with gentle movement and synchronized ambient sound.' \
  --duration 5 --seed 2101 --output outputs/i2v.mp4
```

Use `--dry-run` to inspect the request without contacting a service. The LoRA
path in the request is a **server-side** path. The example saves an MP4 and a
JSON record of parameters, end-to-end elapsed time and server timing headers.
It intentionally does not retry generation automatically.

This selected adapter covers T2VA/FL2VA. Ref2VA uses a different model partition
and adapter; it is outside this handoff profile. Full first-and-last-frame input
support must be validated separately; the example client sends at most one image.

## What TT should reproduce

Start with [TT_PORTING.md](docs/TT_PORTING.md). Preserve the checkpoint, alpha,
schedule, input processing and output media settings before optimizing. The CUDA
image is a reference environment, not a portable TT implementation. Endpoint
creation still requires TT's hardware adaptation and validation. The included
native vLLM API is the reference endpoint contract described in this repository.
