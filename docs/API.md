# NVIDIA reference API

The native service uses `GET /health` and `POST /v1/videos/sync`. The latter waits
for generation and returns the MP4 body (`Content-Type: video/mp4`). This is a
reference evaluation API. It does not include a production billing gateway.
The launcher publishes it on loopback by default. External hosting, authentication
and any TT API adapter are the receiving deployment's responsibility.

The first-frame request is multipart form data with an `input_reference` file.
Text-to-video omits the file. `examples/request.py --dry-run` prints the exact form.

| Field | Selected value |
|---|---|
| `prompt` | Test prompt, unchanged between platforms |
| `width`, `height`, `fps` | `1344`, `768`, `24` |
| `seed` | Explicit, example `2101` |
| `num_inference_steps` | `5` sigma points, giving 4 denoiser evaluations |
| `flow_shift` | `6.0` |
| `extra_params` | JSON: `{"task":"fl2va","duration":5,"audio_flow_shift":3.0}` |
| `lora` | JSON: `{"name":"h3-turbo-v1.2","path":"/models/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors","scale":1.0}` |
| `input_reference` | Optional first-frame image file |

Duration 5/10/15 seconds is explicit in our examples. The pipeline internally
aligns frame counts, so report actual frame count and duration rather than assuming
exactly `seconds × 24` frames. The sync timeout is 3000 seconds; a timeout does not
prove a submitted job was cancelled. Do not blindly resubmit timed-out requests.

Timing headers, when present, include `X-Inference-Time-S`, `X-Stage-Durations`
and `X-Peak-Memory-Mb`. Compare inference and end-to-end latency separately, and
report cold load/compilation separately from warm requests. The client stores
all returned `X-*` headers without assuming a specific header is available.

Validate outputs with `ffprobe` for 1344×768, 24 FPS, 32 kHz stereo and actual
duration; fully decode both audio and video with `ffmpeg -v error -i output.mp4 -f null -`.
Use the captured batch client for its additional media checks if needed.
