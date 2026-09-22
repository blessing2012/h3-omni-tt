#!/usr/bin/env python3
"""Batch-generate MiniMax-H3 first-frame-to-video-and-audio clips.

The script sends one input image to an already-running vLLM-Omni FL2VA
service through ``POST /v1/videos/sync``. By default it generates 5 s, 10 s,
and 15 s clips at both 1344x768 and 832x480. Every MP4 is checked with
ffprobe for its canvas, 24 FPS video, and 32 kHz stereo audio. Pass
``--turbo`` to activate the supported MiniMax-H3 4-step Turbo LoRA.

Requires: requests and ffprobe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import requests


DEFAULT_PROMPT = (
    "Animate the scene with natural, physically plausible motion. The camera "
    "moves gently while the subject remains visually consistent with the first "
    "frame. Add synchronized environmental sound and subtle background ambience."
)
DURATIONS_SECONDS = (5, 10, 15)
DEFAULT_RESOLUTIONS = ((1344, 768), (832, 480))
DEFAULT_TURBO_LORA_PATH = Path(
    "/models/Minimax-h3-Turbo/"
    "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"
)
SUPPORTED_TURBO_LORA_FILENAME = "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"
TURBO_SIGMA_POINTS = 5
TURBO_DENOISER_EVALUATIONS = 4
TURBO_FLOW_SHIFT = 6.0
MAX_IMAGE_BYTES = 30 * 1024 * 1024
IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


def parse_resolution(value: str) -> tuple[int, int]:
    normalized = value.strip().lower().replace("*", "x")
    try:
        width_text, height_text = normalized.split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"resolution must use WIDTHxHEIGHT, for example 832x480; got {value!r}"
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution width and height must be positive")
    if width % 32 or height % 32:
        raise argparse.ArgumentTypeError("resolution width and height must be divisible by 32")
    if width > 4 * height or height > 4 * width:
        raise argparse.ArgumentTypeError("resolution aspect ratio must be between 1:4 and 4:1")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MiniMax-H3 FL2VA clips at 1344x768 and 832x480 for "
            "5 s, 10 s, and 15 s by default."
        )
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        required=True,
        help="First-frame image (JPG, PNG, WEBP, HEIC, or HEIF; at most 30 MiB)",
    )
    parser.add_argument("--url", default="http://127.0.0.1:26000", help="Server base URL")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Image animation prompt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("minimax_h3_fl2va_batch"),
        help="Directory for generated MP4 files and JSON reports",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Filename prefix; defaults to a mode-specific base or Turbo prefix",
    )
    parser.add_argument(
        "--resolution",
        dest="resolutions",
        action="append",
        type=parse_resolution,
        metavar="WIDTHxHEIGHT",
        help=(
            "Resolution to test; repeat for multiple resolutions. "
            "Defaults to 1344x768 and 832x480."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Legacy single-resolution width; must be used together with --height",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Legacy single-resolution height; must be used together with --width",
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Sigma points: defaults to 50 for base or 5 (4 denoiser evaluations) for Turbo",
    )
    parser.add_argument(
        "--duration",
        dest="durations",
        action="append",
        type=int,
        metavar="SECONDS",
        help="Duration to test; repeat for multiple durations. Defaults to 5, 10, and 15 seconds.",
    )
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=None,
        help="Video sigma shift: defaults to 12 for base or 6 for Turbo",
    )
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Activate the supported MiniMax-H3 FL2VA/T2VA 4-step Turbo LoRA",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=DEFAULT_TURBO_LORA_PATH,
        help="Server-visible path to the native Diffusers MiniMax-H3 Turbo LoRA",
    )
    parser.add_argument("--lora-name", default="h3-turbo-v1.2")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=3000,
        help="HTTP request timeout in seconds (match VLLM_OMNI_VIDEO_SYNC_TIMEOUT)",
    )
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Honor HTTP(S)_PROXY from the environment (disabled by default for local servers)",
    )
    args = parser.parse_args()

    input_image = args.input_image.expanduser()
    if not input_image.is_file():
        parser.error(f"--input-image does not exist or is not a file: {input_image}")
    image_mime_type = IMAGE_MIME_TYPES.get(input_image.suffix.lower())
    if image_mime_type is None:
        parser.error("--input-image must be JPG, JPEG, PNG, WEBP, HEIC, or HEIF")
    image_size = input_image.stat().st_size
    if image_size <= 0:
        parser.error("--input-image must not be empty")
    if image_size > MAX_IMAGE_BYTES:
        parser.error("--input-image must not exceed 30 MiB")
    args.input_image = input_image.resolve()
    args.input_image_mime_type = image_mime_type

    if args.turbo:
        turbo_lora_path = args.lora_path.expanduser()
        if not turbo_lora_path.is_file():
            parser.error(f"--lora-path does not exist or is not a file: {turbo_lora_path}")
        if turbo_lora_path.name != SUPPORTED_TURBO_LORA_FILENAME:
            parser.error(
                "MiniMax-H3 Turbo mode supports only "
                f"{SUPPORTED_TURBO_LORA_FILENAME}; got {turbo_lora_path.name}"
            )
        args.lora_path = turbo_lora_path.resolve()
        if args.steps is None:
            args.steps = TURBO_SIGMA_POINTS
        elif args.steps != TURBO_SIGMA_POINTS:
            parser.error(
                f"MiniMax-H3 4-step Turbo requires --steps {TURBO_SIGMA_POINTS} "
                f"({TURBO_DENOISER_EVALUATIONS} denoiser evaluations)"
            )
        if args.flow_shift is None:
            args.flow_shift = TURBO_FLOW_SHIFT
        elif args.flow_shift != TURBO_FLOW_SHIFT:
            parser.error(f"MiniMax-H3 4-step Turbo requires --flow-shift {TURBO_FLOW_SHIFT:g}")
        if not math.isfinite(args.lora_scale) or args.lora_scale <= 0:
            parser.error("--lora-scale must be finite and positive")
        if not args.lora_name.strip():
            parser.error("--lora-name must not be empty")
    else:
        args.steps = 50 if args.steps is None else args.steps
        args.flow_shift = 12.0 if args.flow_shift is None else args.flow_shift

    if args.output_prefix is None:
        args.output_prefix = "minimax_h3_fl2va_turbo" if args.turbo else "minimax_h3_fl2va"
    if args.fps <= 0 or args.steps <= 0:
        parser.error("--fps and --steps must be positive")
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be specified together")
    if args.resolutions and args.width is not None:
        parser.error("use either --resolution or --width/--height, not both")
    if args.width is not None and args.height is not None:
        try:
            args.resolutions = [parse_resolution(f"{args.width}x{args.height}")]
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    elif not args.resolutions:
        args.resolutions = list(DEFAULT_RESOLUTIONS)
    if not args.durations:
        args.durations = list(DURATIONS_SECONDS)
    if any(duration < 4 or duration > 15 for duration in args.durations):
        parser.error("--duration must be between 4 and 15 seconds")
    if args.fps != 24:
        parser.error("MiniMax-H3 output FPS is fixed at 24")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.output_prefix.strip():
        parser.error("--output-prefix must not be empty")
    return args


def check_health(session: requests.Session, base_url: str) -> float:
    started = time.perf_counter()
    response = session.get(f"{base_url.rstrip('/')}/health", timeout=10)
    response.raise_for_status()
    return time.perf_counter() - started


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_input_image(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    metadata: dict[str, Any] = json.loads(result.stdout)
    streams = metadata.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe could not decode an image stream from {path}")
    stream = streams[0]
    try:
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ffprobe returned invalid input-image dimensions") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"input image has invalid dimensions {width}x{height}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "codec": stream.get("codec_name"),
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
    }


def probe_output(
    output: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_fps: int,
    expected_duration: int,
) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_packets",
        "-show_entries",
        (
            "format=duration:stream=index,codec_type,codec_name,width,height,"
            "r_frame_rate,sample_rate,channels,nb_read_packets"
        ),
        "-of",
        "json",
        str(output),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    metadata: dict[str, Any] = json.loads(result.stdout)

    stream_types = {stream.get("codec_type") for stream in metadata.get("streams", [])}
    missing = {"video", "audio"} - stream_types
    if missing:
        raise RuntimeError(f"generated MP4 is missing required stream(s): {sorted(missing)}")

    video_stream = next(stream for stream in metadata["streams"] if stream.get("codec_type") == "video")
    try:
        actual_size = (int(video_stream.get("width", 0)), int(video_stream.get("height", 0)))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ffprobe returned invalid video dimensions") from exc
    if actual_size != (expected_width, expected_height):
        raise RuntimeError(
            f"generated video size is {actual_size[0]}x{actual_size[1]}, "
            f"expected {expected_width}x{expected_height}"
        )

    frame_rate = video_stream.get("r_frame_rate")
    try:
        actual_fps = float(Fraction(frame_rate))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"ffprobe returned an invalid video frame rate: {frame_rate!r}") from exc
    if abs(actual_fps - expected_fps) > 0.01:
        raise RuntimeError(f"generated video frame rate is {actual_fps:g}, expected {expected_fps}")

    audio_stream = next(stream for stream in metadata["streams"] if stream.get("codec_type") == "audio")
    try:
        sample_rate = int(audio_stream.get("sample_rate", 0))
        channels = int(audio_stream.get("channels", 0))
        packet_count = int(audio_stream.get("nb_read_packets", 0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ffprobe returned invalid audio metadata") from exc
    if (sample_rate, channels) != (32000, 2):
        raise RuntimeError(
            f"generated audio is {sample_rate} Hz with {channels} channel(s), "
            "expected 32000 Hz stereo"
        )
    if packet_count <= 0:
        raise RuntimeError("generated MP4 contains an audio stream but no audio packets")

    try:
        actual_duration = float(metadata.get("format", {}).get("duration"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ffprobe returned an invalid media duration") from exc
    if abs(actual_duration - expected_duration) > 0.75:
        raise RuntimeError(
            f"generated media duration is {actual_duration:.3f}s, "
            f"expected approximately {expected_duration}s"
        )
    return metadata


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_stage_durations(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise TypeError("expected a JSON object")
        return {str(key): float(duration) for key, duration in parsed.items()}
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Ignoring malformed X-Stage-Durations header: {exc}", file=sys.stderr)
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def audio_validation(metadata: dict[str, Any]) -> dict[str, Any]:
    audio_stream = next(stream for stream in metadata["streams"] if stream.get("codec_type") == "audio")
    return {
        "present": True,
        "codec": audio_stream.get("codec_name"),
        "sample_rate": int(audio_stream["sample_rate"]),
        "channels": int(audio_stream["channels"]),
        "packet_count": int(audio_stream["nb_read_packets"]),
    }


def run_case(
    session: requests.Session,
    args: argparse.Namespace,
    *,
    base_url: str,
    duration: int,
    width: int,
    height: int,
    output: Path,
    input_image_info: dict[str, Any],
) -> dict[str, Any]:
    endpoint = f"{base_url}/v1/videos/sync"
    extra_params = {
        "task": "fl2va",
        "duration": float(duration),
        "audio_flow_shift": args.audio_flow_shift,
    }
    form = {
        "prompt": args.prompt,
        "width": str(width),
        "height": str(height),
        "fps": str(args.fps),
        "num_inference_steps": str(args.steps),
        "flow_shift": str(args.flow_shift),
        "seed": str(args.seed),
        "extra_params": json.dumps(extra_params, separators=(",", ":")),
    }
    lora_request: dict[str, Any] | None = None
    if args.turbo:
        lora_request = {
            "name": args.lora_name,
            "path": str(args.lora_path),
            "scale": args.lora_scale,
        }
        form["lora"] = json.dumps(lora_request, separators=(",", ":"))
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    print(
        f"Generating FL2VA {duration}s at {width}x{height}, "
        f"{args.fps} FPS, {args.steps} sigma points"
        f"{' / 4 denoiser evaluations (Turbo)' if args.turbo else ''} ...",
        flush=True,
    )
    try:
        with args.input_image.open("rb") as image_file:
            response = session.post(
                endpoint,
                data=form,
                files={
                    "input_reference": (
                        args.input_image.name,
                        image_file,
                        args.input_image_mime_type,
                    )
                },
                headers={"Accept": "video/mp4"},
                timeout=args.timeout,
            )
        client_elapsed_seconds = time.perf_counter() - started
        response.raise_for_status()
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f"\nResponse: {exc.response.text[:1000]}"
        elapsed = time.perf_counter() - started
        raise RuntimeError(f"request failed after {elapsed:.3f}s: {exc}{detail}") from exc

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("video/mp4"):
        raise RuntimeError(f"unexpected content type {content_type!r}: {response.text[:1000]}")
    if not response.content:
        raise RuntimeError("server returned an empty MP4 response")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".part")
    temporary_output.write_bytes(response.content)
    temporary_output.replace(output)

    media_metadata = probe_output(
        output,
        expected_width=width,
        expected_height=height,
        expected_fps=args.fps,
        expected_duration=duration,
    )
    verified_audio = audio_validation(media_metadata)

    interesting_headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower().startswith("x-")
    }
    server_inference_seconds = parse_optional_float(response.headers.get("x-inference-time-s"))
    stage_durations = parse_stage_durations(response.headers.get("x-stage-durations"))
    peak_memory_mb = parse_optional_float(response.headers.get("x-peak-memory-mb"))
    client_overhead_seconds = (
        client_elapsed_seconds - server_inference_seconds
        if server_inference_seconds is not None
        else None
    )
    report = {
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "input_image": input_image_info,
        "request": {
            "prompt": args.prompt,
            "duration_seconds": duration,
            "width": width,
            "height": height,
            "fps": args.fps,
            "num_inference_steps": args.steps,
            "flow_shift": args.flow_shift,
            "audio_flow_shift": args.audio_flow_shift,
            "seed": args.seed,
            "task": "fl2va",
            "turbo": args.turbo,
            "denoiser_evaluations": (
                TURBO_DENOISER_EVALUATIONS if args.turbo else args.steps
            ),
            "lora": lora_request,
        },
        "timing": {
            "health_check_seconds": None,
            "client_end_to_end_seconds": round(client_elapsed_seconds, 6),
            "server_inference_seconds": server_inference_seconds,
            "client_overhead_seconds": (
                round(client_overhead_seconds, 6)
                if client_overhead_seconds is not None
                else None
            ),
            "stage_durations": stage_durations,
        },
        "server": {
            "request_id": response.headers.get("x-request-id"),
            "model": response.headers.get("x-model"),
            "peak_memory_mb": peak_memory_mb,
        },
        "response_headers": interesting_headers,
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "megabytes": round(output.stat().st_size / 1_000_000, 3),
            "media_duration_seconds": float(media_metadata["format"]["duration"]),
        },
        "audio_validation": verified_audio,
        "ffprobe": media_metadata,
    }
    print(f"{width}x{height} {duration}s generation completed successfully")
    print(f"  client end-to-end: {client_elapsed_seconds:.3f}s")
    if server_inference_seconds is not None:
        print(f"  server inference:  {server_inference_seconds:.3f}s")
        print(f"  client overhead:   {client_overhead_seconds:.3f}s")
    if stage_durations:
        for stage, stage_duration in stage_durations.items():
            unit = "ms" if stage.endswith("_ms") else "s"
            print(f"  {stage + ':':<19}{stage_duration:.3f}{unit}")
    if peak_memory_mb is not None:
        print(f"  peak GPU memory:   {peak_memory_mb:.1f} MiB")
    print(
        f"  audio:             {verified_audio['codec']}, "
        f"{verified_audio['sample_rate']} Hz, {verified_audio['channels']} channels"
    )
    print(f"  output:            {output} ({output.stat().st_size / 1_000_000:.2f} MB)")
    return report


def main() -> int:
    args = parse_args()
    if args.turbo:
        print(f"Request LoRA: name={args.lora_name}, path={args.lora_path}, scale={args.lora_scale}", flush=True)
    base_url = args.url.rstrip("/")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffprobe") is None:
        print("ffprobe is required to validate the input image and generated MP4 files", file=sys.stderr)
        return 2

    try:
        input_image_info = probe_input_image(args.input_image)
    except (OSError, subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Input image validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"Input image: {args.input_image} "
        f"({input_image_info['width']}x{input_image_info['height']}, "
        f"{input_image_info['codec']})"
    )
    if not 1.6 <= input_image_info["aspect_ratio"] <= 1.9:
        print(
            "Warning: the input image is not close to 16:9; explicit 1344x768 and "
            "832x480 outputs may crop or resize the first frame.",
            file=sys.stderr,
        )

    session = requests.Session()
    session.trust_env = args.use_env_proxy
    print(f"Checking {base_url}/health ...", flush=True)
    try:
        health_seconds = check_health(session, base_url)
    except requests.RequestException as exc:
        print(f"Server health check failed: {exc}", file=sys.stderr)
        return 2

    batch_started_at = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []
    failed = 0
    for width, height in args.resolutions:
        for duration in args.durations:
            case_name = f"{width}x{height}_{duration}s"
            output = output_dir / f"{args.output_prefix}_{case_name}.mp4"
            timing_json = output.with_suffix(output.suffix + ".timing.json")
            try:
                report = run_case(
                    session,
                    args,
                    base_url=base_url,
                    duration=duration,
                    width=width,
                    height=height,
                    output=output,
                    input_image_info=input_image_info,
                )
                report["timing"]["health_check_seconds"] = round(health_seconds, 6)
                write_json(timing_json, report)
                print(f"  timing report:     {timing_json}")
                results.append(
                    {
                        "duration_seconds": duration,
                        "resolution": {"width": width, "height": height},
                        "status": "passed",
                        "output": str(output),
                        "timing_json": str(timing_json),
                        "audio_validation": report["audio_validation"],
                        "media_duration_seconds": report["output"]["media_duration_seconds"],
                        "client_end_to_end_seconds": report["timing"]["client_end_to_end_seconds"],
                    }
                )
            except (OSError, subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as exc:
                failed += 1
                print(f"{case_name} generation failed: {exc}", file=sys.stderr)
                write_json(
                    timing_json,
                    {
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "failed",
                        "input_image": input_image_info,
                        "request": {
                            "duration_seconds": duration,
                            "width": width,
                            "height": height,
                            "task": "fl2va",
                            "turbo": args.turbo,
                            "num_inference_steps": args.steps,
                            "flow_shift": args.flow_shift,
                            "lora_path": str(args.lora_path) if args.turbo else None,
                        },
                        "output": {"path": str(output)},
                        "error": str(exc),
                    },
                )
                results.append(
                    {
                        "duration_seconds": duration,
                        "resolution": {"width": width, "height": height},
                        "status": "failed",
                        "output": str(output),
                        "timing_json": str(timing_json),
                        "error": str(exc),
                    }
                )

    total_cases = len(args.resolutions) * len(args.durations)
    summary = {
        "started_at_utc": batch_started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": f"{base_url}/v1/videos/sync",
        "input_image": input_image_info,
        "mode": "turbo_4step" if args.turbo else "base",
        "durations_seconds": list(args.durations),
        "resolutions": [
            {"width": width, "height": height}
            for width, height in args.resolutions
        ],
        "passed": len(results) - failed,
        "failed": failed,
        "all_audio_validated": failed == 0 and all(
            result.get("audio_validation", {}).get("present") is True for result in results
        ),
        "results": results,
    }
    summary_path = output_dir / f"{args.output_prefix}_batch_summary.json"
    write_json(summary_path, summary)
    print(f"Batch summary: {summary_path}")
    print(f"Passed: {summary['passed']}/{total_cases}; failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
