#!/usr/bin/env python3
"""Run five varied five-second MiniMax-H3 Turbo FL2VA requests.

Every request uses a different prompt length and a differently sized version
of the same source image. The output remains fixed at 832x480, five seconds,
and 24 FPS. The script validates every MP4 and records per-case timing so cold
start and shape-dependent performance are easy to compare.

Requires: requests, ffmpeg, and ffprobe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests

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
DURATION_SECONDS = 10
FPS = 24
DEFAULT_RESOLUTION = (1344, 768)  # 16:9 aspect ratio, divisible by 32
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_IMAGE = REPO_ROOT / "i2va_testset/i2va_test_5_helicopter.jpg"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "minimax_h3_fl2va_turbo_5s_varied_benchmark"
VARIED_CASES = (
    {
        "label": "very_short",
        "input_size": (208, 120),
        "prompt": "Animate the scene gently with natural synchronized sound.",
    },
    {
        "label": "short",
        "input_size": (416, 240),
        "prompt": (
            "Animate the still scene with gentle subject motion, a slow camera push, "
            "stable composition, realistic lighting, and synchronized ambient sound."
        ),
    },
    {
        "label": "medium",
        "input_size": (624, 360),
        "prompt": (
            "Animate the still image as a polished cinematic shot. Preserve the subject's "
            "identity, colors, proportions, and background layout while introducing smooth "
            "natural movement. Let the camera move forward very slowly, add subtle parallax "
            "between foreground and background elements, keep edges stable, avoid flicker, "
            "and generate synchronized environmental sound that matches the visible motion."
        ),
    },
    {
        "label": "long",
        "input_size": (832, 480),
        "prompt": (
            "Create a realistic five-second cinematic animation from this first frame. Keep "
            "the main subject visually consistent in every frame, preserving its shape, "
            "texture, color palette, facial or structural details, and position within the "
            "composition. Introduce restrained, physically plausible motion rather than large "
            "changes. Use a slow and steady camera push with mild horizontal parallax, subtle "
            "depth changes, and stable perspective. Background elements may move slightly in "
            "response to wind or environmental activity, but they must not warp, duplicate, "
            "or disappear. Maintain coherent illumination, realistic shadows, clean edges, "
            "and temporal stability without flicker. Add stereo environmental ambience and "
            "small synchronized sound details that correspond precisely to visible motion."
        ),
    },
    {
        "label": "very_long",
        "input_size": (1248, 720),
        "prompt": (
            "Transform this single first frame into a carefully controlled five-second "
            "cinematic sequence while treating the source image as the authoritative visual "
            "reference. Preserve the exact identity and recognizable appearance of the main "
            "subject, including its geometry, proportions, materials, colors, fine textures, "
            "and relationship to every nearby object. Begin with an almost imperceptible pause, "
            "then introduce smooth, physically plausible movement that develops gradually and "
            "never becomes abrupt. The camera should execute a slow forward drift combined with "
            "a very small lateral move, producing convincing foreground and background parallax "
            "while retaining the original framing. Keep the horizon, perspective, object "
            "boundaries, and lighting direction stable. Allow secondary elements to respond "
            "naturally through subtle wind, reflections, shifting highlights, or tiny "
            "environmental motion, but do not invent new subjects or remove existing details. "
            "Avoid morphing, duplication, unstable anatomy, texture crawling, exposure pulses, "
            "camera shake, sudden cuts, and temporal flicker. Preserve sharp but natural image "
            "detail throughout the sequence. Generate a coherent 32 kHz stereo soundtrack with "
            "quiet environmental ambience, gentle spatial depth, and precisely timed sound cues "
            "for each visible movement, without speech, music, clipping, or abrupt changes in "
            "loudness. End on a stable frame that remains compositionally consistent with the "
            "original image and the preceding motion."
        ),
    },
)


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
    result = subprocess.run(
        [
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )
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
    result = subprocess.run(
        [
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )
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
            f"generated audio is {sample_rate} Hz with {channels} channel(s), expected 32000 Hz stereo"
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
        "lora": json.dumps(
            {
                "name": args.lora_name,
                "path": str(args.lora_path),
                "scale": args.lora_scale,
            },
            separators=(",", ":"),
        ),
    }
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    print(
        f"Generating FL2VA {duration}s at {width}x{height}, {args.fps} FPS, "
        f"{args.steps} sigma points / {TURBO_DENOISER_EVALUATIONS} denoiser evaluations ...",
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
            "turbo": True,
            "denoiser_evaluations": TURBO_DENOISER_EVALUATIONS,
            "lora": {
                "name": args.lora_name,
                "path": str(args.lora_path),
                "scale": args.lora_scale,
            },
        },
        "timing": {
            "health_check_seconds": None,
            "client_end_to_end_seconds": round(client_elapsed_seconds, 6),
            "server_inference_seconds": server_inference_seconds,
            "client_overhead_seconds": (
                round(client_overhead_seconds, 6) if client_overhead_seconds is not None else None
            ),
            "stage_durations": stage_durations,
        },
        "server": {
            "request_id": response.headers.get("x-request-id"),
            "model": response.headers.get("x-model"),
            "peak_memory_mb": peak_memory_mb,
        },
        "response_headers": {
            key: value for key, value in response.headers.items() if key.lower().startswith("x-")
        },
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "megabytes": round(output.stat().st_size / 1_000_000, 3),
            "media_duration_seconds": float(media_metadata["format"]["duration"]),
        },
        "audio_validation": verified_audio,
        "ffprobe": media_metadata,
    }
    print(f"  client end-to-end: {client_elapsed_seconds:.3f}s")
    if server_inference_seconds is not None:
        print(f"  server inference:  {server_inference_seconds:.3f}s")
    if peak_memory_mb is not None:
        print(f"  peak GPU memory:   {peak_memory_mb:.1f} MiB")
    print(f"  output:            {output} ({output.stat().st_size / 1_000_000:.2f} MB)")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send five MiniMax-H3 Turbo FL2VA requests with increasing prompt "
            "lengths and input-image sizes."
        )
    )
    parser.add_argument("--url", default="http://127.0.0.1:26000")
    parser.add_argument(
        "--input-image",
        type=Path,
        default=DEFAULT_INPUT_IMAGE,
        help="Source image used to create all five input-size variants",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=DEFAULT_RESOLUTION,
        metavar="WIDTHxHEIGHT",
        help="Output resolution (default: 832x480)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_TURBO_LORA_PATH)
    parser.add_argument("--lora-name", default="h3-turbo-v1.2")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=3000)
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Honor HTTP(S)_PROXY from the environment",
    )
    args = parser.parse_args()

    args.input_image = args.input_image.expanduser().resolve()
    if not args.input_image.is_file():
        parser.error(f"--input-image does not exist or is not a file: {args.input_image}")
    args.input_image_mime_type = IMAGE_MIME_TYPES.get(args.input_image.suffix.lower())
    if args.input_image_mime_type is None:
        parser.error("--input-image must be JPG, JPEG, PNG, WEBP, HEIC, or HEIF")
    if args.input_image.stat().st_size <= 0:
        parser.error("--input-image must not be empty")
    if args.input_image.stat().st_size > MAX_IMAGE_BYTES:
        parser.error("--input-image must not exceed 30 MiB")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    args.lora_path = args.lora_path.expanduser().resolve()
    if not args.lora_path.is_file():
        parser.error(f"--lora-path does not exist or is not a file: {args.lora_path}")
    if args.lora_path.name != SUPPORTED_TURBO_LORA_FILENAME:
        parser.error(
            f"Turbo benchmark supports only {SUPPORTED_TURBO_LORA_FILENAME}; "
            f"got {args.lora_path.name}"
        )
    if not args.lora_name.strip():
        parser.error("--lora-name must not be empty")
    if not math.isfinite(args.lora_scale) or args.lora_scale <= 0:
        parser.error("--lora-scale must be finite and positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def latency_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "min_seconds": round(min(values), 6),
        "mean_seconds": round(statistics.fmean(values), 6),
        "p50_seconds": round(statistics.median(values), 6),
        "p95_seconds": round(percentile_nearest_rank(values, 0.95), 6),
        "max_seconds": round(max(values), 6),
    }


def display_seconds(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "-"


def prompt_stats(prompt: str) -> dict[str, int]:
    return {"characters": len(prompt), "words": len(prompt.split())}


def create_input_variant(source: Path, output: Path, width: int, height: int) -> None:
    """Resize and center-crop without distorting the source image."""
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            (
                f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={width}:{height}"
            ),
            "-frames:v",
            "1",
            str(output),
        ],
        check=True,
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg did not create input variant: {output}")


def main() -> int:
    args = parse_args()
    print(f"Request LoRA: name={args.lora_name}, path={args.lora_path}, scale={args.lora_scale}", flush=True)
    missing_tools = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing_tools:
        print(f"Required executable(s) not found: {', '.join(missing_tools)}", file=sys.stderr)
        return 2

    try:
        source_image_info = probe_input_image(args.input_image)
    except (OSError, subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Input image validation failed: {exc}", file=sys.stderr)
        return 2

    width, height = args.resolution
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "input_variants"
    cases: list[dict[str, Any]] = []
    try:
        for index, template in enumerate(VARIED_CASES, start=1):
            input_width, input_height = template["input_size"]
            image_path = input_dir / f"input_{index:02d}_{input_width}x{input_height}.png"
            create_input_variant(args.input_image, image_path, input_width, input_height)
            cases.append(
                {
                    **template,
                    "index": index,
                    "image_path": image_path,
                    "image_info": probe_input_image(image_path),
                    "prompt_stats": prompt_stats(template["prompt"]),
                }
            )
    except (OSError, subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Failed to create input-image variants: {exc}", file=sys.stderr)
        return 2

    base_url = args.url.rstrip("/")
    session = requests.Session()
    session.trust_env = args.use_env_proxy

    print(f"Checking {base_url}/health ...", flush=True)
    try:
        health_seconds = check_health(session, base_url)
    except requests.RequestException as exc:
        print(f"Server health check failed: {exc}", file=sys.stderr)
        return 2

    print(f"Source image: {args.input_image}")
    print(
        f"Fixed output: {width}x{height}, {DURATION_SECONDS}s, {FPS} FPS, "
        f"{TURBO_SIGMA_POINTS} sigma points / {TURBO_DENOISER_EVALUATIONS} denoiser evaluations"
    )
    print("Five calls use increasing prompt lengths and uploaded input-image sizes.")
    print("Call 1 is the cold-start candidate after a server restart.\n")

    request_args = SimpleNamespace(
        prompt="",
        input_image=None,
        input_image_mime_type="image/png",
        fps=FPS,
        steps=TURBO_SIGMA_POINTS,
        flow_shift=TURBO_FLOW_SHIFT,
        audio_flow_shift=3.0,
        seed=args.seed,
        timeout=args.timeout,
        turbo=True,
        lora_path=args.lora_path,
        lora_name=args.lora_name,
        lora_scale=args.lora_scale,
    )

    started_at = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []
    for case in cases:
        index = case["index"]
        label = case["label"]
        input_width, input_height = case["input_size"]
        request_args.prompt = case["prompt"]
        request_args.input_image = case["image_path"]
        output = output_dir / f"case_{index:02d}_{label}_{input_width}x{input_height}.mp4"
        timing_json = output.with_suffix(".mp4.timing.json")
        print(
            f"[{index}/{len(cases)}] {label}: input={input_width}x{input_height}, "
            f"prompt={case['prompt_stats']['words']} words / "
            f"{case['prompt_stats']['characters']} characters",
            flush=True,
        )
        try:
            report = run_case(
                session,
                request_args,
                base_url=base_url,
                duration=DURATION_SECONDS,
                width=width,
                height=height,
                output=output,
                input_image_info=case["image_info"],
            )
            report["benchmark"] = {
                "index": index,
                "label": label,
                "cold_start_candidate": index == 1,
                "prompt_stats": case["prompt_stats"],
                "input_size": {"width": input_width, "height": input_height},
            }
            report["timing"]["health_check_seconds"] = round(health_seconds, 6)
            write_json(timing_json, report)
            results.append(
                {
                    "index": index,
                    "label": label,
                    "cold_start_candidate": index == 1,
                    "prompt": case["prompt"],
                    "prompt_stats": case["prompt_stats"],
                    "input_image": case["image_info"],
                    "status": "passed",
                    "client_end_to_end_seconds": report["timing"]["client_end_to_end_seconds"],
                    "server_inference_seconds": report["timing"]["server_inference_seconds"],
                    "stage_durations": report["timing"]["stage_durations"],
                    "peak_memory_mb": report["server"]["peak_memory_mb"],
                    "output": str(output),
                    "timing_json": str(timing_json),
                }
            )
        except (OSError, subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"Case {index} ({label}) failed: {exc}", file=sys.stderr)
            failure = {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "index": index,
                "label": label,
                "cold_start_candidate": index == 1,
                "prompt": case["prompt"],
                "prompt_stats": case["prompt_stats"],
                "input_image": case["image_info"],
                "status": "failed",
                "error": str(exc),
                "output": str(output),
                "timing_json": str(timing_json),
            }
            write_json(timing_json, failure)
            results.append(failure)

    passed = [result for result in results if result["status"] == "passed"]
    client_values = [float(result["client_end_to_end_seconds"]) for result in passed]
    server_values = [
        float(result["server_inference_seconds"])
        for result in passed
        if result.get("server_inference_seconds") is not None
    ]
    client_mean = statistics.fmean(client_values) if client_values else None
    aggregate = {
        "client_latency": latency_stats(client_values),
        "server_inference_latency": latency_stats(server_values),
        "mean_realtime_factor": (
            round(client_mean / DURATION_SECONDS, 6) if client_mean is not None else None
        ),
        "mean_generated_video_seconds_per_wall_second": (
            round(DURATION_SECONDS / client_mean, 6)
            if client_mean is not None and client_mean > 0
            else None
        ),
    }
    summary = {
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": f"{base_url}/v1/videos/sync",
        "source_image": source_image_info,
        "fixed_output": {
            "width": width,
            "height": height,
            "duration_seconds": DURATION_SECONDS,
            "fps": FPS,
            "sigma_points": TURBO_SIGMA_POINTS,
            "denoiser_evaluations": TURBO_DENOISER_EVALUATIONS,
            "flow_shift": TURBO_FLOW_SHIFT,
            "seed": args.seed,
            "lora_path": str(args.lora_path),
            "lora_name": args.lora_name,
            "lora_scale": args.lora_scale,
        },
        "experiment": (
            "Prompt length and uploaded input-image size change together. The first call is "
            "also the cold-start candidate, so the run shows combined behavior rather than "
            "isolating one independent variable."
        ),
        "total_requests": len(cases),
        "passed": len(passed),
        "failed": len(results) - len(passed),
        "aggregate": aggregate,
        "results": results,
    }
    summary_path = output_dir / "benchmark_5s_varied_summary.json"
    write_json(summary_path, summary)

    print("\nFive varied 5-second FL2VA results")
    print("run  label       input       words  chars  client_s  server_s  peak_mb  status")
    for result in results:
        image_info = result["input_image"]
        pstats = result["prompt_stats"]
        print(
            f"{result['index']:>3}  {result['label']:<10}  "
            f"{image_info['width']:>4}x{image_info['height']:<4}  "
            f"{pstats['words']:>5}  {pstats['characters']:>5}  "
            f"{display_seconds(result.get('client_end_to_end_seconds')):>8}  "
            f"{display_seconds(result.get('server_inference_seconds')):>8}  "
            f"{display_seconds(result.get('peak_memory_mb')):>7}  "
            f"{result['status']}"
        )
    if aggregate["client_latency"] is not None:
        stats = aggregate["client_latency"]
        print(
            "All-case client latency (cases are intentionally different): "
            f"mean={stats['mean_seconds']:.3f}s, p50={stats['p50_seconds']:.3f}s, "
            f"p95={stats['p95_seconds']:.3f}s"
        )
        print(f"Mean real-time factor: {aggregate['mean_realtime_factor']:.3f}x")
    print(f"Summary: {summary_path}")
    return 0 if len(passed) == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
