#!/usr/bin/env python3
"""One explicit Turbo request; writes MP4 + timing metadata. No automatic retries."""
import argparse
import json
import mimetypes
import time
from contextlib import ExitStack
from pathlib import Path

import requests

LORA_NAME = "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"


def build_form(args):
    return {
        "prompt": args.prompt,
        "width": "1344", "height": "768", "fps": "24",
        "num_inference_steps": "5", "flow_shift": "6.0", "seed": str(args.seed),
        "extra_params": json.dumps({"task": "fl2va", "duration": args.duration, "audio_flow_shift": 3.0}),
        "lora": json.dumps({"name": "h3-turbo-v1.2", "path": args.server_lora_path, "scale": 1.0}),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:26000")
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", type=Path, help="Optional first-frame image; omit for text-to-video")
    p.add_argument("--duration", type=int, choices=[5, 10, 15], default=5)
    p.add_argument("--seed", type=int, default=2101)
    p.add_argument("--server-lora-path", default="/models/Minimax-h3-Turbo/" + LORA_NAME,
                   help="Path as seen by the SERVER, not by this client")
    p.add_argument("--output", type=Path, default=Path("outputs/example.mp4"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    form = build_form(args)
    if args.dry_run:
        print(json.dumps({"endpoint": args.url.rstrip("/") + "/v1/videos/sync", "form": form,
                          "first_frame": str(args.image) if args.image else None}, indent=2))
        return
    with ExitStack() as stack:
        session = stack.enter_context(requests.Session())
        session.trust_env = False
        files = None
        if args.image:
            files = {"input_reference": (args.image.name, stack.enter_context(args.image.open("rb")),
                     mimetypes.guess_type(args.image.name)[0] or "application/octet-stream")}
        started = time.perf_counter()
        response = session.post(args.url.rstrip("/") + "/v1/videos/sync", data=form, files=files,
                                headers={"Accept": "video/mp4"}, timeout=3000)
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        if not response.headers.get("content-type", "").startswith("video/mp4") or not response.content:
            raise RuntimeError("Expected a nonempty video/mp4 response")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(response.content)
        result = {"request": form, "client_elapsed_seconds": elapsed, "bytes": len(response.content),
                  "server_headers": {k: v for k, v in response.headers.items() if k.lower().startswith("x-")}}
        args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {args.output}; request-to-download elapsed {elapsed:.2f}s")


if __name__ == "__main__":
    main()
