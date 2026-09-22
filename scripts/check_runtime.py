#!/usr/bin/env python3
"""Fail if Python imports the image's unmodified package instead of this source."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from vllm_omni.diffusion.models.minimax_h3 import lora
    provenance = json.loads((ROOT / "provenance.json").read_text())
    relative = "vllm_omni/diffusion/models/minimax_h3/lora.py"
    expected = (ROOT / "vendor/vllm-omni" / relative).resolve()
    actual = Path(lora.__file__).resolve()
    if actual != expected:
        raise RuntimeError(f"Wrong imported source: {actual}; expected {expected}. Set PYTHONPATH.")
    if hashlib.sha256(actual.read_bytes()).hexdigest() != provenance["source_files_from_leo"][relative]:
        raise RuntimeError("LoRA loader differs from the captured handoff version")
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    if lora._TURBO_ARTIFACT_ALPHAS.get(manifest["lora"]["filename"]) != 8 or lora._TURBO_RANK != 128:
        raise RuntimeError("Missing v1.2 alpha=8 / rank=128 support")
    versions = {x: importlib.metadata.version(x) for x in provenance["observed_packages"]}
    if versions != provenance["observed_packages"]:
        raise RuntimeError(f"Dependency versions differ from the pinned image: {versions}")
    print(json.dumps({"runtime_check": "passed", "lora_alpha": 8, "lora_rank": 128,
                      "loaded_from_handoff_source": True, "versions": versions}))


if __name__ == "__main__":
    main()
