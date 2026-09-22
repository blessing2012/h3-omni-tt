#!/usr/bin/env bash
set -euo pipefail
handoff_root="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${handoff_root}/vendor/vllm-omni${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
python3 "${handoff_root}/scripts/check_runtime.py"
cd "${handoff_root}/vendor/vllm-omni"
exec python3 - <<'PY'
import pytest
from vllm.lora import lora_model
# vLLM's CUDA wheel enables pinned host memory even when all GPUs are hidden.
# Disable ONLY this host-memory optimization in the CPU test process. The LoRA
# loader, alpha/rank arithmetic, packing and every assertion remain unchanged.
lora_model.PIN_MEMORY = False
raise SystemExit(pytest.main([
    "-q", "-o", "addopts=", "-p", "no:cacheprovider",
    "tests/diffusion/models/minimax_h3/test_minimax_h3_lora.py",
]))
PY
