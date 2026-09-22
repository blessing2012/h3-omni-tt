#!/usr/bin/env bash
set -euo pipefail
handoff_root="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${handoff_root}/vendor/vllm-omni${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
model_root="${MODEL_ROOT:-/models}"
python3 "${handoff_root}/scripts/check_runtime.py"
python3 "${handoff_root}/scripts/download_models.py" --root "${model_root}" --verify-only
exec vllm serve "${model_root}/MiniMax-H3/FL2VA" \
  --trust-remote-code --host 0.0.0.0 --port 26000 \
  --task-type fl2va --num-gpus 8 --tensor-parallel-size 4 \
  --usp 2 --ring 1 --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN --omni --lora-backend peft \
  --lora-path "${model_root}/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"
