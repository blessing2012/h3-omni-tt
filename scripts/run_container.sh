#!/usr/bin/env bash
set -euo pipefail
handoff_root="$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-check}"
image="vllm/vllm-omni@sha256:6f8be103eaf0055448cf7578cfd621405fd669079d4361bd58896326b2bf722a"
args=(run --rm --ipc=host --workdir /workspace/h3
  --mount "type=bind,source=${handoff_root},target=/workspace/h3,readonly"
  -e PYTHONPATH=/workspace/h3/vendor/vllm-omni -e PYTHONDONTWRITEBYTECODE=1)
case "${mode}" in
  check) args+=(-e CUDA_VISIBLE_DEVICES=); command=(python3 /workspace/h3/scripts/check_runtime.py) ;;
  test) args+=(-e CUDA_VISIBLE_DEVICES=); command=(bash /workspace/h3/scripts/cpu_tests.sh) ;;
  serve)
    weights_dir="${WEIGHTS_DIR:-${handoff_root}/weights}"
    [[ -d "${weights_dir}" ]] || { echo "Download models first; missing ${weights_dir}" >&2; exit 1; }
    weights_dir="$(cd "${weights_dir}" && pwd)"
    args+=(--gpus all --name "${CONTAINER_NAME:-h3-vllm-omni-tt}"
      -p "${BIND_ADDRESS:-127.0.0.1}:26000:26000"
      --mount "type=bind,source=${weights_dir},target=/models,readonly")
    command=(bash /workspace/h3/scripts/serve.sh) ;;
  *) echo "Usage: $0 {check|test|serve}" >&2; exit 2 ;;
esac
exec docker "${args[@]}" --entrypoint "${command[0]}" "${image}" "${command[@]:1}"
