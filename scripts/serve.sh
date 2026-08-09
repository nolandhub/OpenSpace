#!/usr/bin/env bash
# ABOUTME: Build image rồi chạy Triton server với model_repository của project
# ABOUTME: Truyền tên model vào để chỉ load riêng model đó, ví dụ: ./scripts/serve.sh asr_encoder

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker build -t triton-voice -f "$ROOT/docker/Dockerfile" "$ROOT/docker"

ARGS=(--model-repository=/models)
if [ $# -gt 0 ]; then
  # Chế độ explicit: chỉ load đúng model được chỉ định, tiện lúc debug
  ARGS+=(--model-control-mode=explicit)
  for m in "$@"; do ARGS+=(--load-model="$m"); done
fi

# Chỉ cấp tty khi chạy tương tác, để script tự động hoá gọi được
TTY=()
[ -t 0 ] && TTY=(-it)

docker run --gpus all --rm "${TTY[@]}" --net host --shm-size 1g \
  --name triton-voice-server \
  -v "$ROOT/model_repository:/models" \
  triton-voice tritonserver "${ARGS[@]}"
