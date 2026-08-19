#!/usr/bin/env bash
# ABOUTME: Build image rồi chạy Triton server với model_repository của project
# ABOUTME: Truyền tên model vào để chỉ load riêng model đó, ví dụ: ./scripts/serve_triton.sh asr_streaming

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker build -t triton-voice -f "$ROOT/docker/Dockerfile" "$ROOT/docker"

# Mặc định Triton chỉ phơi counter cộng dồn, từ đó chỉ tính ra được MEAN latency.
# summary_latencies cho quantile thật; quantile viết kèm sai số cho phép.
ARGS=(--model-repository=/models
      --metrics-config summary_latencies=true
      --metrics-config 'summary_quantiles=0.5:0.05,0.95:0.01,0.99:0.001')
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
  -v "$ROOT/serving:/opt/serving/serving:ro" \
  triton-voice tritonserver "${ARGS[@]}"
