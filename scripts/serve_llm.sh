#!/usr/bin/env bash
# ABOUTME: Chạy vLLM (API OpenAI-compatible) trong container riêng, cạnh Triton trên cùng GPU
# ABOUTME: Dùng: ./scripts/serve_llm.sh | MODEL=Qwen/Qwen3-1.7B-AWQ GPU_FRACTION=0.6 ./scripts/serve_llm.sh

set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
PORT=${PORT:-8080}
IMAGE=${IMAGE:-vllm/vllm-openai:latest}
GPU_FRACTION=${GPU_FRACTION:-0.5}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
# Tắt CUDA graph để lấy lại vài trăm MiB - trên card 4GB đây không phải tuỳ chọn.
# Trên Thor thì đặt ENFORCE_EAGER=0, ở đó bộ nhớ dư và CUDA graph có lời thật.
ENFORCE_EAGER=${ENFORCE_EAGER:-1}

# Triton phải load xong TRƯỚC. vLLM đo bộ nhớ trống đúng lúc khởi động rồi
# preallocate KV cache và không trả lại. Dựng ngược thứ tự thì vLLM chiếm luôn
# phần Triton chưa kịp xin, và Triton chết giữa request chứ không chết lúc load.
if ! docker ps --format '{{.Names}}' | grep -qx triton-voice-server; then
  echo "cảnh báo: triton-voice-server chưa chạy. Chạy ./scripts/serve.sh trước," >&2
  echo "         đợi nó load xong cả 2 model rồi hãy dựng vLLM." >&2
fi

# Chặn sớm thay vì để vLLM chạy 60s rồi ném OOM khó đọc. GPU_FRACTION là phần
# của TỔNG VRAM, không phải phần còn trống - card 4GB với 0.5 là xin 2GB dù chỉ
# còn 1GB. Chừa headroom: số "đang dùng" của TTS là lúc nghỉ, chạy câu dài còn hơn.
read -r TOTAL FREE <<<"$(nvidia-smi --query-gpu=memory.total,memory.free \
  --format=csv,noheader,nounits | head -1 | tr -d ',')"
WANT=$(awk -v t="$TOTAL" -v f="$GPU_FRACTION" 'BEGIN { printf "%d", t * f }')
echo "VRAM: tổng ${TOTAL}MiB, trống ${FREE}MiB, vLLM xin ${WANT}MiB (GPU_FRACTION=$GPU_FRACTION)"
if [ "$WANT" -gt "$FREE" ]; then
  echo "dừng: xin ${WANT}MiB nhưng chỉ còn ${FREE}MiB." >&2
  echo "      Hạ GPU_FRACTION xuống $(awk -v f="$FREE" -v t="$TOTAL" \
    'BEGIN { printf "%.2f", (f * 0.9) / t }') hoặc dùng model nhỏ hơn." >&2
  exit 1
fi

EAGER=()
[ "$ENFORCE_EAGER" = 1 ] && EAGER=(--enforce-eager)

TTY=()
[ -t 0 ] && TTY=(-it)

mkdir -p "$HF_CACHE"

# -p thay vì --net host như serve.sh: vLLM chỉ cần đúng một cổng, mà Triton đã
# giữ 8000/8001/8002 ở chế độ host. Công bố cổng tường minh thì không đụng được.
docker run --gpus all --rm "${TTY[@]}" --ipc host \
  --name triton-voice-llm \
  -p "$PORT:$PORT" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "$IMAGE" \
  --model "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_FRACTION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  "${EAGER[@]}" \
  "$@"
