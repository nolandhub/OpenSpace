#!/usr/bin/env bash
# ABOUTME: Chạy perf_analyzer cho một model với input thật, in p50/90/95/99 + throughput + GPU
# ABOUTME: Dùng: scripts/perf.sh asr_streaming | CONCURRENCY=1:8 scripts/perf.sh tts

set -euo pipefail

MODEL=${1:-}
shift || true
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
PA=${PA:-$ROOT/.venv/bin/perf_analyzer}
URL=${URL:-localhost:8001}
METRICS_URL=${METRICS_URL:-http://localhost:8002/metrics}

CONCURRENCY=${CONCURRENCY:-}

# Mỗi model một thư mục dưới bench/: gen_input.py sinh input, results/ nhận kết quả
BENCH_DIR=$ROOT/bench/$MODEL
RESULTS=$BENCH_DIR/results

case "$MODEL" in
  asr_streaming)
    # Audio thật, dạng data lồng - mỗi mảng con là một sequence, các bước đúng
    # thứ tự. Dữ liệu ngẫu nhiên cho số lạc quan ~3.5% vì vòng greedy phát ít
    # token non-blank hơn khi nghe nhiễu.
    "$PY" "$BENCH_DIR/gen_input.py" --out "$RESULTS/input.json" --streams 4 --chunks 50
    # --shape bắt buộc: AUDIO_CHUNK khai dims [-1], perf_analyzer không tự đoán được
    ARGS=(-m asr_streaming -i grpc --streaming
          --input-data "$RESULTS/input.json" --shape AUDIO_CHUNK:3200)
    CCU=${CONCURRENCY:-1:4}
    WARMUP_N=50   # chunk 200ms, 50 request chỉ tốn nửa giây
    ;;
  tts)
    "$PY" "$BENCH_DIR/gen_input.py" --out "$RESULTS/input.json"
    # tts non-decoupled và max_batch_size 0: một request trả trọn file WAV. Mỗi
    # request mất vài giây nên phải đếm theo số request, không theo cửa sổ thời
    # gian - mặc định 5s/cửa sổ sẽ không bao giờ đủ mẫu để ổn định.
    ARGS=(-m tts -i grpc --input-data "$RESULTS/input.json"
          --measurement-mode count_windows --measurement-request-count 10)
    CCU=${CONCURRENCY:-1:4}
    WARMUP_N=2    # một câu mất ~2.7s, 50 request thành 2 phút chết
    ;;
  *)
    echo "dùng: [CONCURRENCY=1:8] scripts/perf.sh {asr_streaming|tts} [cờ bổ sung]" >&2
    exit 2
    ;;
esac

# Warmup: một lượt ngắn, kết quả bỏ đi. Không có nó thì mức CCU đầu tiên gánh
# trọn chi phí nạp ONNX/CUDA và đọc ra chậm hơn thực tế. Không dùng
# --warmup-request-count được vì perf_analyzer từ chối cờ đó khi quét nhiều mức.
echo "warmup..."
"$PA" "${ARGS[@]}" -u "$URL" --request-count "$WARMUP_N" --concurrency-range 1 >/dev/null

# --verbose-csv để CSV có luôn GPU util, không phải đọc stdout
"$PA" "${ARGS[@]}" -u "$URL" --concurrency-range "$CCU" --percentile=95 \
  --collect-metrics --metrics-url "$METRICS_URL" --verbose-csv \
  -f "$RESULTS/perf.csv" "$@"

# perf_analyzer xếp hàng theo throughput tăng dần nên bảng đọc ra lộn xộn - sắp
# lại theo CCU. Latency trong CSV là microsecond, GPU util là "<uuid>:<số>;".
echo
{
  printf 'CCU\tinfer/s\tp50_ms\tp90_ms\tp95_ms\tp99_ms\tqueue_ms\tGPU%%\n'
  awk -F, '
    NR==1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
    {
      gpu = $col["Avg GPU Utilization"]; sub(/.*:/, "", gpu); sub(/;/, "", gpu)
      # %.3f cho throughput: tts chạy quanh 0.36-0.37 infer/s, %.1f làm tròn cả
      # hai mức thành 0.4 và bảng trông như throughput không đổi
      printf "%d\t%.3f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.0f\n",
        $col["Concurrency"], $col["Inferences/Second"],
        $col["p50 latency"] / 1000, $col["p90 latency"] / 1000,
        $col["p95 latency"] / 1000, $col["p99 latency"] / 1000,
        $col["Server Queue"] / 1000, gpu * 100
    }' "$RESULTS/perf.csv" | sort -n
} | column -t
