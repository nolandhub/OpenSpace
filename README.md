# Triton Voice Serving

ASR streaming (Zipformer RNN-T) và TTS (ZipVoice) tiếng Việt trên một
[Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/).

| model | backend | vào → ra |
|---|---|---|
| `asr_streaming` | Python, GPU ×2, sequence batching | chunk audio 16kHz → partial transcript |
| `tts` | Python, GPU ×1, không batch | text (+ giọng mẫu tuỳ chọn) → waveform 24kHz |

Client dùng gRPC (8001); HTTP (8000) và metrics (8002) cũng mở.

LLM chạy **server riêng** — vLLM với API OpenAI-compatible ở cổng 8080, không nằm
trong `model_repository`. Lý do: [`docs/llm-serving.md`](docs/llm-serving.md).

## Chuẩn bị

    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
    ./scripts/fetch_models.sh        # tải trọng số, chỉ cần 1 lần

`CHUNK_VARIANT=16|32|64 ./scripts/fetch_models.sh` chọn biến thể chunk của encoder
streaming — 16 là latency thấp nhất, cũng là mặc định.

## Chạy

    ./scripts/serve_triton.sh                    # dựng image và load cả 2 model
    ./scripts/serve_triton.sh asr_streaming      # chỉ load 1 model, dùng khi debug
    ./scripts/serve_llm.sh                # vLLM ở cổng 8080 — chạy SAU khi Triton load xong

Thứ tự bắt buộc: vLLM đo VRAM trống lúc khởi động rồi giữ luôn, dựng trước Triton là
Triton chết giữa request. `MODEL=` đổi model, `GPU_FRACTION=` đổi phần VRAM xin.

## Dùng

    # ASR streaming - partial transcript hiện dần như phụ đề, --fast để gửi dồn
    .venv/bin/python client/asr_streaming_client.py tests/assets/sample_vi.wav

    # TTS với giọng mẫu đóng gói sẵn
    .venv/bin/python client/tts_client.py --text "Xin chào" --out ra.wav

    # LLM - in dần từng token, --no-think tắt reasoning của Qwen3
    .venv/bin/python client/llm_client.py --prompt "Thủ đô Việt Nam là gì?" --no-think

    # LLM hội thoại nhiều lượt - giữ ngữ cảnh, tự cắt lịch sử cho vừa cửa sổ
    .venv/bin/python client/llm_client.py --chat --no-think \
        --system "Trả lời ngắn gọn bằng tiếng Việt."

    # TTS clone giọng từ file mẫu bất kỳ
    .venv/bin/python client/tts_client.py --text "Hôm nay trời đẹp." \
        --prompt tests/assets/sample_vi.wav \
        --prompt-text "$(cat tests/assets/sample_vi.txt)" --out clone.wav

## Test

    .venv/bin/pytest tests/ -m "not integration"   # unit, không cần server
    .venv/bin/pytest tests/                        # đầy đủ, cần server đang chạy

## Benchmark

    ./scripts/perf.sh asr_streaming                      # p50/90/95/99, throughput, GPU util
    ./scripts/perf.sh tts
    .venv/bin/python bench/asr_streaming/metrics.py      # first-chunk latency + WER
    .venv/bin/python bench/tts/metrics.py                # RTF

perf_analyzer lo mọi chỉ số ở tầng request; `bench/` chỉ chứa cái nó không thấy được.
Cách đọc kết quả: `bench/README.md`.

## Monitoring

    ./scripts/serve_monitoring.sh        # Prometheus 9090 + Grafana 3000

Grafana `http://localhost:3000`, hai dashboard:

- **Voice Serving** — view sản phẩm: RPS, p50/95/99, CCU, queue depth, RTF,
  TTFT/TPOT, GPU, error rate cho cả ASR, TTS và LLM.
- **Triton** — view nội tại server: latency tách thành queue / compute_input /
  compute_infer / compute_output, batch size thực tế, CPU và GPU của Triton.

Triton và vLLM phải chạy trước thì target mới UP.

RTF và CCU do `serving/metrics.py` tự phát — Triton không biết audio dài bao
nhiêu, cũng không biết bao nhiêu phiên đang sống. Cách đọc: `docs/observability.md`.

## Tài liệu

| file | nội dung |
|---|---|
| `Architect.md` | kiến trúc, flow từng model, nút thắt hiện tại |
| `docs/ensemble-vs-one-backend.md` | vì sao một Python backend chứ không tách tầng |
| `docs/llm-serving.md` | vì sao vLLM đứng riêng, và Thor đổi gì |
| `docs/observability.md` | cách chạy monitoring, cách đọc từng chỉ số, giới hạn |
| `bench/README.md` | ranh giới perf_analyzer / script tự viết, kết quả |
| `docs/superpowers/specs/` | thiết kế gốc và lý do từng quyết định cấu hình |
