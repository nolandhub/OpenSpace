# Triton Voice Serving

ASR streaming (Zipformer RNN-T) và TTS (ZipVoice) tiếng Việt trên một
[Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/).

| model | backend | vào → ra |
|---|---|---|
| `asr_streaming` | Python, GPU ×2, sequence batching | chunk audio 16kHz → partial transcript |
| `tts` | Python, GPU ×1, không batch | text (+ giọng mẫu tuỳ chọn) → waveform 24kHz |

Client dùng gRPC (8001); HTTP (8000) và metrics (8002) cũng mở.

## Chuẩn bị

    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
    ./scripts/fetch_models.sh        # tải trọng số, chỉ cần 1 lần

`CHUNK_VARIANT=16|32|64 ./scripts/fetch_models.sh` chọn biến thể chunk của encoder
streaming — 16 là latency thấp nhất, cũng là mặc định.

## Chạy

    ./scripts/serve.sh                    # dựng image và load cả 2 model
    ./scripts/serve.sh asr_streaming      # chỉ load 1 model, dùng khi debug

## Dùng

    # ASR streaming - partial transcript hiện dần như phụ đề, --fast để gửi dồn
    .venv/bin/python client/asr_streaming_client.py tests/assets/sample_vi.wav

    # TTS với giọng mẫu đóng gói sẵn
    .venv/bin/python client/tts_client.py --text "Xin chào" --out ra.wav

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

## Tài liệu

| file | nội dung |
|---|---|
| `Architect.md` | kiến trúc, flow từng model, nút thắt hiện tại |
| `docs/ensemble-vs-one-backend.md` | vì sao một Python backend chứ không tách tầng |
| `bench/README.md` | ranh giới perf_analyzer / script tự viết, kết quả |
| `docs/superpowers/specs/` | thiết kế gốc và lý do từng quyết định cấu hình |
