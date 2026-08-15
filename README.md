# Triton Voice Serving

Serving ASR (Zipformer RNN-T) và TTS (ZipVoice) tiếng Việt trên Triton Inference Server.

- Thiết kế: `docs/superpowers/specs/2026-08-09-triton-voice-serving-design.md`
- Kế hoạch triển khai: `docs/superpowers/plans/2026-08-09-triton-voice-serving.md`
- Thiết kế streaming ASR: `docs/superpowers/specs/2026-08-10-streaming-asr-design.md`

## Chuẩn bị

    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
    ./scripts/fetch_models.sh          # tải trọng số, chỉ cần 1 lần
    .venv/bin/python scripts/prepare_assets.py   # audio mẫu tiếng Việt

## Chạy

    ./scripts/serve.sh                 # dựng image và chạy cả 5 model
    ./scripts/serve.sh asr_encoder     # chỉ load 1 model, dùng khi debug

## Dùng

    .venv/bin/python client/asr_client.py tests/assets/sample_vi.wav
    .venv/bin/python client/tts_client.py --text "Xin chào" --out ra.wav

    # TTS clone giọng từ file mẫu bất kỳ
    .venv/bin/python client/tts_client.py --text "Hôm nay trời đẹp." \
        --prompt tests/assets/sample_vi.wav \
        --prompt-text "$(cat tests/assets/sample_vi.txt)" --out clone.wav

    # ASR streaming - partial transcript hiện dần như phụ đề
    .venv/bin/python client/asr_streaming_client.py tests/assets/sample_vi.wav

## Test

    .venv/bin/pytest tests/ -m "not integration"   # unit, không cần server
    .venv/bin/pytest tests/                        # đầy đủ, cần server đang chạy

## Benchmark

    # latency mỗi chunk 200ms + RTF, quét 1-4 phiên đồng thời
    .venv/bin/python bench/stream_bench.py --ccu 1 2 3 4 --duration 60

Kết quả đã đo và phần phân tích: `bench/README.md`

## Kiến trúc

    asr (ensemble)
     ├── asr_feature   Python, CPU ×4      fbank 80 chiều
     ├── asr_encoder   ONNX, GPU           dynamic batching 5ms
     └── asr_scorer    Python, GPU ×2      greedy search + decoder/joiner ONNX

    asr_streaming     Python, GPU ×1       sequence batcher; chunk audio → partial transcript

    tts               Python, GPU ×1       espeak-ng vi → ZipVoice → vocos
