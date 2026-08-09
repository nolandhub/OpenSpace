# Triton Voice Serving

Serving ASR (Zipformer RNN-T) và TTS (ZipVoice) tiếng Việt trên Triton Inference Server.

- Thiết kế: `docs/superpowers/specs/2026-08-09-triton-voice-serving-design.md`
- Kế hoạch triển khai: `docs/superpowers/plans/2026-08-09-triton-voice-serving.md`

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

## Test

    .venv/bin/pytest tests/ -m "not integration"   # unit, không cần server
    .venv/bin/pytest tests/                        # đầy đủ, cần server đang chạy

## Benchmark

    .venv/bin/python bench/gen_input.py
    .venv/bin/python bench/bench.py e1    # quét max_queue_delay
    .venv/bin/python bench/bench.py e1b   # quét concurrency client
    .venv/bin/python bench/bench.py e2    # quét instance_group.count của TTS
    .venv/bin/python bench/bench.py e3    # breakdown từng tầng ensemble

Kết quả đã đo và phần phân tích: `bench/README.md`

## Kiến trúc

    asr (ensemble)
     ├── asr_feature   Python, CPU ×4      fbank 80 chiều
     ├── asr_encoder   ONNX, GPU           dynamic batching 5ms
     └── asr_scorer    Python, GPU ×2      greedy search + decoder/joiner ONNX

    tts               Python, GPU ×1       espeak-ng vi → ZipVoice → vocos
