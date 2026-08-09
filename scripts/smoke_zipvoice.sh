#!/usr/bin/env bash
# ABOUTME: Chạy thử ZipVoice bằng CLI gốc, bên ngoài Triton
# ABOUTME: Xác nhận checkpoint + tokenizer + vocoder hoạt động trước khi bọc vào Triton

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TTS="$ROOT/model_repository/tts/1"

docker run --gpus all --rm \
  -v "$TTS:/tts" \
  triton-voice \
  python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --model-dir /tts \
    --checkpoint-name model.pt \
    --tokenizer espeak \
    --lang vi \
    --vocoder-path /tts/vocos \
    --prompt-wav /tts/assets/prompt.wav \
    --prompt-text "$(cat "$TTS/assets/prompt.txt")" \
    --text "Xin chào, tôi là trợ lý ảo." \
    --res-wav-path /tts/assets/smoke_out.wav \
    --num-step 8
