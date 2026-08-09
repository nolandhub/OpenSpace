#!/usr/bin/env bash
# ABOUTME: Tải trọng số ASR/TTS từ HuggingFace về đúng vị trí trong model_repository
# ABOUTME: Chạy lại nhiều lần được - file đã có thì bỏ qua

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$ROOT/model_repository"

dl() {  # dl <url> <đích>
  if [ -f "$2" ]; then echo "bỏ qua $2"; return; fi
  mkdir -p "$(dirname "$2")"
  echo "tải $2"
  curl -fL --progress-bar "$1" -o "$2"
}

ASR=https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h/resolve/main
dl "$ASR/encoder-epoch-20-avg-10.onnx" "$REPO/asr_encoder/1/model.onnx"
dl "$ASR/decoder-epoch-20-avg-10.onnx" "$REPO/asr_scorer/1/decoder.onnx"
dl "$ASR/joiner-epoch-20-avg-10.onnx"  "$REPO/asr_scorer/1/joiner.onnx"
dl "$ASR/bpe.model"                     "$REPO/asr_scorer/1/bpe.model"

TTS=https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h/resolve/main
dl "$TTS/iter-525000-avg-2.pt" "$REPO/tts/1/model.pt"
dl "$TTS/tokens.txt"           "$REPO/tts/1/tokens.txt"
# ZipVoice bắt buộc file cấu hình phải tên là model.json, repo gốc đặt là config.json
dl "$TTS/config.json"          "$REPO/tts/1/model.json"

# Vocoder không nằm trong repo của ZipVoice tiếng Việt, phải lấy riêng
VOC=https://huggingface.co/charactr/vocos-mel-24khz/resolve/main
dl "$VOC/config.yaml"       "$REPO/tts/1/vocos/config.yaml"
dl "$VOC/pytorch_model.bin" "$REPO/tts/1/vocos/pytorch_model.bin"

# Ensemble không có trọng số nhưng Triton vẫn bắt buộc có thư mục version
mkdir -p "$REPO/asr/1"
echo "xong"
