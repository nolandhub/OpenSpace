# ABOUTME: Sinh file JSON đầu vào cho perf_analyzer
# ABOUTME: perf_analyzer cần dữ liệu thật đúng shape, không tự sinh được

import json
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from client.common import MAX_FRAMES, NUM_MEL_BINS, pad_wav  # noqa: E402

# ASR: một request chứa waveform đã đệm về 16 giây + độ dài thật
wav, sr = sf.read(ROOT / "tests/assets/sample_vi.wav", dtype="float32")
padded, real_len = pad_wav(wav)
(ROOT / "bench/input_asr.json").write_text(
    json.dumps({"data": [{"WAV": padded.tolist(), "WAV_LEN": [real_len]}]})
)
print("đã ghi bench/input_asr.json")

# TTS: một câu tiếng Việt, dùng giọng mẫu mặc định của model
(ROOT / "bench/input_tts.json").write_text(
    json.dumps({"data": [{"TEXT": ["Xin chào, tôi là trợ lý ảo."]}]})
)
print("đã ghi bench/input_tts.json")

# Encoder đo tách riêng: đo qua ensemble thì tầng scorer nghẽn trước,
# batching của encoder không bao giờ lộ ra. Nội dung không ảnh hưởng thời gian
# chạy nên dùng số ngẫu nhiên cố định seed.
import numpy as np  # noqa: E402

rng = np.random.default_rng(0)
x = rng.standard_normal((MAX_FRAMES, NUM_MEL_BINS)).astype(np.float32)
(ROOT / "bench/input_encoder.json").write_text(
    json.dumps({"data": [{"x": x.reshape(-1).tolist(), "x_lens": [MAX_FRAMES]}]})
)
print("đã ghi bench/input_encoder.json")
