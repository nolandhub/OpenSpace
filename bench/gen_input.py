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

# Scorer nhận ENCODER_OUT - tensor trung gian, không tồn tại dưới dạng file.
# RNN-T greedy search chạy số bước phụ thuộc nội dung tensor (mỗi bước phát một
# token hoặc blank) nên tensor ngẫu nhiên cho thời gian giải mã không đại diện.
# Phải bắt tensor thật từ server, dùng đúng file wav mà các thí nghiệm khác dùng.
# Bước này để cuối cùng: ba file trên không cần server, hỏng ở đây vẫn còn dùng được.
import tritonclient.grpc as grpcclient  # noqa: E402

client = grpcclient.InferenceServerClient("localhost:8001")
try:
    ready = client.is_server_ready()
except Exception as e:
    raise SystemExit(
        f"Không kết nối được Triton tại localhost:8001 ({e}).\n"
        "Chạy scripts/serve.sh rồi gọi lại lệnh này để sinh input_scorer.json."
    )
if not ready:
    raise SystemExit(
        "Triton chưa sẵn sàng. Chạy scripts/serve.sh rồi gọi lại lệnh này."
    )

feature_inputs = [
    grpcclient.InferInput("WAV", [1, len(padded)], "FP32"),
    grpcclient.InferInput("WAV_LEN", [1, 1], "INT32"),
]
feature_inputs[0].set_data_from_numpy(padded.reshape(1, -1))
feature_inputs[1].set_data_from_numpy(np.array([[real_len]], dtype=np.int32))
feature_out = client.infer("asr_feature", feature_inputs)
speech = feature_out.as_numpy("SPEECH")
speech_len = feature_out.as_numpy("SPEECH_LEN").reshape(1, 1).astype(np.int64)

encoder_inputs = [
    grpcclient.InferInput("x", list(speech.shape), "FP32"),
    grpcclient.InferInput("x_lens", [1, 1], "INT64"),
]
encoder_inputs[0].set_data_from_numpy(speech)
encoder_inputs[1].set_data_from_numpy(speech_len)
encoder_out = client.infer("asr_encoder", encoder_inputs).as_numpy("encoder_out")

frames = encoder_out.shape[1]
(ROOT / "bench/input_scorer.json").write_text(
    json.dumps(
        {
            "data": [
                {
                    "ENCODER_OUT": encoder_out.reshape(-1).tolist(),
                    "ENCODER_OUT_LEN": [frames],
                }
            ]
        }
    )
)
print(f"đã ghi bench/input_scorer.json (ENCODER_OUT shape {frames}x512)")
