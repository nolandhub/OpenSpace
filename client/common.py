# ABOUTME: Hằng số và hàm dùng chung cho client và test
# ABOUTME: Mọi nơi phải dùng cùng bộ hằng số này, lệch một chỗ là batching hỏng

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
MAX_DURATION_S = 16
MAX_SAMPLES = SAMPLE_RATE * MAX_DURATION_S   # 256000
FRAME_SHIFT_MS = 10
MAX_FRAMES = MAX_DURATION_S * 1000 // FRAME_SHIFT_MS   # 1600
NUM_MEL_BINS = 80


def load_wav_16k(path) -> np.ndarray:
    """Đọc file wav bất kỳ về mono 16kHz float32 - tần số duy nhất ASR nhận.

    TTS xuất ra 24kHz, nên muốn đưa audio vừa sinh quay lại ASR để kiểm tra thì
    bắt buộc phải hạ tần số. resample_poly có lọc chống aliasing sẵn.
    """
    wav, sample_rate = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)   # trộn stereo về mono
    if sample_rate != SAMPLE_RATE:
        wav = resample_poly(wav, SAMPLE_RATE, sample_rate)
    return np.asarray(wav, dtype=np.float32)


def pad_wav(wav: np.ndarray) -> tuple[np.ndarray, int]:
    """Đệm waveform về đúng MAX_SAMPLES, trả về (mảng đã đệm, độ dài thật).

    Mọi request phải cùng shape thì dynamic batcher của Triton mới gom được.
    Đây là lý do tồn tại của hàm này - xem mục 7 của spec.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) > MAX_SAMPLES:
        raise ValueError(
            f"Audio dài {len(wav) / SAMPLE_RATE:.1f}s, vượt giới hạn {MAX_DURATION_S}s"
        )
    padded = np.zeros(MAX_SAMPLES, dtype=np.float32)
    padded[: len(wav)] = wav
    return padded, len(wav)
