# ABOUTME: Hằng số và hàm dùng chung cho client và test
# ABOUTME: Mọi nơi phải dùng cùng bộ hằng số này, lệch một chỗ là batching hỏng

import numpy as np

SAMPLE_RATE = 16000
MAX_DURATION_S = 16
MAX_SAMPLES = SAMPLE_RATE * MAX_DURATION_S   # 256000
FRAME_SHIFT_MS = 10
MAX_FRAMES = MAX_DURATION_S * 1000 // FRAME_SHIFT_MS   # 1600
NUM_MEL_BINS = 80


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
