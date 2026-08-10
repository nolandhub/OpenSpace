# ABOUTME: Logic thuần cho streaming ASR - fbank tính dần và greedy search theo chunk
# ABOUTME: Không import Triton/ONNX; decoder/joiner truyền vào dạng hàm như greedy_search.py của asr_scorer

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
FRAME_SHIFT = 160        # 10ms
FRAME_LENGTH = 400       # 25ms
# snip_edges=False (khớp asr_feature): khung j có tâm tại j*160+80, cửa sổ [tâm-200, tâm+200)
_CENTER = FRAME_SHIFT // 2        # 80
_HALF_WINDOW = FRAME_LENGTH // 2  # 200


def offline_fbank(samples: np.ndarray) -> np.ndarray:
    """Fbank cả câu, đúng tham số asr_feature dùng - chuẩn đối chiếu cho bản streaming."""
    wav = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
    feat = kaldi.fbank(
        wav,
        num_mel_bins=NUM_MEL_BINS,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        sample_frequency=SAMPLE_RATE,
        snip_edges=False,
    )
    return feat.numpy()


class StreamingFbank:
    """Fbank tính dần theo chunk, kết quả khớp offline_fbank trên cùng audio.

    Mẹo: mỗi lần vẫn gọi kaldi.fbank trên buffer, nhưng chỉ phát những khung mà
    cửa sổ nằm trọn trong phần mẫu thật (không dính reflection ở mép buffer).
    Buffer giữ lại 1 khung ngữ cảnh trái, cắt theo bội FRAME_SHIFT để lưới khung
    cục bộ trùng lưới toàn cục. Reflection thật chỉ còn ở đầu stream và ở flush -
    đúng hai chỗ offline cũng reflect, nên kết quả trùng nhau.
    """

    def __init__(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0    # vị trí tuyệt đối của _buf[0], luôn là bội của FRAME_SHIFT
        self._next_frame = 0   # chỉ số khung toàn cục kế tiếp chưa phát

    def _emit_until(self, last_frame: int) -> np.ndarray:
        """Phát các khung [_next_frame, last_frame] rồi cắt bớt buffer bên trái."""
        if last_frame < self._next_frame or len(self._buf) < FRAME_LENGTH:
            return np.zeros((0, NUM_MEL_BINS), dtype=np.float32)
        feat = offline_fbank(self._buf)
        base = self._buf_start // FRAME_SHIFT
        out = feat[self._next_frame - base : last_frame - base + 1].copy()
        self._next_frame = last_frame + 1
        # Keep at least 2 frame-shifts worth of context to ensure buffer stays >= FRAME_LENGTH
        keep_from = max(0, (self._next_frame - 2) * FRAME_SHIFT)
        self._buf = self._buf[keep_from - self._buf_start :]
        self._buf_start = keep_from
        return out

    def accept_waveform(self, samples: np.ndarray) -> np.ndarray:
        """Nạp thêm mẫu, trả về các khung mới đã chắc chắn (không đổi về sau)."""
        if len(samples):
            self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32)])
        abs_end = self._buf_start + len(self._buf)
        # khung j chắc chắn khi cửa sổ của nó kết thúc trước mẫu cuối đang có
        return self._emit_until((abs_end - _CENTER - _HALF_WINDOW) // FRAME_SHIFT)

    def flush(self) -> np.ndarray:
        """Hết audio - phát nốt khung đuôi, dùng reflection ở đuôi đúng như offline."""
        if len(self._buf) < FRAME_LENGTH:
            # quá ngắn để kaldi.fbank xử lý; trả empty
            return np.zeros((0, NUM_MEL_BINS), dtype=np.float32)
        abs_end = self._buf_start + len(self._buf)
        total = (abs_end + _CENTER) // FRAME_SHIFT   # công thức số khung của snip_edges=False
        return self._emit_until(total - 1)
