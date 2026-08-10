# ABOUTME: Unit test cho StreamingFbank - so khớp từng số với kaldi.fbank offline
# ABOUTME: Không cần GPU/server; lệch khung ở mép chunk là transcript hỏng âm thầm nên phải khớp tuyệt đối

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "model_repository/asr_streaming/1"))
from streaming_search import StreamingFbank, offline_fbank  # noqa: E402


def _random_audio(seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(16000 * seconds)) * 0.1).astype(np.float32)


def _run_chunked(wav, chunk_sizes):
    """Đẩy wav qua StreamingFbank theo các cỡ chunk xoay vòng, gom hết khung phát ra."""
    fb = StreamingFbank()
    outs = []
    pos = 0
    i = 0
    while pos < len(wav):
        n = chunk_sizes[i % len(chunk_sizes)]
        outs.append(fb.accept_waveform(wav[pos : pos + n]))
        pos += n
        i += 1
    outs.append(fb.flush())
    return np.concatenate(outs)


def test_uniform_chunks_match_offline():
    wav = _random_audio()
    got = _run_chunked(wav, [3200])
    want = offline_fbank(wav)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


def test_ragged_chunks_match_offline():
    """Cỡ chunk cố tình lệch bội số khung để ép mọi nhánh bookkeeping."""
    wav = _random_audio(seed=1)
    got = _run_chunked(wav, [123, 7, 4000, 160, 1601])
    want = offline_fbank(wav)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


def test_tiny_first_chunk_emits_nothing():
    fb = StreamingFbank()
    out = fb.accept_waveform(np.zeros(100, dtype=np.float32))
    assert out.shape == (0, 80)


def test_flush_completes_frame_count():
    wav = _random_audio(seconds=0.5, seed=2)
    got = _run_chunked(wav, [1000])
    assert got.shape[0] == offline_fbank(wav).shape[0]
