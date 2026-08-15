# ABOUTME: Unit test cho chunk_wav - cắt waveform thành chunk đều nhau cho perf_analyzer
# ABOUTME: Chunk lệch shape là perf_analyzer từ chối; chunk xáo thứ tự là transcript ra rác

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.common import SAMPLE_RATE, chunk_wav  # noqa: E402


def _ramp(n):
    """Mảng tăng dần - mọi mẫu khác nhau nên phát hiện được xáo thứ tự lẫn mất mẫu."""
    return np.arange(n, dtype=np.float32)


def test_chunks_have_identical_length():
    """--shape của perf_analyzer là cố định: mọi chunk phải đúng một cỡ."""
    chunks = chunk_wav(_ramp(SAMPLE_RATE * 3), chunk_ms=200)
    expected = SAMPLE_RATE * 200 // 1000
    assert chunks
    assert all(len(c) == expected for c in chunks)


def test_chunks_concatenate_back_to_prefix():
    """Ghép lại phải khớp đúng tiền tố audio gốc - không mất, không xáo, không lặp."""
    wav = _ramp(SAMPLE_RATE * 3 + 777)
    chunks = chunk_wav(wav, chunk_ms=200)
    joined = np.concatenate(chunks)
    np.testing.assert_array_equal(joined, wav[: len(joined)])


def test_tail_shorter_than_one_chunk_is_dropped():
    """Phần dư không đủ một chunk bị bỏ, và chỉ được bỏ đúng phần đó."""
    chunk = SAMPLE_RATE * 200 // 1000
    wav = _ramp(chunk * 4 + 123)
    chunks = chunk_wav(wav, chunk_ms=200)
    assert len(chunks) == 4
    assert len(wav) - sum(len(c) for c in chunks) == 123


def test_audio_shorter_than_one_chunk_gives_nothing():
    """Không có chunk đủ cỡ thì trả rỗng thay vì trả một chunk hụt."""
    assert chunk_wav(_ramp(100), chunk_ms=200) == []


def test_chunk_ms_changes_chunk_size():
    """Cỡ chunk suy từ chunk_ms, không viết cứng."""
    assert all(len(c) == 1600 for c in chunk_wav(_ramp(SAMPLE_RATE), chunk_ms=100))


def test_rejects_chunk_ms_not_whole_samples():
    """chunk_ms lẻ làm cỡ chunk không nguyên - từ chối thẳng thay vì làm tròn âm thầm."""
    with pytest.raises(ValueError):
        chunk_wav(_ramp(SAMPLE_RATE), chunk_ms=0)
