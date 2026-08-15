# ABOUTME: Integration test cho phần gọi server của bench/stream_bench.py
# ABOUTME: Cần Triton đang chạy - kiểm việc ghép request_id với partial trả về

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench.stream_bench import run_ccu, run_stream  # noqa: E402
from client.common import chunk_wav, load_wav_16k  # noqa: E402

pytestmark = pytest.mark.integration

ASSETS = Path(__file__).parent / "assets"
CHUNK_MS = 200


@pytest.fixture(scope="module")
def chunks():
    return chunk_wav(load_wav_16k(ASSETS / "sample_vi.wav"), CHUNK_MS)[:8]


def test_every_chunk_gets_its_own_latency(triton, chunks):
    # Ghép sai request_id sẽ lộ ra ở đây: thiếu khoá, hoặc latency âm
    lat = run_stream("localhost:8001", "asr_streaming", 990001, chunks, CHUNK_MS,
                     threading.Barrier(1), timeout_s=30)
    assert sorted(lat) == list(range(len(chunks)))
    assert all(v > 0 for v in lat.values())


def test_latency_is_far_below_chunk_duration(triton, chunks):
    # Một chunk phải xử lý xong nhanh hơn nhiều so với 200ms audio nó mang theo,
    # nếu không thì streaming không thể theo kịp thời gian thực
    lat = run_stream("localhost:8001", "asr_streaming", 990002, chunks, CHUNK_MS,
                     threading.Barrier(1), timeout_s=30)
    assert max(lat.values()) < CHUNK_MS


def test_parallel_streams_stay_separate(triton, chunks):
    # Hai phiên chạy chồng lấn không được lẫn partial của nhau
    per_stream = run_ccu("localhost:8001", "asr_streaming", 2, chunks, CHUNK_MS, 990010, 30)
    assert len(per_stream) == 2
    assert all(sorted(lat) == list(range(len(chunks))) for lat in per_stream)
