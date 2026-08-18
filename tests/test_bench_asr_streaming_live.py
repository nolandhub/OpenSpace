# ABOUTME: Integration test cho phần gọi server của bench/asr_streaming/metrics.py
# ABOUTME: Cần Triton đang chạy - kiểm việc ghép request_id với partial trả về

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench.asr_streaming.metrics import first_chunk_latencies, run_ccu, run_stream  # noqa: E402
from client.common import chunk_wav, load_wav_16k  # noqa: E402

pytestmark = pytest.mark.integration

ASSETS = Path(__file__).parent / "assets"
CHUNK_MS = 200


@pytest.fixture(scope="module")
def chunks():
    return chunk_wav(load_wav_16k(ASSETS / "sample_vi.wav"), CHUNK_MS)[:8]


def test_every_chunk_gets_its_own_latency(triton, chunks):
    # Ghép sai request_id sẽ lộ ra ở đây: thiếu khoá, hoặc latency âm
    r = run_stream("localhost:8001", "asr_streaming", 990001, chunks, CHUNK_MS,
                   threading.Barrier(1), timeout_s=30)
    assert sorted(r.latencies) == list(range(len(chunks)))
    assert all(v > 0 for v in r.latencies.values())


def test_transcript_comes_back_non_empty(triton, chunks):
    # WER tính trên transcript của chunk cuối - rỗng thì WER luôn bằng 1.0 mà
    # không ai biết là do model kém hay do lấy nhầm response
    r = run_stream("localhost:8001", "asr_streaming", 990002, chunks, CHUNK_MS,
                   threading.Barrier(1), timeout_s=30)
    assert r.transcript.strip()


def test_parallel_streams_stay_separate(triton, chunks):
    # Hai phiên chạy chồng lấn không được lẫn partial của nhau
    results = run_ccu("localhost:8001", "asr_streaming", 2, chunks, CHUNK_MS, 990010, 30)
    assert len(results) == 2
    assert all(sorted(r.latencies) == list(range(len(chunks))) for r in results)
    assert len(first_chunk_latencies(results)) == 2
