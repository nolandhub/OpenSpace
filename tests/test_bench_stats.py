# ABOUTME: Unit test cho bench/stats.py - thống kê latency
# ABOUTME: Không cần Triton server

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench.stats import latency_stats  # noqa: E402


def test_percentiles_on_known_array():
    # Mảng 1..100: numpy nội suy tuyến tính cho đáp án biết trước
    stats = latency_stats(list(range(1, 101)))
    assert stats["p50_ms"] == 50.5
    assert stats["p90_ms"] == 90.1
    assert stats["p95_ms"] == 95.05
    assert stats["p99_ms"] == 99.01
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 100.0


def test_reports_sample_count():
    # samples cần để đọc p99 trung thực: dưới ~100 mẫu thì p99 không phải phân vị thật
    assert latency_stats([1.0, 2.0, 3.0])["samples"] == 3
