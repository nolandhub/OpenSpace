# ABOUTME: Unit test cho bench/stats.py - tính latency từ profile export
# ABOUTME: Không cần Triton server, dùng fixture JSON viết tay

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench.stats import latency_stats, parse_profile_export  # noqa: E402


def write_export(tmp_path, requests, boundaries):
    """Ghi file export tối giản theo schema của perf_analyzer 2.60.0."""
    path = tmp_path / "export.json"
    path.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment": {"mode": "concurrency", "value": 1},
                        "requests": requests,
                        "window_boundaries": boundaries,
                    }
                ],
                "version": "2.60.0",
            }
        )
    )
    return path


def test_latency_from_last_response_timestamp(tmp_path):
    # timestamp tính bằng nanosecond, kết quả phải ra millisecond
    path = write_export(
        tmp_path,
        [{"timestamp": 1_000_000, "response_timestamps": [3_000_000]}],
        [],
    )
    assert parse_profile_export(path) == [2.0]


def test_multiple_response_timestamps_uses_last(tmp_path):
    # Request có nhiều response (streaming): latency tính tới mốc cuối cùng
    path = write_export(
        tmp_path,
        [{"timestamp": 0, "response_timestamps": [1_000_000, 5_000_000]}],
        [],
    )
    assert parse_profile_export(path) == [5.0]


def test_requests_outside_stable_windows_excluded(tmp_path):
    # 5 mốc biên, STABLE_WINDOWS=3 -> mốc bắt đầu là boundaries[-4] = 200
    requests = [
        {"timestamp": 150, "response_timestamps": [150 + 9_000_000]},   # loại
        {"timestamp": 250, "response_timestamps": [250 + 1_000_000]},
        {"timestamp": 350, "response_timestamps": [350 + 2_000_000]},
        {"timestamp": 450, "response_timestamps": [450 + 3_000_000]},
    ]
    path = write_export(tmp_path, requests, [100, 200, 300, 400, 500])
    assert parse_profile_export(path) == [1.0, 2.0, 3.0]


def test_few_boundaries_keeps_all_requests(tmp_path):
    # Không đủ mốc biên để cắt 3 cửa sổ thì lấy tất, đừng trả rỗng
    requests = [
        {"timestamp": 0, "response_timestamps": [1_000_000]},
        {"timestamp": 10, "response_timestamps": [10 + 2_000_000]},
    ]
    path = write_export(tmp_path, requests, [100, 200])
    assert parse_profile_export(path) == [1.0, 2.0]


def test_empty_stable_window_raises(tmp_path):
    # Không mẫu nào thì phải nổ rõ ràng, không trả 0 âm thầm
    path = write_export(tmp_path, [], [])
    with pytest.raises(ValueError, match="không có request"):
        parse_profile_export(path)


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
