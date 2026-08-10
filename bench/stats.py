# ABOUTME: Tính min/max/percentile latency từ file profile export của perf_analyzer
# ABOUTME: Hàm thuần - không subprocess, không mạng, test được khi server tắt

import json
from pathlib import Path

import numpy as np

# perf_analyzer xét ổn định trên 3 cửa sổ đo cuối. Lấy đúng 3 cửa sổ đó thì
# thống kê tự tính mới so được với số nó in ra stdout (xem cross_check ở bench.py).
STABLE_WINDOWS = 3


def parse_profile_export(path) -> list[float]:
    """Latency (ms) của từng request trong các cửa sổ đo ổn định.

    perf_analyzer ghi timestamp theo nanosecond. Một request có thể có nhiều
    mốc response (streaming), latency tính tới mốc cuối cùng.
    """
    data = json.loads(Path(path).read_text())
    experiment = data["experiments"][0]
    requests = experiment["requests"]
    boundaries = experiment.get("window_boundaries") or []

    # Cần STABLE_WINDOWS+1 mốc để cắt ra STABLE_WINDOWS cửa sổ cuối. Ít hơn thì
    # phép đo chưa chạy đủ lâu, lấy tất còn hơn trả rỗng.
    if len(boundaries) > STABLE_WINDOWS:
        start = boundaries[-(STABLE_WINDOWS + 1)]
        requests = [r for r in requests if r["timestamp"] >= start]

    latencies = [
        (r["response_timestamps"][-1] - r["timestamp"]) / 1e6
        for r in requests
        if r.get("response_timestamps")
    ]
    if not latencies:
        raise ValueError(f"{path}: không có request nào trong cửa sổ ổn định")
    return latencies


def latency_stats(latencies) -> dict:
    """Sáu thống kê latency. samples để biết p99 có ý nghĩa hay không."""
    a = np.asarray(latencies, dtype=float)
    p50, p90, p95, p99 = np.percentile(a, [50, 90, 95, 99])
    return {
        "p50_ms": round(float(p50), 2),
        "p90_ms": round(float(p90), 2),
        "p95_ms": round(float(p95), 2),
        "p99_ms": round(float(p99), 2),
        "min_ms": round(float(a.min()), 2),
        "max_ms": round(float(a.max()), 2),
        "samples": int(a.size),
    }
