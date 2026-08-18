# ABOUTME: Thống kê latency thuần - min/max/percentile từ một dãy số đo được
# ABOUTME: Hàm thuần - không subprocess, không mạng, test được khi server tắt

import numpy as np


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
