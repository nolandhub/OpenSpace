# ABOUTME: Percentile thuần - dùng chung cho first-chunk latency (ASR) và RTF (TTS)
# ABOUTME: Hàm thuần - không subprocess, không mạng, test được khi server tắt

import numpy as np


def p50_p95(values) -> tuple[float, float]:
    """Điển hình và tệ-trong-thực-tế. Hai mốc là đủ cho phép đo vài chục mẫu.

    Không có p99 ở đây: script này chỉ đo những chỉ số perf_analyzer không thấy
    được, mà mấy chỉ số đó lấy mẫu thưa - p99 trên 40 mẫu chỉ là max đội lốt.

    Không làm tròn: latency tính bằng ms cần 2 chữ số, RTF quanh 0.05 thì 2 chữ
    số là mất sạch thông tin. Nơi hiển thị tự quyết định độ chính xác của mình.
    """
    a = np.asarray(values, dtype=float)
    if not a.size:
        raise ValueError("không có mẫu nào để tính percentile")
    p50, p95 = np.percentile(a, [50, 95])
    return float(p50), float(p95)
