# ABOUTME: Unit test cho bench/common/stats.py - percentile dùng chung
# ABOUTME: Không cần Triton server

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench.common.stats import p50_p95  # noqa: E402


def test_percentiles_on_known_array():
    # Mảng 1..100: numpy nội suy tuyến tính cho đáp án biết trước
    assert p50_p95(list(range(1, 101))) == pytest.approx((50.5, 95.05))


def test_single_sample_gives_that_value_twice():
    assert p50_p95([7.0]) == (7.0, 7.0)


def test_empty_input_is_rejected():
    # np.percentile trên mảng rỗng trả nan kèm warning - nan lọt vào bảng thì im lặng sai
    with pytest.raises(ValueError, match="không có mẫu"):
        p50_p95([])
