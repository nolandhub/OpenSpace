# ABOUTME: Unit test cho phần tính toán thuần của bench/asr_streaming/metrics.py
# ABOUTME: Không cần Triton server - dữ liệu đo được thay bằng dãy số viết tay

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench.asr_streaming.gen_input import payload  # noqa: E402
from bench.asr_streaming.metrics import (  # noqa: E402
    StreamResult,
    first_chunk_latencies,
    render_matrix,
    send_deadlines,
    summarize_run,
)


# ---------- send_deadlines ----------


def test_deadlines_are_absolute_not_cumulative():
    # Mốc thứ i phải là t0 + i*chunk chính xác. Cộng dồn 0.2 sẽ trôi dần vì nhị phân.
    deadlines = send_deadlines(4, 200, t0=10.0)
    assert deadlines == pytest.approx([10.0, 10.2, 10.4, 10.6])


def test_deadline_spacing_holds_over_many_chunks():
    # 300 chunk = 60s audio; mốc cuối phải đúng 59.8s sau t0, không lệch
    deadlines = send_deadlines(300, 200, t0=0.0)
    assert deadlines[-1] == pytest.approx(59.8)


# ---------- first_chunk_latencies ----------


def test_takes_chunk_zero_of_every_session():
    results = [
        StreamResult(latencies={0: 12.0, 1: 5.0}),
        StreamResult(latencies={0: 9.0, 1: 6.0}),
    ]
    assert first_chunk_latencies(results) == [12.0, 9.0]


def test_raises_when_a_session_lost_its_first_chunk():
    # Thiếu chunk 0 nghĩa là partial không về - im lặng bỏ qua sẽ làm sai số liệu
    with pytest.raises(ValueError, match="chunk 0"):
        first_chunk_latencies([StreamResult(latencies={1: 10.0})])


# ---------- summarize_run ----------


def test_row_carries_only_ccu_and_two_first_chunk_percentiles():
    row = summarize_run(ccu=2, results=[StreamResult(latencies={0: float(i)}) for i in range(1, 101)])
    assert set(row) == {"ccu", "first_p50_ms", "first_p95_ms"}
    assert row["ccu"] == 2
    assert row["first_p50_ms"] == 50.5
    assert row["first_p95_ms"] == 95.05


def test_later_chunks_never_reach_the_row():
    # Latency chunk giữa stream là việc của perf_analyzer, không phải của script này
    results = [StreamResult(latencies={0: 10.0, 1: 999.0, 2: 999.0})]
    row = summarize_run(ccu=1, results=results)
    assert row["first_p50_ms"] == 10.0
    assert row["first_p95_ms"] == 10.0


# ---------- render_matrix ----------


def test_matrix_has_one_row_per_ccu_level():
    rows = [summarize_run(ccu=c, results=[StreamResult(latencies={0: float(c)})]) for c in (1, 2, 3, 4)]
    table = render_matrix(rows)
    body = [ln for ln in table.splitlines() if ln.startswith("| ")]
    assert len(body) == 1 + 4
    assert [ln.split("|")[1].strip() for ln in body[1:]] == ["1", "2", "3", "4"]


def test_matrix_shows_only_first_chunk_columns():
    rows = [summarize_run(ccu=1, results=[StreamResult(latencies={0: 9.0})])]
    header = render_matrix(rows).splitlines()[0]
    assert [c.strip() for c in header.strip("|").split("|")] == [
        "CCU", "first-chunk p50 ms", "first-chunk p95 ms"
    ]


# ---------- gen_input.payload ----------


def test_each_stream_is_a_nested_list_of_steps_in_order():
    """perf_analyzer đọc `data` lồng: mỗi mảng con là MỘT sequence, các phần tử
    là các bước theo đúng thứ tự. Phẳng một tầng thì nó hiểu thành nhiều sequence
    riêng lẻ mỗi cái một bước, và state của encoder không bao giờ tích luỹ."""
    chunks = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    data = payload(chunks, n_streams=2)["data"]
    assert len(data) == 2
    assert data[0] == [{"AUDIO_CHUNK": [1.0, 2.0]}, {"AUDIO_CHUNK": [3.0, 4.0]}]


def test_every_stream_carries_the_same_audio():
    data = payload([np.array([1.0]), np.array([2.0])], n_streams=3)["data"]
    assert data[0] == data[1] == data[2]


def test_payload_rejects_empty_audio():
    with pytest.raises(ValueError, match="chunk"):
        payload([], n_streams=1)
