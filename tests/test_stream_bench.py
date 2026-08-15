# ABOUTME: Unit test cho phần tính toán thuần của bench/stream_bench.py
# ABOUTME: Không cần Triton server - dữ liệu đo được thay bằng dãy số viết tay

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench.stream_bench import (  # noqa: E402
    render_matrix,
    repeat_to_duration,
    sanity_warnings,
    send_deadlines,
    split_first_chunk,
    stats_delta,
    summarize_run,
)


def make_stats(name="asr_streaming", batches=(), **ns):
    """Dựng payload get_inference_statistics(as_json=True) tối giản.

    Số đếm của Triton về dạng JSON là chuỗi, không phải int - giữ đúng vậy.
    `batches` là danh sách (batch_size, số lần chạy, tổng ns) cho batch_stats.
    """
    return {
        "model_stats": [
            {
                "name": name,
                "version": "1",
                "inference_stats": {
                    "success": {"count": str(ns["success"]), "ns": "0"},
                    "queue": {"count": str(ns["success"]), "ns": str(ns["queue"])},
                    "compute_input": {"count": str(ns["success"]), "ns": str(ns["cin"])},
                    "compute_infer": {"count": str(ns["success"]), "ns": str(ns["cinf"])},
                    "compute_output": {"count": str(ns["success"]), "ns": str(ns["cout"])},
                },
                "batch_stats": [
                    {
                        "batch_size": str(size),
                        "compute_infer": {"count": str(count), "ns": str(total_ns)},
                    }
                    for size, count, total_ns in batches
                ],
            }
        ]
    }


# ---------- repeat_to_duration ----------


def test_repeat_tiles_source_to_reach_duration():
    # 4 mẫu ở 4Hz = 1s; xin 2.5s phải ra 10 mẫu lặp vòng
    wav = np.array([1, 2, 3, 4], dtype=np.float32)
    out = repeat_to_duration(wav, 2.5, sample_rate=4)
    assert out.tolist() == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]


def test_repeat_truncates_when_source_longer():
    wav = np.array([1, 2, 3, 4], dtype=np.float32)
    assert repeat_to_duration(wav, 0.5, sample_rate=4).tolist() == [1, 2]


def test_repeat_rejects_nonpositive_duration():
    with pytest.raises(ValueError, match="seconds"):
        repeat_to_duration(np.zeros(4, dtype=np.float32), 0, sample_rate=4)


# ---------- send_deadlines ----------


def test_deadlines_are_absolute_not_cumulative():
    # Mốc thứ i phải là t0 + i*chunk chính xác. Cộng dồn 0.2 sẽ trôi dần vì nhị phân.
    deadlines = send_deadlines(4, 200, t0=10.0)
    assert deadlines == pytest.approx([10.0, 10.2, 10.4, 10.6])


def test_deadline_spacing_holds_over_many_chunks():
    # 300 chunk = 60s audio; mốc cuối phải đúng 59.8s sau t0, không lệch
    deadlines = send_deadlines(300, 200, t0=0.0)
    assert deadlines[-1] == pytest.approx(59.8)


# ---------- split_first_chunk ----------


def test_split_pulls_chunk_zero_of_every_stream():
    per_stream = [{0: 50.0, 1: 10.0, 2: 12.0}, {0: 60.0, 1: 11.0}]
    rest, firsts = split_first_chunk(per_stream)
    assert sorted(rest) == [10.0, 11.0, 12.0]
    assert firsts == [50.0, 60.0]


def test_split_raises_when_a_stream_lost_its_first_chunk():
    # Thiếu chunk 0 nghĩa là partial không về - im lặng bỏ qua sẽ làm sai số liệu
    with pytest.raises(ValueError, match="chunk 0"):
        split_first_chunk([{1: 10.0}])


# ---------- stats_delta ----------


def test_delta_divides_by_number_of_new_requests():
    before = make_stats(success=100, queue=1_000_000, cin=0, cinf=0, cout=0)
    after = make_stats(
        success=110, queue=21_000_000, cin=5_000_000, cinf=40_000_000, cout=5_000_000
    )
    d = stats_delta(before, after, "asr_streaming")
    assert d["requests"] == 10
    assert d["queue_ms"] == pytest.approx(2.0)      # 20ms / 10 request
    assert d["compute_ms"] == pytest.approx(5.0)    # 50ms / 10 request


def test_busy_time_comes_from_batch_stats_not_per_request_credit():
    """Triton cộng compute cho TỪNG request trong batch, nên inference_stats
    đếm lặp. Thời gian server thật sự bận chỉ có ở batch_stats."""
    before = make_stats(success=0, queue=0, cin=0, cinf=0, cout=0, batches=[(2, 0, 0)])
    # 10 request chạy trong 5 lần execute batch 2, mỗi lần 6ms -> bận thật 30ms.
    # inference_stats ghi 60ms vì mỗi request được ghi công trọn lần execute.
    after = make_stats(
        success=10, queue=0, cin=0, cinf=60_000_000, cout=0, batches=[(2, 5, 30_000_000)]
    )
    d = stats_delta(before, after, "asr_streaming")
    assert d["compute_total_s"] == pytest.approx(0.03)   # KHÔNG phải 0.06
    assert d["compute_ms"] == pytest.approx(6.0)         # ghi công mỗi request vẫn là 6ms


def test_delta_reports_average_batch_size():
    # batch_avg cho biết batcher gom được bao nhiêu sequence mỗi lần execute
    before = make_stats(success=0, queue=0, cin=0, cinf=0, cout=0, batches=[])
    after = make_stats(
        success=10, queue=0, cin=0, cinf=0, cout=0, batches=[(1, 2, 1000), (3, 2, 2000)]
    )
    d = stats_delta(before, after, "asr_streaming")
    assert d["executions"] == 4
    assert d["batch_avg"] == pytest.approx(2.5)   # 10 request / 4 lần execute


def test_delta_treats_missing_fields_as_zero():
    # MessageToJson bỏ hẳn field có giá trị 0, không phải lúc nào cũng đủ khoá
    before = {"model_stats": [{"name": "m", "inference_stats": {}}]}
    after = {"model_stats": [{"name": "m", "inference_stats": {"success": {"count": "4"}}}]}
    d = stats_delta(before, after, "m")
    assert d["requests"] == 4
    assert d["queue_ms"] == 0.0
    assert d["compute_ms"] == 0.0


def test_delta_raises_when_no_request_ran():
    s = make_stats(success=5, queue=0, cin=0, cinf=0, cout=0)
    with pytest.raises(ValueError, match="không có request"):
        stats_delta(s, s, "asr_streaming")


def test_delta_raises_when_model_absent():
    s = make_stats(success=5, queue=0, cin=0, cinf=0, cout=0)
    with pytest.raises(ValueError, match="tts"):
        stats_delta(s, s, "tts")


# ---------- summarize_run ----------


def test_rtf_p95_is_latency_p95_over_chunk_ms():
    # Gộp chunk của cả 2 phiên: rest = 1..100ms -> p95 = 95.05ms; chia 200ms ra đúng 0.475
    per_stream = [
        {0: 300.0, **{i: float(i) for i in range(1, 51)}},
        {0: 310.0, **{i: float(i) for i in range(51, 101)}},
    ]
    row = summarize_run(ccu=2, chunk_ms=200, per_stream=per_stream, audio_total_s=20.0, server=None)
    assert row["p95_ms"] == 95.05
    assert row["rtf_p95"] == pytest.approx(0.475)
    assert row["samples"] == 100


def test_first_chunk_reported_apart_and_kept_out_of_percentiles():
    # 300ms của chunk mở phiên không được lẫn vào max của các chunk còn lại
    per_stream = [{0: 300.0, 1: 10.0, 2: 20.0}]
    row = summarize_run(ccu=1, chunk_ms=200, per_stream=per_stream, audio_total_s=0.6, server=None)
    assert row["first_ms"] == 300.0
    assert row["max_ms"] == 20.0


def test_first_chunk_column_takes_the_worst_stream():
    per_stream = [{0: 100.0, 1: 5.0}, {0: 250.0, 1: 6.0}]
    row = summarize_run(ccu=2, chunk_ms=200, per_stream=per_stream, audio_total_s=0.8, server=None)
    assert row["first_ms"] == 250.0


def test_rtf_stream_is_server_compute_over_audio_duration():
    # 4s compute cho 20s audio -> server ăn 20% thời gian thực
    server = {
        "requests": 100,
        "executions": 50,
        "batch_avg": 2.0,
        "queue_ms": 1.0,
        "compute_ms": 40.0,
        "compute_total_s": 4.0,
    }
    per_stream = [{0: 50.0, 1: 10.0}]
    row = summarize_run(ccu=1, chunk_ms=200, per_stream=per_stream, audio_total_s=20.0, server=server)
    assert row["rtf_stream"] == pytest.approx(0.2)
    assert row["queue_ms"] == 1.0
    assert row["compute_ms"] == 40.0
    assert row["batch_avg"] == 2.0


def test_server_columns_empty_when_stats_unavailable():
    row = summarize_run(ccu=1, chunk_ms=200, per_stream=[{0: 1.0, 1: 2.0}], audio_total_s=0.4, server=None)
    assert row["rtf_stream"] is None
    assert row["queue_ms"] is None
    assert row["batch_avg"] is None


# ---------- render_matrix ----------


def test_matrix_has_one_row_per_ccu_level():
    rows = [
        summarize_run(ccu=c, chunk_ms=200, per_stream=[{0: 9.0, 1: float(c)}], audio_total_s=0.4, server=None)
        for c in (1, 2, 3, 4)
    ]
    table = render_matrix(rows, chunk_ms=200)
    # Đường kẻ markdown bắt đầu bằng "|-" nên không lọt vào đây; còn lại header + 4 hàng
    body = [ln for ln in table.splitlines() if ln.startswith("| ")]
    assert len(body) == 1 + 4
    assert [ln.split("|")[1].strip() for ln in body[1:]] == ["1", "2", "3", "4"]
    assert "RTF p95" in table


# ---------- sanity_warnings ----------


def test_queue_longer_than_max_latency_is_flagged():
    """Đã gặp thật: một lần chạy báo queue 162ms trong khi max latency chỉ 124ms.
    Request không thể chờ lâu hơn tổng đời của nó - cửa sổ thống kê bị dính
    request ngoài phép đo. Số vô lý mà lọt vào bảng thì cả bảng mất tin cậy."""
    server = {"requests": 10, "executions": 10, "batch_avg": 1.0,
              "queue_ms": 162.69, "compute_ms": 17.14, "compute_total_s": 0.1}
    row = summarize_run(ccu=2, chunk_ms=200, per_stream=[{0: 5.0, 1: 124.18}],
                        audio_total_s=0.4, server=server)
    warnings = sanity_warnings(row)
    assert len(warnings) == 1
    assert "queue" in warnings[0] and "CCU 2" in warnings[0]


def test_no_warning_when_queue_fits_inside_latency():
    server = {"requests": 10, "executions": 10, "batch_avg": 1.0,
              "queue_ms": 6.85, "compute_ms": 14.0, "compute_total_s": 0.1}
    row = summarize_run(ccu=2, chunk_ms=200, per_stream=[{0: 5.0, 1: 79.55}],
                        audio_total_s=0.4, server=server)
    assert sanity_warnings(row) == []


def test_no_warning_when_server_stats_absent():
    row = summarize_run(ccu=1, chunk_ms=200, per_stream=[{0: 1.0, 1: 2.0}],
                        audio_total_s=0.4, server=None)
    assert sanity_warnings(row) == []


def test_matrix_carries_the_warning_so_it_cannot_be_read_without_it():
    server = {"requests": 10, "executions": 10, "batch_avg": 1.0,
              "queue_ms": 500.0, "compute_ms": 1.0, "compute_total_s": 0.1}
    rows = [summarize_run(ccu=2, chunk_ms=200, per_stream=[{0: 5.0, 1: 20.0}],
                          audio_total_s=0.4, server=server)]
    assert "CẢNH BÁO" in render_matrix(rows, chunk_ms=200)


def test_matrix_prints_dash_for_missing_server_stats():
    rows = [summarize_run(ccu=1, chunk_ms=200, per_stream=[{0: 9.0, 1: 1.0}], audio_total_s=0.4, server=None)]
    assert "| - |" in render_matrix(rows, chunk_ms=200)
