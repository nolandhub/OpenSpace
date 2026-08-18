# ABOUTME: Unit test cho phần tính toán thuần của bench/tts/metrics.py
# ABOUTME: Không cần Triton server

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench.tts.gen_input import payload  # noqa: E402
from bench.tts.metrics import render_table, rtf, summarize_run  # noqa: E402


# ---------- rtf ----------


def test_rtf_is_compute_time_over_generated_audio_length():
    # 2s tính toán cho 4s audio -> RTF 0.5, sinh nhanh gấp đôi thời gian thực
    assert rtf(latency_s=2.0, audio_s=4.0) == 0.5


def test_rtf_above_one_means_slower_than_realtime():
    assert rtf(latency_s=6.0, audio_s=4.0) == 1.5


def test_rtf_rejects_empty_audio():
    # Model trả WAV rỗng thì RTF là vô cực - phải nổ chứ không được ghi vào bảng
    with pytest.raises(ValueError, match="audio"):
        rtf(latency_s=2.0, audio_s=0.0)


# ---------- summarize_run ----------


def test_row_carries_only_the_two_rtf_percentiles():
    # RTF giữ 3 chữ số: model nhanh cho RTF quanh 0.05, làm tròn 2 chữ số là mất sạch
    row = summarize_run([float(i) / 100 for i in range(1, 101)])
    assert row == {"rtf_p50": 0.505, "rtf_p95": 0.95}


def test_summarize_rejects_empty_measurement():
    with pytest.raises(ValueError, match="không có"):
        summarize_run([])


# ---------- render_table ----------


def test_table_shows_only_the_rtf_columns():
    header = render_table(summarize_run([0.5, 0.6])).splitlines()[0]
    assert [c.strip() for c in header.strip("|").split("|")] == ["RTF p50", "RTF p95"]


# ---------- gen_input.payload ----------


def test_each_text_is_one_request():
    # tts không phải sequence model nên `data` phẳng một tầng, perf_analyzer
    # xoay vòng qua các phần tử cho từng request
    assert payload(["xin chào", "tạm biệt"])["data"] == [
        {"TEXT": ["xin chào"]}, {"TEXT": ["tạm biệt"]}
    ]


def test_payload_rejects_empty_text_list():
    with pytest.raises(ValueError, match="text"):
        payload([])
