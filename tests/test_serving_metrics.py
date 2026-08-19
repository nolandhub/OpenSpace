# ABOUTME: Test cho serving/metrics.py - chạy được khi server tắt vì pb_utils được tiêm
# ABOUTME: Hai test quan trọng nhất: label của CCU/RTF, và MetricFamily không bị GC

import gc
import weakref

import pytest

from bench.tts.metrics import rtf as bench_rtf
from serving.metrics import (
    ASR_RTF_BUCKETS,
    CCU_TTL_S,
    TTS_RTF_BUCKETS,
    ModelMetrics,
    rtf,
)


def make_fake_api():
    """Giả lập pb_utils. Trả (api, families) - families là dict tên -> family đã tạo."""
    families = {}

    class FakeMetric:
        def __init__(self, labels, buckets):
            self.labels = labels
            self.buckets = buckets
            self.observed = []
            self.set_values = []

        def observe(self, value):
            self.observed.append(value)

        def set(self, value):
            self.set_values.append(value)

    class FakeMetricFamily:
        COUNTER, GAUGE, HISTOGRAM = "COUNTER", "GAUGE", "HISTOGRAM"

        def __init__(self, name, description, kind):
            self.name = name
            self.description = description
            self.kind = kind
            self.metrics = []
            families[name] = self

        def Metric(self, labels, buckets=None):
            m = FakeMetric(labels, buckets)
            self.metrics.append(m)
            return m

    class FakeApi:
        MetricFamily = FakeMetricFamily

    return FakeApi, families


@pytest.fixture
def tts_metrics():
    api, families = make_fake_api()
    return ModelMetrics(api, "tts", "tts_0_0", TTS_RTF_BUCKETS), families


# ------------------------------------------------------------------ rtf thuần


def test_rtf_khop_voi_bench():
    """Cùng công thức với bench/tts/metrics.py, chỉ khác chỗ không làm tròn.

    Hai nguồn phải so được với nhau, lệch công thức là so nhầm mà không ai biết.
    """
    assert round(rtf(0.86, 1.0), 3) == bench_rtf(0.86, 1.0)
    assert round(rtf(2.5, 3.0), 3) == bench_rtf(2.5, 3.0)


def test_rtf_khong_lam_tron():
    """Histogram cần giá trị thô để rơi đúng bucket."""
    assert rtf(1.0, 3.0) == pytest.approx(1 / 3)


def test_rtf_audio_rong_nem_loi():
    with pytest.raises(ValueError):
        rtf(0.5, 0.0)
    with pytest.raises(ValueError):
        rtf(0.5, -1.0)


def test_buckets_tang_dan():
    for buckets in (ASR_RTF_BUCKETS, TTS_RTF_BUCKETS):
        assert buckets == sorted(buckets)
        assert len(set(buckets)) == len(buckets)


def test_ttl_khop_config_pbtxt():
    """CCU_TTL_S phải soi gương max_sequence_idle_microseconds.

    Lệch nhau thì query Grafana bỏ qua instance sai thời điểm - CCU sai âm thầm.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    cfg = (root / "model_repository/asr_streaming/config.pbtxt").read_text()
    micros = int(re.search(r"max_sequence_idle_microseconds:\s*(\d+)", cfg).group(1))
    assert CCU_TTL_S == micros / 1_000_000


# --------------------------------------------------------------- ModelMetrics


def test_rtf_khong_co_label_instance(tts_metrics):
    """HISTOGRAM observe() cộng dồn đúng qua process nên label chung là được.

    Thêm instance vào đây sẽ chẻ histogram ra thành nhiều series vô ích.
    """
    _, families = tts_metrics
    (metric,) = families["voice_rtf"].metrics
    assert metric.labels == {"model": "tts"}


def test_ccu_co_label_model_instance(tts_metrics):
    """GAUGE set() dùng label chung thì 2 instance ghi đè nhau - CCU sẽ sai.

    asr_streaming có count: 2 nên đây là ca thật, không phải giả định. Tên
    label là "model_instance" chứ không phải "instance" - Prometheus tự
    chiếm tên "instance" cho địa chỉ target lúc scrape.
    """
    _, families = tts_metrics
    for name in ("voice_ccu", "voice_ccu_updated_at"):
        (metric,) = families[name].metrics
        assert metric.labels == {"model": "tts", "model_instance": "tts_0_0"}


def test_hai_gauge_ccu_cung_label(tts_metrics):
    """PromQL join `on(model, model_instance)` - lệch label thì join rỗng, CCU đọc 0."""
    _, families = tts_metrics
    assert families["voice_ccu"].metrics[0].labels == (
        families["voice_ccu_updated_at"].metrics[0].labels
    )


def test_kind_dung(tts_metrics):
    _, families = tts_metrics
    assert families["voice_rtf"].kind == "HISTOGRAM"
    assert families["voice_ccu"].kind == "GAUGE"
    assert families["voice_ccu_updated_at"].kind == "GAUGE"


def test_buckets_truyen_vao_histogram(tts_metrics):
    _, families = tts_metrics
    assert families["voice_rtf"].metrics[0].buckets == TTS_RTF_BUCKETS


def test_observe_rtf_ghi_gia_tri_dung(tts_metrics):
    metrics, families = tts_metrics
    metrics.observe_rtf(0.86, 1.0)
    assert families["voice_rtf"].metrics[0].observed == [pytest.approx(0.86)]


def test_set_ccu_cap_nhat_ca_hai_gauge(tts_metrics):
    """Gộp làm một lời gọi để hai gauge không bao giờ lệch nhau."""
    metrics, families = tts_metrics
    metrics.set_ccu(3)
    assert families["voice_ccu"].metrics[0].set_values == [3]
    stamps = families["voice_ccu_updated_at"].metrics[0].set_values
    assert len(stamps) == 1
    import time as _time

    assert abs(stamps[0] - _time.time()) < 5  # unix epoch, không phải monotonic


def test_giu_tham_chieu_metric_family():
    """Stub Triton: 'MetricFamily' bị xoá trước 'Metric' thì Metric vô hiệu.

    Viết MetricFamily(...).Metric(...) rồi chỉ giữ Metric là family bị GC ngay.
    Lỗi này im lặng lúc chạy thật, test unit thường không thấy - nên bắt ở đây.
    """
    api, families = make_fake_api()
    m = ModelMetrics(api, "tts", "tts_0_0", TTS_RTF_BUCKETS)
    refs = {name: weakref.ref(f) for name, f in families.items()}
    families.clear()
    gc.collect()
    alive = {name for name, ref in refs.items() if ref() is not None}
    assert alive == set(refs), f"family bị GC: {set(refs) - alive}"
    assert m is not None
