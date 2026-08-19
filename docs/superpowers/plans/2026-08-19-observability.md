# Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prometheus scrape liên tục Triton (`:8002`) và vLLM (`:8080`), Grafana hiện một dashboard phủ 14 chỉ số vận hành của cả ba component.

**Architecture:** 9/14 chỉ số đã có sẵn trên hai endpoint. RTF và CCU thì không — Triton mù về nội dung tensor và không biết số phiên đang sống, nên hai model Python backend tự phát metric qua `pb_utils.MetricFamily`. Logic thuần nằm ở `serving/metrics.py` (mount vào container) để test được khi server tắt, đúng tiền lệ `streaming_search.py`. Prometheus + Grafana chạy `network_mode: host` trong một compose riêng, không đụng `serve_llm.sh`.

**Tech Stack:** Triton 25.01 (`pb_utils.MetricFamily`), vLLM 0.27.1, Prometheus, Grafana, docker compose, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-observability-design.md`

## Global Constraints

- Comment và docstring **tiếng Việt**, identifier tiếng Anh. Mọi file mới mở bằng 2 dòng `# ABOUTME:`.
- Metric tự phát dùng tiền tố `voice_` — tách hẳn khỏi `nv_` (Triton) và `vllm:` (vLLM).
- Tên metric Triton **không có hậu tố `_total`**: `nv_inference_request_success`, `nv_inference_request_failure`.
- `nv_inference_request_failure` có label `reason` ∈ {`BACKEND`,`OTHER`,`CANCELED`,`REJECTED`}. **Mọi phép tính đụng nó phải bọc `sum()`** để bỏ `reason`, nếu không phép cộng với `success` lệch label và trả rỗng.
- Metric per-model của Triton có label `version="1"`. GPU metric dùng label `gpu_uuid`.
- `voice_ccu` **bắt buộc** có label `instance`; `voice_rtf` **không** được có. Lý do ở spec §5.3.
- `ModelMetrics` **phải giữ tham chiếu** tới mọi `MetricFamily` — mất tham chiếu thì family bị GC và `Metric` vô hiệu lúc chạy.
- `CCU_TTL_S = 60.0` phải khớp `max_sequence_idle_microseconds: 60000000` trong `model_repository/asr_streaming/config.pbtxt` và con số trong PromQL của dashboard.
- Chỉ **cộng thêm** vào `model.py`, không sửa luồng inference. Không đổi `config.pbtxt`, `docker/Dockerfile`, `scripts/serve_llm.sh`.
- TDD: viết test trước, chạy xác nhận đỏ, mới code.
- HTTP dùng `urllib.request`, **không** thêm `requests`/`httpx` — quy ước đã ghi ở `client/llm_client.py:8`. Dependency mới duy nhất được phép là `PyYAML`.
- Chạy test bằng `.venv/bin/pytest`. Unit: `-m "not integration"`.

---

## File Structure

| File | Trạng thái | Trách nhiệm |
|---|---|---|
| `serving/__init__.py` | tạo | đánh dấu package |
| `serving/metrics.py` | tạo | hằng số TTL, buckets, `rtf()`, `ModelMetrics` — thuần, không import `pb_utils` |
| `scripts/serve.sh` | sửa | mount `serving/`, bật summary latencies |
| `model_repository/asr_streaming/1/model.py` | sửa | gọi `observe_rtf()` mỗi chunk, `set_ccu()` cuối `execute()` |
| `model_repository/tts/1/model.py` | sửa | gọi `observe_rtf()` mỗi request, `set_ccu()` vào/ra |
| `docker/monitoring/prometheus.yml` | tạo | 2 job, scrape 5s |
| `docker/monitoring/docker-compose.yml` | tạo | prometheus + grafana, `network_mode: host` |
| `docker/monitoring/grafana/provisioning/datasources/prometheus.yml` | tạo | datasource mặc định |
| `docker/monitoring/grafana/provisioning/dashboards/voice.yml` | tạo | trỏ tới thư mục dashboard |
| `docker/monitoring/build_dashboard.py` | tạo | sinh dashboard JSON từ bảng panel — DRY, review được |
| `docker/monitoring/grafana/dashboards/voice-serving.json` | tạo (sinh ra) | dashboard, commit kèm |
| `scripts/serve_monitoring.sh` | tạo | kiểm cổng + `docker compose up -d` |
| `tests/test_serving_metrics.py` | tạo | unit — `serving/metrics.py` |
| `tests/test_monitoring_config.py` | tạo | unit — config, dashboard, chống drift |
| `tests/test_metrics_endpoint.py` | tạo | integration — metric có thật trên `:8002` |
| `requirements.txt` | sửa | thêm `PyYAML` — dependency mới duy nhất của plan |
| `docs/observability.md` | tạo | cách chạy, cách đọc, giới hạn |
| `README.md` | sửa | mục Monitoring + bảng tài liệu |

---

### Task 1: `serving/metrics.py` — module thuần

Không cần server, không cần Triton. Đây là nền của Task 2 và 3.

**Files:**
- Create: `serving/__init__.py`, `serving/metrics.py`
- Test: `tests/test_serving_metrics.py`

**Interfaces:**
- Consumes: không có
- Produces:
  - `CCU_TTL_S: float = 60.0`
  - `ASR_RTF_BUCKETS: list[float]`, `TTS_RTF_BUCKETS: list[float]`
  - `rtf(compute_s: float, audio_s: float) -> float`
  - `ModelMetrics(metric_api, model: str, instance: str, rtf_buckets: list[float])`
    với `observe_rtf(compute_s: float, audio_s: float) -> None` và `set_ccu(n: int) -> None`

- [ ] **Step 1: Viết test đỏ**

Tạo `tests/test_serving_metrics.py`:

```python
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


def test_ccu_co_label_instance(tts_metrics):
    """GAUGE set() dùng label chung thì 2 instance ghi đè nhau - CCU sẽ sai.

    asr_streaming có count: 2 nên đây là ca thật, không phải giả định.
    """
    _, families = tts_metrics
    for name in ("voice_ccu", "voice_ccu_updated_at"):
        (metric,) = families[name].metrics
        assert metric.labels == {"model": "tts", "instance": "tts_0_0"}


def test_hai_gauge_ccu_cung_label(tts_metrics):
    """PromQL join `on(model, instance)` - lệch label thì join rỗng, CCU đọc 0."""
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
```

- [ ] **Step 2: Chạy để xác nhận đỏ**

Run: `.venv/bin/pytest tests/test_serving_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'serving'`

- [ ] **Step 3: Viết implementation tối thiểu**

Tạo `serving/__init__.py`:

```python
# ABOUTME: Package cho code dùng chung giữa các model Triton, mount vào /opt/serving
# ABOUTME: Không đặt trong model_repository vì Triton quét thư mục con ở đó như model
```

Tạo `serving/metrics.py`:

```python
# ABOUTME: Metric runtime cho Triton Python backend - RTF và CCU, thứ nv_* không thấy được
# ABOUTME: pb_utils tiêm từ ngoài để test chạy được khi server tắt

import time

# Soi gương max_sequence_idle_microseconds trong asr_streaming/config.pbtxt.
# Query Grafana dùng đúng con số này để bỏ qua instance im lặng; lệch là CCU sai
# âm thầm. test_serving_metrics.py và test_monitoring_config.py canh cả ba nơi.
CCU_TTL_S = 60.0

# Hai thang lệch hẳn một bậc (bench/: ASR quanh 0.05, TTS quanh 0.86) nên không
# dùng chung buckets được - dùng chung thì một trong hai dồn hết vào 1-2 bucket.
ASR_RTF_BUCKETS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0]
TTS_RTF_BUCKETS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]


def rtf(compute_s: float, audio_s: float) -> float:
    """Thời gian xử lý / độ dài audio. Dưới 1.0 là nhanh hơn thời gian thực.

    Cùng công thức bench/tts/metrics.py:rtf() để hai nguồn so được với nhau,
    chỉ khác: không làm tròn, vì histogram cần giá trị thô mới rơi đúng bucket.
    """
    if audio_s <= 0:
        raise ValueError(f"audio dài {audio_s}s - không tính được RTF")
    return compute_s / audio_s


class ModelMetrics:
    """Ba metric family Triton không tự có. Một đối tượng cho mỗi model instance.

    metric_api: đối tượng có .MetricFamily kèm .GAUGE/.HISTOGRAM. Production
    truyền thẳng pb_utils; test truyền fake. Tiêm thay vì import ở đầu file vì
    pb_utils chỉ tồn tại bên trong container Triton.
    """

    def __init__(self, metric_api, model: str, instance: str, rtf_buckets):
        family = metric_api.MetricFamily

        # Giữ family làm thuộc tính là BẮT BUỘC, không phải cho gọn. Stub Triton:
        # "The 'MetricFamily' object should be deleted AFTER its corresponding
        # 'Metric' objects have been deleted." Viết family(...).Metric(...) rồi
        # chỉ giữ Metric thì family bị GC và metric chết im lặng lúc chạy thật.
        self._rtf_family = family(
            name="voice_rtf",
            description="Thời gian xử lý chia độ dài audio",
            kind=family.HISTOGRAM,
        )
        # RTF để label chung: observe() của HISTOGRAM cộng dồn đúng qua nhiều
        # process, bucket gộp chuẩn. Tách theo instance chỉ chẻ vụn vô ích.
        self._rtf = self._rtf_family.Metric(
            labels={"model": model}, buckets=list(rtf_buckets)
        )

        # CCU thì ngược lại. GAUGE set() với label chung sẽ ghi đè giữa các
        # instance, giá trị cuối là của process nào chạy sau. asr_streaming có
        # count: 2 nên đây là ca thật. Tách theo instance, cộng lại ở PromQL.
        ccu_labels = {"model": model, "instance": instance}
        self._ccu_family = family(
            name="voice_ccu",
            description="Số phiên đang sống trên một model instance",
            kind=family.GAUGE,
        )
        self._ccu = self._ccu_family.Metric(labels=dict(ccu_labels))

        self._ccu_at_family = family(
            name="voice_ccu_updated_at",
            description="Unix timestamp lần cuối voice_ccu được cập nhật",
            kind=family.GAUGE,
        )
        self._ccu_at = self._ccu_at_family.Metric(labels=dict(ccu_labels))

    def observe_rtf(self, compute_s: float, audio_s: float) -> None:
        self._rtf.observe(rtf(compute_s, audio_s))

    def set_ccu(self, n: int) -> None:
        """Set cả hai gauge trong một lần gọi - tách ra là chúng lệch nhau.

        time.time() chứ không monotonic: giá trị này đem so với time() của
        Prometheus ở PromQL nên phải cùng gốc unix epoch.
        """
        self._ccu.set(n)
        self._ccu_at.set(time.time())
```

- [ ] **Step 4: Chạy để xác nhận xanh**

Run: `.venv/bin/pytest tests/test_serving_metrics.py -v`
Expected: PASS, 13 test

- [ ] **Step 5: Commit**

```bash
git add serving/ tests/test_serving_metrics.py
git commit -m "Add serving/metrics.py: RTF và CCU cho Triton Python backend

pb_utils tiêm từ ngoài để test chạy được khi server tắt. CCU tách label
instance vì GAUGE set() dùng label chung sẽ ghi đè giữa 2 instance của
asr_streaming; RTF để label chung vì HISTOGRAM observe() cộng dồn đúng."
```

---

### Task 2: Instrument `asr_streaming` + mount vào `serve.sh`

**Files:**
- Modify: `scripts/serve.sh`
- Modify: `model_repository/asr_streaming/1/model.py`
- Test: `tests/test_metrics_endpoint.py`

**Interfaces:**
- Consumes: `serving.metrics.ModelMetrics`, `ASR_RTF_BUCKETS` (Task 1)
- Produces: `voice_rtf{model="asr_streaming"}`, `voice_ccu{model="asr_streaming",instance=...}`, `voice_ccu_updated_at{...}` trên `:8002/metrics`; `nv_inference_request_summary_us{quantile}`

- [ ] **Step 1: Viết test đỏ**

Tạo `tests/test_metrics_endpoint.py`:

```python
# ABOUTME: Integration - khẳng định metric tự phát có thật trên :8002/metrics
# ABOUTME: Cần Triton đang chạy; skip nếu chưa

import urllib.request

import pytest

METRICS_URL = "http://localhost:8002/metrics"


def scrape():
    """Đọc :8002/metrics. urllib chứ không requests - repo giữ quy ước stdlib
    cho HTTP, xem comment đầu client/llm_client.py."""
    with urllib.request.urlopen(METRICS_URL, timeout=3) as r:
        return r.read().decode("utf-8")


@pytest.fixture(scope="session")
def metrics_text():
    try:
        return scrape()
    except Exception as e:
        pytest.skip(f"Không lấy được metrics tại {METRICS_URL}: {e}")


def _sample(text, name, **labels):
    """Trả giá trị float của một sample, hoặc None nếu không có dòng nào khớp."""
    for line in text.splitlines():
        if not line.startswith(name + "{"):
            continue
        if all(f'{k}="{v}"' in line for k, v in labels.items()):
            return float(line.rsplit(" ", 1)[1])
    return None


@pytest.mark.integration
def test_summary_latency_duoc_bat(metrics_text):
    """Cần --metrics-config summary_latencies=true, không có thì chỉ tính ra mean."""
    for q in ("0.5", "0.95", "0.99"):
        assert f'quantile="{q}"' in metrics_text, f"thiếu quantile {q}"


@pytest.mark.integration
def test_asr_co_metric_tu_phat(metrics_text):
    assert "voice_rtf_bucket" in metrics_text
    assert _sample(metrics_text, "voice_ccu", model="asr_streaming") is not None
    assert _sample(metrics_text, "voice_ccu_updated_at", model="asr_streaming") is not None


@pytest.mark.integration
def test_ccu_co_label_instance_tren_server_that(metrics_text):
    """asr_streaming có count: 2 - thiếu label instance là 2 process ghi đè nhau."""
    lines = [l for l in metrics_text.splitlines() if l.startswith("voice_ccu{")]
    assert lines, "không có sample voice_ccu nào"
    assert all("instance=" in l for l in lines), lines


@pytest.mark.integration
def test_rtf_khong_co_label_instance_tren_server_that(metrics_text):
    lines = [l for l in metrics_text.splitlines() if l.startswith("voice_rtf_bucket{")]
    assert lines, "không có sample voice_rtf_bucket nào"
    assert not any("instance=" in l for l in lines), lines


@pytest.mark.integration
def test_stream_asr_lam_tang_rtf_count(triton, metrics_text):
    """Chạy một stream thật rồi khẳng định histogram nhích lên và CCU về 0."""
    import numpy as np
    import tritonclient.grpc as grpcclient

    before = _sample(metrics_text, "voice_rtf_count", model="asr_streaming") or 0.0

    audio = np.zeros(3200, dtype=np.float32)  # 200ms im lặng ở 16kHz
    corrid = 987654
    for i in range(3):
        inp = grpcclient.InferInput("AUDIO_CHUNK", [len(audio)], "FP32")
        inp.set_data_from_numpy(audio)
        triton.infer(
            "asr_streaming",
            [inp],
            sequence_id=corrid,
            sequence_start=(i == 0),
            sequence_end=(i == 2),
        )

    after_text = scrape()
    after = _sample(after_text, "voice_rtf_count", model="asr_streaming") or 0.0
    assert after > before, f"voice_rtf_count không tăng: {before} -> {after}"

    ccu = sum(
        float(l.rsplit(" ", 1)[1])
        for l in after_text.splitlines()
        if l.startswith("voice_ccu{") and 'model="asr_streaming"' in l
    )
    assert ccu == 0, f"stream đã END mà CCU vẫn {ccu}"
```

- [ ] **Step 2: Chạy để xác nhận đỏ**

Dựng server hiện tại rồi chạy:

```bash
./scripts/serve.sh asr_streaming    # terminal khác, đợi load xong
.venv/bin/pytest tests/test_metrics_endpoint.py -v
```
Expected: FAIL — `test_summary_latency_duoc_bat` và các test `voice_*` đều đỏ (chưa có flag, chưa có metric)

- [ ] **Step 3: Sửa `scripts/serve.sh`**

Thêm mount `serving/` — đặt ngay dưới dòng mount `model_repository`:

```bash
docker run --gpus all --rm "${TTY[@]}" --net host --shm-size 1g \
  --name triton-voice-server \
  -v "$ROOT/model_repository:/models" \
  -v "$ROOT/serving:/opt/serving/serving:ro" \
  triton-voice tritonserver "${ARGS[@]}"
```

Thêm hai flag metrics vào `ARGS` — ngay sau dòng khởi tạo `ARGS`:

```bash
# Mặc định Triton chỉ phơi counter cộng dồn, từ đó chỉ tính ra được MEAN latency.
# summary_latencies cho quantile thật; quantile viết kèm sai số cho phép.
ARGS=(--model-repository=/models
      --metrics-config summary_latencies=true
      --metrics-config 'summary_quantiles=0.5:0.05,0.95:0.01,0.99:0.001')
```

Mount đích là `/opt/serving/serving` (không phải `/opt/serving`) để import trong container
giống hệt import trong test: `from serving.metrics import ...`.

- [ ] **Step 4: Sửa `model_repository/asr_streaming/1/model.py`**

Thêm hằng số cạnh `STATE_TTL_S`:

```python
SAMPLE_RATE = 16000          # mẫu số của RTF; khớp fbank và client
```

Thêm import — ngay dưới khối `sys.path.insert` đang có:

```python
sys.path.insert(0, "/opt/serving")
from serving.metrics import ASR_RTF_BUCKETS, ModelMetrics  # noqa: E402
```

Trong `initialize()`, thêm dòng cuối (sau `self.streams = {}`):

```python
self.metrics = ModelMetrics(
    pb_utils, "asr_streaming", args["model_instance_name"], ASR_RTF_BUCKETS
)
```

Trong `_handle()`, bọc phần tính toán. Thay khối từ `new_feat = stream.fbank...` tới `self._advance(...)` bằng:

```python
t0 = time.perf_counter()
new_feat = stream.fbank.accept_waveform(audio)
if end:
    tail = stream.fbank.flush()
    if len(tail):
        new_feat = np.concatenate([new_feat, tail]) if len(new_feat) else tail
self._advance(stream, new_feat, flush=end)
# Chunk rỗng (END không kèm audio) là hợp lệ - bỏ qua thay vì chia cho 0.
if len(audio):
    self.metrics.observe_rtf(time.perf_counter() - t0, len(audio) / SAMPLE_RATE)
```

Trong `execute()`, thêm trước `return responses`:

```python
# Sau vòng lặp: chunk có END đã xoá state của nó, số này mới đúng.
self.metrics.set_ccu(len(self.streams))
return responses
```

- [ ] **Step 5: Chạy lại test**

```bash
./scripts/serve.sh asr_streaming     # dựng lại để nạp flag + mount mới
.venv/bin/pytest tests/test_metrics_endpoint.py -v
```
Expected: PASS toàn bộ (test `voice_ccu` của tts sẽ chưa có — Task 3)

- [ ] **Step 6: Xác nhận không làm chậm đường nóng**

```bash
./scripts/perf.sh asr_streaming
```
Expected: throughput vẫn quanh 138 infer/s (`bench/asr_streaming/README.md`). Tụt quá 5% thì dừng lại điều tra trước khi đi tiếp — spec R7.

- [ ] **Step 7: Commit**

```bash
git add scripts/serve.sh model_repository/asr_streaming/1/model.py tests/test_metrics_endpoint.py
git commit -m "Instrument asr_streaming: voice_rtf per chunk, voice_ccu per instance

serve.sh mount serving/ vào /opt/serving/serving để import trong container
giống hệt trong test, và bật summary_latencies - mặc định chỉ có counter
cộng dồn nên chỉ tính ra được mean chứ không có p50/95/99."
```

---

### Task 3: Instrument `tts`

**Files:**
- Modify: `model_repository/tts/1/model.py`
- Test: `tests/test_metrics_endpoint.py` (thêm test)

**Interfaces:**
- Consumes: `serving.metrics.ModelMetrics`, `TTS_RTF_BUCKETS` (Task 1); mount `/opt/serving` (Task 2)
- Produces: `voice_rtf{model="tts"}`, `voice_ccu{model="tts",instance=...}`

- [ ] **Step 1: Viết test đỏ**

Thêm vào cuối `tests/test_metrics_endpoint.py`:

```python
@pytest.mark.integration
def test_tts_co_metric_tu_phat(metrics_text):
    assert _sample(metrics_text, "voice_ccu", model="tts") is not None
    assert _sample(metrics_text, "voice_rtf_bucket", model="tts", le="1") is not None


@pytest.mark.integration
def test_tts_rtf_tang_sau_mot_cau(triton):
    """RTF đo trong execute() nên KHÔNG có RTT gRPC - thấp hơn bench/tts một chút.

    Ở đây chỉ khẳng định nó có ghi nhận, không so số với bench.
    """
    import numpy as np
    import tritonclient.grpc as grpcclient

    before = _sample(scrape(), "voice_rtf_count", model="tts") or 0.0

    inp = grpcclient.InferInput("TEXT", [1], "BYTES")
    inp.set_data_from_numpy(np.array(["Xin chào".encode("utf-8")], dtype=object))
    triton.infer("tts", [inp])

    after = _sample(scrape(), "voice_rtf_count", model="tts") or 0.0
    assert after > before, f"voice_rtf_count của tts không tăng: {before} -> {after}"


@pytest.mark.integration
def test_tts_ccu_ve_0_khi_ranh(triton, metrics_text):
    """execute() set 0 trong finally - xong request là phải về 0."""
    total = sum(
        float(l.rsplit(" ", 1)[1])
        for l in metrics_text.splitlines()
        if l.startswith("voice_ccu{") and 'model="tts"' in l
    )
    assert total == 0, f"tts rảnh mà CCU vẫn {total}"
```

- [ ] **Step 2: Chạy để xác nhận đỏ**

```bash
./scripts/serve.sh                   # cả 2 model
.venv/bin/pytest tests/test_metrics_endpoint.py -k tts -v
```
Expected: FAIL — `voice_ccu{model="tts"}` chưa tồn tại

- [ ] **Step 3: Sửa `model_repository/tts/1/model.py`**

Thêm `import time` vào khối import chuẩn (sau `import os`), và thêm import module dùng chung sau khối import zipvoice:

```python
import sys

sys.path.insert(0, "/opt/serving")
from serving.metrics import TTS_RTF_BUCKETS, ModelMetrics  # noqa: E402
```

Trong `initialize()`, thêm dòng cuối:

```python
self.metrics = ModelMetrics(
    pb_utils, "tts", args["model_instance_name"], TTS_RTF_BUCKETS
)
```

Sửa `execute()` — bọc cả thân hàm để CCU luôn về 0:

```python
def execute(self, requests):
    responses = []
    self.metrics.set_ccu(len(requests))
    try:
        for request in requests:
            ...                       # giữ nguyên toàn bộ phần đang có
    finally:
        # finally chứ không phải cuối hàm: request lỗi giữa chừng mà không về 0
        # thì gauge treo mãi ở giá trị cũ.
        self.metrics.set_ccu(0)
    return responses
```

Bên trong vòng lặp, bọc thời gian sinh. Thay khối `try: ... finally: xoá tmp` bằng:

```python
t0 = time.perf_counter()
try:
    with torch.inference_mode():
        generate_sentence(
            save_path=out_path,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            text=text,
            model=self.model,
            vocoder=self.vocoder,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            device=self.device,
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
            sampling_rate=SAMPLING_RATE,
        )
    wav, _ = sf.read(out_path, dtype="float32")
finally:
    for path in tmp_files:
        if os.path.exists(path):
            os.remove(path)
# Cùng công thức bench/tts/metrics.py nhưng đo trong execute() nên không tính
# RTT gRPC - số ở đây thấp hơn bench một chút, đó là đúng chứ không phải lệch.
self.metrics.observe_rtf(time.perf_counter() - t0, len(wav) / SAMPLING_RATE)
```

- [ ] **Step 4: Chạy lại test**

```bash
./scripts/serve.sh
.venv/bin/pytest tests/test_metrics_endpoint.py -v
```
Expected: PASS toàn bộ

- [ ] **Step 5: Đối chiếu số với bench**

```bash
.venv/bin/python bench/tts/metrics.py
curl -s localhost:8002/metrics | grep 'voice_rtf_sum{model="tts"}'
curl -s localhost:8002/metrics | grep 'voice_rtf_count{model="tts"}'
```
Expected: `sum/count` (RTF trung bình phía server) **thấp hơn** `rtf_p50` của bench, chênh cỡ vài phần trăm. Cao hơn bench là sai — dừng lại điều tra.

- [ ] **Step 6: Commit**

```bash
git add model_repository/tts/1/model.py tests/test_metrics_endpoint.py
git commit -m "Instrument tts: voice_rtf per request, voice_ccu quanh execute()

set_ccu(0) đặt trong finally chứ không cuối hàm - request lỗi giữa chừng
mà không về 0 thì gauge treo mãi ở giá trị cũ."
```

---

### Task 4: Prometheus

**Files:**
- Create: `docker/monitoring/prometheus.yml`, `docker/monitoring/docker-compose.yml`, `scripts/serve_monitoring.sh`
- Test: `tests/test_monitoring_config.py`

**Interfaces:**
- Consumes: endpoint `:8002` (Task 2, 3) và `:8080` (vLLM có sẵn)
- Produces: Prometheus ở `:9090` với 2 job `triton` và `vllm`

- [ ] **Step 1: Thêm PyYAML vào `requirements.txt`**

Test đọc `prometheus.yml` và `docker-compose.yml` nên cần parser YAML — stdlib
không có. Đây là dependency mới **duy nhất** của cả plan; HTTP vẫn dùng
`urllib.request` theo quy ước ở `client/llm_client.py`.

Thêm dòng vào `requirements.txt` (giữ nguyên thứ tự các dòng khác):

```
PyYAML==6.0.2
```

Rồi: `.venv/bin/pip install -r requirements.txt`

- [ ] **Step 3: Viết test đỏ**

Tạo `tests/test_monitoring_config.py`:

```python
# ABOUTME: Unit test cho config monitoring - không cần server nào chạy
# ABOUTME: Nhiệm vụ chính là chống drift giữa CCU_TTL_S, config.pbtxt và dashboard

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
MON = ROOT / "docker" / "monitoring"


@pytest.fixture(scope="module")
def prom_cfg():
    return yaml.safe_load((MON / "prometheus.yml").read_text())


def test_scrape_interval_5s(prom_cfg):
    """CCU và queue depth là gauge tức thời - 15s mặc định là trượt mất spike."""
    assert prom_cfg["global"]["scrape_interval"] == "5s"


def test_dung_hai_job(prom_cfg):
    jobs = {j["job_name"]: j for j in prom_cfg["scrape_configs"]}
    assert set(jobs) == {"triton", "vllm"}
    assert jobs["triton"]["static_configs"][0]["targets"] == ["localhost:8002"]
    assert jobs["vllm"]["static_configs"][0]["targets"] == ["localhost:8080"]


def test_compose_dung_host_network():
    """Triton chạy --net host nên :8002 chỉ thấy từ host namespace."""
    compose = yaml.safe_load((MON / "docker-compose.yml").read_text())
    for name, svc in compose["services"].items():
        assert svc.get("network_mode") == "host", f"{name} không dùng host network"
```

- [ ] **Step 3: Chạy để xác nhận đỏ**

Run: `.venv/bin/pytest tests/test_monitoring_config.py -v`
Expected: FAIL — `FileNotFoundError: docker/monitoring/prometheus.yml`

- [ ] **Step 4: Tạo `docker/monitoring/prometheus.yml`**

```yaml
# ABOUTME: Scrape config - Triton :8002 và vLLM :8080, cả hai qua localhost
# ABOUTME: Chạy network_mode host nên localhost ở đây là host thật

global:
  # 5s chứ không phải 15s mặc định: voice_ccu và nv_inference_pending_request_count
  # là gauge tức thời, 15s là trượt mất đúng cái spike cần nhìn.
  scrape_interval: 5s
  evaluation_interval: 15s

scrape_configs:
  - job_name: triton
    static_configs:
      - targets: ["localhost:8002"]

  - job_name: vllm
    static_configs:
      - targets: ["localhost:8080"]
```

- [ ] **Step 5: Tạo `docker/monitoring/docker-compose.yml`**

```yaml
# ABOUTME: Prometheus + Grafana cho stack voice serving, dựng độc lập với Triton/vLLM
# ABOUTME: Dùng: ./scripts/serve_monitoring.sh

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: triton-voice-prometheus
    restart: unless-stopped
    # host network vì Triton chạy --net host, :8002 không thấy được từ bridge.
    # Bridge sẽ phải dựa vào host.docker.internal - thêm một lớp hỏng được mà
    # không đổi lại gì.
    network_mode: host
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.path=/prometheus
      - --storage.tsdb.retention.time=15d

  grafana:
    image: grafana/grafana:latest
    container_name: triton-voice-grafana
    restart: unless-stopped
    network_mode: host
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana-data:/var/lib/grafana
    environment:
      # Máy dev một người dùng - bắt đăng nhập chỉ tổ vướng.
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: Admin
      GF_AUTH_DISABLE_LOGIN_FORM: "true"

volumes:
  prometheus-data:
  grafana-data:
```

- [ ] **Step 6: Tạo `scripts/serve_monitoring.sh`**

```bash
#!/usr/bin/env bash
# ABOUTME: Dựng Prometheus (9090) + Grafana (3000) scrape Triton và vLLM
# ABOUTME: Dùng: ./scripts/serve_monitoring.sh | ./scripts/serve_monitoring.sh down

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE="$ROOT/docker/monitoring/docker-compose.yml"

if [ "${1:-up}" = "down" ]; then
  docker compose -f "$COMPOSE" down
  exit 0
fi

# Cả hai container dùng host network nên cổng đụng là chết im lặng trong log
# container. Chặn ở đây để báo cho rõ.
for port in 9090 3000; do
  if ss -ltn "sport = :$port" | grep -q LISTEN; then
    echo "dừng: cổng $port đang bị chiếm." >&2
    echo "      ss -ltnp 'sport = :$port'  để xem tiến trình nào." >&2
    exit 1
  fi
done

docker compose -f "$COMPOSE" up -d

echo
echo "Prometheus  http://localhost:9090/targets"
echo "Grafana     http://localhost:3000"
echo
echo "Triton và vLLM phải chạy sẵn thì target mới UP:"
echo "  ./scripts/serve.sh  &&  ./scripts/serve_llm.sh"
```

Rồi: `chmod +x scripts/serve_monitoring.sh`

- [ ] **Step 7: Chạy test và dựng thật**

```bash
.venv/bin/pytest tests/test_monitoring_config.py -v
./scripts/serve_monitoring.sh
sleep 15
curl -s localhost:9090/api/v1/targets | python3 -m json.tool | grep -E '"health"|"job"'
```
Expected: test PASS; cả hai target `"health": "up"` (vLLM `down` nếu chưa chạy `serve_llm.sh` — chấp nhận ở bước này)

- [ ] **Step 8: Commit**

```bash
git add requirements.txt docker/monitoring/prometheus.yml \
        docker/monitoring/docker-compose.yml \
        scripts/serve_monitoring.sh tests/test_monitoring_config.py
git commit -m "Add Prometheus scraping Triton và vLLM

network_mode host vì Triton chạy --net host nên :8002 không thấy được từ
bridge. scrape_interval 5s chứ không 15s: CCU và queue depth là gauge tức
thời, 15s trượt mất spike."
```

---

### Task 5: Grafana — dashboard 14 chỉ số

Dashboard JSON viết tay là ~1200 dòng lặp lại, không ai review nổi. Sinh từ một bảng panel: bảng đó mới là thứ đáng đọc, và test kiểm được PromQL trực tiếp.

**Files:**
- Create: `docker/monitoring/build_dashboard.py`
- Create: `docker/monitoring/grafana/provisioning/datasources/prometheus.yml`
- Create: `docker/monitoring/grafana/provisioning/dashboards/voice.yml`
- Create: `docker/monitoring/grafana/dashboards/voice-serving.json` (sinh ra, commit kèm)
- Test: `tests/test_monitoring_config.py` (thêm test)

**Interfaces:**
- Consumes: `serving.metrics.CCU_TTL_S` (Task 1); metric của Task 2, 3, 4
- Produces: dashboard `voice-serving` trong Grafana

- [ ] **Step 1: Viết test đỏ**

Thêm vào `tests/test_monitoring_config.py`:

```python
import json
import subprocess
import sys

DASHBOARD = MON / "grafana" / "dashboards" / "voice-serving.json"

# Chốt từ dump :8002/metrics và source vLLM 0.27.1 (spec §8). Panel dùng tên
# ngoài danh sách này gần như chắc chắn là gõ nhầm - Grafana sẽ im lặng hiện
# "No data" chứ không báo lỗi.
KNOWN_METRICS = {
    "nv_inference_request_success",
    "nv_inference_request_failure",
    "nv_inference_request_summary_us",
    "nv_inference_pending_request_count",
    "nv_gpu_utilization",
    "nv_gpu_memory_used_bytes",
    "nv_gpu_memory_total_bytes",
    "voice_rtf_bucket",
    "voice_ccu",
    "voice_ccu_updated_at",
    "vllm:request_success_total",
    "vllm:e2e_request_latency_seconds_bucket",
    "vllm:time_to_first_token_seconds_bucket",
    "vllm:inter_token_latency_seconds_bucket",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
}


@pytest.fixture(scope="module")
def dashboard():
    return json.loads(DASHBOARD.read_text())


def _panels(dash):
    for panel in dash["panels"]:
        yield panel
        for sub in panel.get("panels", []):
            yield sub


def _exprs(dash):
    for panel in _panels(dash):
        for target in panel.get("targets", []):
            yield panel["title"], target["expr"]


def test_dashboard_khop_generator():
    """JSON commit kèm phải đúng bằng output của generator - không sửa tay."""
    out = subprocess.run(
        [sys.executable, str(MON / "build_dashboard.py"), "--stdout"],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout
    assert json.loads(out) == json.loads(DASHBOARD.read_text()), (
        "voice-serving.json lệch với build_dashboard.py - chạy lại generator"
    )


def test_moi_panel_co_datasource(dashboard):
    for panel in _panels(dashboard):
        if panel.get("type") == "row":
            continue
        assert panel.get("datasource"), f"panel thiếu datasource: {panel['title']}"


def test_moi_metric_deu_co_that(dashboard):
    for title, expr in _exprs(dashboard):
        used = set(re.findall(r"[a-zA-Z_:][a-zA-Z0-9_:]*", expr))
        unknown = {
            m for m in used
            if (m.startswith(("nv_", "vllm", "voice_")) and m not in KNOWN_METRICS)
        }
        assert not unknown, f"panel {title!r} dùng metric lạ: {unknown}"


def test_moi_phep_tinh_failure_deu_boc_sum(dashboard):
    """nv_inference_request_failure có label `reason`.

    Không bọc sum() thì phép cộng với success lệch label, Prometheus không
    match được cặp nào và trả rỗng - panel "No data" chứ không phải số sai.
    """
    for title, expr in _exprs(dashboard):
        for m in re.finditer(r"nv_inference_request_failure", expr):
            prefix = expr[: m.start()]
            assert "sum(" in prefix, f"panel {title!r}: failure không bọc sum()"


def test_ttl_trong_dashboard_khop_hang_so():
    """Số 60 nằm ở 3 nơi - config.pbtxt, CCU_TTL_S, PromQL. Lệch là CCU sai âm thầm."""
    from serving.metrics import CCU_TTL_S

    exprs = [e for _, e in _exprs(json.loads(DASHBOARD.read_text()))]
    ccu_exprs = [e for e in exprs if "voice_ccu_updated_at" in e]
    assert ccu_exprs, "không panel nào lọc CCU theo tuổi"
    for expr in ccu_exprs:
        found = re.search(r"<\s*bool\s+([\d.]+)", expr)
        assert found, f"query CCU thiếu ngưỡng bool: {expr}"
        assert float(found.group(1)) == CCU_TTL_S


def test_phu_du_14_chi_so(dashboard):
    """Mỗi chỉ số trong spec §2 phải có ít nhất một panel."""
    titles = " | ".join(p["title"] for p in _panels(dashboard)).lower()
    for keyword in ("rps", "success", "error", "p50", "p95", "p99",
                    "gpu util", "gpu mem", "queue", "ccu", "rtf", "ttft", "tpot"):
        assert keyword in titles, f"thiếu panel cho: {keyword}"
```

- [ ] **Step 2: Chạy để xác nhận đỏ**

Run: `.venv/bin/pytest tests/test_monitoring_config.py -v`
Expected: FAIL — `FileNotFoundError: .../voice-serving.json`

- [ ] **Step 3: Tạo `docker/monitoring/build_dashboard.py`**

```python
#!/usr/bin/env python3
# ABOUTME: Sinh voice-serving.json từ bảng panel - JSON viết tay ~1200 dòng không review nổi
# ABOUTME: Chạy: python3 docker/monitoring/build_dashboard.py [--stdout]

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serving.metrics import CCU_TTL_S  # noqa: E402

DS = {"type": "prometheus", "uid": "prometheus"}
OUT = Path(__file__).parent / "grafana" / "dashboards" / "voice-serving.json"

# Instance nào không được chạm trong CCU_TTL_S giây thì nhân 0. Mô phỏng đúng
# cái _sweep() sẽ làm nếu nó được chạy - xem spec §6.
def ccu(selector: str) -> str:
    return (
        f"sum(voice_ccu{selector} * on(model, instance) "
        f"(time() - voice_ccu_updated_at < bool {CCU_TTL_S:g}))"
    )


# sum() bắt buộc quanh failure: nó có label `reason`, không bọc thì phép cộng
# với success lệch label và Prometheus trả rỗng.
def error_rate(model: str) -> str:
    f = f'sum(rate(nv_inference_request_failure{{model="{model}"}}[1m]))'
    s = f'sum(rate(nv_inference_request_success{{model="{model}"}}[1m]))'
    return f"{f} / clamp_min({s} + {f}, 1e-9)"


def triton_rows(model: str) -> list:
    sel = f'{{model="{model}"}}'
    q = lambda p: f'nv_inference_request_summary_us{{model="{model}",quantile="{p}"}} / 1000'
    return [
        (f"{model} · RPS", [f"rate(nv_inference_request_success{sel}[1m])"], "reqps"),
        (f"{model} · Latency p50/p95/p99", [q("0.5"), q("0.95"), q("0.99")], "ms"),
        (f"{model} · CCU", [ccu(sel)], "short"),
        (f"{model} · Queue depth", [f"nv_inference_pending_request_count{sel}"], "short"),
        (
            f"{model} · RTF p50/p95/p99",
            [
                f"histogram_quantile({p}, sum by(le) (rate(voice_rtf_bucket{sel}[5m])))"
                for p in ("0.5", "0.95", "0.99")
            ],
            "none",
        ),
        (f"{model} · Error rate", [error_rate(model)], "percentunit"),
    ]


LLM_E2E = "rate(vllm:e2e_request_latency_seconds_bucket[5m])"
ROWS = [
    (
        "OVERVIEW",
        [
            (
                "RPS tổng",
                [
                    "sum(rate(nv_inference_request_success[1m]))"
                    " + sum(rate(vllm:request_success_total[1m]))"
                ],
                "reqps",
            ),
            (
                "Success rate (Triton)",
                [
                    "sum(rate(nv_inference_request_success[1m]))"
                    " / clamp_min(sum(rate(nv_inference_request_success[1m]))"
                    " + sum(rate(nv_inference_request_failure[1m])), 1e-9)"
                ],
                "percentunit",
            ),
            (
                "Error rate (Triton)",
                [
                    "sum(rate(nv_inference_request_failure[1m]))"
                    " / clamp_min(sum(rate(nv_inference_request_success[1m]))"
                    " + sum(rate(nv_inference_request_failure[1m])), 1e-9)"
                ],
                "percentunit",
            ),
            ("CCU tổng", [f"{ccu('')} + sum(vllm:num_requests_running)"], "short"),
        ],
    ),
    ("ASR_STREAMING", triton_rows("asr_streaming")),
    ("TTS", triton_rows("tts")),
    (
        "LLM (vLLM)",
        [
            ("llm · RPS", ["sum(rate(vllm:request_success_total[1m]))"], "reqps"),
            (
                "llm · Latency p50/p95/p99",
                [f"histogram_quantile({p}, sum by(le) ({LLM_E2E}))" for p in ("0.5", "0.95", "0.99")],
                "s",
            ),
            ("llm · CCU", ["vllm:num_requests_running"], "short"),
            ("llm · Queue depth", ["vllm:num_requests_waiting"], "short"),
            (
                "llm · TTFT p50/p95/p99",
                [
                    f"histogram_quantile({p}, sum by(le) "
                    f"(rate(vllm:time_to_first_token_seconds_bucket[5m])))"
                    for p in ("0.5", "0.95", "0.99")
                ],
                "s",
            ),
            (
                "llm · TPOT p50/p95/p99",
                [
                    f"histogram_quantile({p}, sum by(le) "
                    f"(rate(vllm:inter_token_latency_seconds_bucket[5m])))"
                    for p in ("0.5", "0.95", "0.99")
                ],
                "s",
            ),
            (
                "llm · Error rate (proxy: abort)",
                [
                    'sum(rate(vllm:request_success_total{finished_reason="abort"}[1m]))'
                    " / clamp_min(sum(rate(vllm:request_success_total[1m])), 1e-9)"
                ],
                "percentunit",
            ),
        ],
    ),
    (
        "GPU (chung cả máy)",
        [
            ("GPU Utilization", ["nv_gpu_utilization * 100"], "percent"),
            (
                "GPU Memory usage",
                ["nv_gpu_memory_used_bytes / nv_gpu_memory_total_bytes * 100"],
                "percent",
            ),
            ("GPU Memory used", ["nv_gpu_memory_used_bytes"], "bytes"),
        ],
    ),
]


def build() -> dict:
    panels, pid, y = [], 1, 0
    for row_title, specs in ROWS:
        panels.append(
            {
                "type": "row", "title": row_title, "id": pid,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
                "collapsed": False, "panels": [],
            }
        )
        pid += 1
        y += 1
        for i, (title, exprs, unit) in enumerate(specs):
            panels.append(
                {
                    "type": "timeseries", "title": title, "id": pid,
                    "datasource": DS,
                    "gridPos": {"h": 8, "w": 8, "x": (i % 3) * 8, "y": y + (i // 3) * 8},
                    "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
                    "targets": [
                        {"datasource": DS, "expr": e, "refId": chr(65 + n),
                         "legendFormat": "{{model}}{{instance}}{{quantile}}"}
                        for n, e in enumerate(exprs)
                    ],
                }
            )
            pid += 1
        y += ((len(specs) + 2) // 3) * 8
    return {
        "uid": "voice-serving",
        "title": "Voice Serving",
        "tags": ["triton", "vllm", "voice"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "10s",
        "time": {"from": "now-30m", "to": "now"},
        "panels": panels,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="in ra thay vì ghi file")
    args = ap.parse_args()
    text = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    if args.stdout:
        print(text, end="")
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text)
        print(f"đã ghi {OUT}")
```

- [ ] **Step 4: Tạo provisioning**

`docker/monitoring/grafana/provisioning/datasources/prometheus.yml`:

```yaml
# ABOUTME: Datasource Prometheus, uid cố định để dashboard JSON trỏ thẳng vào
# ABOUTME: localhost:9090 vì Grafana chạy network_mode host

apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    uid: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
```

`docker/monitoring/grafana/provisioning/dashboards/voice.yml`:

```yaml
# ABOUTME: Nạp mọi dashboard JSON trong /var/lib/grafana/dashboards lúc khởi động
# ABOUTME: allowUiUpdates false - dashboard sinh từ build_dashboard.py, sửa tay sẽ bị ghi đè

apiVersion: 1

providers:
  - name: voice
    orgId: 1
    type: file
    disableDeletion: false
    allowUiUpdates: false
    updateIntervalSeconds: 30
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: false
```

- [ ] **Step 5: Sinh dashboard và chạy test**

```bash
.venv/bin/python docker/monitoring/build_dashboard.py
.venv/bin/pytest tests/test_monitoring_config.py -v
```
Expected: PASS toàn bộ

- [ ] **Step 6: Xác nhận bằng mắt trên Grafana thật**

```bash
./scripts/serve.sh                       # terminal 1
./scripts/serve_llm.sh                   # terminal 2, sau khi Triton load xong
./scripts/serve_monitoring.sh            # terminal 3
.venv/bin/python client/asr_streaming_client.py tests/assets/sample_vi.wav
.venv/bin/python client/tts_client.py --text "Xin chào" --out /tmp/ra.wav
.venv/bin/python client/llm_client.py --prompt "Thủ đô Việt Nam?" --no-think
```

Mở `http://localhost:3000` → dashboard **Voice Serving**.
Expected: **không panel nào "No data"**. Panel nào trống thì ghi lại tên rồi soi query đó ở `http://localhost:9090/graph` trước khi đi tiếp.

- [ ] **Step 7: Commit**

```bash
git add docker/monitoring/build_dashboard.py docker/monitoring/grafana \
        tests/test_monitoring_config.py
git commit -m "Add Grafana dashboard sinh từ generator, 14 chỉ số

JSON viết tay ~1200 dòng lặp lại không ai review nổi; bảng panel trong
build_dashboard.py mới là thứ đáng đọc. Test khẳng định JSON commit kèm
đúng bằng output generator, mọi metric có thật, và mọi phép tính đụng
nv_inference_request_failure đều bọc sum() vì nó có label reason."
```

---

### Task 6: Tài liệu

**Files:**
- Create: `docs/observability.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: mọi thứ từ Task 1–5
- Produces: không có

- [ ] **Step 1: Tạo `docs/observability.md`**

```markdown
# Observability

Prometheus scrape liên tục, Grafana hiện 14 chỉ số của cả ba component.
Thiết kế và lý do từng quyết định: `superpowers/specs/2026-08-19-observability-design.md`.

## Chạy

    ./scripts/serve.sh              # Triton trước
    ./scripts/serve_llm.sh          # rồi vLLM
    ./scripts/serve_monitoring.sh   # Prometheus 9090 + Grafana 3000

Grafana `http://localhost:3000` → dashboard **Voice Serving**. Không cần đăng nhập.
Tắt: `./scripts/serve_monitoring.sh down`.

## Ranh giới với `bench/`

|  | `bench/` + perf_analyzer | ở đây |
|---|---|---|
| Khi nào có số | chạy tay, tải tổng hợp | liên tục, traffic thật |
| Sống bao lâu | một lần chạy | chuỗi thời gian |
| Trả lời | "trần của hệ ở đâu" | "ngay bây giờ hệ thế nào" |

Không thay thế nhau.

## Metric tự phát

Triton mù về nội dung tensor nên không biết audio dài bao nhiêu; và số phiên
đang sống chỉ `self.streams` trong process Python biết. Hai thứ đó do
`serving/metrics.py` phát qua `pb_utils.MetricFamily`, xuất hiện ngay trên
`:8002/metrics` cạnh `nv_*`.

| Metric | Kind | Labels |
|---|---|---|
| `voice_rtf` | histogram | `model` |
| `voice_ccu` | gauge | `model`, `instance` |
| `voice_ccu_updated_at` | gauge | `model`, `instance` |

`voice_ccu` **phải** có label `instance` còn `voice_rtf` **không** — hai instance
của `asr_streaming` cùng `set()` vào một gauge sẽ ghi đè nhau, còn `observe()`
của histogram thì cộng dồn đúng. Chi tiết ở spec §5.3.

## Sửa dashboard

Sửa bảng panel trong `docker/monitoring/build_dashboard.py`, rồi:

    .venv/bin/python docker/monitoring/build_dashboard.py
    .venv/bin/pytest tests/test_monitoring_config.py

Sửa thẳng trong UI Grafana sẽ bị ghi đè (`allowUiUpdates: false`).

## Đọc số cho đúng

- **CCU nghĩa là "sequence chưa hết hạn"**, không phải "người thật đang nói".
  Client chết mà chưa quá 60s thì Triton vẫn giữ slot, vẫn được đếm.
- **ASR RTF lưỡng đỉnh.** Encoder chỉ chạy khi đủ 45 khung, ~2/5 số chunk không
  kích nên gần như miễn phí. Đừng đọc p50 như chi phí trung bình mỗi chunk.
- **TTS RTF thấp hơn `bench/tts`** vài phần trăm: đo trong `execute()` nên không
  tính RTT gRPC. Lệch đó là đúng.
- **GPU là của cả máy.** Triton và vLLM dùng chung card, không tách được.
- **Error rate LLM là proxy.** vLLM không đếm HTTP 4xx/5xx; panel dùng tỉ lệ
  `finished_reason="abort"`.
- **`CANCELED` nằm trong tổng lỗi của Triton** — client streaming ngắt giữa
  chừng cũng sinh ra nó, không phải lúc nào cũng là lỗi thật.

## Sửa lỗi thường gặp

| Triệu chứng | Nguyên nhân |
|---|---|
| Panel p50/95/99 trống | `serve.sh` thiếu `--metrics-config summary_latencies=true` |
| Panel `voice_*` trống | `serve.sh` thiếu mount `serving/` vào `/opt/serving/serving` |
| CCU luôn bằng 0 | join `on(model, instance)` hỏng — hai gauge lệch label |
| Target `vllm` DOWN | chưa chạy `serve_llm.sh`, hoặc `PORT` khác 8080 |
| `serve_monitoring.sh` báo cổng bị chiếm | 9090/3000 đang có tiến trình khác |
```

- [ ] **Step 2: Sửa `README.md`**

Thêm mục **Monitoring** ngay sau mục **Benchmark**:

```markdown
## Monitoring

    ./scripts/serve_monitoring.sh        # Prometheus 9090 + Grafana 3000

Grafana `http://localhost:3000` → dashboard **Voice Serving**: RPS, p50/95/99,
CCU, queue depth, RTF, TTFT/TPOT, GPU, error rate cho cả ASR, TTS và LLM.
Triton và vLLM phải chạy trước thì target mới UP.

RTF và CCU do `serving/metrics.py` tự phát — Triton không biết audio dài bao
nhiêu, cũng không biết bao nhiêu phiên đang sống. Cách đọc: `docs/observability.md`.
```

Thêm hai dòng vào bảng **Tài liệu** ở cuối:

```markdown
| `docs/observability.md` | cách chạy monitoring, cách đọc từng chỉ số, giới hạn |
```

- [ ] **Step 3: Chạy toàn bộ test**

```bash
.venv/bin/pytest tests/ -m "not integration" -v
.venv/bin/pytest tests/ -v                      # cần Triton + vLLM chạy
```
Expected: PASS toàn bộ, không có test cũ nào đỏ thêm

- [ ] **Step 4: Commit**

```bash
git add docs/observability.md README.md
git commit -m "Document observability: cách chạy, cách đọc, giới hạn

Mục 'Đọc số cho đúng' là phần đáng giá nhất - CCU nghĩa là sequence chưa
hết hạn chứ không phải người thật đang nói, và ASR RTF lưỡng đỉnh theo
thiết kế nên p50 không phải chi phí trung bình mỗi chunk."
```

---

## Kiểm sau khi xong

- [ ] `.venv/bin/pytest tests/ -m "not integration"` xanh
- [ ] `.venv/bin/pytest tests/` xanh (Triton + vLLM đang chạy)
- [ ] `http://localhost:9090/targets` — cả `triton` và `vllm` đều UP
- [ ] Dashboard **Voice Serving** không panel nào "No data" sau khi chạy cả 3 client
- [ ] `./scripts/perf.sh asr_streaming` — throughput không tụt quá 5% so với 138 infer/s
- [ ] `git status` sạch
