# Observability — Design Spec

**Ngày:** 2026-08-19
**Người thực hiện:** Nhân (VF-KPTX-VPTAITX)
**Trạng thái:** Design đã chốt, chờ implementation plan
**Kế thừa:** `2026-08-10-streaming-asr-design.md`, `2026-08-09-triton-voice-serving-design.md`

---

## 1. Bối cảnh

Hệ đang chạy ba thành phần: `asr_streaming` và `tts` trên một Triton (`:8002/metrics`), vLLM đứng riêng (`:8080/metrics`). Cả hai đã phơi Prometheus format sẵn nhưng **chưa ai scrape** — số liệu chỉ tồn tại lúc chạy `scripts/perf.sh` hoặc `bench/*/metrics.py` bằng tay, và biến mất ngay sau đó.

Khác biệt căn bản với `bench/`:

| | `bench/` + perf_analyzer | Observability |
|---|---|---|
| Khi nào có số | chạy tay, tải tổng hợp | liên tục, **traffic thật** |
| Sống bao lâu | một lần chạy | chuỗi thời gian, so được hôm qua với hôm nay |
| Trả lời câu hỏi | "trần của hệ ở đâu" | "ngay bây giờ hệ đang thế nào" |

Hai thứ **không thay thế nhau**. `bench/README.md` giữ nguyên vai trò đo trần; spec này lo phần theo dõi liên tục.

## 2. Mục tiêu

Dựng Prometheus scrape cả Triton lẫn vLLM, cộng một Grafana dashboard duy nhất phủ **14 chỉ số** đã chốt:

| # | Chỉ số | # | Chỉ số |
|---|---|---|---|
| 1 | RPS | 8 | GPU Memory Usage |
| 2 | Success Rate | 9 | Queue Depth (Pending Requests) |
| 3 | Error Rate | 10 | CCU |
| 4 | P50 Latency | 11 | ASR RTF |
| 5 | P95 Latency | 12 | TTS RTF |
| 6 | P99 Latency | 13 | TTFT (LLM) |
| 7 | GPU Utilization | 14 | TPOT (LLM) |

**Thành công khi:**

1. `./scripts/serve_monitoring.sh` dựng xong, Prometheus báo cả 2 target `UP`
2. Grafana `:3000` tự nạp datasource và dashboard, không cần bấm gì
3. Cả 14 panel có số thật khi chạy `client/*.py` — không panel nào "No data"
4. Toàn bộ test mục 10 xanh, test cũ không đỏ thêm cái nào

**Không phải mục tiêu:** alerting/Alertmanager, log aggregation, tracing, tách GPU theo tiến trình, multi-host.

## 3. Kiến trúc

```
┌── host network ───────────────────────────────────────────────┐
│                                                               │
│  Triton   :8000 :8001 :8002 ◄──────┐                          │
│           (--net host)             │ scrape 5s                │
│                                    │                          │
│  vLLM     :8080             ◄──────┤                          │
│           (-p 8080:8080)           │                          │
│                                    │                          │
│  Prometheus :9090 ─────────────────┘                          │
│           (network_mode: host)     ▲                          │
│                                    │ query                    │
│  Grafana    :3000 ─────────────────┘                          │
│           (network_mode: host)                                │
└───────────────────────────────────────────────────────────────┘
```

**Vì sao `network_mode: host` cho cả hai container mới:** Triton chạy `--net host` nên `:8002` chỉ nhìn thấy từ host namespace. Prometheus ở bridge network sẽ phải dựa vào `host.docker.internal` + `--add-host=host-gateway` — thêm một lớp có thể hỏng mà không đổi lại gì. Host net thì scrape thẳng `localhost`.

**Đánh đổi:** cổng 9090 và 3000 phải trống trên host. Script kiểm trước và báo lỗi rõ ràng thay vì để container chết im lặng.

**Không đụng `serve_llm.sh`.** `serve.sh` nhận đúng hai thay đổi (mục 8).

## 4. Bản đồ 14 chỉ số → nguồn

Tên metric dưới đây đã verify trực tiếp trên image đang dùng: Triton **25.01**, vLLM **0.27.1**.

| # | Chỉ số | Triton | vLLM | Tình trạng |
|---|---|---|---|---|
| 1 | RPS | `nv_inference_request_success` | `vllm:request_success_total` | sẵn |
| 2 | Success Rate | success / (success+failure) | — | sẵn |
| 3 | Error Rate | `nv_inference_request_failure{reason}` | `{finished_reason="abort"}` (proxy) | sẵn / hạn chế |
| 4–6 | p50/95/99 | `nv_inference_request_summary_us{quantile}` | `vllm:e2e_request_latency_seconds_bucket` | **cần flag** |
| 7 | GPU Util | `nv_gpu_utilization` | — | sẵn |
| 8 | GPU Mem | `nv_gpu_memory_used_bytes` / `_total_bytes` | — | sẵn |
| 9 | Queue Depth | `nv_inference_pending_request_count` | `vllm:num_requests_waiting` | sẵn |
| 10 | CCU | **`voice_ccu`** (tự phát) | `vllm:num_requests_running` | **cần code** |
| 11 | ASR RTF | **`voice_rtf{model="asr_streaming"}`** | — | **cần code** |
| 12 | TTS RTF | **`voice_rtf{model="tts"}`** | — | **cần code** |
| 13 | TTFT | — | `vllm:time_to_first_token_seconds_bucket` | sẵn |
| 14 | TPOT | — | `vllm:inter_token_latency_seconds_bucket` | sẵn |

Tổng: **9 sẵn, 3 cần flag, 2 cần code.**

## 5. Quyết định: metric tự phát

### 5.1 Vì sao Triton không tự có RTF và CCU

RTF cần **độ dài audio** — Triton mù về nội dung tensor, đúng lý do `bench/README.md` đã ghi. CCU cần **số sequence đang sống**, thứ chỉ `self.streams` trong process Python biết; `nv_inference_pending_request_count` là độ sâu hàng đợi, ngữ nghĩa khác hẳn.

Triton Python backend có `pb_utils.MetricFamily` — metric tự đăng ký xuất hiện ngay trên `:8002/metrics` cạnh `nv_*`. Không cần exporter riêng, không cần cổng mới.

### 5.2 Ba family

| Family | Kind | Labels | Buckets |
|---|---|---|---|
| `voice_rtf` | HISTOGRAM | `{model}` | tách theo model, xem 5.4 |
| `voice_ccu` | GAUGE | `{model, instance}` | — |
| `voice_ccu_updated_at` | GAUGE | `{model, instance}` | — |

### 5.3 Vì sao CCU phải có label `instance` còn RTF thì không

Đây **không phải sở thích** — đã probe trên `triton-voice:latest` với model 2 instance, kết quả:

```
probe_ccu{model="mm"}                    2     ← 2 instance cùng ghi vào MỘT metric
probe_ccu{instance="mm_0_0",model="mm"}  7
probe_ccu{instance="mm_0_1",model="mm"}  7     ← tách label mới riêng
probe_rtf_count{model="mm"}              2     ← observe() cộng dồn đúng qua process
```

Kết luận:

- **GAUGE + label chung = hỏng.** Hai instance `set()` vào cùng một metric thì ghi đè nhau, giá trị cuối là của process nào chạy sau. `asr_streaming` có `count: 2` nên đây là ca thật, không phải giả định. → CCU **bắt buộc** label `instance`, cộng lại ở PromQL.
- **HISTOGRAM + label chung = đúng.** `observe()` cộng dồn qua process, bucket gộp chuẩn. → RTF không cần tách instance.

Đăng ký MetricFamily trùng tên từ nhiều instance **không lỗi** — Triton gộp theo tên. Cái nguy hiểm là gộp *im lặng*, không phải crash.

### 5.4 Định nghĩa và buckets

| Model | Công thức | Đo ở đâu | Buckets |
|---|---|---|---|
| `asr_streaming` | `compute_s / (len(chunk)/16000)` | mỗi chunk, trong `_handle()` | `[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0]` |
| `tts` | `gen_s / (len(wav)/24000)` | mỗi request, trong `execute()` | `[0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]` |

Buckets tách vì hai thang lệch nhau hẳn một bậc — số từ `bench/`: ASR quanh 0.05, TTS quanh 0.86.

**Công thức TTS khớp `bench/tts/metrics.py:rtf()` có chủ ý**, để hai nguồn so được với nhau. Khác một điểm phải nhớ: bench đo từ client nên **có RTT gRPC**, metric này đo trong `execute()` nên **không có**. Số của metric sẽ thấp hơn bench một chút — lệch đó là đúng, không phải bug.

**ASR RTF lưỡng đỉnh theo thiết kế.** Encoder chỉ chạy khi buffer đủ 45 khung, mà mỗi chunk 200ms chỉ nạp 20 khung → chỉ ~3/5 số chunk kích được encoder (`Architect.md`, mục 5). Chunk không kích thì gần như miễn phí. Nên histogram có hai cụm: đuôi thấp là chunk rỗng việc, phần còn lại mới là chi phí thật. p50 vẫn rơi vào cụm có encoder (60% > 50%) nên đọc được; nhưng **không được diễn giải p50 ASR RTF như chi phí trung bình mỗi chunk audio.**

### 5.5 Interface `ModelMetrics`

`pb_utils` chỉ tồn tại bên trong container Triton, nên nó được **tiêm vào** chứ không import ở module — đó là điều kiện để `test_serving_metrics.py` chạy được khi server tắt.

```python
class ModelMetrics:
    def __init__(self, metric_api, model: str, instance: str, rtf_buckets: list[float]):
        """metric_api: module/đối tượng có .MetricFamily với .GAUGE/.HISTOGRAM.
        Production truyền pb_utils; test truyền fake ghi lại mọi lời gọi."""

    def observe_rtf(self, compute_s: float, audio_s: float) -> None: ...
    def set_ccu(self, n: int) -> None:
        """Set voice_ccu = n VÀ voice_ccu_updated_at = time.time() trong cùng một lần gọi.
        Gộp làm một để hai gauge không bao giờ lệch nhau - xem R1."""
```

`instance` lấy từ `args["model_instance_name"]` trong `initialize()`.

**Bắt buộc giữ tham chiếu tới `MetricFamily`.** Stub của Triton ném:

> *Invalid metric operation as the corresponding 'MetricFamily' has been deleted. The 'MetricFamily' object should be deleted AFTER its corresponding 'Metric' objects have been deleted.*

Nên viết `metric_api.MetricFamily(...).Metric(...)` rồi chỉ giữ `Metric` là **sai** — family bị GC, metric thành vô hiệu lúc chạy. `ModelMetrics` phải giữ cả family lẫn metric làm thuộc tính.

## 6. CCU và bài toán giá trị đóng băng

### 6.1 Vấn đề

`voice_ccu` chỉ cập nhật khi `execute()` chạy. Không traffic thì không có lần set nào, gauge giữ giá trị cũ vô hạn — Prometheus không đánh stale vì Triton vẫn phơi metric đều đặn.

Ba tình huống, **chỉ một hỏng**:

| Tình huống | Kết quả |
|---|---|
| Stream kết thúc bằng `END` | **Đúng.** `del self.streams[corrid]` nằm trong `_handle()`, tức bên trong `execute()` — lần set cuối đã đọc số đã giảm |
| Traffic dừng sau khi mọi stream END | **Vô hại.** Gauge đã là 0, đóng băng ở 0 |
| **Stream chết không gửi `END` + instance đó im lặng** | **Hỏng.** `_sweep()` chỉ chạy ở `execute():181`; không request thì không sweep, gauge giữ N |

Ca hỏng gây hại thế nào: luôn báo **thừa**, không bao giờ thiếu. Và vì `sum by(model)` cộng cả 2 instance, một instance im lặng ôm 3 stream ma làm bẩn tổng **ngay cả khi hệ đang bận** — thực tế 2 CCU mà dashboard đọc 5.

### 6.2 Vì sao không thêm thread nền

Cách hiển nhiên là chạy timer trong `model.py` gọi `_sweep()` định kỳ. Loại vì: thêm thread vào Python backend đang chạy ổn định để làm đẹp một panel là đổi rủi ro thật lấy lợi ích hiển thị. `model.py` đang giữ state không khoá, dựa hoàn toàn vào việc sequence batcher tuần tự hoá truy cập.

### 6.3 Phương án: hết hạn phía query

Set thêm một gauge cùng lúc, cùng chỗ:

```python
self.ccu.set(len(self.streams))
self.ccu_at.set(time.time())      # wall clock — KHÔNG phải monotonic
```

`time.time()` chứ không `time.monotonic()` vì phải so được với `time()` của Prometheus. `model.py` đang dùng `monotonic` cho `last_seen` — hai cái này khác mục đích, không dùng lẫn.

```promql
sum by (model) (
  voice_ccu * on(model, instance) (time() - voice_ccu_updated_at < bool 60)
)
```

Instance nào không được chạm trong `CCU_TTL_S` giây thì nhân 0, biến khỏi tổng.

### 6.4 Vì sao đúng

Không tồn tại trạng thái thứ ba:

| Instance | `_sweep()` chạy? | Giá trị gauge | Query |
|---|---|---|---|
| Đang bận | có (`execute():181`) | **đúng** — orphan đã bị xoá | giữ nguyên |
| Im lặng ≥ TTL | không | có thể sai | **nhân 0** |

Query chỉ đang mô phỏng lại `_sweep()` từ bên ngoài, đúng ngưỡng mà `_sweep()` dùng.

Con số 60 không tuỳ tiện. `config.pbtxt` đặt `max_sequence_idle_microseconds: 60000000`, và proto upstream định nghĩa:

> *The maximum time, in microseconds, that a sequence is allowed to be idle before it is aborted. […] the inference server will free the sequence slot allocated by the sequence and make it available for another sequence. If not specified (or specified as zero) a default value of 1000000 (1 second) is used.* — `model_config.proto:1557`

Nghĩa là instance im lặng 60s thì **theo luật của chính Triton**, mọi sequence trên đó đã bị abort. Báo 0 không phải đoán.

**Lợi ích ngoài dự tính:** nếu Triton sập, Prometheus vẫn trả mẫu cuối trong 5 phút (lookback delta) — gauge trần sẽ đọc CCU cao giả suốt 5 phút. Nhưng `voice_ccu_updated_at` cũng đóng băng theo, nên sau 60s tích thành 0. Dashboard tự lành khi server chết.

### 6.5 Ghi chú: vì sao `max_sequence_idle` là 60s chứ không phải mặc định 1s

Mặc định 1 giây quá ngắn cho voice. Người nói có khoảng lặng — ngập ngừng, lấy hơi, nghĩ câu tiếp. Để mặc định thì Triton abort sequence giữa lúc user đang nghĩ, chunk kế tiếp rơi vào nhánh `model.py:148`:

```python
if start or corrid not in self.streams:
    if not start:
        pb_utils.Logger.log_warn(f"... chunk không có state (corrid={corrid}), khởi tạo lại")
```

State mới tinh → mất cache encoder và hypothesis → **transcript đứt đoạn giữa câu**. Không crash, chỉ ra kết quả sai — kiểu lỗi tệ nhất.

60s là ngưỡng "người này đã bỏ đi", không phải "người này đang nghĩ". Đánh đổi: 8 stream ma chặn slot tới 60s (`max_candidate_sequences: 8`). Máy dev 1 GPU thì chấp nhận; production đông người phải đo lại.

## 7. Cấu trúc thư mục (phần thêm mới)

```
serving/
  __init__.py
  metrics.py                                 module thuần: CCU_TTL_S, buckets, rtf(), ModelMetrics
docker/monitoring/
  docker-compose.yml                         prometheus + grafana, network_mode: host
  prometheus.yml                             2 job, scrape_interval 5s
  grafana/provisioning/datasources/prometheus.yml
  grafana/provisioning/dashboards/voice.yml
  grafana/dashboards/voice-serving.json      dashboard duy nhất, 5 row
scripts/serve_monitoring.sh                  wrapper: kiểm cổng + docker compose up -d
tests/test_serving_metrics.py                unit
tests/test_monitoring_config.py              unit — chống drift
tests/test_metrics_endpoint.py               integration
```

**`serving/` nằm ngoài `model_repository/` có lý do bắt buộc:** Triton quét mọi thư mục con của model repository như một model; thư mục không có `config.pbtxt` làm server chết lúc load. Nên module dùng chung không thể đặt trong đó.

`model.py` nạp bằng `sys.path.insert(0, "/opt/serving")` — đúng idiom nó đang dùng cho thư mục model, và tránh phải ghi đè `PYTHONPATH` bằng `-e` (sẽ xoá mất đường dẫn ZipVoice mà Dockerfile đặt).

## 8. Thay đổi vào code đang chạy

Nguyên tắc: **chỉ cộng thêm, không sửa luồng inference.**

| File | Thay đổi |
|---|---|
| `scripts/serve.sh` | thêm `-v "$ROOT/serving:/opt/serving:ro"`; thêm `--metrics-config summary_latencies=true` và `--metrics-config 'summary_quantiles=0.5:0.05,0.95:0.01,0.99:0.001'` |
| `model_repository/asr_streaming/1/model.py` | `initialize()`: dựng `ModelMetrics`. `_handle()`: bọc thời gian, `observe_rtf()`. `execute()`: `set_ccu(len(self.streams))` ở cuối |
| `model_repository/tts/1/model.py` | `initialize()`: dựng `ModelMetrics`. `execute()`: `set_ccu(len(requests))` đầu hàm, bọc thời gian sinh rồi `observe_rtf()` mỗi request, `set_ccu(0)` trong `finally` |

Không đổi `config.pbtxt`, không đổi `docker/Dockerfile`, không đổi `scripts/serve_llm.sh`.

**Bước 0 — ĐÃ XONG (2026-08-19).** Dump `:8002/metrics` từ `triton-voice:latest` cho kết quả:

- Counter Triton **không có hậu tố `_total`**: `nv_inference_request_success`, `nv_inference_request_failure`
- `nv_inference_request_failure` **có thêm label `reason`** — `BACKEND` / `OTHER` / `CANCELED` / `REJECTED`
- Mọi metric per-model có label `version="1"`
- GPU metric dùng label **`gpu_uuid`**, không phải chỉ số card
- Summary ra đúng `quantile="0.5"/"0.95"/"0.99"` với flag ở trên

Label `reason` là chỗ suýt hỏng: `rate(failure) + rate(success)` là cộng hai vector **lệch label**, Prometheus không match được cặp nào và trả rỗng — panel "No data" chứ không phải số sai. Mọi phép tính đụng `failure` phải bọc `sum()` để bỏ `reason` trước. Xem §9.

`CANCELED` đáng ngờ có phải lỗi không — client streaming ngắt kết nối cũng sinh ra nó. Giữ trong tổng lỗi ở bản đầu, nhưng tách một series riêng trong panel để nhìn thấy tỉ trọng.

## 9. Query và dashboard

Một dashboard `voice-serving`, 5 row, xếp theo component.

**Row OVERVIEW**
```promql
# RPS tổng
sum(rate(nv_inference_request_success[1m])) + sum(rate(vllm:request_success_total[1m]))
# CCU tổng
sum(voice_ccu * on(model,instance) (time() - voice_ccu_updated_at < bool 60))
  + sum(vllm:num_requests_running)
```

**Row ASR_STREAMING / TTS** — `$model` dưới đây là **chỗ điền lúc sinh JSON**, không phải template variable của Grafana. Mỗi row viết cứng tên model của nó; hai row không dùng chung biến, để chọn được buckets và trục y riêng cho từng model.
```promql
rate(nv_inference_request_success{model="$model"}[1m])
nv_inference_request_summary_us{model="$model",quantile="0.99"} / 1000        # → ms
sum(voice_ccu{model="$model"} * on(model,instance) (time() - voice_ccu_updated_at < bool 60))
nv_inference_pending_request_count{model="$model"}
histogram_quantile(0.95, sum by(le) (rate(voice_rtf_bucket{model="$model"}[5m])))
# sum() bắt buộc: nv_inference_request_failure có label `reason`, không bọc thì
# phép cộng dưới đây lệch label và trả về rỗng
sum(rate(nv_inference_request_failure{model="$model"}[1m]))
  / clamp_min(
      sum(rate(nv_inference_request_success{model="$model"}[1m]))
      + sum(rate(nv_inference_request_failure{model="$model"}[1m])), 1e-9)
```

**Row LLM**
```promql
histogram_quantile(0.99, sum by(le) (rate(vllm:e2e_request_latency_seconds_bucket[5m])))
histogram_quantile(0.95, sum by(le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))    # TTFT
histogram_quantile(0.95, sum by(le) (rate(vllm:inter_token_latency_seconds_bucket[5m])))    # TPOT
vllm:num_requests_running        # CCU
vllm:num_requests_waiting        # queue
```

**Row GPU**
```promql
# label là gpu_uuid; hai metric cùng label nên phép chia match được
nv_gpu_utilization * 100
nv_gpu_memory_used_bytes / nv_gpu_memory_total_bytes * 100
```

## 10. Test

TDD như cũ: viết test trước, xác nhận đỏ, mới code.

| File | Loại | Kiểm gì |
|---|---|---|
| `test_serving_metrics.py` | unit | `rtf()` khớp `bench/tts/metrics.py:rtf()` trên cùng input; `audio_s <= 0` ném lỗi; buckets tăng dần; `ModelMetrics` gọi fake API — **assert `voice_ccu` có label `instance`, `voice_rtf` thì không** (chính là cái bẫy mục 5.3); `set_ccu()` set cả hai gauge trong một lần gọi |
| `test_monitoring_config.py` | unit | `prometheus.yml` parse được, đúng 2 target, `scrape_interval` 5s; dashboard JSON hợp lệ, mọi panel có datasource, mọi `expr` chỉ dùng tên metric trong whitelist đã verify ở bước 0; **`CCU_TTL_S` == số trong PromQL của dashboard == `max_sequence_idle_microseconds` parse từ `config.pbtxt`** |
| `test_metrics_endpoint.py` | integration | `:8002/metrics` có `voice_rtf_bucket`, `voice_ccu`, `voice_ccu_updated_at`, `nv_inference_request_summary_us{quantile="0.99"}`; sau một stream ASR thật thì `voice_rtf_count` tăng và `voice_ccu` về 0 |

Fixture `triton`/`llm` trong `conftest.py` dùng lại; thêm fixture `metrics_text` scrape `:8002`. Test integration đánh dấu `@pytest.mark.integration` như cũ.

Test drift ở `test_monitoring_config.py` là chỗ đáng giá nhất trong ba file — nó chặn đúng rủi ro R2 dưới đây.

## 11. Giới hạn đã biết

Ghi ra để người đọc dashboard không hiểu sai, không phải để xin lỗi.

| Giới hạn | Chi tiết |
|---|---|
| **GPU không tách được theo tiến trình** | `nv_gpu_utilization` là toàn card, mà Triton và vLLM dùng chung. Panel GPU là chỉ số của cả máy, không quy trách nhiệm cho component nào |
| **Error rate LLM yếu hơn Triton** | vLLM 0.27.1 không đếm HTTP 4xx/5xx, chỉ có `vllm:request_success{finished_reason=stop\|length\|abort}`. Dùng tỉ lệ `abort` làm proxy. Muốn số thật phải chèn reverse proxy đo status code — ngoài phạm vi |
| **CCU thừa trong cửa sổ ≤60s** | Client chết mà instance vẫn bận thì orphan còn được đếm tới khi `_sweep()` xoá. Nhưng trong cửa sổ đó Triton **cũng đang coi sequence là sống** và vẫn giữ slot — nên con số khớp định nghĩa của Triton. Hiểu đúng: CCU nghĩa là *"sequence chưa hết hạn"*, không phải *"người thật đang nói"* |
| **ASR RTF lưỡng đỉnh** | ~2/5 số chunk không kích encoder nên gần như miễn phí. Xem 5.4 |
| **Summary quantile không cộng được** | `nv_inference_request_summary_us` là Summary, quantile tính sẵn phía server. Một server thì không sao; nhiều server sau này thì không `avg()` được, phải đổi sang histogram |

## 12. Rủi ro

| Rủi ro | Mức | Xử lý |
|---|---|---|
| **R1 — Label join sai làm CCU đọc 0** | **Cao** | `on(model, instance)` mà hai family lệch label thì join rỗng → CCU = 0. Sai theo chiều **báo thiếu** và **im lặng**, nguy hiểm hơn báo thừa. Chặn: hai gauge tạo cùng một chỗ, dùng chung một dict labels; `test_serving_metrics.py` assert đúng chuyện đó |
| **R2 — Số 60 nằm ở 3 nơi** | **Cao** | `config.pbtxt`, `CCU_TTL_S`, PromQL trong dashboard. Lệch nhau là sai âm thầm. Chặn: hằng số về `serving/metrics.py`, `test_monitoring_config.py` assert cả ba khớp |
| R3 — Tên counter Triton khác dự đoán | ~~Trung bình~~ **đã đóng** | Bước 0 đã chạy, kết quả trong §8 |
| **R8 — MetricFamily bị GC làm Metric vô hiệu** | **Cao** | Không giữ tham chiếu family thì metric chết im lặng lúc chạy, test unit không thấy. Chặn: `ModelMetrics` giữ cả family; test dùng `weakref` + `gc.collect()` khẳng định family còn sống |
| R4 — Flag `summary_latencies` là experimental | Trung bình | Đã probe chạy đúng trên 25.01, ra đúng `quantile="0.5"/"0.95"/"0.99"`. Nếu bản sau bỏ thì rơi về mean từ counter, panel p-tiles mất — ghi rõ trong doc để không tưởng là hỏng code |
| R5 — Cổng 9090/3000 bị chiếm | Thấp | `serve_monitoring.sh` kiểm trước, báo lỗi rõ |
| R6 — Lệch đồng hồ container/host | Thấp | Cùng kernel clock nên không lệch. Nếu sau này tách Prometheus sang máy khác thì query 6.3 hỏng — ghi vào doc |
| R7 — Chi phí metric trong đường nóng | Thấp | `observe()`/`set()` ghi shared memory, không I/O. Nhưng ASR gọi mỗi chunk — đo lại `bench/asr_streaming` sau khi implement, so với trần 138 infer/s hiện có |

## 13. Ngoài phạm vi

- Alertmanager và rule cảnh báo
- Log aggregation (Loki) và tracing (Tempo/OTel)
- DCGM exporter để tách GPU theo tiến trình
- Reverse proxy đo HTTP status của vLLM
- Recording rules để nén truy vấn nặng
- Multi-host, service discovery động
- Đổi Summary sang Histogram cho latency Triton

## 14. Quy ước làm việc

Như các spec trước: identifier tiếng Anh, comment/docstring tiếng Việt, mở file bằng 2 dòng `# ABOUTME:`, giải thích tập trung ở mức config và kiến trúc. Tên metric tự phát dùng tiền tố `voice_` để tách hẳn khỏi `nv_` của Triton và `vllm:` của vLLM.
