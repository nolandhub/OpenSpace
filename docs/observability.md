# Observability

Prometheus scrape liên tục, Grafana có hai dashboard:

| board | mức | trả lời |
|---|---|---|
| **Voice Serving** | sản phẩm | ASR, TTS, LLM đang phục vụ thế nào — RPS, latency, CCU, RTF, TTFT/TPOT |
| **Triton** | nội tại server | thời gian trôi đi đâu trong Triton — queue vs compute, batch size, CPU/GPU |

Tách hai board vì hai mức trừu tượng khác nhau, không phải vì board cũ dài quá:
nhìn "TTS chậm" là câu hỏi của board đầu, "TTS chậm vì xếp hàng hay vì model"
là câu hỏi của board sau.

Thiết kế và lý do từng quyết định: `superpowers/specs/2026-08-19-observability-design.md`.

## Chạy

    ./scripts/serve_triton.sh       # Triton trước
    ./scripts/serve_llm.sh          # rồi vLLM
    ./scripts/serve_monitoring.sh   # Prometheus 9090 + Grafana 3000

Grafana `http://localhost:3000` → **Voice Serving** hoặc **Triton**. Không cần
đăng nhập. Tắt: `./scripts/serve_monitoring.sh down`.

## Ranh giới với `bench/`

|  | `bench/` + perf_analyzer | ở đây |
|---|---|---|
| Khi nào có số | chạy tay, tải tổng hợp | liên tục, traffic thật |
| Sống bao lâu | một lần chạy | chuỗi thời gian |
| Trả lời | "trần của hệ ở đâu" | "ngay bây giờ hệ thế nào" |

Không thay thế nhau.

## Board Triton

Bốn row, tất cả đọc từ `nv_*` — không cần thêm cờ nào cho `serve_triton.sh`,
`--metrics-config summary_latencies=true` đang bật là đủ.

| Row | Panel | Ghi chú |
|---|---|---|
| HEALTH | Target UP, Error rate, Model load | `up{job="triton"}` là thứ duy nhất phân biệt "hệ rảnh" với "hệ tắt" — thiếu nó thì Triton chết là cả board cùng hiện No data mà không chỗ nào nói vì sao |
| LATENCY | Request/Queue p50-95-99, Queue time trung bình, Queue depth, Latency breakdown ×2 | breakdown là lý do board này tồn tại |
| THROUGHPUT | RPS, Inference vs execution, Batch size trung bình | |
| RESOURCE | GPU util/memory/power, CPU util/memory | |

**Latency breakdown** tách `request` thành bốn chặng Triton đo sẵn:

    request = queue + compute_input + compute_infer + compute_output

Đo trên máy này, ASR chạy client `--fast` (dồn chunk, không chờ thời gian thực):

| chặng | asr_streaming | tts |
|---|---|---|
| queue | 169 ms | 0.02 ms |
| compute_input | 0.02 ms | 3.6 ms |
| compute_infer | 17 ms | 1975 ms |
| compute_output | 0.08 ms | 0.25 ms |

Hai cột nói hai chuyện khác hẳn nhau: ASR gần như toàn bộ thời gian là **xếp
hàng** (vì `--fast` dồn chunk vào một lúc, chạy đúng nhịp 200ms thì cột này về
gần 0), TTS gần như toàn bộ là **compute** — một instance, không batch, request
sau chờ request trước xong.

**Batch size trung bình** = `nv_inference_count / nv_inference_exec_count`.
Hiện đang là **1.0** cho cả hai model, kể cả `asr_streaming` vốn khai
`max_candidate_sequences: 8` — nghĩa là sequence batcher chưa gộp được gì, mỗi
lần execute chỉ một request. Bình thường khi chỉ có một client; đáng ngờ nếu
nhiều client cùng nói mà số vẫn bằng 1.

## Metric tự phát

Triton mù về nội dung tensor nên không biết audio dài bao nhiêu; và số phiên
đang sống chỉ `self.streams` trong process Python biết. Hai thứ đó do
`serving/metrics.py` phát qua `pb_utils.MetricFamily`, xuất hiện ngay trên
`:8002/metrics` cạnh `nv_*`.

| Metric | Kind | Labels |
|---|---|---|
| `voice_rtf` | histogram | `model` |
| `voice_ccu` | gauge | `model`, `model_instance` |
| `voice_ccu_updated_at` | gauge | `model`, `model_instance` |

`voice_ccu` **phải** có label `model_instance` còn `voice_rtf` **không** — hai
instance của `asr_streaming` cùng `set()` vào một gauge sẽ ghi đè nhau, còn
`observe()` của histogram thì cộng dồn đúng. Chi tiết ở spec §5.3.

Tên label là `model_instance` chứ không phải `instance`: Prometheus tự gắn
label `instance` = địa chỉ target vào mọi metric lúc scrape, đụng tên với
label tự phát thì bị ghi đè mất giá trị thật.

## Sửa dashboard

Sửa bảng panel trong `docker/monitoring/build_dashboard.py` — `ROWS` cho board
Voice Serving, `SERVER_ROWS` cho board Triton — rồi:

    .venv/bin/python docker/monitoring/build_dashboard.py            # sinh cả hai
    .venv/bin/python docker/monitoring/build_dashboard.py --stdout --board triton
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
- **Panel quantile trống lúc hệ rảnh là ĐÚNG.** Quantile của summary tính trên
  sliding window; window rỗng thì Triton trả `Nan`, không phải 0. Vì vậy
  "Queue p50/p95/p99" có ô "Queue time trung bình" đứng cạnh — cặp
  `_sum`/`_count` là counter thường nên luôn có số.
- **`nv_gpu_power_limit` đọc ra 0 W trên GPU laptop này**, `nvidia-smi` cũng trả
  `power.limit [N/A]` (chỉ `enforced.power.limit` có số, 60 W). Panel GPU power
  vì thế chỉ vẽ usage.
- **`CANCELED` nằm trong tổng lỗi của Triton** — client streaming ngắt giữa
  chừng cũng sinh ra nó, không phải lúc nào cũng là lỗi thật.

## Sửa lỗi thường gặp

| Triệu chứng | Nguyên nhân |
|---|---|
| Panel p50/95/99 trống | `serve_triton.sh` thiếu `--metrics-config summary_latencies=true` |
| Panel `voice_*` trống | `serve_triton.sh` thiếu mount `serving/` vào `/opt/serving/serving` |
| CCU luôn bằng 0 | join `on(model, model_instance)` hỏng — hai gauge lệch label |
| Target `vllm` DOWN | chưa chạy `serve_llm.sh`, hoặc `PORT` khác 8080 |
| `serve_monitoring.sh` báo cổng bị chiếm | 9090/3000 đang có tiến trình khác |
| Board **Triton** không thấy trong Grafana | provisioning quét thư mục mỗi 30s (`updateIntervalSeconds`) — đợi, hoặc restart Grafana |
| Board này nạp đè board kia | hai JSON trùng `uid`; `test_hai_board_khac_uid` canh chuyện này |
