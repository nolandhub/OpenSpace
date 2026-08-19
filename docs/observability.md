# Observability

Prometheus scrape liên tục, Grafana hiện 14 chỉ số của cả ba component.
Thiết kế và lý do từng quyết định: `superpowers/specs/2026-08-19-observability-design.md`.

## Chạy

    ./scripts/serve_triton.sh       # Triton trước
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
| `voice_ccu` | gauge | `model`, `model_instance` |
| `voice_ccu_updated_at` | gauge | `model`, `model_instance` |

`voice_ccu` **phải** có label `model_instance` còn `voice_rtf` **không** — hai
instance của `asr_streaming` cùng `set()` vào một gauge sẽ ghi đè nhau, còn
`observe()` của histogram thì cộng dồn đúng. Chi tiết ở spec §5.3.

Tên label là `model_instance` chứ không phải `instance`: Prometheus tự gắn
label `instance` = địa chỉ target vào mọi metric lúc scrape, đụng tên với
label tự phát thì bị ghi đè mất giá trị thật.

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
| Panel p50/95/99 trống | `serve_triton.sh` thiếu `--metrics-config summary_latencies=true` |
| Panel `voice_*` trống | `serve_triton.sh` thiếu mount `serving/` vào `/opt/serving/serving` |
| CCU luôn bằng 0 | join `on(model, model_instance)` hỏng — hai gauge lệch label |
| Target `vllm` DOWN | chưa chạy `serve_llm.sh`, hoặc `PORT` khác 8080 |
| `serve_monitoring.sh` báo cổng bị chiếm | 9090/3000 đang có tiến trình khác |
