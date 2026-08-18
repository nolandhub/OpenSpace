# Benchmark

Một nguyên tắc: **perf_analyzer lo mọi thứ đo được ở tầng request. `bench/` chỉ chứa
cái nó không thấy.**

perf_analyzer mù về loại model — nó biết "một request vào, một response ra", không biết
trong tensor là audio hay text. Nên latency, throughput, queue/compute và GPU util thì nó
làm hết; cái gì cần biết *bên trong* request là gì thì phải tự viết.

| Model | perf_analyzer đo | `bench/<model>/metrics.py` đo | Vì sao nó không đo được |
|---|---|---|---|
| [`asr_streaming`](asr_streaming/README.md) | p50/p90/p95/p99, infer/s, queue, compute, GPU util | **first-chunk latency** | gộp mọi request trong một sequence thành một phân phối |
| | | **WER** | vứt toàn bộ output tensor |
| [`tts`](tts/README.md) | như trên | **RTF** | không đọc output nên không biết WAV trả về dài bao nhiêu |

## Bố cục

    scripts/perf.sh <model>       entry point perf_analyzer, một dispatcher cho mọi model
    bench/common/stats.py         p50_p95, dùng chung
    bench/<model>/metrics.py      chỉ số riêng của model đó
    bench/<model>/gen_input.py    sinh input JSON audio/text thật cho perf_analyzer
    bench/<model>/results/        metrics.md (commit) · perf.csv, input.json (bỏ qua)
    bench/<model>/README.md       kết quả và cách đọc của riêng model đó

Thêm model mới = thêm thư mục `bench/<tên model>/` theo bộ khung trên, cộng một nhánh
`case` trong `scripts/perf.sh`.

## Chạy

    ./scripts/perf.sh asr_streaming                   # ~75s
    ./scripts/perf.sh tts                             # ~3 phút
    CONCURRENCY=1:8 ./scripts/perf.sh asr_streaming   # đổi dải tải
    .venv/bin/python bench/asr_streaming/metrics.py
    .venv/bin/python bench/tts/metrics.py

`perf.sh` tự gọi `gen_input.py` của model để nạp **audio và text thật**. Dữ liệu ngẫu
nhiên mặc định của perf_analyzer cho số lạc quan ~3.5% ở asr: vòng greedy của transducer
phát ít token non-blank hơn khi nghe nhiễu, nên tốn ít lượt decoder hơn.

## Kết quả

Máy đo: NVIDIA RTX 3050 Laptop 4GB. `asr_streaming` 2 instance, `tts` 1 instance.

| Model | Kết luận một dòng |
|---|---|
| [`asr_streaming`](asr_streaming/README.md) | trần **138 infer/s = 27.7x thời gian thực**. GPU mới 77% — nút thắt là vòng `for` tuần tự trong `execute()`, không phải GPU |
| [`tts`](tts/README.md) | **bão hoà ngay ở CCU 1**, GPU 98%. RTF 0.86 — vừa đủ một phiên real-time, không có dư địa |

## Ba cái bẫy đã gặp

1. **Chạy hai benchmark cùng lúc.** `tts` chiếm 98% GPU, đo `asr` song song ra 27 infer/s
   thay vì 88 — mà perf_analyzer vẫn báo "ổn định". Mỗi lần chỉ chạy một model.

2. **Throttle nhiệt.** Đo lúc GPU 87°C, clock tụt 2100 → 1275 MHz, số sai 2-3 lần. Kiểm
   trước khi tin bảng:

       nvidia-smi --query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv

   `clocks_throttle_reasons.active` khác `0x1` (GpuIdle) nghĩa là đang bị hãm.

3. **"Ổn định" không phải "đúng".** perf_analyzer dừng đo khi 3 cửa sổ liền nhau lệch dưới
   `--stability-percentage` (mặc định 10%). Nếu GPU chậm suốt cả lát thời gian đó thì cả ba
   cửa sổ đều đồng ý ở con số sai.

## Nguồn

- [Performance Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/README.html) — measurement mode, stabilization
- [Metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html) — cái `--collect-metrics` lấy từ `:8002/metrics`
