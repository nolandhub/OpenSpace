# tts — benchmark

Nguyên tắc phân công và bố cục chung: [`../README.md`](../README.md).

    ./scripts/perf.sh tts                      # p50/90/95/99, throughput, GPU util
    .venv/bin/python bench/tts/metrics.py      # RTF

## Kết quả

RTX 3050 Laptop, `instance_group.count: 1`, ZipVoice `num_step: 16`, GPU nguội.

| CCU | infer/s | p50 ms | p95 ms | queue ms | GPU |
|---|---|---|---|---|---|
| 1 | 0.385 | 2568.68 | 2646.43 | 0.05 | 98% |
| 2 | 0.380 | 5292.78 | 5303.31 | 2538.0 | 98% |
| 3 | 0.375 | 7942.63 | 7965.84 | 5112.6 | 98% |
| 4 | 0.375 | 10686.17 | 10887.07 | 7733.0 | 97% |

`metrics.py`: **RTF p50 = 0.859, p95 = 0.962** (12 lần sinh, GPU nguội).

## Đọc kết quả

**Bão hoà tuyệt đối ngay ở CCU 1.** Throughput còn giảm nhẹ (0.385 → 0.375) trong khi
latency tăng **tuyến tính đúng bậc**: 2.57 → 5.29 → 7.94 → 10.69s, mỗi bậc cộng thêm ~2.6s
= đúng một lượt sinh. GPU 97-98% suốt từ CCU 1.

Từ client thứ hai trở đi, không ai được phục vụ nhanh hơn — chỉ có hàng đợi dài thêm. Ở
CCU 4, người dùng chờ 10.69s để nhận ~2.9s tiếng, trong đó 7.73s là ngồi chờ.

Khác hẳn kiểu bão hoà của `asr_streaming` (ở đó GPU mới 77% và nút thắt nằm ở CPU/vòng lặp).

**RTF 0.86 nghĩa là vừa đủ, không có dư địa.** Sinh 1 giây audio tốn 0.86 giây → trần
1.16x thời gian thực. Một phiên real-time thì kịp; hai phiên hỏng ngay, đúng như bảng trên.

**Và cái "vừa đủ" đó không bền.** Đo lại ngay sau một đợt tải kéo dài, RTF lên **1.24** —
chậm hơn thời gian thực. `nvidia-smi` báo `HW Thermal Slowdown: Not Active` nhưng counter
`SW Thermal Slowdown` đã cộng 1.18s và clock tụt 2100 → 1747 MHz. Trên GPU laptop, con số
0.86 chỉ đúng lúc máy nguội.

Tăng `instance_group.count` không giúp gì khi GPU đã 98% — hai instance chỉ chia nhau cùng
một lượng compute. Muốn nhiều phiên thì phải đổi model hoặc thêm GPU.

## Việc nên làm tiếp

- Đo `zipvoice_distill` (`num_step: 8`, `guidance_scale: 3.0` trong `config.pbtxt`) xem RTF
  có xuống dưới 0.5 không. Đổi tham số phải reload model nên không quét được trong một lần chạy.
- **Bộ text hiện tại chỉ có 1 câu 11 từ** (`tests/assets/sample_vi.txt`) nên perf_analyzer
  xoay vòng đúng câu đó — RTF chỉ đúng cho độ dài ấy. Cần bộ phủ 5-40 từ, có số, viết tắt,
  từ nước ngoài, vì RTF đi theo số token sau normalize.

## Ghi chú phép đo

- `--measurement-mode count_windows`: mỗi request mất vài giây, cửa sổ 5s mặc định không
  bao giờ đủ mẫu để ổn định.
- Input JSON dạng `data` **phẳng** một tầng — `tts` không phải sequence model, perf_analyzer
  xoay vòng qua các phần tử cho từng request.
- **Warmup chỉ 2 request** (asr là 50): một câu mất ~2.7s, 50 request thành 2 phút chết.
- Cột `infer/s` in `%.3f`: các mức chỉ chênh nhau 0.01, `%.1f` làm tròn hết thành 0.4 và
  bảng trông như throughput không đổi.
- RTF bỏ lần sinh đầu: nó gánh chi phí nạp vocoder lên GPU và kéo lệch p50.
- Mẫu số của RTF lấy từ chính output: `len(WAV) / SAMPLE_RATE`. Độ dài audio đổi theo từng
  câu nên không thể coi là hằng số — đây là lý do perf_analyzer không tính được RTF.
- Kiểm `nvidia-smi --query-gpu=clocks.sm,temperature.gpu` trước khi tin bảng.
