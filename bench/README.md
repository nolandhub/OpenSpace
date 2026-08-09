# Kết quả benchmark

Đo ngày 2026-08-09 trên RTX 3050 Laptop 4GB, Triton 2.54.0, container dựng từ `nvcr.io/nvidia/tritonserver:25.01-py3`.

Số liệu thô: `results/results.csv`. Output đầy đủ của E3: `results/e3_perf_analyzer.txt`.

## Cách chạy lại

```bash
.venv/bin/python bench/gen_input.py     # sinh dữ liệu mẫu, chỉ cần 1 lần
.venv/bin/python bench/bench.py e1      # quét max_queue_delay
.venv/bin/python bench/bench.py e1b     # quét concurrency client
.venv/bin/python bench/bench.py e2      # quét instance_group.count của TTS
.venv/bin/python bench/bench.py e3      # breakdown từng tầng của ensemble
```

E1 và E2 tự sửa `config.pbtxt` và khởi động lại container giữa các lần đo, rồi trả config về mặc định khi xong.

---

## E3 — Tầng nào là nút cổ chai

Ensemble `asr`, concurrency 8.

| model | chờ (ms) | tính (ms) | batch tb |
|---|---|---|---|
| `asr_feature` | 2.5 | 18.8 | 1.00 |
| `asr_encoder` | 14.4 | 46.4 | 1.18 |
| **`asr_scorer`** | **168.4** | 79.3 | 1.00 |

Tổng: 23.8 rps, p95 434 ms.

**Nút cổ chai là `asr_scorer`, không phải encoder.** Nó chờ 168 ms để được tính 79 ms — dồn hàng gấp đôi thời gian làm việc thật.

Đây đúng là hình dạng đã dự đoán từ lúc thiết kế: encoder gánh gần hết FLOPs (46 ms tính, một lượt duy nhất cho cả câu), còn scorer chạy ~300 bước lặp tuần tự với op tí hon nên bị chi phối bởi overhead launch kernel chứ không phải khối lượng tính.

Chính vì hai tầng nghẽn theo hai cơ chế trái ngược nhau mà việc tách chúng thành hai model Triton riêng là bắt buộc — mỗi tầng cần một cách chữa khác nhau.

**Hướng chữa tiếp:** tăng `instance_group.count` của `asr_scorer` (đang là 2). Đây là loại nghẽn mà instance count chữa được, khác hẳn trường hợp TTS ở E2.

---

## E1 — `max_queue_delay_microseconds` (nút vặn sai)

Concurrency cố định 8, quét delay.

| đo gì | delay | batch tb | rps | p50 ms | p95 ms |
|---|---|---|---|---|---|
| `asr_encoder` trực tiếp | 0 | 4.01 | 37.26 | 212 | 216 |
| | 1 ms | 4.01 | 37.23 | 213 | 215 |
| | 5 ms | 4.06 | 37.13 | 214 | 236 |
| | 20 ms | 4.83 | 37.77 | 212 | 214 |
| qua ensemble `asr` | 0 | 1.00 | 23.82 | 333 | 338 |
| | 1 ms | 1.01 | 23.70 | 335 | 339 |
| | 5 ms | 1.19 | 23.55 | 331 | 446 |
| | 20 ms | 1.80 | 23.99 | 325 | 431 |

**Delay gần như không đổi gì.** Hai điều giải thích:

1. **Khi request đã tự xếp hàng thì không cần chờ.** Gọi thẳng encoder ở concurrency 8, hàng đợi luôn đầy nên batcher gom được 4 ngay cả với `delay=0`. Delay chỉ có tác dụng khi request tới nhỏ giọt — đúng tình huống qua ensemble, nơi scorer đẩy áp lực ngược lên làm encoder nhận từng cái một (batch 1.00 → 1.80 khi delay đi từ 0 lên 20 ms).

2. **Ở concurrency cố định, batch bị chặn bởi số request đang bay, không phải bởi thời gian chờ.** Với 8 request in-flight và ~2 batch đang chạy, batch không thể vượt 4 dù chờ bao lâu.

Cái giá của latency thì thấy rõ: p95 qua ensemble xấu đi từ 338 lên 446 ms khi bật delay 5 ms.

**Chi phí của hai tầng Python:** 37.3 rps khi gọi encoder trực tiếp, còn 23.8 rps qua cả chuỗi — `asr_feature` + `asr_scorer` ăn mất 36% throughput.

---

## E1b — concurrency client (nút vặn đúng)

`asr_encoder` gọi trực tiếp, `delay` cố định 5 ms.

| concurrency | batch tb | rps | p50 ms | p95 ms |
|---|---|---|---|---|
| 1 | 1.00 | 28.83 | 33.7 | 33.9 |
| 2 | 1.19 | 34.40 | 56.4 | 60.2 |
| 4 | 2.40 | 35.28 | 112.5 | 113.6 |
| 8 | 4.21 | 37.32 | 213.8 | 264.9 |
| 16 | 8.00 | 38.63 | 411.2 | 413.7 |

Batch leo từ 1.00 lên đúng trần `max_batch_size: 8`.

**Đổi lại: throughput chỉ tăng 34% trong khi latency tăng 12 lần.**

Lý do nằm ở chính quyết định thiết kế của project: mọi request đều được đệm về **16 giây cố định** để dynamic batcher gom được (spec §7). Hệ quả là mỗi request mang một tensor `(1600, 80)` — đủ lớn để lấp GPU ngay ở batch 1. Batching không còn gì để thu hồi.

Với audio thật 3 giây không đệm, batch-1 sẽ để GPU trống nhiều và batching mới có lãi lớn. Đây là lập luận trực tiếp cho việc thay đệm cố định bằng **bucketing hoặc ragged batching** — đã ghi ở spec §12.

---

## E2 — `instance_group.count` của TTS

| instances | concurrency | rps | p50 ms | VRAM (MB) |
|---|---|---|---|---|
| 1 | 1 | 0.475 | 2 096 | 1 317 |
| 1 | 2 | 0.475 | 4 205 | 1 317 |
| 1 | 4 | 0.466 | 8 496 | 1 317 |
| 2 | 1 | 0.475 | 2 098 | 2 082 |
| 2 | 2 | 0.449 | 4 414 | 2 096 |
| 2 | 4 | 0.449 | 8 830 | 2 110 |
| 4 | 1 | 0.480 | 2 053 | 3 663 |
| 4 | 2 | 0.453 | 4 417 | 3 691 |
| 4 | 4 | 0.444 | 8 905 | 3 691 |

Hai điều đọc được:

**Với 1 instance, throughput là hằng số còn latency tăng tuyến tính** — 0.475 rps ở mọi mức concurrency, p50 đi 2.1 → 4.2 → 8.5 giây. ZipVoice chạy tuần tự, thêm client chỉ là thêm người xếp hàng.

**Tăng instance không mua được gì.** Từ 1 lên 4 instance, throughput đứng nguyên ~0.45–0.48 rps, còn VRAM leo từ 1 317 lên 3 691 MB — tức 90% của card 4 096 MB. Mỗi instance tốn thêm ~780 MB để đổi lấy con số âm.

Lý do: ZipVoice ở `num_step=16` chạy 16 lượt flow-matching tuần tự qua decoder 512 chiều. GPU đã bận kín với một request; nhiều instance chỉ chia nhau cùng đám SM.

---

## Kết luận chung

Ba thí nghiệm hội tụ về **một** nguyên tắc:

> Batching và model concurrency chỉ có lãi khi GPU đang rảnh. Chúng thu hồi thời gian chết, chứ không tạo thêm năng lực tính toán.

Áp vào từng tầng của hệ thống này:

| tầng | GPU rảnh vì | nút vặn đúng |
|---|---|---|
| `asr_feature` | chạy CPU, không đụng GPU | số instance CPU |
| `asr_encoder` | không rảnh — compute-bound do đệm 16 giây | bỏ đệm cố định trước, rồi mới tính tới batching |
| `asr_scorer` | rảnh nhiều — 300 bước lặp op tí hon, kẹt ở kernel launch | **số instance** (đang 2, nên tăng) |
| `tts` | không rảnh — 16 bước flow matching nặng | không có nút nào; phải giảm `num_step` hoặc dùng bản distill |

Việc tách ASR thành ba model Triton riêng là điều kiện cần để có bảng này — gộp một cục thì chỉ đo được một con số tổng, không biết chữa ở đâu.

## Hạn chế của phép đo

- Máy chỉ có một GPU dùng chung với desktop, nên `--stability-percentage` phải nới lên 25%. Cột VRAM có nhiễu từ tiến trình khác (thấy rõ ở E1: 2 049 → 3 069 MB giữa hai lần đo cùng cấu hình).
- Mỗi cấu hình chỉ đo một lần, không lặp lại để lấy khoảng tin cậy.
- E2 dùng đúng một câu đầu vào; độ dài câu ảnh hưởng mạnh tới thời gian TTS nên con số tuyệt đối chỉ đúng cho câu đó.
