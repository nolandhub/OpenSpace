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
.venv/bin/python bench/bench.py e4      # ma trận đầy đủ 5 component × CCU 1-4
```

E1 và E2 tự sửa `config.pbtxt` và khởi động lại container giữa các lần đo, rồi trả config về mặc định khi xong.

---

## E4 — Ma trận 5 component × CCU 1–4

Đo ngày 2026-08-10. Số liệu thô: `results/matrix.csv`, bảng dán được: `results/matrix.md`, log: `results/e4_run.log`.

Khác E1–E3, E4 **không sửa `config.pbtxt` nào** — nó đo đúng cấu hình đang commit, mỗi component gọi trực tiếp (trừ dòng `asr` là cả ensemble).

### asr (ensemble)

| CCU | rps | P50 | P90 | P95 | P99 | min | max | mẫu |
|---|---|---|---|---|---|---|---|---|
| 1 | 18.75 | 46.5 | 47.1 | 47.4 | 48.0 | 45.9 | 51.6 | 150 |
| 2 | 21.42 | 86.0 | 89.1 | 92.4 | 98.1 | 49.3 | 138.1 | 150 |
| 3 | 21.42 | 120.7 | 130.2 | 147.5 | 180.4 | 51.0 | 193.5 | 150 |
| 4 | 21.42 | 160.7 | 183.7 | 245.2 | 365.8 | 46.7 | 413.9 | 150 |

### asr_feature

| CCU | rps | P50 | P90 | P95 | P99 | min | max | mẫu |
|---|---|---|---|---|---|---|---|---|
| 1 | 119.83 | 6.8 | 11.7 | 12.3 | 13.5 | 3.4 | 15.4 | 120 |
| 2 | 59.98 | 18.9 | 30.9 | 35.4 | 46.2 | 3.3 | 80.3 | 120 |
| 3 | 24.00 | 99.5 | 178.8 | 197.3 | 232.2 | 3.6 | 241.2 | 120 |
| 4 | 29.99 | 115.7 | 164.4 | 182.0 | 220.8 | 3.6 | 228.6 | 120 |

### asr_encoder

| CCU | rps | P50 | P90 | P95 | P99 | min | max | mẫu |
|---|---|---|---|---|---|---|---|---|
| 1 | 25.00 | 33.4 | 33.8 | 33.9 | 34.2 | 33.1 | 35.2 | 100 |
| 2 | 24.99 | 59.2 | 59.5 | 59.6 | 59.8 | 58.4 | 60.0 | 100 |
| 3 | 33.32 | 85.0 | 85.2 | 85.4 | 86.2 | 35.4 | 86.3 | 100 |
| 4 | 33.30 | 109.7 | 110.3 | 110.5 | 111.1 | 108.7 | 111.3 | 100 |

### asr_scorer

| CCU | rps | P50 | P90 | P95 | P99 | min | max | mẫu |
|---|---|---|---|---|---|---|---|---|
| 1 | 24.00 | 33.8 | 34.6 | 35.0 | 35.5 | 32.7 | 35.6 | 120 |
| 2 | 19.99 | 92.0 | 93.1 | 93.1 | 95.1 | 91.0 | 95.1 | 120 |
| 3 | 19.99 | 92.8 | 182.2 | 182.6 | 183.8 | 90.6 | 184.1 | 120 |
| 4 | 20.00 | 181.7 | 182.4 | 182.6 | 182.9 | 88.0 | 185.5 | 120 |

### tts

| CCU | rps | P50 | P90 | P95 | P99 | min | max | mẫu |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.48 | 2 062.7 | 2 076.5 | 2 080.1 | 2 082.4 | 2 017.8 | 2 082.4 | 48 |
| 2 | 0.48 | 4 146.1 | 4 164.3 | 4 182.3 | 4 209.2 | 2 065.8 | 4 209.2 | 48 |
| 3 | 0.48 | 6 270.2 | 6 288.2 | 6 309.7 | 6 316.1 | 2 082.1 | 6 316.1 | 48 |
| 4 | 0.48 | 8 338.6 | 8 347.5 | 8 349.5 | 8 352.0 | 2 088.8 | 8 352.0 | 48 |

### 1. Component nào latency tăng nhanh nhất theo CCU

Tỉ lệ P50 ở CCU 4 so với CCU 1:

| component | P50 tăng | rps CCU1 → CCU4 |
|---|---|---|
| **`asr_feature`** | **× 17,0** | 119,8 → 30,0 (**sụt 4 lần**) |
| `asr_scorer` | × 5,4 | 24,0 → 20,0 |
| `tts` | × 4,0 | 0,48 → 0,48 |
| `asr` (ensemble) | × 3,5 | 18,8 → 21,4 |
| `asr_encoder` | × 3,3 | 25,0 → 33,3 |

`asr_feature` thắng áp đảo, và nó là trường hợp duy nhất **throughput đi lùi**: 119,8 rps ở CCU 1 xuống 24–30 rps ở CCU 3–4. Bốn component kia throughput đứng hoặc tăng — latency chỉ tăng vì xếp hàng, năng lực xử lý không mất. Ở `asr_feature` thì năng lực bị **phá**.

Nguyên nhân là tranh chấp CPU. `asr_feature` cấu hình `KIND_CPU, count: 4`, mỗi instance là một tiến trình Python riêng gọi `torchaudio.compliance.kaldi.fbank`, mà torch mặc định mở thread theo số core. Ở CCU 1 chỉ một instance làm việc → được trọn CPU → 6,8 ms. Ở CCU 3–4 cả bốn instance chạy cùng lúc, mỗi cái lại bung thread → oversubscribe core, chi phí đổi ngữ cảnh ăn hết phần việc thật.

Bằng chứng chốt hạ nằm ở cột `min`: **3,3–3,6 ms ở mọi mức CCU**. Việc tính fbank vẫn nhanh y nguyên. Toàn bộ 115 ms ở P50 là chờ và giành CPU, không phải tính.

Đây là phát hiện mà E1–E3 không thấy: E3 đo `asr_feature` chỉ 2,5 ms chờ / 18,8 ms tính và xếp nó là tầng vô hại. Đo riêng ở CCU cao mới lộ ra nó mong manh nhất trong cả năm component. Hướng chữa: ghim `torch.set_num_threads(1)` trong `asr_feature/1/model.py` rồi để `count: 4` lo phần song song — chưa làm ở vòng này, ghi lại làm việc tiếp theo.

**Đối chiếu E1b:** `asr_encoder` ở đây throughput 25,0 → 33,3 rps, **+33%**; E1b đo độc lập hôm trước ra 28,8 → 38,6 rps, **+34%**. Hai lần chạy khác nhau, cùng một kết luận — con số đáng tin.

### 2. `asr_feature` (CPU ×4) so với `asr_scorer` (GPU ×2)

Hai kiểu suy giảm khác nhau về bản chất:

`asr_scorer` suy giảm **có thể dự đoán**. Throughput giữ phẳng ~20 rps ở cả bốn mức, latency tăng theo **bậc thang** chứ không mượt: 34 ms → 92 ms → 92/182 ms → 182 ms. Bậc thang đó là dấu vân tay của hàng đợi trước một số worker cố định. Rõ nhất ở CCU 3: P50 = 92,8 ms nhưng P90 = 182,2 ms — với 3 request và 2 instance, một request được phục vụ ngay, hai request kia phải đợi một lượt. Sang CCU 4 thì mọi request đều đợi đúng một lượt, phân bố lại gọn về 182 ms.

`asr_feature` suy giảm **bệnh lý**. Throughput không giữ được mà sụt 4 lần. Thêm request vào không chỉ làm chúng phải chờ — nó làm hệ thống xử lý chậm đi thật.

Bài học rút ra là một mệnh đề về `instance_group`:

> Tăng số instance chỉ chia được việc nếu tài nguyên bên dưới còn rảnh. `count: 2` trên GPU cho hàng đợi sạch. `count: 4` trên CPU mà mỗi instance tự bung thread thì bốn instance giành nhau đúng số core đó.

Nói cách khác `count` không phải nút vặn độc lập — nó chỉ đúng khi đi kèm việc chặn song song ở tầng dưới.

### 3. Khoảng cách P99 ↔ max nói gì về độ ổn định

| component (CCU 4) | P99 | max | khoảng cách |
|---|---|---|---|
| `asr_encoder` | 111,1 | 111,3 | **0,2 ms** |
| `asr_scorer` | 182,9 | 185,5 | 2,6 ms |
| `asr_feature` | 220,8 | 228,6 | 7,8 ms |
| `asr` (ensemble) | 365,8 | 413,9 | **48,1 ms** |

`asr_encoder` gần như không có đuôi — 0,2 ms giữa P99 và max, tức không tồn tại request nào cá biệt. Hợp lý: một lượt matmul cỡ cố định trên tensor đã đệm về 16 giây, không có nhánh rẽ nào để mà biến động.

Ensemble ngược lại có đuôi dài nhất, 48 ms. Nó không có nguồn biến động riêng — nó **cộng dồn** biến động của cả ba tầng, và request xấu nhất là request gặp lúc xấu ở nhiều tầng cùng lúc. Đây là lý do thực dụng để đo từng tầng chứ không chỉ đo đầu-cuối: p99 của chuỗi luôn tệ hơn p99 của bất kỳ mắt nào trong chuỗi.

`tts` phải đọc riêng: **P99 = max chính xác ở cả bốn mức** (2 082,4 = 2 082,4). Đây không phải phát hiện gì về TTS, mà là hệ quả của cỡ mẫu 48 — phân vị 99 của 48 mẫu rơi đúng vào giá trị lớn nhất. Xem hạn chế bên dưới.

Cột `min` của `tts` cũng đáng chú ý: đứng ở ~2 020–2 090 ms trong khi P50 leo 2 063 → 8 339 ms. Ở mọi mức CCU vẫn luôn có request không phải chờ ai — đúng hình dạng "một worker tuần tự, phần còn lại xếp hàng" mà E2 đã kết luận.

### Về cảnh báo lệch percentile trong log

Log có **một** dòng cảnh báo:

```
! asr@4 p99_ms: export 328.2 ms vs perf_analyzer 365.8 ms (lệch 10%)
```

Đã truy nguyên, và **không phải lỗi dữ liệu**: cột `mẫu` của dòng đó là 150, đúng bằng `--request-count 150`, nên tập mẫu dùng để tính min/max trùng khít tập perf_analyzer báo cáo. Lệch đến từ **quy ước ước lượng phân vị**: numpy nội suy tuyến tính (p99 của 150 mẫu rơi giữa phần tử thứ 148 và 149), perf_analyzer lấy theo thứ hạng nguyên. Ở CCU 4 đuôi rất nặng (P95 245 ms, max 414 ms) nên hai phần tử liền kề cách nhau vài chục ms.

`min`/`max` của dòng này **vẫn tin được** — chúng là thống kê thứ tự thuần, không nội suy nên không có chỗ cho hai quy ước bất đồng. Cái cần nới là ngưỡng 5%: nó quá chặt cho p99 ở cỡ mẫu nhỏ với đuôi nặng.

### Hạn chế riêng của E4

1. **Cỡ mẫu nhỏ: 100–150 request mỗi ô, TTS 48.** P99 vì thế chỉ là một hai request chậm nhất, không phải phân vị thật — thấy rõ nhất ở `tts` nơi P99 trùng khít max. Cột `mẫu` có trong bảng đúng để đọc kèm.
2. **`--request-count` tắt vòng lặp ổn định của perf_analyzer**, nên `--stability-percentage 25` không còn tác dụng ở E4. Số đo là một lát cắt cố định, **không phải giá trị đã hội tụ** — khác E1–E3 ở điểm này, cần nhớ khi đặt hai bảng cạnh nhau.
3. **`max` không tái lập được** giữa các lần chạy.
4. **Toàn bộ là số phía server**, chưa tính overhead client.
5. **Vì sao cỡ mẫu phải nhỏ:** `--profile-export-file` nhúng nguyên tensor input vào mỗi bản ghi và giữ hết trong RAM client — đo được 6,4 MB dữ liệu và ~21 MB RAM cho **mỗi** request của `asr_encoder`. Cửa sổ 20 giây ở 25 rps cần ≈ 31 GB cho ba cửa sổ; máy có 15 GB và một lần chạy đã bị kernel oom-kill ở 6,6 GB RSS. Nghẽn là **RAM của máy chạy client, không phải VRAM** (VRAM lúc load cả 6 model chỉ dùng 2,3/4,1 GB). Ai chạy lại trên máy nhiều RAM hơn thì nâng `total_requests` trong `e4()` lên để P99 có nghĩa hơn.

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
| `asr_feature` | chạy CPU, không đụng GPU | số instance CPU — **nhưng xem E4**: `count: 4` mà không chặn thread của torch thì bốn instance giành core, throughput sụt 4 lần |
| `asr_encoder` | không rảnh — compute-bound do đệm 16 giây | bỏ đệm cố định trước, rồi mới tính tới batching |
| `asr_scorer` | rảnh nhiều — 300 bước lặp op tí hon, kẹt ở kernel launch | **số instance** (đang 2, nên tăng) |
| `tts` | không rảnh — 16 bước flow matching nặng | không có nút nào; phải giảm `num_step` hoặc dùng bản distill |

Việc tách ASR thành ba model Triton riêng là điều kiện cần để có bảng này — gộp một cục thì chỉ đo được một con số tổng, không biết chữa ở đâu.

## Hạn chế của phép đo

- Máy chỉ có một GPU dùng chung với desktop, nên `--stability-percentage` phải nới lên 25%. Cột VRAM có nhiễu từ tiến trình khác (thấy rõ ở E1: 2 049 → 3 069 MB giữa hai lần đo cùng cấu hình).
- Mỗi cấu hình chỉ đo một lần, không lặp lại để lấy khoảng tin cậy.
- E2 dùng đúng một câu đầu vào; độ dài câu ảnh hưởng mạnh tới thời gian TTS nên con số tuyệt đối chỉ đúng cho câu đó.
