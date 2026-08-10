# Benchmark Matrix — Design Spec

**Ngày:** 2026-08-10
**Người thực hiện:** Nhân (VF-KPTX-VPTAITX)
**Trạng thái:** Design đã chốt, chờ implementation plan

---

## 1. Bối cảnh

Mentor xem qua số liệu `bench/README.md` (E1–E3) và giao lại yêu cầu đo cụ thể:

> Đo cho tất cả các thành phần của model (ví dụ asr: ensemble, feature, encoder, scorer).
> Metric gồm hai phần: **Latency** (P50, P90, P95, P99, min, max) và **Throughput**.
> Kịch bản test với **CCU 1–4**.
> Output là một ma trận các metric khớp với từng CCU từ 1 đến 4, cho cả ASR và TTS.

Số liệu hiện có không khớp ở bốn điểm:

| Khoảng trống | Hiện tại | Yêu cầu |
|---|---|---|
| Độ phủ component | chỉ `asr`, `asr_encoder`, `tts` | thêm `asr_feature`, `asr_scorer` |
| Percentile | p50, p95 | p50, p90, p95, p99, min, max |
| Mức CCU | rời rạc theo từng thí nghiệm (8; 1–16; 1–4) | thống nhất 1, 2, 3, 4 |
| Hình dạng output | ba bảng theo ba câu hỏi khác nhau | một ma trận component × CCU |

Thống kê từng tầng ở E3 là **trung bình** queue/compute lấy từ counter của Triton, không phải percentile — nên không dùng thay thế được.

## 2. Mục tiêu

Một lệnh sinh ra ma trận 5 component × 4 mức CCU × 7 metric, đủ để gửi mentor.

**Thành công khi:**

1. `python bench/bench.py e4` chạy xong không cần can thiệp tay, sinh `bench/results/matrix.csv` và `bench/results/matrix.md`
2. Ma trận có đủ 20 dòng (5 component × CCU 1–4), mỗi dòng đủ P50/P90/P95/P99/min/max/throughput
3. Percentile tự tính từ profile export khớp trong 5% với cả bốn percentile perf_analyzer tự in — lệch thì cảnh báo
4. Test đơn vị của `bench/stats.py` xanh, chạy được không cần server
5. E1–E3 và `results.csv` cũ không bị ảnh hưởng

**Không phải mục tiêu:**

- Đo lại E1–E3 theo format mới — số cũ vẫn là bằng chứng trả lời câu *tại sao*
- Quét `instance_group.count` hay `max_queue_delay` trong E4 — E4 đo hệ ở cấu hình mặc định
- Lặp nhiều lần mỗi cấu hình để lấy khoảng tin cậy — xem mục 8
- Đo latency phía client Python (`client/asr_client.py`) — xem mục 8

## 3. Quyết định kiến trúc

### 3.1 Thêm E4, không sửa E1–E3

E4 là một thí nghiệm mới trong `bench/bench.py`, ghi ra file kết quả **riêng** (`matrix.csv`), dùng chung hạ tầng `restart_server()` / `vram_mb()` / `run_perf()`.

Hai phương án bị loại:

- **Mở rộng `COLUMNS` toàn cục:** thêm 4 cột percentile vào bộ cột dùng chung sẽ làm header lệch khi E1–E3 append vào `results.csv` đang tồn tại, và để lại 4 cột rỗng cho mọi dòng cũ.
- **Viết `bench/matrix.py` độc lập:** ranh giới sạch hơn nhưng phải lặp lại `restart_server`, `vram_mb`, `run_perf` — ba hàm đã kiểm chứng qua E1–E3.

### 3.2 Hai nguồn số liệu, mỗi nguồn làm việc nó làm tốt nhất

Kiểm tra output thô của E3 (`bench/results/e3_perf_analyzer.txt`) cho thấy perf_analyzer **đã in sẵn cả bốn percentile** cùng số request:

```
Request count: 1726
Throughput: 23.829 infer/sec
p50 latency: 327724 usec
p90 latency: 362057 usec
p95 latency: 434014 usec
p99 latency: 584353 usec
```

Thiếu đúng `min` và `max`. Nên phân vai:

| Metric | Nguồn |
|---|---|
| throughput, P50, P90, P95, P99 | stdout của perf_analyzer — số có thẩm quyền, không phải suy diễn gì |
| min, max, số mẫu | profile export, tự tính từ timestamp từng request |

Đây là lý do quan trọng: cắt cửa sổ ổn định (mục 3.3) là bước phải phỏng đoán ngữ nghĩa của `window_boundaries`. Nếu bốn percentile chính cũng phụ thuộc bước đó thì một phỏng đoán sai sẽ âm thầm làm sai toàn bộ ma trận. Tách vai như trên thì phỏng đoán sai chỉ ảnh hưởng `min`/`max`, còn phép so chéo ở 3.6 biến nó thành lỗi nhìn thấy được.

Bản perf_analyzer 2.60.0 trong `.venv` có `--profile-export-file`, xuất JSON chứa timestamp **từng request**:

```
experiments[].experiment.{mode, value}
experiments[].requests[].{timestamp, response_timestamps, request_inputs, response_outputs}
experiments[].window_boundaries
```

Latency mỗi request = `response_timestamps[-1] - timestamp` (nanosecond). Có toàn bộ phân bố thì tính được mọi percentile, min và max.

Giữ perf_analyzer làm bộ tạo tải thay vì tự viết client Python: số liệu E4 so sánh trực tiếp được với E1–E3, và không phải kiểm chứng lại một bộ sinh tải mới từ đầu.

### 3.3 Chỉ tính trên cửa sổ ổn định

perf_analyzer chạy nhiều cửa sổ đo và chỉ coi các cửa sổ cuối là ổn định. Tính thống kê trên toàn bộ request trong file export sẽ trộn cả giai đoạn warmup vào — làm `max` vô nghĩa và `p99` bị kéo lệch.

`parse_profile_export` dùng `window_boundaries` để chỉ lấy request nằm trong **3 cửa sổ cuối** — bằng số cửa sổ perf_analyzer dùng để xét ổn định. Con số 3 đặt thành hằng số `STABLE_WINDOWS` kèm chú thích lý do; nếu file có ít hơn 4 mốc biên thì lấy toàn bộ request.

Đây là chỗ dễ chọn sai nhất trong cả thiết kế, và cũng là chỗ được phép kiểm chứng chéo ở 3.6 canh: chọn sai cửa sổ thì p50/p95 tự tính sẽ lệch khỏi số perf_analyzer in ra và cảnh báo nổ ngay.

Cộng thêm một lượt warmup bỏ đi cho mỗi component (mục 3.5).

### 3.4 Input cho `asr_scorer` bắt từ server thật

`asr_scorer` nhận `ENCODER_OUT` — tensor trung gian `[T, 512]` do encoder sinh ra, không có sẵn dưới dạng file.

RNN-T greedy search chạy số bước phụ thuộc nội dung tensor (mỗi bước phát một token hoặc blank), nên tensor ngẫu nhiên sẽ cho thời gian giải mã không đại diện. `gen_input.py` gọi `asr_feature` rồi `asr_encoder` trên server đang chạy với đúng `tests/assets/sample_vi.wav` mà E1–E3 dùng, lưu `ENCODER_OUT` thật vào `bench/input_scorer.json`.

Hệ quả: bước này cần server. Ba file input offline (`input_asr.json`, `input_tts.json`, `input_encoder.json`) được ghi **trước**, bước bắt scorer chạy cuối, để `gen_input.py` vẫn hữu ích khi server tắt.

Shape `[T, 512]` không hardcode ở hai nơi. `bench.py` đọc `input_scorer.json` và suy `T = len(ENCODER_OUT) / 512`, nên file input và tham số `--shape` không thể lệch nhau.

### 3.5 Warmup

Lần suy luận đầu của mỗi model gánh chi phí một lần: ONNX Runtime dựng execution plan, CUDA nạp kernel, Python backend nạp checkpoint lười. Chi phí này rơi trọn vào `max` — đúng metric mentor yêu cầu.

Mỗi component chạy một lượt perf_analyzer ngắn ở CCU 1 và **bỏ kết quả** trước khi vào vòng đo thật.

### 3.6 Tự kiểm chứng cách cắt cửa sổ

Từ profile export, tính lại cả bốn percentile P50/P90/P95/P99 rồi so với số perf_analyzer in ra. Lệch quá 5% ở bất kỳ percentile nào thì in cảnh báo kèm cả hai giá trị.

Bốn điểm so trên cùng một phân bố là phép kiểm chặt: nếu `min`/`max` đáng tin thì bốn percentile tự tính phải khớp, vì chúng đến từ đúng tập mẫu đó. Cảnh báo nổ nghĩa là tập mẫu dùng cho `min`/`max` không phải tập perf_analyzer đã báo cáo — lúc đó `min`/`max` mới là số cần xem lại, còn ma trận vẫn dùng được nhờ 3.2.

## 4. Ma trận đo

5 component, mỗi component ở CCU 1, 2, 3, 4:

| Component | Model Triton | Input | `--shape` | Ghi chú |
|---|---|---|---|---|
| ensemble ASR | `asr` | `input_asr.json` | `WAV:256000` | đầu-cuối, gồm cả 3 tầng |
| feature | `asr_feature` | `input_asr.json` | `WAV:256000` | Python backend, KIND_CPU count=4 |
| encoder | `asr_encoder` | `input_encoder.json` | `x:1600,80` | ONNX, KIND_GPU count=1 |
| scorer | `asr_scorer` | `input_scorer.json` | suy từ file | Python backend, KIND_GPU count=2 |
| TTS | `tts` | `input_tts.json` | — | ZipVoice, num_step=16 |

`asr_feature` nhận đúng cặp `WAV` + `WAV_LEN` như ensemble nên dùng lại được `input_asr.json`.

E4 **không sửa `config.pbtxt`** nào — đo hệ ở cấu hình mặc định đang commit. Nhờ vậy chỉ cần khởi động server một lần cho cả thí nghiệm, khác E1/E2 phải restart giữa mỗi bước.

### 4.1 Chế độ đo theo component

TTS chạy ~2.4 giây một câu, ở CCU 1 chỉ ~0.47 rps. Cửa sổ 20 giây cho ra khoảng 9 mẫu — quá ít để percentile p99 có nghĩa và để perf_analyzer hội tụ.

| Component | Chế độ | Tham số |
|---|---|---|
| `asr`, `asr_feature`, `asr_encoder`, `asr_scorer` | time_windows | `--measurement-interval 20000` |
| `tts` | count_windows | `--measurement-request-count 16` |

Giữ `--stability-percentage 25` như E1–E3: máy chỉ có một GPU dùng chung với desktop, để mặc định 10% thì không hội tụ.

Ước lượng thời gian chạy: ASR ~20 phút (16 lần đo), TTS ~8 phút, tổng ~30 phút.

## 5. Kiến trúc code

Tách phần tính toán khỏi phần chạy đo — phần tính toán là chỗ duy nhất có logic đáng test.

### 5.1 `bench/stats.py` (mới)

Hàm thuần, không subprocess, không mạng:

```
parse_profile_export(path) -> list[float]
    Đọc JSON export, trả về latency (ms) của các request trong cửa sổ ổn định.

latency_stats(latencies) -> dict
    {p50_ms, p90_ms, p95_ms, p99_ms, min_ms, max_ms, samples}
```

`samples` là số mẫu dùng để tính — cần để đọc `p99` một cách trung thực: dưới ~100 mẫu thì p99 chỉ là một hai request cuối, không phải phân vị thật. TTS ở `request_count=16` chắc chắn rơi vào trường hợp này, và đó là thông tin phải nói với mentor chứ không phải che đi.

Quy ước percentile: dùng `numpy.percentile` với nội suy tuyến tính (mặc định). Chọn numpy vì đã là dependency của project (`numpy==1.26.4`).

### 5.2 `bench/bench.py` (sửa)

- `parse_summary()` regex stdout lấy throughput + P50/P90/P95/P99 + request count.
- `run_perf()` thêm tham số `export_path`; khi có thì bổ sung `--profile-export-file`, gọi `stats.py` lấy `min`/`max`/`samples`, và chạy kiểm chứng chéo 3.6.
- `MATRIX_COLUMNS` và `write_matrix_csv()` riêng cho `matrix.csv`.
- `write_matrix_md()` sinh bảng markdown nhóm theo component.
- `e4()`: khởi động server một lần, lặp 5 component × CCU 1–4, warmup rồi đo.

`COLUMNS`, `write_csv`, `print_table` của E1–E3 giữ nguyên.

### 5.3 `bench/gen_input.py` (sửa)

Thêm bước cuối sinh `input_scorer.json` bằng `tritonclient.grpc`: gọi `asr_feature` → lấy `SPEECH`/`SPEECH_LEN` → gọi `asr_encoder` → lưu `encoder_out`.

Nếu server không sẵn sàng: in hướng dẫn chạy `scripts/serve.sh` rồi thoát khác 0. Ba file trước đó đã ghi xong nên chạy lại chỉ tốn vài giây.

## 6. Output

### 6.1 `bench/results/matrix.csv`

20 dòng, cột: `component`, `concurrency`, `throughput_rps`, `p50_ms`, `p90_ms`, `p95_ms`, `p99_ms`, `min_ms`, `max_ms`, `vram_mb`.

Ghi đè (không append) — E4 là một ma trận trọn vẹn, không phải log tích lũy như `results.csv`.

### 6.2 `bench/results/matrix.md`

Bảng markdown nhóm theo component, dán trực tiếp vào chat gửi mentor được:

```
## asr (ensemble)

| CCU | throughput (rps) | P50 | P90 | P95 | P99 | min | max |
|---|---|---|---|---|---|---|---|
| 1 | ... | ... | ... | ... | ... | ... | ... |
```

### 6.3 File export thô

`bench/results/export/<component>_ccu<N>.json` giữ lại để kiểm tra lại phép tính khi cần.

## 7. Test

Theo TDD: viết test trước, xác nhận fail, rồi mới viết `stats.py`.

### 7.1 `tests/test_bench_stats.py` — unit, không cần server

| Test | Kiểm điều gì |
|---|---|
| `test_latency_from_last_response_timestamp` | latency = `response_timestamps[-1] - timestamp`, đổi ns → ms đúng |
| `test_multiple_response_timestamps_uses_last` | request nhiều response (streaming) lấy mốc cuối |
| `test_requests_outside_stable_windows_excluded` | request trước `window_boundaries` ổn định bị loại |
| `test_percentiles_on_known_array` | p50/p90/p95/p99/min/max đúng trên mảng biết trước đáp án |
| `test_reports_sample_count` | `samples` bằng số mẫu thực dùng, để đọc p99 khỏi bị hiểu sai |
| `test_empty_stable_window_raises` | không có mẫu nào thì báo lỗi rõ ràng, không trả 0 âm thầm |

Dùng fixture JSON viết tay trong test, không phụ thuộc file export thật.

### 7.2 Kiểm chứng khi chạy thật

Phép so chéo ở 3.6 là lớp kiểm tra tích hợp: nó chạy mỗi lần đo, so số tự tính với số perf_analyzer in ra.

Không thêm test integration mới cần server — `tests/test_asr*.py` và `tests/test_tts.py` đã phủ phần model chạy đúng.

## 8. Hạn chế của phép đo

Ghi rõ để trình bày với mentor, không giấu:

- **`max` không phải số tái lập được.** Nó là đúng một request tệ nhất, chịu ảnh hưởng của nhiễu desktop và scheduler. Warmup và lọc cửa sổ ổn định chỉ giảm được phần warmup. Nên đọc `max` như chỉ dấu outlier, `p99` mới là số dùng để cam kết.
- **Mỗi cấu hình đo một lần**, không có khoảng tin cậy. Kế thừa hạn chế của E1–E3.
- **Máy một GPU dùng chung với desktop.** Cột VRAM có nhiễu từ tiến trình khác; `--stability-percentage` phải nới lên 25%.
- **Chỉ một mẫu đầu vào mỗi component.** ASR luôn đệm về 16 giây cố định nên độ dài audio thật không ảnh hưởng; nhưng TTS thì thời gian phụ thuộc mạnh vào độ dài câu, nên số TTS chỉ đúng cho câu mẫu đó.
- **Toàn bộ số liệu là phía server.** perf_analyzer nói gRPC trực tiếp với Triton, không tính overhead của `client/asr_client.py` (đọc file, tiền xử lý) lẫn độ trễ mạng. Latency người dùng cảm nhận sẽ cao hơn.
- **CCU 1–4 là dải hẹp.** E1b đã cho thấy `asr_encoder` chỉ chạm trần `max_batch_size: 8` ở CCU 16; ở CCU ≤ 4 batching chưa bộc lộ hết. Đây là dải mentor yêu cầu, không phải dải tìm giới hạn hệ thống.
