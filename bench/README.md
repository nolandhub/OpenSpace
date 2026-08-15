# Benchmark streaming ASR

Đo `asr_streaming` theo cách nó thật sự được dùng: cắt audio thành chunk 200ms,
gửi đúng nhịp thời gian thực qua một gRPC stream, đo latency **từng chunk**.

    .venv/bin/python bench/stream_bench.py --ccu 1 2 3 4 --duration 60

Kết quả ghi vào `results/streaming.md`, latency thô từng chunk vào
`results/streaming_ccu{N}.csv`.

## Đọc bảng thế nào

Một chunk mang 200ms audio. Server phải xử lý xong nó trong dưới 200ms, nếu
không audio dồn lại nhanh hơn tốc độ tiêu thụ và độ trễ tăng vô hạn.

| Cột | Nghĩa |
|---|---|
| `p50..max ms` | latency mỗi chunk: `t_nhận_partial − t_gửi_chunk`, gộp mọi phiên, **trừ chunk 0** |
| `RTF p95` | `p95 ms / 200`. Ngưỡng đậu/rớt. RTF là latency chia hằng số nên chỉ giữ một mốc |
| `first-chunk ms` | latency chunk `START` - chi phí mở phiên, tách ra để không làm bẩn p99 |
| `RTF stream` | thời gian server thật sự bận / độ dài audio. Đo công suất tiêu thụ, không đo độ trễ |
| `queue ms` | request chờ trước khi được xử lý. Cao = thiếu instance |
| `compute ms` | ghi công mỗi request. Tăng theo `batch avg`, xem cảnh báo bên dưới |
| `batch avg` | số sequence gom được mỗi lần `execute()` |

`RTF stream` lấy từ `batch_stats`, không lấy từ `inference_stats`: Triton ghi
trọn thời gian một lần `execute()` cho **từng** request trong batch, nên
`inference_stats` đếm lặp - đo thử thấy 86.4s trong khi server chỉ bận 47.6s.

## Kết quả

NVIDIA RTX 3050 Laptop, chunk 200ms, 60s audio mỗi phiên, ~300 chunk/phiên.
Bảng đầy đủ ở `results/streaming.md`.

| CCU | p50 ms | p95 ms | max ms | RTF p95 | RTF stream | queue ms | compute ms | batch avg |
|---|---|---|---|---|---|---|---|---|
| 1 | 18.85 | 23.68 | 43.6 | 0.118 | 0.067 | 0.05 | 13.32 | 1.0 |
| 2 | 20.68 | 45.28 | 99.79 | 0.226 | 0.068 | 6.91 | 13.65 | 1.01 |
| 3 | 21.92 | 68.38 | 111.01 | 0.342 | 0.068 | 9.12 | 22.68 | 1.51 |
| 4 | 23.97 | 92.77 | 164.55 | 0.464 | 0.074 | 11.47 | 32.73 | 1.87 |

**Cả 4 mức đều kịp thời gian thực.** CCU 4 dùng hết 46% ngân sách 200ms.

## Phân tích

### Latency lưỡng đỉnh - chunk 200ms lệch nhịp encoder

Latency 14 chunk đầu của một phiên:

    2.8  2.4  18.5  22.6  2.3  20.4  2.2  18.2  19.3  2.0  18.7  19.2  2.1  18.7
    └ rẻ ┘    └ đắt ┘

Encoder cần 45 frame mới chạy được một bước, mỗi bước nuốt 32 frame
(`decode_chunk_len`). Chunk 200ms chỉ nạp 20 frame. Nên khoảng 2/5 số chunk chỉ
cộng fbank rồi trả partial cũ (~2ms), 3/5 còn lại kích encoder + greedy (~19ms).

Hệ quả khi đọc bảng: `p50` nằm ở nhóm đắt, không phải "chi phí trung bình một
chunk". Phân bố là hỗn hợp hai nhóm, không phải một.

Đây cũng là chỗ chỉnh được: `chunk_ms` bằng 320 sẽ khớp `decode_chunk_len`, mỗi
chunk chạy đúng một bước encoder, latency đều, và số lượt round-trip giảm 38%.
Đổi lại độ trễ cảm nhận tăng vì phụ đề chỉ cập nhật mỗi 320ms. Chưa đo.

### Nút thắt là xếp hàng, không phải compute

`RTF stream` phẳng 0.067→0.074 qua cả 4 mức: mỗi giây audio luôn tốn server
~0.07 giây làm việc, bất kể tải. Theo con số đó thì compute chỉ bão hoà ở
khoảng 14 phiên đồng thời.

Nhưng latency xấu đi sớm hơn thế nhiều, và xấu theo một kiểu rất đặc trưng:

    p50:  18.85 → 20.68 → 21.92 → 23.97    (+27%)
    p95:  23.68 → 45.28 → 68.38 → 92.77    (+292%)

`p50` gần như đứng yên - phần lớn chunk không đụng ai. `p95` tăng gấp 4 - các
chunk kích encoder của nhiều phiên rơi trúng nhau và phải nối đuôi, vì
`instance_group.count: 1` và `execute()` duyệt từng request trong batch bằng
vòng `for` tuần tự (`model_repository/asr_streaming/1/model.py:182`).
`queue ms` 0.05 → 11.47 nói đúng điều đó.

Ngoại suy tuyến tính `RTF p95` (+0.115 mỗi CCU) thì chạm 1.0 quanh **CCU 8-9**,
trùng với trần `max_candidate_sequences: 8`. Coi đây là cận trên lạc quan: hàng
đợi tăng nhanh dần khi gần bão hoà, và mốc trên 4 chưa hề được đo.

### `compute ms` tăng không có nghĩa model chậm đi

13.32 → 32.73 nhìn như model chậm gấp 2.5. Không phải. `batch avg` cùng lúc
tăng 1.0 → 1.87, và request nằm trong batch lớn thì được ghi công nhiều hơn
(thiên lệch theo cỡ batch). Thước đo không thiên lệch là `RTF stream`, và nó
chỉ nhích 10%.

Con số 10% đó mới là chi phí thật của việc chạy 4 phiên chung một instance.

## Việc nên làm tiếp

- `instance_group.count: 2` - đánh thẳng vào `queue ms`, thứ đang chặn CCU
- Quét `chunk_ms` 200/320/640 - kiểm giả thuyết lệch nhịp encoder ở trên
- Đo CCU 6 và 8 - phần ngoại suy hiện chưa có gì chống lưng
- Batch thật trong `execute()` thay vì vòng `for` tuần tự, nếu encoder chịu batch

## Ghi chú phép đo

- Audio lặp từ `tests/assets/sample_vi_long.wav` cho đủ 60s: p99 cần đủ mẫu, mà
  file gốc chỉ 18.5s. Lặp làm audio thiếu tự nhiên nhưng thứ đang đo là thời
  gian xử lý, không phải độ chính xác transcript.
- Nhịp gửi tính theo mốc tuyệt đối `t0 + i×200ms`, không cộng dồn `sleep(200ms)`.
- Warmup một phiên ngắn trước khi đo, nếu không mức CCU chạy đầu gánh trọn chi
  phí khởi động ONNX/CUDA.
- Đã gặp một lần `queue ms` báo 162.69 trong khi `max ms` chỉ 124.18 - bất khả
  thi, do cửa sổ `get_inference_statistics` dính request ngoài phép đo. Không
  tái hiện được. `sanity_warnings()` giờ chặn và in cảnh báo ngay trong file
  kết quả thay vì để số vô lý nằm im.
