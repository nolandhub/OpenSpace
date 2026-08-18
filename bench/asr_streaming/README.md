# asr_streaming — benchmark

Nguyên tắc phân công và bố cục chung: [`../README.md`](../README.md).

    ./scripts/perf.sh asr_streaming                      # p50/90/95/99, throughput, GPU util
    .venv/bin/python bench/asr_streaming/metrics.py      # first-chunk latency + WER

## Kết quả

RTX 3050 Laptop, `instance_group.count: 2`, chunk 200ms, audio thật, GPU nguội.

| CCU | infer/s | p50 ms | p90 ms | p95 ms | p99 ms | queue ms | GPU |
|---|---|---|---|---|---|---|---|
| 1 | 88.4 | 15.35 | 19.30 | 19.72 | 20.57 | 0.06 | 35% |
| 2 | 132.7 | 20.25 | 24.24 | 26.41 | 30.60 | 0.13 | 67% |
| 3 | 134.8 | 22.10 | 42.61 | 45.54 | 51.93 | 6.75 | 74% |
| 4 | **138.4** | 25.33 | 45.04 | 48.06 | 56.21 | 12.82 | 77% |

Một request mang 200ms audio → **138 infer/s = 27.7x thời gian thực**, hoặc **27 phiên
real-time đồng thời** (mỗi phiên gửi 5 chunk/giây).

`metrics.py`:

| CCU | first-chunk p50 ms | first-chunk p95 ms |
|---|---|---|
| 1 | 2.42 | 4.61 |
| 2 | 3.08 | 4.71 |
| 3 | 3.98 | 10.40 |
| 4 | 4.46 | 8.16 |

## Đọc kết quả

**Hai chế độ rõ rệt.** CCU 1→2: queue vẫn ~0, latency tăng hoàn toàn do compute
(10.3 → 13.4ms) — hai instance cùng chạy trên một GPU nên mỗi cái chậm đi, đổi lại
throughput +50%. Từ CCU 3: hết instance, queue nhảy 0.13 → 6.75 → 12.82ms còn throughput
đứng yên. Mọi tải thêm vào đều thành thời gian chờ.

Cộng lại kiểm chứng được: ở CCU 3, `compute 14.9 + queue 6.75 ≈ 21.6ms` ≈ p50 22.10ms.

**p95 phồng nhanh hơn p50 nhiều** (+213% so với +65% qua 4 mức) vì xếp hàng đánh vào đuôi:
request nào rơi đúng lúc cả hai instance đang bận thì phải chờ trọn một lượt compute.

**Vẫn còn xa hạn real-time.** p99 ở CCU 4 là 56.21ms, hạn là 200ms — còn dư 3.5 lần.

**first-chunk rẻ và không phải vấn đề.** 2.42 → 4.46ms, nhỏ hơn p50 chunk thường một bậc.
Chunk mở phiên chỉ nạp 20 khung, chưa đủ 45 để chạy một bước encoder, nên nó chỉ tốn fbank.
Dựng state cho phiên mới gần như không tốn gì.

**Nút thắt không phải GPU.** Trần 138 infer/s đạt khi GPU mới 77%. Nguyên nhân ở
`model_repository/asr_streaming/1/model.py`: `execute()` nhận cả batch rồi duyệt tuần tự
`for request in requests`, nên encoder chạy 8 lần với batch=1 thay vì 1 lần với batch=8.

## WER

**2.56%** trên `sample_vi_long.wav` — **con số này không dùng được để đánh giá chất lượng.**

`tests/assets/sample_vi_long.wav` lấy từ [`doof-ferb/fpt_fosd`](https://huggingface.co/datasets/doof-ferb/fpt_fosd),
mà FPT nằm trong danh sách 6000h training của
[`hynt/Zipformer-30M-RNNT-Streaming-6000h`](https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h).
Đây là đo trên chính dữ liệu đã train. Tác giả công bố **14.29%** cho biến thể streaming
chunk-32 trên VLSP2023-PublicTest — đó mới là dải thật.

Muốn số có nghĩa thì cần bộ held-out. Ứng viên: **VIVOS** (test 760 utterance / 19 speaker
/ 7722 từ) — không có trong danh sách training. Tránh FLEURS, BUD500, GigaSpeech2-vi,
ViVoice, PhoAudiobook, VLSP2020/21/23: đều đã nhiễm.

## Việc nên làm tiếp

1. **Stack request trong `execute()`** — đo trực tiếp trên `encoder.onnx`: batch 1 tốn
   11.00ms, batch 8 tốn 14.37ms (**1.80ms mỗi chunk**). Gộp 8 chunk nhanh hơn 6.1 lần so
   với 8 lần gọi riêng. Encoder 30M tham số fp16 ≈ 60MB trọng số phải đọc từ VRAM mỗi lần
   gọi để xử lý 7KB audio — batch amortize việc đọc đó. Ước tính ở tầng hệ thống là 2-2.5x
   (fbank chạy CPU và greedy là vòng lặp, cả hai không batch thẳng được).
2. **WER trên bộ held-out** — xem mục trên.

## Ghi chú phép đo

- `--shape AUDIO_CHUNK:3200` bắt buộc: `dims: [-1]` thì perf_analyzer không đoán được cỡ.
- Input JSON dùng dạng `data` **lồng** — mỗi mảng con là một sequence, các phần tử là các
  bước theo thứ tự. Phẳng một tầng thì perf_analyzer hiểu thành nhiều sequence riêng lẻ mỗi
  cái một bước, encoder không bao giờ tích luỹ state.
- **Warmup là một lượt `perf_analyzer` riêng** trong `perf.sh`, không phải cờ
  `--warmup-request-count` — perf_analyzer từ chối cờ đó khi quét nhiều mức CCU
  (`Error: --warmup-request-count not supported with multiple concurrency values in one run`).
- `--verbose-csv` để GPU util vào thẳng CSV; không có nó thì phải bóc từ stdout.
- perf_analyzer xếp hàng trong CSV theo throughput tăng dần, nên `perf.sh` sắp lại theo CCU
  trước khi in. Đọc file thô sẽ thấy thứ tự 1, 3, 4, 2.
- WER chạy một phiên đầy đủ, một mình — đây là phép đo chất lượng, không phải phép đo tải.
- first-chunk cần nhiều lần **mở** phiên chứ không cần phiên dài, nên mỗi lượt chỉ gửi 10
  chunk rồi đóng.
