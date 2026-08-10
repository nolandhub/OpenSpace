# Kiến trúc & Luồng hoạt động — Triton Voice Serving

Tài liệu này giải thích ngắn gọn hệ thống hoạt động ra sao. Lý do đằng sau từng quyết định cấu hình nằm ở `docs/superpowers/specs/2026-08-09-triton-voice-serving-design.md` — file này chỉ tóm tắt phần "chạy như thế nào".

## Tổng quan

Một Triton Inference Server duy nhất, phục vụ 2 model tiếng Việt độc lập, không liên quan nhau:

- **`asr`** — ensemble 3 tầng (Zipformer RNN-T, non-streaming)
- **`tts`** — 1 Python backend (ZipVoice, zero-shot)

Client nói chuyện qua gRPC (cổng 8001). HTTP (8000) và metrics (8002) cũng mở nhưng client dùng gRPC.

## Sơ đồ tổng thể

```
client (gRPC)
  │
  ├──► asr  (ensemble, bản thân không có trọng số)
  │      asr_feature ──► asr_encoder ──► asr_scorer
  │      Python CPU×4    ONNX GPU×1      Python GPU×2
  │      (fbank 80d)     (dynamic batch)  (greedy search)
  │
  └──► tts  (1 Python backend, GPU×1)
         espeak-ng (vi) ──► ZipVoice (flow matching) ──► vocos ──► waveform 24kHz
```

## Flow ASR

1. **Client tiền xử lý.** Đọc wav bất kỳ → resample 16kHz mono → đệm cố định về 16 giây (256.000 mẫu, `client/common.py`). Gửi `WAV` (đã đệm) + `WAV_LEN` (độ dài thật). Đệm cố định tồn tại vì một lý do duy nhất: dynamic batcher của Triton chỉ gom được các request **cùng shape**.

2. **`asr_feature`** (Python, CPU×4) — tính fbank 80 chiều, nhưng chỉ trên phần audio thật (cắt theo `WAV_LEN` trước khi tính, bỏ phần đệm). Luôn xuất `SPEECH` shape cố định `(1600, 80)` + `SPEECH_LEN` (số khung thật) — nhờ vậy tầng sau luôn thấy cùng shape.

3. **`asr_encoder`** (ONNX, GPU, `dynamic_batching` 5ms) — nhận `SPEECH` cùng shape từ nhiều request đang chờ, Triton tự gom thành 1 batch thật cho backend `onnxruntime`. Đây là ~95% FLOPs của cả pipeline — một lượt matmul lớn cho cả câu. `x_lens` dùng để mask phần đệm nên kết quả không bị sai dù input đã đệm.

4. **`asr_scorer`** (Python, GPU×2) — chạy **greedy search** tuần tự theo từng frame (`greedy_search.py`), tách hoàn toàn khỏi Triton nên test được bằng numpy thuần:
   - Khởi tạo ngữ cảnh = 2 token blank, chạy `decoder` một lần.
   - Với mỗi frame `t` của `encoder_out`: đưa `(encoder_out[t], decoder_out)` qua `joiner` → argmax ra 1 token.
   - Token là blank → bỏ qua, sang frame tiếp theo. Token ≠ blank → nối vào lịch sử, chạy lại `decoder` (chỉ lúc này mới cần, đa số frame là blank nên phần lớn thời gian bỏ qua được bước này).
   - Hết vòng lặp → `sentencepiece.decode()` token id thành chuỗi tiếng Việt.

   Tầng này rẻ về khối lượng tính (op tí hon) nhưng chạy ~300 bước lặp tuần tự mỗi câu → bị chi phối bởi overhead launch kernel, không batch được.

5. `TRANSCRIPT` trả ngược qua ensemble về client.

## Flow ASR streaming (`asr_streaming`)

Model thứ ba, độc lập với ensemble `asr`. Cùng họ Zipformer RNN-T nhưng là checkpoint
streaming (`hynt/Zipformer-30M-RNNT-Streaming-6000h`, ONNX fp16, biến thể chunk-16-left-128).

1. **Client mở một gRPC stream**, cắt audio thành chunk (~200ms), gửi kèm `sequence_id`
   + cờ start/end. Không đệm gì cả — trò đệm 16s chỉ tồn tại vì dynamic batcher.
2. **Sequence batcher** (`oldest`) ghim mọi chunk của một stream vào cùng model instance,
   đúng thứ tự, tối đa 1 request đang bay mỗi stream → state per-stream trong process là hợp lệ.
3. `model.py` giữ state theo CORRID: fbank tính dần (`StreamingFbank`, khớp từng số với fbank
   offline), buffer khung, cache tensor của encoder streaming, hypothesis greedy. Đủ `T` khung
   → một bước encoder (tiêu thụ `decode_chunk_len` khung, phần dư là lookahead) → greedy
   search đi tiếp trên đoạn encoder_out mới.
4. Mỗi chunk trả **partial transcript** (transcript-tới-hiện-tại); request mang cờ END flush
   nốt buffer (đệm LOG_EPS) rồi trả bản final, state bị xoá.
5. Stream chết không END: Triton hủy sequence sau `max_sequence_idle` 60s; sweep trong
   `model.py` xoá state mồ côi tương ứng.

Vì sao 1 Python backend thay vì tách 3 tầng như `asr`: mỗi bước encoder giờ chỉ ăn vài trăm
ms audio, compute tí hon — tách tầng không mua được batching đáng kể mà phải đẩy cache
encoder qua ranh giới model mỗi chunk. Chi tiết và các phương án đã loại: spec 2026-08-10.

## Flow TTS

1. Client gửi `TEXT` (+ optional `PROMPT_WAV`, `PROMPT_TEXT`, `NUM_STEPS`, `SPEED`, `GUIDANCE_SCALE`). Không truyền prompt → dùng giọng mẫu đóng gói sẵn trong `model_repository/tts/1/assets/` (ZipVoice zero-shot **bắt buộc** phải có giọng mẫu để bám theo).

2. `model.py` (Python, GPU×1), một request là một lượt xử lý đầy đủ:
   - `EspeakTokenizer` (espeak-ng, `lang=vi`) chuyển text → phoneme.
   - `ZipVoice` chạy **flow matching** N bước (mặc định 16, đổi được qua `config.pbtxt` hoặc `NUM_STEPS` mỗi request) — từ nhiễu, sinh dần mel-spectrogram có điều kiện theo phoneme + giọng mẫu.
   - `vocos` (vocoder) chuyển mel-spectrogram → waveform 24kHz.
   - Input/output trung gian ghi tạm ra `/dev/shm` (tmpfs, RAM) vì hàm `generate_sentence` của ZipVoice nhận đường dẫn file chứ không nhận tensor trực tiếp.

3. Trả `WAV` + `SAMPLE_RATE=24000` về client.

`max_batch_size: 0` — không batch, vì câu dài ngắn khác nhau đệm sẽ rất phí. Song song lấy qua `instance_group.count`, không qua batching.

## Vì sao ASR tách 3 model Triton thay vì gộp 1 Python backend

Ba lý do cứng (chi tiết ở spec §5):

- **Python backend không tự batch** — Triton chỉ tự gộp request cho backend `onnxruntime`/`tensorrt`. Gộp cả 3 tầng vào 1 Python backend nghĩa là mất dynamic batching hoàn toàn.
- **Queue delay đặt sai chỗ** nếu gộp — nơi cần gom (`encoder`) khác nơi rẻ để chờ (`feature`).
- **VRAM** — muốn nhiều instance CPU cho `feature` thì kéo theo nhiều bản `encoder` nằm trên GPU. Với 4GB không chấp nhận được.

## Số đo thực tế (RTX 3050 Laptop 4GB, đo 2026-08-09)

Phát hiện quan trọng nhất, ngược với suy đoán ban đầu: **nút cổ chai của ASR là `asr_scorer`, không phải `asr_encoder`** — dù encoder gánh ~95% FLOPs.

| model | chờ hàng đợi (ms) | tính (ms) | batch tb |
|---|---|---|---|
| `asr_feature` | 2.5 | 18.8 | 1.00 |
| `asr_encoder` | 14.4 | 46.4 | 1.18 |
| **`asr_scorer`** | **168.4** | 79.3 | 1.00 |

`scorer` chờ 168ms để được tính 79ms — dồn hàng gấp đôi thời gian làm việc thật, vì mỗi instance xử lý tuần tự (~300 bước lặp/câu) và không batch được.

Vặn `max_queue_delay_microseconds` gần như vô dụng ở concurrency cố định (E1) — batch bị chặn bởi số request đang bay, không phải bởi thời gian chờ. Nút vặn đúng là concurrency client (E1b): batch leo từ 1.00 lên trần `max_batch_size: 8`, nhưng đổi lại throughput chỉ +34% trong khi p95 tăng 12 lần — hệ quả trực tiếp của việc đệm cố định 16 giây làm mỗi request đã đủ lớn để lấp GPU ngay ở batch 1.

Đầy đủ 3 thí nghiệm (E1, E2, E3) và phân tích: `bench/README.md`.

## Tài liệu liên quan

- Thiết kế đầy đủ, lý do từng quyết định cấu hình: `docs/superpowers/specs/2026-08-09-triton-voice-serving-design.md`
- Kế hoạch triển khai: `docs/superpowers/plans/2026-08-09-triton-voice-serving.md`
- Kết quả benchmark + phân tích: `bench/README.md`
