# Streaming ASR — Design Spec

**Ngày:** 2026-08-10
**Người thực hiện:** Nhân (VF-KPTX-VPTAITX)
**Trạng thái:** Design đã chốt, chờ implementation plan
**Kế thừa:** `2026-08-09-triton-voice-serving-design.md` (v1, non-streaming)

---

## 1. Bối cảnh

V1 đã serving thành công ASR non-streaming (`hynt/Zipformer-30M-RNNT-6000h`) theo dạng full audio: client gửi cả câu, nhận cả transcript. Mục 12 của spec v1 đã ghi sẵn bước tiếp theo: **streaming ASR bằng `hynt/Zipformer-30M-RNNT-Streaming-6000h`**.

Repo streaming trên HF **đã có sẵn ONNX export**, không cần export lại:

```
encoder-epoch-31-avg-11-chunk-{16,32,64}-left-128.fp16.onnx
decoder-epoch-31-avg-11-chunk-{16,32,64}-left-128.fp16.onnx
joiner-epoch-31-avg-11-chunk-{16,32,64}-left-128.fp16.onnx
bpe.model
```

Chỉ có bản fp16 — chạy trên GPU (CUDAExecutionProvider) nên không thành vấn đề.

Streaming thay đổi hợp đồng một cách căn bản:

| | Non-streaming (v1) | Streaming |
|---|---|---|
| Request | 1 request = cả câu | 1 stream = nhiều chunk nhỏ |
| State | không có | fbank buffer, encoder cache, hypothesis — sống qua các chunk |
| Scheduler | dynamic batcher | **sequence batcher** |
| Kết quả | 1 transcript cuối | partial transcript mỗi chunk, final khi kết thúc |

Cơ chế Triton tương ứng: `sequence_batching` — ghim mọi chunk của một stream vào cùng một model instance, đúng thứ tự, tối đa một request đang bay mỗi stream. Đó là điều kiện để giữ state per-stream đúng đắn.

## 2. Mục tiêu

Thêm model `asr_streaming` chạy **song song** với `asr` hiện có. Không đụng vào `asr`, `tts`, hay benchmark E1–E3.

**Thành công khi:**

1. `tritonserver` load sạch `asr_streaming` cùng các model cũ
2. `client/asr_streaming_client.py` gửi wav theo chunk qua một gRPC stream → nhận partial transcript dần dần, final đúng khi kết thúc
3. Toàn bộ test ở mục 8 xanh, test cũ không đỏ thêm cái nào

**Không phải mục tiêu (xem mục 12):** mic client thật, VAD/endpointing, ensemble + implicit state, TensorRT.

## 3. Quyết định kiến trúc

**Một Python backend duy nhất, giữ toàn bộ state trong process** (phương án A, đã chốt sau khi cân với 2 phương án khác).

```
client ══ gRPC stream ═══► asr_streaming  (Python, GPU ×1, sequence batcher)
        chunk + seq_id       │
        + START/END          │  state[corrid]:
                             │    fbank buffer ─► encoder.onnx (fp16, cached) ─► greedy step
        ◄════════════════════│                                                      │
        partial TRANSCRIPT   └──────────────────────────── sentencepiece ◄──────────┘
```

**Vì sao không tách 3 tầng như v1.** Lý do tách của v1 là dynamic batching cho encoder — encoder ăn ~95% FLOPs khi xử lý cả câu 16s một lượt. Ở streaming, mỗi lần gọi encoder chỉ là một chunk ~vài trăm ms, compute tí hon; và số đo E3 của chính project này đã chỉ ra scorer tuần tự mới là nút cổ chai chứ không phải encoder. Tách tầng lúc này mua được gần như zero batching benefit, nhưng phải trả giá bằng việc đẩy ~35 cache tensor của encoder qua ranh giới model **mỗi chunk**. State để một chỗ, mọi thứ đơn giản và test được.

**Hai phương án đã loại:**

- **B — ensemble + implicit state:** encoder thành model `onnxruntime` riêng, Triton tự quản cache tensor qua config `state`, batch được giữa các stream. Nhiều máy móc Triton nhất, nhưng ~35 khai báo state phải khớp từng tên với ONNX, ba model stateful phải đồng bộ, ensemble + sequence batcher nhiều cạnh sắc. Quá nặng cho v1 trên 4GB VRAM với vài stream đồng thời. Ghi làm đường nâng cấp (mục 12).
- **C — BLS orchestrator:** Python giữ state, gọi encoder model riêng qua BLS mỗi chunk. Cache round-trip Python↔encoder mỗi vài trăm ms — gánh phần lớn độ phức tạp của B mà hưởng rất ít lợi ích của nó.

## 4. Tài sản model

Tải từ `hynt/Zipformer-30M-RNNT-Streaming-6000h` về `model_repository/asr_streaming/1/`:

- `encoder.onnx`, `decoder.onnx`, `joiner.onnx` — bản `chunk-16-left-128.fp16` (mặc định), biến `CHUNK_VARIANT` trong `fetch_models.sh` đổi được sang 32/64
- `bpe.model`

**Vì sao chunk-16:** biến thể latency thấp nhất trong ba bản có sẵn. Số trong tên file là frame nội bộ của encoder, **không** phải ms; độ dài chunk thật (`decode_chunk_len`, tính theo mẫu 16kHz) và shape input `T` nằm trong metadata của file ONNX — đọc bằng `scripts/inspect_onnx.py` **trước khi** viết code, đúng nếp "kiểm shape trước config" của v1. Checkpoint không commit vào git, như cũ.

## 5. Interface

### `asr_streaming`

| Hướng | Tên | Kiểu | Shape | Ghi chú |
|---|---|---|---|---|
| in | `AUDIO_CHUNK` | FP32 | `[-1]` | mono 16kHz, độ dài **bất kỳ** ≥ 0 |
| out | `TRANSCRIPT` | STRING | `[1]` | transcript-tới-hiện-tại; ở request END là bản final |

Danh tính stream (`sequence_id`, cờ start/end) đi trong metadata gRPC của request, không phải tensor — client đặt qua `async_stream_infer(sequence_id=..., sequence_start=..., sequence_end=...)`.

**Hai lựa chọn có chủ đích:**

1. **Client không cần biết chunk size của encoder.** Model tự buffer mẫu audio, chỉ chạy encoder khi gom đủ `decode_chunk_len`. Client gửi 100ms hay 500ms một lần đều đúng — hợp đồng dễ tính, encoder đổi biến thể chunk không phải sửa client.
2. **Không đệm gì cả.** Trò đệm 16s của v1 tồn tại chỉ để phục vụ dynamic batcher; model này không dùng dynamic batcher nên bỏ.

## 6. Quyết định cấu hình

```
name: "asr_streaming"
backend: "python"
max_batch_size: 8
sequence_batching {
  oldest { max_candidate_sequences: 8 }
  max_sequence_idle_microseconds: 60000000
}
instance_group [ { kind: KIND_GPU, count: 1 } ]
```

| Quyết định | Lý do |
|---|---|
| `oldest` thay vì `direct` | Cho phép chunk của **nhiều stream khác nhau** vào chung một lượt `execute()` — code Python loop qua list request, mỗi request tra state theo CORRID. `direct` chia slot cứng, phức tạp hơn mà không cần cho backend Python tự quản state. |
| `max_candidate_sequences: 8` | Trần số stream đồng thời, khớp `max_batch_size`. Vượt trần thì stream mới xếp hàng chờ — hành vi đo được, là biến số cho E4. |
| `max_sequence_idle: 60s` | Stream chết không gửi END thì Triton tự hủy sequence, tránh giữ slot vĩnh viễn. |
| `count: 1` | Mỗi instance là một process ôm encoder + toàn bộ state của các stream nó phụ trách. VRAM 4GB, v1 chưa cần hơn. Tăng count là knob của E4. |

**State per-stream** (dict trong `model.py`, khoá CORRID):

- buffer mẫu audio thô chưa tiêu thụ (fbank cần chồng lấn 15ms cửa sổ giữa các chunk)
- cache tensor của encoder — khởi tạo zeros, shape đọc từ chính input của ONNX
- hypothesis greedy search: list token + `decoder_out` gần nhất

Sinh khi START, xoá khi END. Kèm sweep theo timestamp trong `execute()` xoá state mồ côi quá 60s — soi gương `max_sequence_idle` để dict không rò rỉ khi stream chết giữa chừng.

## 7. Cấu trúc thư mục (phần thêm mới)

```
model_repository/asr_streaming/
├── config.pbtxt
└── 1/
    ├── model.py               # glue Triton: control tensor, state dict, response
    ├── streaming_search.py    # hàm thuần: incremental fbank + greedy search step
    ├── encoder.onnx           # fetch, không commit
    ├── decoder.onnx
    ├── joiner.onnx
    └── bpe.model
client/asr_streaming_client.py
tests/test_streaming_search.py
tests/test_streaming_fbank.py
tests/test_asr_streaming.py
```

`streaming_search.py` theo đúng nguyên tắc của `greedy_search.py` v1: hàm thuần nhận numpy + callable, không import Triton, test được không cần GPU lẫn server. Khác biệt duy nhất: nhận và trả **state** thay vì chạy trọn vẹn một lượt.

## 8. Client

`client/asr_streaming_client.py`:

- Đọc wav → 16kHz mono (tái dùng `client/common.py`, **không** đệm)
- Cắt chunk ~200ms, gửi tuần tự qua một stream: `start_stream()` + `async_stream_infer`
- Mặc định gửi theo nhịp thời gian thật (ngủ 200ms giữa chunk) để mô phỏng mic; cờ `--fast` gửi dồn cho test
- Partial in đè dòng hiện tại, final in ra kèm xuống dòng

## 9. Test

TDD như cũ: viết test trước, xác nhận đỏ, mới code.

| File | Loại | Kiểm gì |
|---|---|---|
| `test_streaming_search.py` | unit | Encoder-out giả cắt nhiều đoạn, gọi search step nhiều lần → chuỗi token **giống hệt** chạy một lần trên toàn bộ; context và `decoder_out` mang đúng qua ranh giới chunk; chuỗi toàn blank → rỗng |
| `test_streaming_fbank.py` | unit | Fbank tính từng khúc == fbank tính cả file (sai số float cho phép). Đây là chỗ dễ sai nhất — lệch frame ở mép chunk là transcript hỏng âm thầm |
| `test_asr_streaming.py` | integration | Chunk `sample_vi.wav` qua stream thật → final khớp chuỗi kỳ vọng; partial cuối == final; số token chỉ tăng đơn điệu |

**Lưu ý trung thực:** checkpoint streaming là model khác (epoch-31, kiến trúc streaming) — chuỗi transcript kỳ vọng phải ghi lại từ chính nó chạy lần đầu, không tái dùng chuỗi của bản non-streaming.

Test integration đánh dấu `@pytest.mark.integration` như cũ.

## 10. Xử lý lỗi

| Tình huống | Xử lý |
|---|---|
| Stream chết không END | Triton hủy sau `max_sequence_idle` 60s; sweep trong `model.py` xoá state mồ côi |
| Chunk đến không có state (server restart giữa stream) | Khởi tạo state mới + log warning, không crash |
| Chunk rỗng / END không kèm audio | Hợp lệ — flush buffer còn lại, trả transcript hiện có |
| Lỗi trong một sequence | Error response cho riêng sequence đó, stream khác không ảnh hưởng |

## 11. Benchmark (E4 — tuỳ chọn, không chặn v1)

N stream đồng thời (1→8, client mô phỏng nhịp thật), đo p95 latency mỗi chunk và RTF tổng. Vặn `count` instance_group {1, 2} nếu VRAM cho phép. Đây là "model concurrency" phiên bản streaming, nối tiếp E1–E3.

## 12. Ngoài phạm vi

- Ensemble + implicit state (phương án B) — nâng cấp khi cần batch encoder giữa nhiều stream
- VAD / endpointing tự cắt câu
- Mic client thật (giao thức không đổi, chỉ là nguồn audio khác)
- TensorRT cho encoder streaming
- Bản int8 của checkpoint streaming (nếu HF bổ sung)

## 13. Rủi ro

| Rủi ro | Mức | Xử lý |
|---|---|---|
| Shape/metadata ONNX streaming khác dự đoán (số cache tensor, `T`, `decode_chunk_len`) | Trung bình | `inspect_onnx.py` đọc metadata **trước khi** viết config và code — chặn sớm như v1 đã làm với joiner |
| Incremental fbank lệch với offline fbank | Trung bình | `test_streaming_fbank.py` viết trước, đối chiếu số trực tiếp |
| fp16 trên CUDA EP cho kết quả lệch bản fp32 | Thấp | Chỉ có fp16 để dùng; nếu transcript sai hệ thống thì thử jit `.pt` làm đối chứng |
| Sequence batcher + Python backend có hành vi control tensor không như tài liệu | Thấp | Smoke test 2 stream đan xen ngay sau khi dựng skeleton, trước khi viết logic đầy đủ |

## 14. Quy ước làm việc

Như v1: identifier tiếng Anh, comment/docstring tiếng Việt, mở file bằng 2 dòng `# ABOUTME:`, giải thích tập trung ở mức config và kiến trúc.
