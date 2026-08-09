# Triton Voice Serving — Design Spec

**Ngày:** 2026-08-09
**Người thực hiện:** Nhân (VF-KPTX-VPTAITX)
**Trạng thái:** Design đã chốt, chờ implementation plan

---

## 1. Bối cảnh

Mentor giao hai model tiếng Việt và yêu cầu serving thử bằng Triton Inference Server:

| Vai trò | Model | HuggingFace |
|---|---|---|
| ASR | Zipformer 30M RNN-T, non-streaming, 6000h | `hynt/Zipformer-30M-RNNT-6000h` |
| TTS | ZipVoice zero-shot, 2500h | `hynt/ZipVoice-Vietnamese-2500h` |

Việc này nằm trong phần được phân công của Nhân trong dự án voice agent: **model serving — architecture, batching, ensemble, model concurrency**.

Đây là project **dựng mới hoàn toàn**. Không kế thừa, không sửa đổi `~/Projects/voice-agent-lab`.

## 2. Mục tiêu

Một Triton server chạy được cả hai model, có client gọi được, và có số đo chứng minh hiểu ba cơ chế được giao.

**Thành công khi:**

1. `tritonserver` load sạch cả hai model, `/v2/health/ready` trả 200
2. `client/asr_client.py` gửi file wav tiếng Việt → nhận transcript đúng
3. `client/tts_client.py` gửi câu tiếng Việt → nhận file wav nghe được
4. `bench/` sinh ra bảng số cho ba thí nghiệm ở mục 10
5. Toàn bộ test ở mục 9 xanh

**Không phải mục tiêu:**

- Streaming ASR (checkpoint mentor giao là bản non-streaming; bản streaming là repo khác)
- Ghép ASR → TTS thành pipeline nối chuỗi
- Tích hợp LiveKit/Pipecat — đó là phần của Hồng Anh
- Chạy trên Jetson Thor — giai đoạn sau, xem mục 12
- Tối ưu TensorRT — xem mục 12

## 3. Môi trường

| Hạng mục | Quyết định |
|---|---|
| Máy | Laptop local, RTX 3050 4GB VRAM, x86_64, Fedora |
| Chạy trong | Docker, image nền `nvcr.io/nvidia/tritonserver:25.01-py3` |
| Cổng | 8000 HTTP, 8001 gRPC, 8002 metrics |
| Client protocol | gRPC |

**Vì sao Docker chứ không cài native:** Triton Python backend cần `torchaudio`, `onnxruntime-gpu`, `sentencepiece`, `espeak-ng`. Trên Fedora với gcc 16, việc dựng chuỗi dependency này đã có tiền lệ hỏng (vụ nvcc ở lab trước). Container cho môi trường xác định và bê sang máy khác được.

## 4. Tài sản model

**ASR** — repo đã có sẵn ONNX, không cần export:

```
encoder-epoch-20-avg-10.onnx    92 MB
decoder-epoch-20-avg-10.onnx     5 MB
joiner-epoch-20-avg-10.onnx      4 MB
bpe.model                      268 KB
```
Bản `.int8.onnx` cũng có; v1 dùng bản fp32, int8 để dành cho thí nghiệm sau.

**TTS** — chỉ có checkpoint PyTorch:

```
iter-525000-avg-2.pt           491 MB
tokens.txt                     2.5 KB
```

Thiếu hai thứ, phải bổ sung:
- **Vocoder** — ZipVoice dùng vocos, không nằm trong repo này, tải riêng
- **espeak-ng bản `vi`** — tokenizer của model, cài trong Dockerfile

Checkpoint **không commit vào git**. `scripts/fetch_models.sh` tải từ HF về đúng vị trí trong `model_repository/`.

## 5. Kiến trúc

### ASR — ensemble 4 model

```
                 ┌──────────────────────────────────────┐
client ─request─►│ asr  (ensemble, không có trọng số)   │
                 │                                      │
                 │  asr_feature   WAV      → SPEECH     │  Python, CPU ×4
                 │       ↓                              │
                 │  asr_encoder   SPEECH   → ENC        │  ONNX, GPU, batching
                 │       ↓                              │
                 │  asr_scorer    ENC      → TRANSCRIPT │  Python, GPU ×2
                 └──────────────────────────────────────┘
                            │
client ◄─response───────────┘
```

**Vì sao tách ba tầng thay vì gộp một Python backend:**

| Tầng | Đặc tính tính toán | Cấu hình cần |
|---|---|---|
| `asr_feature` | DSP thuần, CPU, không tham số, request độc lập | nhiều instance CPU |
| `asr_encoder` | ~95% FLOPs, matmul lớn, một lần cho cả câu | dynamic batching, GPU |
| `asr_scorer` | vòng lặp tuần tự theo frame, op tí hon, latency do kernel launch overhead | nhiều instance, batching vô dụng |

Ba lý do cứng khiến không gộp được:

1. **Python backend không tự batch.** Triton chỉ tự gộp request thành batch cho backend `onnxruntime`/`tensorrt`. Với backend `python`, Triton đưa vào một `list` request và code tự xử lý. Gộp cả ba tầng vào một Python backend nghĩa là **không có dynamic batching ở bất kỳ đâu**.
2. **Queue delay đặt sai chỗ.** `dynamic_batching` gom request ở cửa vào của model. Gộp lại thì cửa vào là feature extraction — phải chờ 5ms để gom cho một việc tốn 3ms, còn nếu bỏ chờ thì encoder mất batching.
3. **VRAM.** Một instance = một process ôm trọn model. Muốn 4 instance để lấp CPU cho feature thì kéo theo 4 bản encoder nằm trên GPU. Với 4GB là không chấp nhận được.

Lợi ích kèm theo: Triton phơi metrics per-model ở `:8002/metrics`, tách ra là đo được từng tầng mà không phải cắm timer thủ công.

`decoder.onnx` và `joiner.onnx` **không** tách thành model Triton riêng. Chúng nằm trong vòng lặp greedy search; tách ra thì mỗi bước lặp phải vượt ranh giới model — đắt hơn nhiều so với lợi thu được.

### TTS — một Python backend

```
client ─request─► tts  (Python, GPU ×1)
                   espeak-ng vi → ZipVoice fp16 → vocos → waveform
```

Flow matching chạy vòng lặp N bước, không map được vào ensemble DAG tĩnh. Phương án tách thành BLS đòi hỏi export ZipVoice sang ONNX trước — chưa ai xác nhận checkpoint tiếng Việt này export sạch, nên để ngoài phạm vi v1.

Interface của `tts` được cố định ngay từ đầu để sau này thay ruột bằng BLS mà client không phải sửa.

## 6. Interface

Đây là hợp đồng với client. Tên cổng phải khớp giữa `config.pbtxt` và code.

### `asr`

| Hướng | Tên | Kiểu | Shape | Ghi chú |
|---|---|---|---|---|
| in | `WAV` | FP32 | `[-1]` | mono 16kHz, đã đệm về độ dài cố định |
| in | `WAV_LEN` | INT32 | `[1]` | số mẫu thật, chưa tính phần đệm |
| out | `TRANSCRIPT` | STRING | `[1]` | UTF-8 |

### `tts`

| Hướng | Tên | Kiểu | Shape | Ghi chú |
|---|---|---|---|---|
| in | `TEXT` | STRING | `[1]` | UTF-8 |
| in | `PROMPT_WAV` | FP32 | `[-1]` | optional, mặc định dùng prompt đóng gói sẵn |
| in | `PROMPT_TEXT` | STRING | `[1]` | optional |
| in | `NUM_STEPS` | INT32 | `[1]` | optional, mặc định 8 |
| out | `WAV` | FP32 | `[-1]` | |
| out | `SAMPLE_RATE` | INT32 | `[1]` | 24000 |

ZipVoice là zero-shot nên bắt buộc có prompt audio. Repo đóng gói sẵn một prompt tiếng Việt trong `model_repository/tts/1/assets/` để gọi trần không cần truyền gì.

## 7. Quyết định cấu hình

Đây là phần cốt lõi của bài. Mỗi giá trị dưới đây là một lựa chọn có lý do, và là biến số cho các thí nghiệm ở mục 10.

| Model | `max_batch_size` | `dynamic_batching` | `instance_group` |
|---|---|---|---|
| `asr_feature` | 8 | không | `KIND_CPU`, count 4 |
| `asr_encoder` | 8 | `max_queue_delay_microseconds: 5000` | `KIND_GPU`, count 1 |
| `asr_scorer` | 8 | không | `KIND_GPU`, count 2 |
| `asr` (ensemble) | 8 | — | — |
| `tts` | 0 | — | `KIND_GPU`, count 1 |

**`tts` đặt `max_batch_size: 0`** — mỗi câu dài ngắn khác nhau hoàn toàn, gom lại phải đệm rất phí. Tắt hẳn cho đơn giản; độ song song lấy qua `instance_group`.

**Xử lý độ dài thay đổi (quan trọng).** Dynamic batcher của Triton chỉ gom được các request **cùng shape**, trừ khi bật ragged batching. Audio dài ngắn khác nhau thì gần như không bao giờ gom được, và thí nghiệm batching sẽ ra số vô nghĩa.

Quyết định v1: **đệm về độ dài cố định**.
- Client đệm `WAV` về `MAX_DURATION_S = 16` giây (256.000 mẫu), gửi kèm `WAV_LEN`
- `asr_feature` luôn xuất ra `SPEECH` shape `(B, 1600, 80)`, kèm `SPEECH_LEN` mang độ dài thật
- Encoder do đó luôn thấy cùng shape → dynamic batching hoạt động thật
- Encoder dùng `x_lens` để mask phần đệm, kết quả không sai

Đánh đổi: câu 3 giây vẫn trả tiền tính toán cho 16 giây. Chấp nhận được vì bench dùng audio đồng đều độ dài nên không méo số. Hướng nâng cấp — bucketing hoặc ragged batching — ghi ở mục 12.

## 8. Cấu trúc thư mục

```
triton-voice-serving/
├── README.md
├── .gitignore                      chặn *.onnx, *.pt, *.wav lớn
├── docker/
│   ├── Dockerfile                  tritonserver + espeak-ng + python deps
│   └── compose.yaml
├── scripts/
│   ├── fetch_models.sh             tải HF → model_repository/*/1/
│   └── serve.sh                    dựng image + chạy container
├── model_repository/
│   ├── asr_feature/
│   │   ├── config.pbtxt
│   │   └── 1/model.py
│   ├── asr_encoder/
│   │   ├── config.pbtxt
│   │   └── 1/model.onnx
│   ├── asr_scorer/
│   │   ├── config.pbtxt
│   │   └── 1/{model.py, decoder.onnx, joiner.onnx, bpe.model}
│   ├── asr/
│   │   ├── config.pbtxt            ensemble
│   │   └── 1/                      rỗng, nhưng Triton bắt buộc phải có
│   └── tts/
│       ├── config.pbtxt
│       └── 1/{model.py, model.pt, tokens.txt, assets/}
├── client/
│   ├── asr_client.py
│   └── tts_client.py
├── tests/
│   ├── test_greedy_search.py       unit, không cần GPU lẫn server
│   ├── test_asr.py                 integration
│   ├── test_tts.py                 integration
│   └── assets/sample_vi.wav
└── bench/
    ├── bench.py
    └── results/
```

## 9. Test

Theo TDD: viết test trước, xác nhận đỏ, rồi mới code.

| File | Loại | Nội dung |
|---|---|---|
| `test_greedy_search.py` | unit | Vòng lặp greedy tách khỏi Triton, cho logits giả. Kiểm: chuỗi toàn blank → text rỗng; token phát ra đúng thứ tự; decoder chỉ được gọi lại khi gặp token khác blank |
| `test_asr.py` | integration | Gửi `sample_vi.wav` → transcript khớp chuỗi mong đợi |
| `test_tts.py` | integration | Sinh wav từ câu tiếng Việt → đúng 24kHz, không NaN, độ dài tỉ lệ hợp lý với số ký tự |

Test integration đánh dấu `@pytest.mark.integration`, yêu cầu server đang chạy. `pytest -m "not integration"` chạy được trên máy không có GPU.

Để `test_greedy_search.py` chạy độc lập, vòng lặp greedy phải nằm trong một hàm thuần tuý nhận numpy array, tách khỏi class `TritonPythonModel`. `model.py` chỉ import và gọi.

## 10. Benchmark

Ba thí nghiệm, mỗi cái ứng với một chủ đề được phân công.

**E1 — Dynamic batching (ASR).** Vặn `max_queue_delay_microseconds` qua {0, 1000, 5000, 20000} với concurrency cố định 8. Đo throughput và p50/p95. Kỳ vọng: throughput tăng theo delay rồi bão hoà, p95 xấu dần.

**E2 — Model concurrency (TTS).** Vặn `instance_group.count` qua {1, 2, 4}, concurrency client 1→8. Đo throughput, p95, và VRAM. Kỳ vọng: tăng đến khi chạm trần 4GB.

**E3 — Ensemble breakdown (ASR).** Ở tải cố định, đọc `nv_inference_queue_duration_us` và `nv_inference_compute_infer_duration_us` cho từng model con. Chỉ ra tầng nào là nút cổ chai.

Thí nghiệm phụ nếu còn thời gian: `num_steps` của TTS qua {4, 8, 16, 32} → RTF và chất lượng nghe.

Dùng `perf_analyzer` có sẵn trong image làm công cụ chính. `bench/bench.py` chỉ điều phối và gom kết quả ra CSV. Giữ format cột giống `voice-agent-lab/results/results.csv` để đặt cạnh nhau so sánh được.

## 11. Rủi ro

| Rủi ro | Mức | Xử lý |
|---|---|---|
| **API inference của ZipVoice chưa rõ** — không chắc repo `k2-fsa/ZipVoice` có hàm gọn để import hay phải vendor code vào `model.py` | Cao | Đọc repo và chạy được ZipVoice ngoài Triton **trước khi** viết `tts/1/model.py`. Nếu không import sạch được thì vendor phần tối thiểu vào `model_repository/tts/1/zipvoice/` và ghi rõ nguồn |
| **Tên cổng và shape của ONNX** — export của icefall có bản joiner nhận `(N, C)`, có bản nhận `(N, 1, 1, C)` | Trung bình | Kiểm bằng `onnx.load` ngay bước đầu, trước khi viết `config.pbtxt`. Sai tên thì Triton không load; sai shape thì lỗi lúc chạy |
| **4GB VRAM** — ASR ~120MB + TTS fp16 ~250MB thì thoải mái, nhưng E2 đẩy `count` lên 4 sẽ chạm trần | Trung bình | Đo VRAM trong E2 và ghi lại điểm gãy — đó là số liệu có giá trị, không phải thất bại |
| **vocos vocoder tải riêng** | Thấp | Thêm vào `fetch_models.sh` |

## 12. Ngoài phạm vi v1

Ghi lại để không quên, không làm ở vòng này:

- Export ZipVoice sang ONNX rồi tách thành BLS (đo được từng stage của flow matching)
- Encoder ONNX → TensorRT engine (chỉ động vào thư mục `asr_encoder/2/`)
- Ragged batching hoặc bucketing thay cho đệm cố định 16 giây
- Bản `.int8.onnx` của ASR — so accuracy đổi lấy tốc độ
- Streaming ASR bằng `hynt/Zipformer-30M-RNNT-Streaming-6000h`
- Port sang Jetson Thor (bản Triton igpu, ARM64)

## 13. Quy ước làm việc

- Giải thích tập trung ở mức `config.pbtxt` và kiến trúc. Code Python là đường ống, không mổ từng dòng.
- File code mở đầu bằng 2 dòng `# ABOUTME:` theo quy ước chung.
- **Ngôn ngữ trong code:** mọi identifier viết bằng **tiếng Anh** — tên hàm, tên biến, tên tham số, tên hàm test, khoá dict, hằng số. **Comment và docstring viết bằng tiếng Việt.** Không có ngoại lệ; một comment tiếng Việt giải thích hàm tên tiếng Anh là đúng chuẩn, ngược lại là sai.
