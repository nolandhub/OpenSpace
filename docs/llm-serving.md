# vLLM đứng riêng hay nằm trong Triton?

Trả lời câu: *"đã có Triton rồi, sao không load luôn LLM vào `model_repository`?"*

Triton **có** backend vLLM (`nvcr.io/nvidia/tritonserver:xx.yy-vllm-python-py3`). Câu hỏi
không phải làm được hay không, mà là nó mua được gì.

## Triton mua được scheduling — vLLM từ chối dùng

Giá trị của Triton nằm ở per-model scheduler: sequence batcher ghim state cho
`asr_streaming`, dynamic batcher gộp request. Đó là thứ hai model hiện tại sống nhờ.

vLLM mang theo continuous batching + paged attention của riêng nó. `vllm_backend` là một
Python backend bọc `AsyncLLMEngine` chạy decoupled mode: request vào là forward thẳng
xuống engine, Triton không xếp hàng, không gộp batch, không quyết định gì.

Chiếu đúng bộ quy tắc ở [`ensemble-vs-one-backend.md`](ensemble-vs-one-backend.md), câu hỏi
luôn quy về *"các tầng có thật sự muốn scheduling khác nhau không?"*. Đây là ca cực đoan
nhất của câu đó: tầng này muốn scheduling **riêng tuyệt đối**. Triton ở giữa là một proxy.

## Dependency đâm nhau ngay ở Dockerfile

`docker/Dockerfile` ghim cứng, và cả hai ghim đều có lý do sống còn ghi ngay tại chỗ:

    PIP_CONSTRAINT=numpy<2          # tritonserver + cupy + torch 2.5.1 đều đòi
    torch==2.5.1 (cu124)            # ZipVoice; thả nổi thì pip kéo CUDA 13 về

vLLM ghim torch theo từng minor của chính nó và ra bản mới vài tuần một lần. Chung một
image nghĩa là mỗi lần nâng vLLM là một lần đánh cược vào ZipVoice và onnxruntime-gpu.

## VRAM: đo trên máy dev (RTX 3050 Laptop, 4096 MiB)

| process | VRAM |
|---|---|
| `tritonserver` | 134 MiB |
| `asr_streaming` ×2 instance | 216 + 214 MiB |
| `tts` (ZipVoice + vocos) | 848 MiB |
| **còn trống** | **2321 MiB** |

vLLM preallocate KV cache lúc khởi động theo `gpu_memory_utilization` — tính trên **tổng**
VRAM, không phải phần còn trống — rồi giữ luôn, không trả lại.

Hai hệ quả đi thẳng vào `scripts/serve_llm.sh`:

1. **Thứ tự khởi động là ràng buộc, không phải thói quen.** Triton phải load xong cả 2
   model trước. Dựng ngược thì vLLM đo được nhiều bộ nhớ trống hơn thực tế, chiếm phần
   Triton chưa kịp xin, và Triton chết giữa request chứ không chết lúc load.
2. **848 MiB của `tts` là số lúc nghỉ.** Flow matching câu dài còn hơn — phải chừa headroom
   chứ không ép `GPU_FRACTION` sát mép.

## API surface

vLLM standalone cho sẵn `/v1/chat/completions`, SSE streaming, tool calling, structured
output, prefix caching, metrics Prometheus riêng. Qua Triton thì nhận endpoint `generate`
của Triton; muốn OpenAI-compatible phải dựng thêm frontend — mà frontend đó cũng là một
process riêng.

## Cái mất khi tách — và vì sao nó rẻ

Mất một endpoint duy nhất, một cổng metrics, và khả năng nối ASR→LLM→TTS bằng BLS ngay
trong server.

Nhưng cổng số 5 của bộ quy tắc — *"việc tách có ép dùng biểu diễn dữ liệu tệ hơn không?"* —
ở đây trả lời **không**. Payload giữa ba chặng là **text**, vài KB. Không có data locality
nào để giữ. Đây đúng là chỗ ngược với `asr_streaming`, nơi tách encoder ép 1.3 MB cache qua
lại để cõng 7 KB audio. Tách ở đây miễn phí; orchestration đẩy lên tầng app.

## Lên Jetson AGX Thor thì đổi gì

**Lý do VRAM chết.** 128GB LPDDR5X unified — capacity hết là ràng buộc.

**Nhưng tranh chấp đổi từ capacity sang bandwidth.** CPU và GPU dùng chung một bus
(~270 GB/s, kiểm lại datasheet). Decode của LLM bị chặn bởi bandwidth chứ không phải
compute: `tok/s ≈ băng thông / kích thước trọng số`. Model 8B fp16 (16GB) trần lý thuyết
~17 tok/s; cũng 8B ở int4 (~4.5GB) thì ~60 tok/s. **Chọn model theo ngân sách bandwidth,
không theo "nó vừa bộ nhớ"** — 128GB nhét vừa con 70B nhưng nó chạy ~3 tok/s.

Và ASR/TTS ăn chung đúng cái bus đó. Tách process thì còn đo được ai ăn bao nhiêu.

**Hai lý do mới, mạnh hơn cả ba lý do trên:**

*Đường migration khác nhau hoàn toàn.* Thor là aarch64 + JetPack 7 + CUDA 13.
`docker/Dockerfile` phải viết lại sạch: torch lấy từ index Jetson của NVIDIA chứ không
phải `download.pytorch.org/whl/cu124`, `onnxruntime-gpu` phải là build Jetson, Triton phải
là image `-igpu`. vLLM thì lấy container aarch64 dựng sẵn từ jetson-ai-lab. Gộp một image
nghĩa là ZipVoice kẹt thì LLM kẹt theo, và ngược lại.

*Trên Thor thì bật lại CUDA graph.* `ENFORCE_EAGER=0` — `--enforce-eager` chỉ là thuế phải
trả trên card 4GB.

## Nguồn

- [vLLM backend cho Triton](https://github.com/triton-inference-server/vllm_backend)
- [vLLM — Conserving Memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory.html)
- [Business Logic Scripting](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/bls.html)
