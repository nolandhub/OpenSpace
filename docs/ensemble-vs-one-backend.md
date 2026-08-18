# Ensemble hay một Python backend?

Trả lời câu: *"sao không tách `asr_streaming` thành nhiều model riêng cho encoder,
decoder, joiner?"*

## Ensemble mua được đúng một thứ

**Scheduling riêng cho từng tầng** — instance count, batching policy, device, metrics,
version. Hết. Module hoá / test được / code sạch thì tách file trong một backend là đủ.

Nên câu hỏi luôn quy về: *các tầng có thật sự muốn scheduling khác nhau không?*

## `decoder` + `joiner`: không tách được

Greedy search của transducer là vòng lặp có số bước phụ thuộc dữ liệu: mỗi khung encoder →
joiner → nếu ra token non-blank thì nạp vào decoder rồi **chạy joiner lại** → tới khi ra blank.

Ensemble không diễn đạt được cái đó. NVIDIA nói thẳng:

> Triton's ensemble feature supports many use cases where multiple models are composed into
> a pipeline (or more generally a DAG, directed acyclic graph). **However, there are many
> other use cases that are not supported because as part of the model pipeline they require
> loops, conditionals (if-then-else), data-dependent control-flow** and other custom logic
> to be intermixed with model execution.
>
> — [Business Logic Scripting](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/bls.html)

Đường duy nhất là BLS — một Python model gọi `pb_utils.InferenceRequest` sang decoder/joiner.
Tức là đổi N lần gọi in-process thành N lần gọi qua Triton, mỗi khung một lần. Tệ hơn
nghiêm ngặt.

Đây cũng là lý do `tts` là một backend: flow matching N bước cũng là vòng lặp.

## `encoder`: tách được, nhưng không lời

Encoder streaming **không phải** hàm `audio → feature` thuần. Nó là
`(audio, cache) → (feature, cache')`. Số lấy từ chính file ONNX:

| | non-streaming encoder | streaming encoder |
|---|---|---|
| inputs | 2 (`x`, `x_lens`) | **75** (`x` + 74 tensor cache) |
| outputs | 2 | **75** |
| payload thật/bước | cả câu | `x` = **7 KB** (45 khung) |
| cache kèm theo | không | **0.66 MB** vào, ngần ấy ra |

74 tensor đó là `cached_key`, `cached_val1/2`, `cached_nonlin_attn`, `cached_conv1/2` × 12
layer, cộng `embed_states` + `processed_lens`.

Hai đường tách:

1. **Cache đi qua ranh giới dạng tensor** — ~1.3 MB qua lại để cõng 7 KB audio, tỷ lệ 190:1.
2. **Implicit state management** — khai `state` trong `sequence_batching`, Triton giữ cache
   hộ server-side, không tốn wire.

Đường 2 sạch, nhưng có hai ràng buộc:

> Currently, only **onnxruntime_backend, tensorrt_backend, and pytorch_backend** support
> implicit state.
>
> — [Implicit State Management](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/implicit_state_management.html)

Nghĩa là encoder phải được phục vụ bằng ONNX backend — đúng kịch bản tách. **Nhưng
`StreamingFbank` và `SearchState` của greedy vẫn phải sống ở đâu đó**, và Python backend
không dùng được implicit state. Vẫn cần một Python backend điều phối theo sequence.

**Tách encoder không bỏ được BE Python — chỉ thêm một round-trip mỗi bước encoder.**

Không phải vì "tốn công đồng bộ CORRID": sequence batcher làm đúng việc đó hộ, tự ghim
mọi chunk cùng `sequence_id` vào một instance, đúng thứ tự
([Triton Architecture](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html)).

## Cái tách thật sự mua được — và nó đã có sẵn

Với streaming, thứ đáng giá nhất là **gộp bước encoder của nhiều stream vào một lần gọi GPU**.
Đo trực tiếp trên `encoder.onnx` (RTX 3050, fp16, CUDA EP, 50 lần sau 10 lần warmup):

| batch | ms/lần gọi | ms mỗi chunk |
|---|---|---|
| 1 | 11.00 | 11.00 |
| 2 | 12.03 | 6.01 |
| 4 | 12.94 | 3.24 |
| 8 | 14.37 | **1.80** |

8 chunk trong một lần gọi tốn 14.37ms; tám lần gọi riêng tốn 88ms — **nhanh hơn 6.1 lần**.
Encoder 30M tham số fp16 ≈ 60MB trọng số phải đọc từ VRAM mỗi lần gọi để xử lý 7KB audio.
Batch 8 đọc một lần rồi làm 8 lần phép tính: +31% thời gian cho 8 lần công việc.

**Nhưng batch đó không cần tách mới có.** `config.pbtxt` đã bật:

    max_batch_size: 8
    sequence_batching { oldest { max_candidate_sequences: 8 } }

> With the Oldest scheduling strategy the sequence batcher ensures that all inference
> requests in a sequence are routed to the same model instance and then **uses the dynamic
> batcher to batch together multiple inferences from different sequences** into a batch that
> inferences together.
>
> — [Triton Architecture](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html)

Batch đã được đưa tới cửa `execute()`. Vấn đề là `model.py` duyệt `for request in requests`
xử lý tuần tự — encoder chạy 8 lần với batch=1 thay vì 1 lần với batch=8.

Đó chính là lý do throughput chạm trần ~138 infer/s khi GPU mới 77%.

## Vì sao `asr_nonstreaming` từng tách 3 tầng — và số đo nói gì

Giả thuyết ban đầu: encoder gánh ~95% FLOPs, mà dynamic batching chỉ tác dụng ở ranh giới
model, nên tách encoder ra để nó được batch. Đo xong (RTX 3050):

| model | chờ hàng đợi (ms) | tính (ms) | batch tb |
|---|---|---|---|
| `asr_feature` | 2.5 | 18.8 | 1.00 |
| `asr_encoder` | 14.4 | 46.4 | **1.18** |
| **`asr_scorer`** | **168.4** | 79.3 | 1.00 |

Giả thuyết sai: `batch avg` chỉ 1.18, và nút cổ chai là `asr_scorer` — tầng batching vô
dụng theo bản chất (~300 bước lặp tuần tự/câu). Ép concurrency cho batch chạm trần 8:
throughput +34%, p95 ×12.

**Đắt nhất là chỗ không tầng nào chịu.** Batching đòi mọi request cùng shape → đệm cứng 16
giây → câu 3 giây vẫn chạy encoder cho 16 giây. *Điều kiện để bật một tối ưu tốn hơn chính
tối ưu đó.*

Bản ensemble đã gỡ khỏi `model_repository` ở `ba106d9` (code lấy lại được:
`git show cc709ce:model_repository/asr_encoder/config.pbtxt`).

## Quy tắc quyết định

Hỏi theo thứ tự:

1. **State có phải sống qua nhiều lần gọi không?** → có thì one-backend
2. **Control flow có phải DAG tĩnh không?** → không thì one-backend
3. **Có tầng nào tensor-in/tensor-out thuần không?** → có thì tầng đó đáng tách
4. **Các tầng có muốn instance count khác nhau — và tầng cần scale có nhẹ VRAM không?**
5. **Việc tách có ép dùng biểu diễn dữ liệu tệ hơn không?** ← cổng hay quên nhất, và là
   cổng giết ensemble ở đây

(1) và (3) không độc lập: tách model không làm state biến mất, nó biến state thành
**payload**. Cùng một cổng nhìn từ hai phía.

Kèm theo: tensor qua ranh giới phải nhỏ so với công việc hai đầu, và **đừng để vòng lặp
cắt ngang ranh giới**.

## Ba case trong repo

| | state qua lời gọi | DAG tĩnh | tầng ONNX thuần | kết luận |
|---|---|---|---|---|
| `asr_streaming` | có (cache 74 tensor) | — | không (state là chữ ký) | one-backend |
| `tts` | không | **không** (flow matching N bước) | không | one-backend |
| `asr_nonstreaming` | không | có | có (encoder) | one-backend — chết ở cổng 5 |

## Làm lại `asr_nonstreaming` thì làm sao

Một Python backend giống `asr_streaming`: bật `dynamic_batching`, `execute()` gom request
theo bucket độ dài rồi **đệm về max trong bucket** thay vì đệm cứng 16s, dùng lại
`greedy_search_step` gọi một lượt trên cả câu.

`dynamic_batching` chạy cho cả backend `python`, chỉ khác là nó đưa vào `execute()` dạng
**list request** và mình phải tự stack tensor
([Python Backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html)).
Nghĩa là one-backend **vẫn batch được**, không hề đánh đổi batching lấy sự đơn giản.

Mất per-model metrics ở `:8002/metrics` — bù bằng log thời gian 3 chặng fbank/encoder/greedy
qua `pb_utils.Logger`. Đây là mất mát thật duy nhất, vì chính metrics per-model là cách phát
hiện ra `asr_scorer` mới là nút cổ chai.
