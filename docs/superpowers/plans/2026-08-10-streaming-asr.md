# Streaming ASR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm model `asr_streaming` — nhận audio theo chunk qua gRPC stream, trả partial transcript mỗi chunk — chạy song song với ensemble `asr` hiện có.

**Architecture:** Một Python backend duy nhất dùng sequence batcher (`oldest`), giữ toàn bộ state per-stream (fbank tính dần, cache encoder streaming, hypothesis greedy) trong process, khoá theo CORRID. Encoder/decoder/joiner là ONNX fp16 tải sẵn từ `hynt/Zipformer-30M-RNNT-Streaming-6000h`, biến thể `chunk-16-left-128`. Spec: `docs/superpowers/specs/2026-08-10-streaming-asr-design.md`.

**Tech Stack:** Triton Python backend, onnxruntime (CUDA), torchaudio kaldi fbank, sentencepiece, tritonclient gRPC streaming.

## Global Constraints

- Identifier tiếng Anh; comment/docstring tiếng Việt; mọi file code mở đầu 2 dòng `# ABOUTME:` (spec §14)
- KHÔNG dependency mới — image đã có torch, torchaudio, onnxruntime-gpu, sentencepiece; venv đã có tritonclient, pytest
- KHÔNG commit `*.onnx`, `bpe.model` tải về (`.gitignore` đã chặn)
- KHÔNG sửa model `asr`, `tts`, bench hiện có — chỉ đụng file liệt kê trong từng task
- Test integration đánh dấu `@pytest.mark.integration`; `pytest -m "not integration"` phải chạy được không cần server
- Server chạy qua `scripts/serve.sh` (docker, container tên `triton-voice-server`); health: `curl -sf localhost:8000/v2/health/ready`
- Commit message tiếng Anh, KHÔNG kèm bất kỳ trailer `Co-Authored-By` nào

**Lệnh chạy chuẩn:** `PY=.venv/bin/python`, `PYTEST=.venv/bin/pytest` từ gốc repo.

**Restart server (dùng ở Task 4, 5):**

```bash
docker rm -f triton-voice-server 2>/dev/null || true
nohup bash scripts/serve.sh > /tmp/triton-serve.log 2>&1 &
for i in $(seq 90); do curl -sf localhost:8000/v2/health/ready >/dev/null && break; sleep 2; done
curl -sf localhost:8000/v2/models/asr_streaming/ready && echo "asr_streaming READY" \
  || { echo "asr_streaming KHÔNG load — xem /tmp/triton-serve.log"; tail -50 /tmp/triton-serve.log; }
```

---

### Task 1: Tải model streaming + kiểm metadata ONNX

**Files:**
- Modify: `scripts/fetch_models.sh`
- Modify: `scripts/inspect_onnx.py`
- Create: `docs/superpowers/plans/2026-08-10-streaming-asr-notes.md`

**Interfaces:**
- Produces: `model_repository/asr_streaming/1/{encoder,decoder,joiner}.onnx`, `bpe.model` trên đĩa; file notes ghi metadata thật (`decode_chunk_len`, `T`, số state tensor, dtype) mà Task 5 dựa vào để xác nhận giả định của `model.py`.

- [ ] **Step 1: Thêm section streaming vào `fetch_models.sh`**

Chèn sau block `ASR=` (sau dòng `dl "$ASR/bpe.model" ...`):

```bash
# ASR streaming - repo đã export sẵn ONNX fp16 theo từng biến thể chunk.
# CHUNK_VARIANT đổi được: 16 (latency thấp nhất), 32, 64. Xem spec 2026-08-10 mục 4.
CHUNK_VARIANT="${CHUNK_VARIANT:-16}"
STREAM=https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h/resolve/main
SFX="epoch-31-avg-11-chunk-${CHUNK_VARIANT}-left-128.fp16.onnx"
dl "$STREAM/encoder-$SFX" "$REPO/asr_streaming/1/encoder.onnx"
dl "$STREAM/decoder-$SFX" "$REPO/asr_streaming/1/decoder.onnx"
dl "$STREAM/joiner-$SFX"  "$REPO/asr_streaming/1/joiner.onnx"
dl "$STREAM/bpe.model"    "$REPO/asr_streaming/1/bpe.model"
```

- [ ] **Step 2: Cho `inspect_onnx.py` in cả metadata**

Thêm vào cuối vòng `for path in sys.argv[1:]:` (sau vòng in OUT):

```python
    if m.metadata_props:
        print("  META")
        for p in m.metadata_props:
            print(f"       {p.key} = {p.value}")
```

- [ ] **Step 3: Chạy fetch rồi inspect**

```bash
bash scripts/fetch_models.sh
$PY scripts/inspect_onnx.py model_repository/asr_streaming/1/encoder.onnx \
    model_repository/asr_streaming/1/decoder.onnx \
    model_repository/asr_streaming/1/joiner.onnx
```

- [ ] **Step 4: Kiểm 4 cổng chặn — sai cái nào DỪNG LẠI báo ngay, không đi tiếp**

1. Encoder META có khoá `decode_chunk_len` (model.py đọc khoá này lúc initialize)
2. Input `x` của encoder có chiều thứ 2 (T) là **số tĩnh**, không phải `?`
3. Số input của encoder trừ `x` == số output trừ `encoder_out` (map state theo vị trí)
4. Decoder META có `context_size = 2`

- [ ] **Step 5: Ghi output inspect vào notes**

Tạo `docs/superpowers/plans/2026-08-10-streaming-asr-notes.md`, dán nguyên văn output Step 3 vào trong một code block, kèm 3 dòng tóm tắt: giá trị `decode_chunk_len`, giá trị `T`, số lượng state tensor.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_models.sh scripts/inspect_onnx.py docs/superpowers/plans/2026-08-10-streaming-asr-notes.md
git commit -m "Fetch streaming ASR ONNX models and record their metadata"
```

---

### Task 2: `StreamingFbank` — fbank tính dần, khớp offline từng số (TDD)

**Files:**
- Create: `model_repository/asr_streaming/1/streaming_search.py`
- Test: `tests/test_streaming_fbank.py`

**Interfaces:**
- Produces:
  - `offline_fbank(samples: np.ndarray) -> np.ndarray` — fbank cả câu `(T, 80)`, đúng tham số `asr_feature` dùng (num_mel_bins 80, dither 0, `snip_edges=False`)
  - `class StreamingFbank` với `accept_waveform(samples: np.ndarray) -> np.ndarray` (trả các khung mới chắc chắn, `(k, 80)` float32) và `flush() -> np.ndarray` (phát nốt khung đuôi khi hết audio)
  - Hằng `NUM_MEL_BINS = 80`

**Cốt lõi thuật toán** (vì sao khớp được `snip_edges=False` theo chunk): với `snip_edges=False`, khung `j` có tâm tại mẫu `j*160 + 80`, cửa sổ `[tâm-200, tâm+200)`. Mỗi lần vẫn gọi `kaldi.fbank` trên buffer, nhưng **chỉ phát khung có cửa sổ nằm trọn trong mẫu thật** — khung dính reflection ở mép buffer thì chờ. Buffer luôn giữ lại 1 khung ngữ cảnh trái và cắt theo bội 160 để lưới khung cục bộ trùng lưới toàn cục. Reflection thật chỉ xảy ra ở đầu stream (`_buf_start == 0`) và ở `flush()` — đúng hai chỗ offline cũng reflect, nên kết quả trùng nhau tuyệt đối.

- [ ] **Step 1: Viết test đỏ**

Tạo `tests/test_streaming_fbank.py`:

```python
# ABOUTME: Unit test cho StreamingFbank - so khớp từng số với kaldi.fbank offline
# ABOUTME: Không cần GPU/server; lệch khung ở mép chunk là transcript hỏng âm thầm nên phải khớp tuyệt đối

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "model_repository/asr_streaming/1"))
from streaming_search import StreamingFbank, offline_fbank  # noqa: E402


def _random_audio(seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(16000 * seconds)) * 0.1).astype(np.float32)


def _run_chunked(wav, chunk_sizes):
    """Đẩy wav qua StreamingFbank theo các cỡ chunk xoay vòng, gom hết khung phát ra."""
    fb = StreamingFbank()
    outs = []
    pos = 0
    i = 0
    while pos < len(wav):
        n = chunk_sizes[i % len(chunk_sizes)]
        outs.append(fb.accept_waveform(wav[pos : pos + n]))
        pos += n
        i += 1
    outs.append(fb.flush())
    return np.concatenate(outs)


def test_uniform_chunks_match_offline():
    wav = _random_audio()
    got = _run_chunked(wav, [3200])
    want = offline_fbank(wav)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


def test_ragged_chunks_match_offline():
    """Cỡ chunk cố tình lệch bội số khung để ép mọi nhánh bookkeeping."""
    wav = _random_audio(seed=1)
    got = _run_chunked(wav, [123, 7, 4000, 160, 1601])
    want = offline_fbank(wav)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


def test_tiny_first_chunk_emits_nothing():
    fb = StreamingFbank()
    out = fb.accept_waveform(np.zeros(100, dtype=np.float32))
    assert out.shape == (0, 80)


def test_flush_completes_frame_count():
    wav = _random_audio(seconds=0.5, seed=2)
    got = _run_chunked(wav, [1000])
    assert got.shape[0] == offline_fbank(wav).shape[0]
```

- [ ] **Step 2: Chạy xác nhận đỏ**

Run: `$PYTEST tests/test_streaming_fbank.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'streaming_search'`

- [ ] **Step 3: Viết `streaming_search.py` (phần fbank)**

Tạo `model_repository/asr_streaming/1/streaming_search.py`:

```python
# ABOUTME: Logic thuần cho streaming ASR - fbank tính dần và greedy search theo chunk
# ABOUTME: Không import Triton/ONNX; decoder/joiner truyền vào dạng hàm như greedy_search.py của asr_scorer

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
FRAME_SHIFT = 160        # 10ms
FRAME_LENGTH = 400       # 25ms
# snip_edges=False (khớp asr_feature): khung j có tâm tại j*160+80, cửa sổ [tâm-200, tâm+200)
_CENTER = FRAME_SHIFT // 2        # 80
_HALF_WINDOW = FRAME_LENGTH // 2  # 200


def offline_fbank(samples: np.ndarray) -> np.ndarray:
    """Fbank cả câu, đúng tham số asr_feature dùng - chuẩn đối chiếu cho bản streaming."""
    wav = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
    feat = kaldi.fbank(
        wav,
        num_mel_bins=NUM_MEL_BINS,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        sample_frequency=SAMPLE_RATE,
        snip_edges=False,
    )
    return feat.numpy()


class StreamingFbank:
    """Fbank tính dần theo chunk, kết quả khớp offline_fbank trên cùng audio.

    Mẹo: mỗi lần vẫn gọi kaldi.fbank trên buffer, nhưng chỉ phát những khung mà
    cửa sổ nằm trọn trong phần mẫu thật (không dính reflection ở mép buffer).
    Buffer giữ lại 1 khung ngữ cảnh trái, cắt theo bội FRAME_SHIFT để lưới khung
    cục bộ trùng lưới toàn cục. Reflection thật chỉ còn ở đầu stream và ở flush -
    đúng hai chỗ offline cũng reflect, nên kết quả trùng nhau.
    """

    def __init__(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0    # vị trí tuyệt đối của _buf[0], luôn là bội của FRAME_SHIFT
        self._next_frame = 0   # chỉ số khung toàn cục kế tiếp chưa phát

    def _emit_until(self, last_frame: int) -> np.ndarray:
        """Phát các khung [_next_frame, last_frame] rồi cắt bớt buffer bên trái."""
        if last_frame < self._next_frame:
            return np.zeros((0, NUM_MEL_BINS), dtype=np.float32)
        feat = offline_fbank(self._buf)
        base = self._buf_start // FRAME_SHIFT
        out = feat[self._next_frame - base : last_frame - base + 1].copy()
        self._next_frame = last_frame + 1
        keep_from = max(0, (self._next_frame - 1) * FRAME_SHIFT)
        self._buf = self._buf[keep_from - self._buf_start :]
        self._buf_start = keep_from
        return out

    def accept_waveform(self, samples: np.ndarray) -> np.ndarray:
        """Nạp thêm mẫu, trả về các khung mới đã chắc chắn (không đổi về sau)."""
        if len(samples):
            self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32)])
        abs_end = self._buf_start + len(self._buf)
        # khung j chắc chắn khi cửa sổ của nó kết thúc trước mẫu cuối đang có
        return self._emit_until((abs_end - _CENTER - _HALF_WINDOW) // FRAME_SHIFT)

    def flush(self) -> np.ndarray:
        """Hết audio - phát nốt khung đuôi, dùng reflection ở đuôi đúng như offline."""
        if len(self._buf) <= _HALF_WINDOW:
            # ngắn hơn nửa cửa sổ thì reflection không hợp lệ; bỏ qua phần đuôi vụn này
            return np.zeros((0, NUM_MEL_BINS), dtype=np.float32)
        abs_end = self._buf_start + len(self._buf)
        total = (abs_end + _CENTER) // FRAME_SHIFT   # công thức số khung của snip_edges=False
        return self._emit_until(total - 1)
```

- [ ] **Step 4: Chạy xác nhận xanh**

Run: `$PYTEST tests/test_streaming_fbank.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add model_repository/asr_streaming/1/streaming_search.py tests/test_streaming_fbank.py
git commit -m "Add StreamingFbank matching offline kaldi fbank chunk-exactly"
```

---

### Task 3: Greedy search theo chunk (TDD)

**Files:**
- Modify: `model_repository/asr_streaming/1/streaming_search.py` (thêm vào cuối file)
- Test: `tests/test_streaming_search.py`

**Interfaces:**
- Consumes: `greedy_search(encoder_out, num_frames, run_decoder, run_joiner, blank_id, context_size)` từ `model_repository/asr_scorer/1/greedy_search.py` (chỉ trong test, làm chuẩn đối chiếu)
- Produces:
  - `@dataclass SearchState` — `hyp: List[int]`, `decoder_out: np.ndarray`
  - `init_search_state(run_decoder, blank_id=0, context_size=2) -> SearchState`
  - `greedy_search_step(encoder_out, state, run_decoder, run_joiner, blank_id=0, context_size=2) -> SearchState` — đi tiếp trên một đoạn `(T, C)`, mutate và trả `state`
  - `emitted_tokens(state, context_size=2) -> List[int]`

- [ ] **Step 1: Viết test đỏ**

Tạo `tests/test_streaming_search.py`:

```python
# ABOUTME: Unit test cho greedy search theo chunk - đối chiếu với greedy_search cả câu của asr_scorer
# ABOUTME: decoder/joiner giả cùng kiểu test_greedy_search.py, không cần GPU/ONNX

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "model_repository/asr_scorer/1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "model_repository/asr_streaming/1"))
from greedy_search import greedy_search  # noqa: E402
from streaming_search import emitted_tokens, greedy_search_step, init_search_state  # noqa: E402

DIM = 4
VOCAB_SIZE = 10


def fake_decoder(_context):
    """Decoder giả - luôn trả vector 0, đủ dùng vì joiner giả không nhìn tới nó."""
    return np.zeros((1, DIM), dtype=np.float32)


def fake_joiner(emitted):
    """Joiner giả phát token theo kịch bản, mỗi lần gọi lấy 1 phần tử - giữ state qua các chunk."""
    state = {"step": 0}

    def _joiner(_enc, _dec):
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        logits[0, emitted[state["step"]]] = 1.0
        state["step"] += 1
        return logits

    return _joiner


def test_chunked_matches_full_utterance():
    """Bất biến quan trọng nhất: cắt encoder_out kiểu gì thì kết quả cũng như chạy một lần."""
    script = [0, 7, 0, 3, 0, 0, 9, 0, 2, 0]
    enc = np.zeros((10, DIM), dtype=np.float32)
    want = greedy_search(enc, 10, fake_decoder, fake_joiner(script))

    state = init_search_state(fake_decoder)
    joiner = fake_joiner(script)
    for part in np.split(enc, [3, 4, 8]):   # các chunk 3, 1, 4, 2 khung
        greedy_search_step(part, state, fake_decoder, joiner)
    assert emitted_tokens(state) == want


def test_empty_chunk_changes_nothing():
    state = init_search_state(fake_decoder)
    greedy_search_step(np.zeros((0, DIM), dtype=np.float32), state, fake_decoder, fake_joiner([]))
    assert emitted_tokens(state) == []


def test_context_carries_across_chunks():
    seen = []

    def recording_decoder(context):
        seen.append(list(context))
        return np.zeros((1, DIM), dtype=np.float32)

    enc = np.zeros((4, DIM), dtype=np.float32)
    state = init_search_state(recording_decoder)
    joiner = fake_joiner([2, 0, 6, 0])
    greedy_search_step(enc[:2], state, recording_decoder, joiner)
    greedy_search_step(enc[2:], state, recording_decoder, joiner)

    # khởi tạo [0,0]; sau token 2 → [0,2]; sau token 6 (chunk hai) → [2,6]
    assert seen == [[0, 0], [0, 2], [2, 6]]
```

- [ ] **Step 2: Chạy xác nhận đỏ**

Run: `$PYTEST tests/test_streaming_search.py -v`
Expected: FAIL — `ImportError: cannot import name 'emitted_tokens'`

- [ ] **Step 3: Thêm phần search vào `streaming_search.py`**

Thêm vào đầu file (mở rộng import hiện có):

```python
from dataclasses import dataclass
from typing import Callable, List
```

Thêm vào cuối file:

```python
@dataclass
class SearchState:
    """Trạng thái greedy search sống qua các chunk của một stream."""

    hyp: List[int]
    decoder_out: np.ndarray


def init_search_state(
    run_decoder: Callable[[List[int]], np.ndarray],
    blank_id: int = 0,
    context_size: int = 2,
) -> SearchState:
    """Ngữ cảnh khởi tạo toàn blank, chạy decoder một lần - như greedy_search làm ở đầu câu."""
    hyp = [blank_id] * context_size
    return SearchState(hyp=hyp, decoder_out=run_decoder(hyp[-context_size:]))


def greedy_search_step(
    encoder_out: np.ndarray,
    state: SearchState,
    run_decoder: Callable[[List[int]], np.ndarray],
    run_joiner: Callable[[np.ndarray, np.ndarray], np.ndarray],
    blank_id: int = 0,
    context_size: int = 2,
) -> SearchState:
    """Đi tiếp vòng greedy trên một đoạn encoder_out (T, C).

    Cùng logic với greedy_search của asr_scorer, chỉ khác: trạng thái nhận vào
    và trả ra thay vì khởi tạo - kết thúc trong một lần gọi.
    """
    for t in range(encoder_out.shape[0]):
        logits = run_joiner(encoder_out[t : t + 1], state.decoder_out)
        token = int(np.argmax(logits.reshape(-1)))
        if token != blank_id:
            state.hyp.append(token)
            state.decoder_out = run_decoder(state.hyp[-context_size:])
    return state


def emitted_tokens(state: SearchState, context_size: int = 2) -> List[int]:
    """Token đã phát, bỏ phần ngữ cảnh blank khởi tạo."""
    return state.hyp[context_size:]
```

- [ ] **Step 4: Chạy xác nhận xanh + toàn bộ unit không vỡ**

Run: `$PYTEST tests/ -m "not integration" -v`
Expected: tất cả pass (gồm 4 test fbank, 3 test search mới, các test cũ)

- [ ] **Step 5: Commit**

```bash
git add model_repository/asr_streaming/1/streaming_search.py tests/test_streaming_search.py
git commit -m "Add chunk-wise greedy search sharing state across calls"
```

---

### Task 4: `config.pbtxt` + skeleton `model.py` + smoke 2 stream đan xen

Spec §13 yêu cầu smoke sequence batcher **trước khi** viết logic nhận dạng đầy đủ — đây chính là task đó. Skeleton chỉ đếm mẫu per-stream và trả `"corrid:tổng_số_mẫu"` để chứng minh state không lẫn giữa các stream.

**Files:**
- Create: `model_repository/asr_streaming/config.pbtxt`
- Create: `model_repository/asr_streaming/1/model.py` (skeleton — Task 5 thay ruột, giữ khung)

**Interfaces:**
- Produces: model `asr_streaming` load được trên Triton với sequence batching; khung `model.py` (dict `self.streams`, `_sweep`, `_flag`, vòng `execute`) mà Task 5 giữ nguyên và chỉ thay phần xử lý audio.

- [ ] **Step 1: Viết `config.pbtxt`**

Tạo `model_repository/asr_streaming/config.pbtxt`:

```
name: "asr_streaming"
backend: "python"
max_batch_size: 8

input [
  {
    name: "AUDIO_CHUNK"
    data_type: TYPE_FP32
    dims: [-1]
  }
]

output [
  {
    name: "TRANSCRIPT"
    data_type: TYPE_STRING
    dims: [1]
  }
]

sequence_batching {
  max_sequence_idle_microseconds: 60000000
  oldest {
    max_candidate_sequences: 8
  }
  control_input [
    {
      name: "START"
      control [ { kind: CONTROL_SEQUENCE_START fp32_false_true: [0, 1] } ]
    },
    {
      name: "END"
      control [ { kind: CONTROL_SEQUENCE_END fp32_false_true: [0, 1] } ]
    },
    {
      name: "READY"
      control [ { kind: CONTROL_SEQUENCE_READY fp32_false_true: [0, 1] } ]
    },
    {
      name: "CORRID"
      control [ { kind: CONTROL_SEQUENCE_CORRID data_type: TYPE_UINT64 } ]
    }
  ]
}

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
```

- [ ] **Step 2: Viết skeleton `model.py`**

Tạo `model_repository/asr_streaming/1/model.py`:

```python
# ABOUTME: Triton Python backend - streaming ASR, nhận audio theo chunk qua sequence batcher
# ABOUTME: Skeleton: mới quản state per-stream, chưa nhận dạng - xem plan Task 5

import time

import numpy as np
import triton_python_backend_utils as pb_utils

STATE_TTL_S = 60.0   # soi gương max_sequence_idle_microseconds trong config.pbtxt


class TritonPythonModel:
    def initialize(self, args):
        self.streams = {}   # corrid -> state của stream đang sống

    def _new_stream(self):
        return {"samples": 0, "last_seen": time.monotonic()}

    def _sweep(self):
        """Xoá state của stream chết không gửi END - nếu không dict rò rỉ vĩnh viễn."""
        now = time.monotonic()
        for k in [k for k, s in self.streams.items() if now - s["last_seen"] > STATE_TTL_S]:
            pb_utils.Logger.log_warn(f"asr_streaming: xoá state mồ côi corrid={k}")
            del self.streams[k]

    @staticmethod
    def _flag(request, name):
        t = pb_utils.get_input_tensor_by_name(request, name)
        return t is not None and bool(t.as_numpy().reshape(-1)[0])

    def execute(self, requests):
        responses = []
        self._sweep()
        for request in requests:
            corrid = int(
                pb_utils.get_input_tensor_by_name(request, "CORRID").as_numpy().reshape(-1)[0]
            )
            start = self._flag(request, "START")
            end = self._flag(request, "END")

            if start or corrid not in self.streams:
                if not start:
                    # server restart giữa stream chẳng hạn - khởi tạo lại thay vì crash
                    pb_utils.Logger.log_warn(
                        f"asr_streaming: chunk không có state (corrid={corrid}), khởi tạo lại"
                    )
                self.streams[corrid] = self._new_stream()
            stream = self.streams[corrid]
            stream["last_seen"] = time.monotonic()

            audio = (
                pb_utils.get_input_tensor_by_name(request, "AUDIO_CHUNK")
                .as_numpy()
                .reshape(-1)
            )
            stream["samples"] += len(audio)

            # Skeleton trả "corrid:tổng_mẫu" để smoke test kiểm state không lẫn giữa stream
            text = f"{corrid}:{stream['samples']}"
            if end:
                del self.streams[corrid]

            out = np.array([[text.encode("utf-8")]], dtype=object)
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("TRANSCRIPT", out)])
            )
        return responses
```

- [ ] **Step 3: Restart server, kiểm model load**

Chạy block "Restart server" ở Global Constraints. Expected: `asr_streaming READY`.

- [ ] **Step 4: Smoke — 2 stream đan xen trên một kênh gRPC**

```bash
$PY - <<'EOF'
import queue
import numpy as np
import tritonclient.grpc as grpcclient

q = queue.Queue()
c = grpcclient.InferenceServerClient("localhost:8001")
c.start_stream(callback=lambda r, e: q.put((r, e)))

def send(sid, n, start, end):
    x = np.zeros((1, n), dtype=np.float32)
    i = grpcclient.InferInput("AUDIO_CHUNK", [1, n], "FP32")
    i.set_data_from_numpy(x)
    c.async_stream_infer("asr_streaming", [i],
                         sequence_id=sid, sequence_start=start, sequence_end=end)

send(11, 100, True, False); send(22, 5, True, False)
send(11, 100, False, False); send(22, 5, False, False)
send(11, 100, False, True);  send(22, 5, False, True)

texts = []
for _ in range(6):
    r, e = q.get(timeout=15)
    assert e is None, e
    texts.append(r.as_numpy("TRANSCRIPT")[0, 0].decode())
c.stop_stream()
print(texts)
assert "11:300" in texts and "22:15" in texts, texts
print("SMOKE OK - state không lẫn giữa 2 stream")
EOF
```

Expected: in danh sách 6 text rồi `SMOKE OK`. Nếu sai — sửa config/model cho đến khi qua, KHÔNG đi tiếp Task 5.

- [ ] **Step 5: Commit**

```bash
git add model_repository/asr_streaming/config.pbtxt model_repository/asr_streaming/1/model.py
git commit -m "Add asr_streaming sequence-batching skeleton with per-stream state"
```

---

### Task 5: Logic nhận dạng đầy đủ + integration test (TDD)

**Files:**
- Test: `tests/test_asr_streaming.py` (viết TRƯỚC)
- Modify: `model_repository/asr_streaming/1/model.py` (thay ruột skeleton, giữ khung `execute`/`_sweep`/`_flag`)

**Interfaces:**
- Consumes: `StreamingFbank`, `offline_fbank` params, `SearchState`, `init_search_state`, `greedy_search_step`, `emitted_tokens`, `NUM_MEL_BINS` từ `streaming_search.py`; metadata đã kiểm ở Task 1 (`decode_chunk_len`, `T`, state map theo vị trí).
- Produces: model `asr_streaming` trả transcript thật — hợp đồng cuối cùng cho client Task 6.

**Ghi chú so với spec §9:** transcript kỳ vọng dùng lại `tests/assets/sample_vi.txt` với ngưỡng word-overlap ≥ 0.6 (đúng kiểu `test_asr.py`) thay vì ghi file kỳ vọng riêng — cùng câu nói, cùng mục đích, khỏi thêm asset trùng lặp.

- [ ] **Step 1: Viết integration test**

Tạo `tests/test_asr_streaming.py`:

```python
# ABOUTME: Integration test cho asr_streaming - gửi wav theo chunk qua gRPC stream
# ABOUTME: Kiểm partial dài dần, final khớp nội dung, 2 stream đan xen không lẫn state

import queue
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.common import SAMPLE_RATE  # noqa: E402

pytestmark = pytest.mark.integration

ASSETS = Path(__file__).parent / "assets"
URL = "localhost:8001"
CHUNK_SAMPLES = 3200   # 200ms @ 16kHz


def _chunks(wav):
    return [wav[i : i + CHUNK_SAMPLES] for i in range(0, len(wav), CHUNK_SAMPLES)]


def _load_sample():
    wav, sample_rate = sf.read(ASSETS / "sample_vi.wav", dtype="float32")
    assert sample_rate == SAMPLE_RATE
    return wav


def _send_chunks(client, grpcclient, parts, seq_id, chunk_index):
    part = parts[chunk_index]
    inp = grpcclient.InferInput("AUDIO_CHUNK", [1, len(part)], "FP32")
    inp.set_data_from_numpy(part.reshape(1, -1))
    client.async_stream_infer(
        "asr_streaming",
        [inp],
        sequence_id=seq_id,
        sequence_start=(chunk_index == 0),
        sequence_end=(chunk_index == len(parts) - 1),
    )


def _stream_transcripts(wav, seq_id):
    """Gửi cả wav qua một stream, trả danh sách transcript theo thứ tự chunk."""
    import tritonclient.grpc as grpcclient

    client = grpcclient.InferenceServerClient(URL)
    q = queue.Queue()
    client.start_stream(callback=lambda result, error: q.put((result, error)))
    parts = _chunks(wav)
    for i in range(len(parts)):
        _send_chunks(client, grpcclient, parts, seq_id, i)
    texts = []
    for _ in parts:
        result, error = q.get(timeout=60)
        assert error is None, error
        texts.append(result.as_numpy("TRANSCRIPT")[0, 0].decode("utf-8"))
    client.stop_stream()
    return texts


def test_partials_grow_and_final_matches_reference(triton):
    wav = _load_sample()
    expected = (ASSETS / "sample_vi.txt").read_text(encoding="utf-8").strip().lower()
    assert expected, "sample_vi.txt rỗng - phải chứa câu được nói trong sample_vi.wav"

    texts = _stream_transcripts(wav, seq_id=101)

    assert len(texts) == len(_chunks(wav))
    lengths = [len(t) for t in texts]
    assert lengths == sorted(lengths), f"partial phải dài dần: {lengths}"

    final = texts[-1].strip().lower()
    assert final, "final transcript rỗng"
    expected_words = set(expected.split())
    overlap = len(expected_words & set(final.split())) / len(expected_words)
    assert overlap >= 0.6, f"mong đợi '{expected}', nhận '{final}'"


def test_two_interleaved_streams_share_no_state(triton):
    """Cùng audio trên 2 stream đan xen từng chunk - greedy tất định nên final phải trùng nhau."""
    import tritonclient.grpc as grpcclient

    wav = _load_sample()
    parts = _chunks(wav)
    clients, queues = {}, {}
    for sid in (201, 202):
        c = grpcclient.InferenceServerClient(URL)
        q = queue.Queue()
        c.start_stream(callback=lambda result, error, q=q: q.put((result, error)))
        clients[sid], queues[sid] = c, q

    for i in range(len(parts)):
        for sid in (201, 202):
            _send_chunks(clients[sid], grpcclient, parts, sid, i)

    finals = {}
    for sid in (201, 202):
        texts = []
        for _ in parts:
            result, error = queues[sid].get(timeout=120)
            assert error is None, error
            texts.append(result.as_numpy("TRANSCRIPT")[0, 0].decode("utf-8"))
        finals[sid] = texts[-1]
        clients[sid].stop_stream()

    assert finals[201] == finals[202]
    assert finals[201].strip()
```

- [ ] **Step 2: Chạy xác nhận đỏ đúng kiểu**

Run: `$PYTEST tests/test_asr_streaming.py -v` (server skeleton Task 4 vẫn đang chạy)
Expected: FAIL — skeleton trả `"101:256000"` dạng đếm mẫu nên overlap < 0.6. (Nếu server không chạy: SKIP — khởi động lại server rồi chạy lại cho ra FAIL thật.)

- [ ] **Step 3: Thay ruột `model.py` bằng logic thật**

Ghi đè `model_repository/asr_streaming/1/model.py` (khung `execute`/`_sweep`/`_flag` giữ nguyên từ skeleton, phần state và xử lý audio là mới):

```python
# ABOUTME: Triton Python backend - streaming ASR: fbank dần + encoder streaming + greedy theo chunk
# ABOUTME: State per-stream khoá theo CORRID; logic thuần nằm ở streaming_search.py

import math
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import triton_python_backend_utils as pb_utils

# Triton không tự thêm thư mục model vào sys.path nên phải tự làm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streaming_search import (  # noqa: E402
    NUM_MEL_BINS,
    StreamingFbank,
    emitted_tokens,
    greedy_search_step,
    init_search_state,
)

BLANK_ID = 0
CONTEXT_SIZE = 2
LOG_EPS = math.log(1e-10)   # giá trị đệm khung cuối, theo quy ước icefall/sherpa
STATE_TTL_S = 60.0          # soi gương max_sequence_idle_microseconds trong config.pbtxt

_ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


class _Stream:
    """State của một stream đang sống - mọi thứ phải nhớ giữa hai chunk."""

    def __init__(self, model):
        self.fbank = StreamingFbank()
        self.feat = np.zeros((0, NUM_MEL_BINS), dtype=np.float32)
        self.enc_states = [
            np.zeros(shape, dtype=dtype) for shape, dtype in model.init_state_specs
        ]
        self.search = init_search_state(model.run_decoder, BLANK_ID, CONTEXT_SIZE)
        self.last_seen = time.monotonic()


class TritonPythonModel:
    def initialize(self, args):
        d = os.path.join(args["model_repository"], args["model_version"])
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.encoder = ort.InferenceSession(os.path.join(d, "encoder.onnx"), providers=providers)
        self.decoder = ort.InferenceSession(os.path.join(d, "decoder.onnx"), providers=providers)
        self.joiner = ort.InferenceSession(os.path.join(d, "joiner.onnx"), providers=providers)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(d, "bpe.model"))

        # Nhịp tiêu thụ khung nằm trong metadata của chính file ONNX (kiểm ở plan Task 1)
        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.decode_chunk_len = int(meta["decode_chunk_len"])

        enc_inputs = self.encoder.get_inputs()
        self.x_name = enc_inputs[0].name
        self.x_dtype = _ORT_TO_NP[enc_inputs[0].type]
        self.T = int(enc_inputs[0].shape[1])   # khung đưa vào mỗi bước; T - decode_chunk_len là lookahead

        # State của encoder: mọi input trừ x, khớp THEO VỊ TRÍ với mọi output trừ encoder_out.
        # Không viết cứng tên/số lượng - export đổi layer thì code vẫn đúng.
        self.state_in_names = [i.name for i in enc_inputs[1:]]
        self.state_out_count = len(self.encoder.get_outputs()) - 1
        assert len(self.state_in_names) == self.state_out_count
        self.init_state_specs = [
            (tuple(s if isinstance(s, int) else 1 for s in i.shape), _ORT_TO_NP[i.type])
            for i in enc_inputs[1:]
        ]

        self.decoder_in = self.decoder.get_inputs()[0].name
        self.joiner_in = [i.name for i in self.joiner.get_inputs()]
        self.joiner_dtype = _ORT_TO_NP[self.joiner.get_inputs()[0].type]

        self.streams = {}   # corrid -> _Stream

    def run_decoder(self, context):
        y = np.array([context], dtype=np.int64)
        out = self.decoder.run(None, {self.decoder_in: y})[0]
        # Một số bản export trả (N, 1, C), bỏ chiều giữa cho khớp joiner
        out = out[:, 0, :] if out.ndim == 3 else out
        return out.astype(np.float32)

    def run_joiner(self, enc_frame, dec_out):
        feeds = {
            self.joiner_in[0]: enc_frame.astype(self.joiner_dtype),
            self.joiner_in[1]: dec_out.astype(self.joiner_dtype),
        }
        return self.joiner.run(None, feeds)[0].astype(np.float32)

    def _encoder_step(self, stream, feat_chunk):
        """Một bước encoder streaming: (T, 80) + cache cũ -> (T', C) + cache mới."""
        feeds = {self.x_name: feat_chunk[None].astype(self.x_dtype)}
        feeds.update(zip(self.state_in_names, stream.enc_states))
        outs = self.encoder.run(None, feeds)
        stream.enc_states = outs[1:]
        return outs[0][0].astype(np.float32)

    def _advance(self, stream, new_feat, flush):
        """Nạp khung mới, chạy encoder đủ số bước, đi tiếp greedy search."""
        if len(new_feat):
            stream.feat = np.concatenate([stream.feat, new_feat])
        while stream.feat.shape[0] >= self.T:
            enc_out = self._encoder_step(stream, stream.feat[: self.T])
            stream.feat = stream.feat[self.decode_chunk_len :]
            greedy_search_step(
                enc_out, stream.search, self.run_decoder, self.run_joiner, BLANK_ID, CONTEXT_SIZE
            )
        if flush and stream.feat.shape[0] > 0:
            # khung cuối không đủ T - đệm LOG_EPS cho đủ một bước encoder chót
            pad = np.full(
                (self.T - stream.feat.shape[0], NUM_MEL_BINS), LOG_EPS, dtype=np.float32
            )
            enc_out = self._encoder_step(stream, np.concatenate([stream.feat, pad]))
            stream.feat = stream.feat[:0]
            greedy_search_step(
                enc_out, stream.search, self.run_decoder, self.run_joiner, BLANK_ID, CONTEXT_SIZE
            )

    def _sweep(self):
        """Xoá state của stream chết không gửi END - nếu không dict rò rỉ vĩnh viễn."""
        now = time.monotonic()
        for k in [k for k, s in self.streams.items() if now - s.last_seen > STATE_TTL_S]:
            pb_utils.Logger.log_warn(f"asr_streaming: xoá state mồ côi corrid={k}")
            del self.streams[k]

    @staticmethod
    def _flag(request, name):
        t = pb_utils.get_input_tensor_by_name(request, name)
        return t is not None and bool(t.as_numpy().reshape(-1)[0])

    def _handle(self, request):
        """Xử lý trọn một chunk của một stream, trả InferenceResponse."""
        corrid = int(
            pb_utils.get_input_tensor_by_name(request, "CORRID").as_numpy().reshape(-1)[0]
        )
        start = self._flag(request, "START")
        end = self._flag(request, "END")

        if start or corrid not in self.streams:
            if not start:
                # server restart giữa stream chẳng hạn - khởi tạo lại thay vì crash
                pb_utils.Logger.log_warn(
                    f"asr_streaming: chunk không có state (corrid={corrid}), khởi tạo lại"
                )
            self.streams[corrid] = _Stream(self)
        stream = self.streams[corrid]
        stream.last_seen = time.monotonic()

        audio = (
            pb_utils.get_input_tensor_by_name(request, "AUDIO_CHUNK")
            .as_numpy()
            .reshape(-1)
            .astype(np.float32)
        )
        new_feat = stream.fbank.accept_waveform(audio)
        if end:
            tail = stream.fbank.flush()
            if len(tail):
                new_feat = np.concatenate([new_feat, tail]) if len(new_feat) else tail
        self._advance(stream, new_feat, flush=end)

        text = self.sp.decode(emitted_tokens(stream.search, CONTEXT_SIZE))
        if end:
            del self.streams[corrid]

        out = np.array([[text.encode("utf-8")]], dtype=object)
        return pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("TRANSCRIPT", out)])

    def execute(self, requests):
        responses = []
        self._sweep()
        for request in requests:
            try:
                responses.append(self._handle(request))
            except Exception as e:
                # lỗi của một sequence không được lây sang các stream khác trong cùng batch (spec §10)
                pb_utils.Logger.log_error(f"asr_streaming: corrid lỗi: {e}")
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))
                )
        return responses
```

- [ ] **Step 4: Restart server, chạy integration test**

Chạy block "Restart server". Rồi:

Run: `$PYTEST tests/test_asr_streaming.py -v`
Expected: 2 passed. Nếu transcript sai hệ thống (overlap thấp): nghi đầu tiên là fbank/nhịp tiêu thụ khung — đối chiếu `decode_chunk_len`/`T` với notes Task 1, xem log `/tmp/triton-serve.log`.

- [ ] **Step 5: Chạy TOÀN BỘ test — cũ không được đỏ thêm**

Run: `$PYTEST tests/ -v`
Expected: tất cả pass (test `asr`/`tts` cũ cần server đang chạy đủ model).

- [ ] **Step 6: Commit**

```bash
git add model_repository/asr_streaming/1/model.py tests/test_asr_streaming.py
git commit -m "Implement streaming recognition in asr_streaming backend"
```

---

### Task 6: Client streaming

**Files:**
- Create: `client/asr_streaming_client.py`

**Interfaces:**
- Consumes: `load_wav_16k`, `SAMPLE_RATE` từ `client/common.py`; hợp đồng `AUDIO_CHUNK`/`TRANSCRIPT` của Task 5.

- [ ] **Step 1: Viết client**

Tạo `client/asr_streaming_client.py`:

```python
# ABOUTME: Client streaming ASR - cắt wav thành chunk, gửi qua một gRPC stream
# ABOUTME: Chạy: python client/asr_streaming_client.py file.wav (--fast để gửi dồn không mô phỏng mic)

import argparse
import queue
import sys
import time
from pathlib import Path

import tritonclient.grpc as grpcclient

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.common import SAMPLE_RATE, load_wav_16k  # noqa: E402


def _print_partial(result):
    # in đè dòng hiện tại để partial chạy như phụ đề trực tiếp
    print(f"\r{result.as_numpy('TRANSCRIPT')[0, 0].decode('utf-8')}", end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="file wav, tần số nào cũng được - tự hạ về 16kHz")
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--chunk-ms", type=int, default=200)
    ap.add_argument("--fast", action="store_true", help="gửi dồn, không ngủ giữa các chunk")
    args = ap.parse_args()

    wav = load_wav_16k(args.wav)
    chunk = SAMPLE_RATE * args.chunk_ms // 1000
    parts = [wav[i : i + chunk] for i in range(0, len(wav), chunk)]

    q = queue.Queue()
    client = grpcclient.InferenceServerClient(args.url)
    client.start_stream(callback=lambda result, error: q.put((result, error)))
    seq_id = int(time.time()) % 2**31 + 1   # đủ khác nhau giữa các lần chạy

    received = 0
    for i, part in enumerate(parts):
        inp = grpcclient.InferInput("AUDIO_CHUNK", [1, len(part)], "FP32")
        inp.set_data_from_numpy(part.reshape(1, -1))
        client.async_stream_infer(
            "asr_streaming",
            [inp],
            sequence_id=seq_id,
            sequence_start=(i == 0),
            sequence_end=(i == len(parts) - 1),
        )
        if not args.fast:
            time.sleep(args.chunk_ms / 1000)
        # in mọi partial đã về trong lúc chờ, không chặn vòng gửi
        while True:
            try:
                result, error = q.get_nowait()
            except queue.Empty:
                break
            if error:
                raise SystemExit(f"lỗi từ server: {error}")
            received += 1
            _print_partial(result)

    while received < len(parts):
        result, error = q.get(timeout=30)
        if error:
            raise SystemExit(f"lỗi từ server: {error}")
        received += 1
        _print_partial(result)
    print()
    client.stop_stream()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Chạy thử cả hai chế độ**

```bash
$PY client/asr_streaming_client.py tests/assets/sample_vi.wav          # nhịp thật, partial chạy dần
$PY client/asr_streaming_client.py tests/assets/sample_vi.wav --fast   # gửi dồn
```

Expected: partial hiện dần trên một dòng, kết thúc bằng transcript final + xuống dòng; hai chế độ ra cùng final (greedy tất định).

- [ ] **Step 3: Commit**

```bash
git add client/asr_streaming_client.py
git commit -m "Add streaming ASR client with live partial display"
```

---

### Task 7: Cập nhật tài liệu

**Files:**
- Modify: `README.md`
- Modify: `Architect.md`

- [ ] **Step 1: README**

Trong mục đầu (danh sách thiết kế) thêm dòng:

```markdown
- Thiết kế streaming ASR: `docs/superpowers/specs/2026-08-10-streaming-asr-design.md`
```

Trong mục "Dùng" thêm:

```markdown
    # ASR streaming - partial transcript hiện dần như phụ đề
    .venv/bin/python client/asr_streaming_client.py tests/assets/sample_vi.wav
```

Trong mục "Kiến trúc" thêm dòng cuối:

```markdown
    asr_streaming     Python, GPU ×1       sequence batcher; chunk audio → partial transcript
```

- [ ] **Step 2: Architect.md — thêm section sau "Flow ASR"**

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add README.md Architect.md
git commit -m "Document asr_streaming flow and usage"
```

---

## Ngoài plan này

- **E4 benchmark** (N stream đồng thời, p95 per-chunk, RTF) — spec §11 đánh dấu tuỳ chọn, làm sau khi v1 chạy nếu còn thời gian.
- Biến thể chunk-32/64, mic client, ensemble + implicit state — spec §12.
