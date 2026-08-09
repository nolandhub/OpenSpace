# Triton Voice Serving — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dựng một Triton Inference Server chạy đồng thời ASR tiếng Việt (Zipformer RNN-T) và TTS tiếng Việt (ZipVoice), kèm bộ số đo chứng minh dynamic batching, ensemble và model concurrency.

**Architecture:** ASR tách thành 4 model Triton — `asr_feature` (Python/CPU), `asr_encoder` (ONNX/GPU, có dynamic batching), `asr_scorer` (Python/GPU, chạy greedy search) và `asr` (ensemble nối ba tầng). TTS là một model Python backend duy nhất bọc ZipVoice PyTorch. Tất cả chạy trong một container dựng từ `nvcr.io/nvidia/tritonserver:25.01-py3`.

**Tech Stack:** Triton Inference Server 25.01, ONNX Runtime (GPU), PyTorch + torchaudio, ZipVoice + Vocos, sentencepiece, espeak-ng, tritonclient[grpc], pytest.

## Global Constraints

Mọi task đều phải tuân thủ những ràng buộc dưới đây.

- **Image nền:** `nvcr.io/nvidia/tritonserver:25.01-py3`. Tên image build ra: `triton-voice`.
- **Phần cứng:** RTX 3050, 4GB VRAM. Mọi cấu hình phải vừa trong 4GB.
- **Cổng:** 8000 HTTP, 8001 gRPC, 8002 metrics. Container chạy `--net host`.
- **Mọi file code (`.py`, `.sh`) mở đầu bằng đúng 2 dòng `# ABOUTME:`** mô tả file đó làm gì.
- **Ngôn ngữ trong code:** identifier viết **tiếng Anh** (tên hàm, biến, tham số, hàm test, khoá dict, hằng số); comment và docstring viết **tiếng Việt**.

> **Lưu ý khi đọc plan này:** một số khối code bên dưới còn đặt tên identifier bằng tiếng Việt — đó là lỗi của bản nháp, đã sửa hết trong code thật. Khi có khác biệt, **lấy file trong repo làm chuẩn**, không lấy plan.
- **Không commit trọng số.** `*.onnx`, `*.pt`, `*.plan`, `*.bin` đã nằm trong `.gitignore`.
- **Hằng số dùng chung — dùng đúng những giá trị này ở mọi nơi:**
  - `SAMPLE_RATE = 16000` (đầu vào ASR)
  - `MAX_DURATION_S = 16` → `MAX_SAMPLES = 256000`
  - `NUM_MEL_BINS = 80`, `FRAME_SHIFT_MS = 10` → `MAX_FRAMES = 1600`
  - `BLANK_ID = 0`, `CONTEXT_SIZE = 2` (stateless decoder)
  - TTS xuất ra `24000` Hz
- **Tên tensor trong `config.pbtxt` phải khớp từng ký tự với tên dùng trong `model.py`.** Đây là nguồn lỗi số một.
- **Client và test chạy trên host** (venv riêng, không cần GPU), server chạy trong container.

---

## File Structure

| File | Trách nhiệm |
|---|---|
| `docker/Dockerfile` | Image: Triton + espeak-ng + torch/onnxruntime + ZipVoice |
| `scripts/fetch_models.sh` | Tải checkpoint từ HuggingFace về đúng vị trí trong `model_repository/` |
| `scripts/inspect_onnx.py` | In tên và shape các cổng của file ONNX |
| `scripts/serve.sh` | Build image + chạy container |
| `model_repository/asr_encoder/config.pbtxt` | Khai báo encoder: ONNX backend, dynamic batching, GPU |
| `model_repository/asr_feature/config.pbtxt` + `1/model.py` | fbank 80 chiều, CPU, nhiều instance |
| `model_repository/asr_scorer/1/greedy_search.py` | **Hàm thuần tuý** — vòng lặp greedy, không dính Triton, test được độc lập |
| `model_repository/asr_scorer/config.pbtxt` + `1/model.py` | Bọc `greedy_search` bằng ONNX sessions thật |
| `model_repository/asr/config.pbtxt` | Ensemble nối ba tầng |
| `model_repository/tts/config.pbtxt` + `1/model.py` | ZipVoice PyTorch → waveform |
| `client/common.py` | Hằng số dùng chung + `pad_wav()` |
| `client/asr_client.py` | CLI: wav → text |
| `client/tts_client.py` | CLI: text → wav |
| `tests/conftest.py` | Fixture kết nối Triton, tự skip nếu server chưa chạy |
| `tests/test_greedy_search.py` | Unit, không cần GPU lẫn server |
| `tests/test_asr.py`, `tests/test_tts.py` | Integration |
| `bench/bench.py` | Điều phối `perf_analyzer`, gom kết quả ra CSV |
| `requirements.txt` | Dependency cho host (client + test + bench) |

---

## Task 1: Bộ khung project và container

**Files:**
- Create: `docker/Dockerfile`
- Create: `scripts/fetch_models.sh`
- Create: `scripts/serve.sh`
- Create: `scripts/inspect_onnx.py`
- Create: `requirements.txt`
- Create: `README.md`
- Create: `model_repository/.gitkeep`

**Interfaces:**
- Consumes: không có
- Produces: image `triton-voice`; các file trọng số nằm tại `model_repository/asr_encoder/1/model.onnx`, `model_repository/asr_scorer/1/{decoder.onnx,joiner.onnx,bpe.model}`, `model_repository/tts/1/{model.pt,tokens.txt,config.json}`, `model_repository/tts/1/vocos/{config.yaml,pytorch_model.bin}`

- [ ] **Step 1: Viết `requirements.txt`**

```
tritonclient[grpc]==2.53.0
soundfile==0.12.1
numpy==1.26.4
pytest==8.3.4
onnx==1.17.0
scipy==1.14.1
```

- [ ] **Step 2: Viết `docker/Dockerfile`**

`docker/Dockerfile` — thư mục `docker/` chỉ chứa đúng file này, không cần file phụ nào:

```dockerfile
FROM nvcr.io/nvidia/tritonserver:25.01-py3

# espeak-ng là tokenizer bắt buộc của ZipVoice cho tiếng Việt
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng espeak-ng-data git \
    && rm -rf /var/lib/apt/lists/*

RUN echo "numpy<2" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# Ghim torch/torchaudio TRƯỚC khi cài ZipVoice.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1 torchaudio==2.5.1

RUN git clone --depth 1 https://github.com/k2-fsa/ZipVoice.git /opt/ZipVoice

RUN pip install --no-cache-dir -r /opt/ZipVoice/requirements.txt

RUN pip install --no-cache-dir \
        onnxruntime-gpu==1.20.2 \
        sentencepiece==0.2.0

RUN pip install --no-cache-dir --no-deps /opt/ZipVoice

ENV PYTHONPATH=/opt/ZipVoice:${PYTHONPATH}

# Chốt lại: nếu bước nào lỡ nâng numpy lên 2.x thì dừng build ngay tại đây
RUN python3 -c "import numpy; assert numpy.__version__.startswith('1.'), numpy.__version__" \
    && python3 -c "import torch, torchaudio, onnxruntime, sentencepiece, vocos, soundfile, zipvoice; print('deps ok')"
```

**Ba cái bẫy đã gặp thật khi dựng, đừng bỏ qua:**

1. `onnxruntime-gpu==1.20.1` **không tồn tại** trên PyPI. Bản gần nhất là `1.20.2`.
2. `requirements.txt` của ZipVoice **không ghim torch**. Để pip tự chọn thì nó kéo torch 2.13 kèm nguyên bộ thư viện CUDA 13 (~900MB riêng cudnn) — vừa phình image vừa đụng với `onnxruntime-gpu` vốn dựng cho CUDA 12. Phải cài torch trước.
3. `lhotse` kéo **numpy 2.x**, trong khi `tritonserver` và `cupy` trong image nền đều yêu cầu `numpy<2`, và torch 2.5.1 biên dịch với numpy 1.x. Không có `PIP_CONSTRAINT` thì Python backend vỡ lúc truyền tensor. Dòng `assert` ở cuối là chốt chặn cho lỗi này.

Trong image nền chỉ có `python3`, **không có** `python`. Mọi lệnh phải gọi `python3`.

- [ ] **Step 3: Build image và xác nhận từng thành phần**

Run:
```bash
docker build -t triton-voice -f docker/Dockerfile docker/
docker run --rm triton-voice espeak-ng --voices=vi
```
Expected: build in ra `deps ok` ở bước cuối, và espeak-ng liệt kê giọng tiếng Việt.

Đừng chạy `docker build ... | tail` — pipe nuốt mất exit code, build hỏng vẫn báo thành công.

- [ ] **Step 4: Viết `scripts/fetch_models.sh`**

```bash
#!/usr/bin/env bash
# ABOUTME: Tải trọng số ASR/TTS từ HuggingFace về đúng vị trí trong model_repository
# ABOUTME: Chạy lại nhiều lần được - file đã có thì bỏ qua

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$ROOT/model_repository"

dl() {  # dl <url> <đích>
  if [ -f "$2" ]; then echo "bỏ qua $2"; return; fi
  mkdir -p "$(dirname "$2")"
  echo "tải $2"
  curl -fL --progress-bar "$1" -o "$2"
}

ASR=https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h/resolve/main
dl "$ASR/encoder-epoch-20-avg-10.onnx" "$REPO/asr_encoder/1/model.onnx"
dl "$ASR/decoder-epoch-20-avg-10.onnx" "$REPO/asr_scorer/1/decoder.onnx"
dl "$ASR/joiner-epoch-20-avg-10.onnx"  "$REPO/asr_scorer/1/joiner.onnx"
dl "$ASR/bpe.model"                     "$REPO/asr_scorer/1/bpe.model"

TTS=https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h/resolve/main
dl "$TTS/iter-525000-avg-2.pt" "$REPO/tts/1/model.pt"
dl "$TTS/tokens.txt"           "$REPO/tts/1/tokens.txt"
dl "$TTS/config.json"          "$REPO/tts/1/config.json"

# Vocoder không nằm trong repo của ZipVoice tiếng Việt, phải lấy riêng
VOC=https://huggingface.co/charactr/vocos-mel-24khz/resolve/main
dl "$VOC/config.yaml"       "$REPO/tts/1/vocos/config.yaml"
dl "$VOC/pytorch_model.bin" "$REPO/tts/1/vocos/pytorch_model.bin"

# Ensemble không có trọng số nhưng Triton vẫn bắt buộc có thư mục version
mkdir -p "$REPO/asr/1"
echo "xong"
```

- [ ] **Step 5: Chạy và kiểm tra kích thước file**

Run:
```bash
chmod +x scripts/fetch_models.sh && ./scripts/fetch_models.sh
find model_repository -type f \( -name '*.onnx' -o -name '*.pt' -o -name '*.bin' -o -name '*.model' \) -exec ls -lh {} \;
```
Expected: `model.onnx` ~92MB, `decoder.onnx` ~5MB, `joiner.onnx` ~4MB, `bpe.model` ~268KB, `tts/1/model.pt` ~491MB, `vocos/pytorch_model.bin` ~54MB.

Nếu file nào ~1KB thì đó là trang lỗi HTML chứ không phải trọng số — kiểm tra lại URL.

- [ ] **Step 6: Viết `scripts/inspect_onnx.py`**

```python
# ABOUTME: In tên và shape các cổng vào/ra của file ONNX
# ABOUTME: Dùng để điền config.pbtxt cho đúng - sai tên là Triton không load

import sys

import onnx


def dims(t):
    return [d.dim_value or d.dim_param or "?" for d in t.type.tensor_type.shape.dim]


for path in sys.argv[1:]:
    m = onnx.load(path)
    print(f"\n=== {path} ===")
    for i in m.graph.input:
        print(f"  IN   {i.name:20s} {dims(i)}")
    for o in m.graph.output:
        print(f"  OUT  {o.name:20s} {dims(o)}")
```

- [ ] **Step 7: Chạy inspect và GHI LẠI kết quả**

Run:
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/inspect_onnx.py \
  model_repository/asr_encoder/1/model.onnx \
  model_repository/asr_scorer/1/decoder.onnx \
  model_repository/asr_scorer/1/joiner.onnx
```

Expected (chuẩn export của icefall):
```
=== model.onnx ===
  IN   x                    ['N', 'T', 80]
  IN   x_lens               ['N']
  OUT  encoder_out          ['N', 'T', 512]
  OUT  encoder_out_lens     ['N']
=== decoder.onnx ===
  IN   y                    ['N', 2]
  OUT  decoder_out          ['N', 1, 512]
=== joiner.onnx ===
  IN   encoder_out          ['N', 512]
  IN   decoder_out          ['N', 512]
  OUT  logit                ['N', 'vocab']
```

**Nếu tên khác với bảng trên, dùng tên thật ở Task 2 và Task 5 thay cho tên trong plan này.** Đây là bước bắt buộc, không được bỏ qua.

- [ ] **Step 8: Viết `scripts/serve.sh`**

```bash
#!/usr/bin/env bash
# ABOUTME: Build image rồi chạy Triton server với model_repository của project
# ABOUTME: Truyền tên model vào để chỉ load riêng model đó, ví dụ: ./scripts/serve.sh asr_encoder

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker build -t triton-voice -f "$ROOT/docker/Dockerfile" "$ROOT/docker"

ARGS=(--model-repository=/models)
if [ $# -gt 0 ]; then
  # Chế độ explicit: chỉ load đúng model được chỉ định, tiện lúc debug
  ARGS+=(--model-control-mode=explicit)
  for m in "$@"; do ARGS+=(--load-model="$m"); done
fi

docker run --gpus all --rm -it --net host --shm-size 1g \
  -v "$ROOT/model_repository:/models" \
  triton-voice tritonserver "${ARGS[@]}"
```

- [ ] **Step 9: Viết `README.md`**

```markdown
# Triton Voice Serving

Serving ASR (Zipformer RNN-T) và TTS (ZipVoice) tiếng Việt trên Triton Inference Server.

Thiết kế: `docs/superpowers/specs/2026-08-09-triton-voice-serving-design.md`

## Chạy

    ./scripts/fetch_models.sh          # tải trọng số, chỉ cần 1 lần
    ./scripts/serve.sh                 # dựng image và chạy server
    ./scripts/serve.sh asr_encoder     # chỉ load 1 model, dùng khi debug

## Dùng

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python client/asr_client.py tests/assets/sample_vi.wav
    .venv/bin/python client/tts_client.py --text "Xin chào" --out ra.wav

## Test

    .venv/bin/pytest tests/ -m "not integration"   # unit, không cần server
    .venv/bin/pytest tests/                        # đầy đủ, cần server đang chạy
```

- [ ] **Step 10: Commit**

```bash
chmod +x scripts/*.sh
git add docker scripts requirements.txt README.md .gitignore docs
git commit -m "feat: bộ khung project, Dockerfile và script tải model"
```

---

## Task 2: `asr_encoder` — model đầu tiên, không viết code Python

**Files:**
- Create: `model_repository/asr_encoder/config.pbtxt`
- Test: `tests/conftest.py`, `tests/test_asr_encoder.py`

**Interfaces:**
- Consumes: `model_repository/asr_encoder/1/model.onnx` từ Task 1
- Produces: model Triton tên `asr_encoder`, nhận `x` FP32 `(B, T, 80)` + `x_lens` INT64 `(B,)`, trả `encoder_out` FP32 `(B, T', C)` + `encoder_out_lens` INT64 `(B,)`

- [ ] **Step 1: Viết `tests/conftest.py`**

```python
# ABOUTME: Fixture dùng chung cho test - kết nối Triton qua gRPC
# ABOUTME: Tự động skip test integration nếu server chưa chạy

import pytest

URL = "localhost:8001"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: cần Triton server đang chạy")


@pytest.fixture(scope="session")
def triton():
    grpc = pytest.importorskip("tritonclient.grpc")
    client = grpc.InferenceServerClient(URL)
    try:
        if not client.is_server_ready():
            pytest.skip(f"Triton chưa sẵn sàng tại {URL}")
    except Exception as e:
        pytest.skip(f"Không kết nối được Triton tại {URL}: {e}")
    return client
```

- [ ] **Step 2: Viết test thất bại `tests/test_asr_encoder.py`**

```python
# ABOUTME: Integration test cho model asr_encoder
# ABOUTME: Đưa tensor ngẫu nhiên vào, kiểm tra shape đầu ra hợp lệ

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_encoder_tra_ve_dung_so_khung(triton):
    import tritonclient.grpc as grpcclient

    batch, frames = 1, 1600
    x = np.random.randn(batch, frames, 80).astype(np.float32)
    x_lens = np.array([[frames]], dtype=np.int64)

    inputs = [
        grpcclient.InferInput("x", list(x.shape), "FP32"),
        grpcclient.InferInput("x_lens", list(x_lens.shape), "INT64"),
    ]
    inputs[0].set_data_from_numpy(x)
    inputs[1].set_data_from_numpy(x_lens)

    out = triton.infer("asr_encoder", inputs)
    enc = out.as_numpy("encoder_out")
    enc_lens = out.as_numpy("encoder_out_lens")

    assert enc.ndim == 3 and enc.shape[0] == batch
    # Zipformer subsample khoảng 4 lần nên số khung ra phải nhỏ hơn hẳn đầu vào
    assert 0 < enc.shape[1] < frames
    assert int(enc_lens.reshape(-1)[0]) == enc.shape[1]
```

- [ ] **Step 3: Chạy test để xác nhận nó đỏ**

Run: `.venv/bin/pytest tests/test_asr_encoder.py -v`
Expected: SKIP (server chưa chạy) — đúng như thiết kế của fixture.

Bật server ở terminal khác: `./scripts/serve.sh asr_encoder`
Chạy lại. Expected: server báo lỗi không load được `asr_encoder` vì chưa có `config.pbtxt`.

- [ ] **Step 4: Viết `model_repository/asr_encoder/config.pbtxt`**

Dùng đúng tên cổng đã in ra ở Task 1 Step 7.

```protobuf
name: "asr_encoder"
backend: "onnxruntime"
max_batch_size: 8

input [
  {
    name: "x"
    data_type: TYPE_FP32
    dims: [-1, 80]
  },
  {
    name: "x_lens"
    data_type: TYPE_INT64
    dims: [1]
    reshape: { shape: [] }
  }
]

output [
  {
    name: "encoder_out"
    data_type: TYPE_FP32
    dims: [-1, -1]
  },
  {
    name: "encoder_out_lens"
    data_type: TYPE_INT64
    dims: [1]
    reshape: { shape: [] }
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
```

Ba điểm cần hiểu ở file này:
- `max_batch_size: 8` bật lên thì `dims` **không kể chiều batch**. `[-1, 80]` nghĩa là tensor thật `(B, T, 80)`.
- `reshape: { shape: [] }` bóc chiều thừa: Triton luôn ghép batch thành `(B, 1)` nhưng ONNX muốn `(B,)`.
- `encoder_out` để `[-1, -1]` cho khỏi phải tra đúng số chiều ẩn — Triton chấp nhận dim động hoàn toàn.

- [ ] **Step 5: Khởi động lại server và chạy test**

Run:
```bash
./scripts/serve.sh asr_encoder     # terminal 1
.venv/bin/pytest tests/test_asr_encoder.py -v   # terminal 2
```
Expected: PASS. Log server có dòng `asr_encoder ... READY`.

Nếu server báo `failed to load`, đọc log — gần như luôn là sai tên cổng so với Step 7 của Task 1.

- [ ] **Step 6: Commit**

```bash
git add model_repository/asr_encoder/config.pbtxt tests/conftest.py tests/test_asr_encoder.py
git commit -m "feat: model asr_encoder chạy ONNX backend với dynamic batching"
```

---

## Task 3: Hàm greedy search thuần tuý

Task này **không đụng đến Triton**. Mục đích là tách vòng lặp decode ra thành hàm thuần tuý để test được mà không cần GPU, server, hay file ONNX.

**Files:**
- Create: `model_repository/asr_scorer/1/greedy_search.py`
- Test: `tests/test_greedy_search.py`

**Interfaces:**
- Consumes: không có
- Produces:
  ```python
  greedy_search(
      encoder_out: np.ndarray,      # (T, C) của MỘT câu
      num_frames: int,
      run_decoder: Callable[[list[int]], np.ndarray],   # nhận ngữ cảnh -> (1, C)
      run_joiner: Callable[[np.ndarray, np.ndarray], np.ndarray],  # (1,C),(1,C) -> (1, V)
      blank_id: int = 0,
      context_size: int = 2,
  ) -> list[int]
  ```
  Task 5 sẽ gọi hàm này với hai closure bọc ONNX session thật.

- [ ] **Step 1: Viết test thất bại `tests/test_greedy_search.py`**

```python
# ABOUTME: Unit test cho vòng lặp greedy search - không cần GPU, server hay ONNX
# ABOUTME: decoder và joiner được thay bằng hàm giả để kiểm tra đúng logic vòng lặp

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "model_repository/asr_scorer/1"))
from greedy_search import greedy_search  # noqa: E402

DIM = 4
VOCAB = 10


def decoder_gia(_ngu_canh):
    """Decoder giả - luôn trả về vector 0, đủ dùng vì joiner giả không nhìn tới nó."""
    return np.zeros((1, DIM), dtype=np.float32)


def joiner_theo_kich_ban(kich_ban):
    """Tạo joiner giả phát ra token theo đúng kịch bản, mỗi lần gọi lấy 1 phần tử."""
    buoc = {"i": 0}

    def _joiner(_enc, _dec):
        logits = np.zeros((1, VOCAB), dtype=np.float32)
        logits[0, kich_ban[buoc["i"]]] = 1.0
        buoc["i"] += 1
        return logits

    return _joiner


def test_toan_blank_thi_khong_ra_token():
    enc = np.zeros((5, DIM), dtype=np.float32)
    ket_qua = greedy_search(enc, 5, decoder_gia, joiner_theo_kich_ban([0] * 5))
    assert ket_qua == []


def test_token_ra_dung_thu_tu():
    enc = np.zeros((5, DIM), dtype=np.float32)
    # khung 0 blank, khung 1 ra token 7, khung 2-3 blank, khung 4 ra token 3
    ket_qua = greedy_search(enc, 5, decoder_gia, joiner_theo_kich_ban([0, 7, 0, 0, 3]))
    assert ket_qua == [7, 3]


def test_chi_goi_lai_decoder_khi_phat_ra_token():
    """Đây là tối ưu chính của vòng lặp: blank thì lịch sử text không đổi nên bỏ qua decoder."""
    so_lan_goi = {"n": 0}

    def dem_decoder(ngu_canh):
        so_lan_goi["n"] += 1
        return np.zeros((1, DIM), dtype=np.float32)

    enc = np.zeros((6, DIM), dtype=np.float32)
    greedy_search(enc, 6, dem_decoder, joiner_theo_kich_ban([0, 5, 0, 0, 9, 0]))

    # 1 lần khởi tạo + 2 lần vì phát ra 2 token
    assert so_lan_goi["n"] == 3


def test_ngu_canh_dua_cho_decoder_la_hai_token_cuoi():
    ngu_canh_da_thay = []

    def ghi_lai_ngu_canh(ngu_canh):
        ngu_canh_da_thay.append(list(ngu_canh))
        return np.zeros((1, DIM), dtype=np.float32)

    enc = np.zeros((4, DIM), dtype=np.float32)
    greedy_search(enc, 4, ghi_lai_ngu_canh, joiner_theo_kich_ban([2, 6, 0, 0]))

    assert ngu_canh_da_thay[0] == [0, 0]   # khởi tạo: toàn blank
    assert ngu_canh_da_thay[1] == [0, 2]   # sau khi phát token 2
    assert ngu_canh_da_thay[2] == [2, 6]   # sau khi phát token 6
    assert all(len(c) == 2 for c in ngu_canh_da_thay)


def test_chi_duyet_dung_so_khung_duoc_yeu_cau():
    """encoder_out có thể dài hơn num_frames vì phần đệm - không được đụng vào phần đó."""
    enc = np.zeros((10, DIM), dtype=np.float32)
    ket_qua = greedy_search(enc, 3, decoder_gia, joiner_theo_kich_ban([1, 2, 3, 4, 5]))
    assert ket_qua == [1, 2, 3]
```

- [ ] **Step 2: Chạy test để xác nhận nó đỏ**

Run: `.venv/bin/pytest tests/test_greedy_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'greedy_search'`

- [ ] **Step 3: Viết `model_repository/asr_scorer/1/greedy_search.py`**

```python
# ABOUTME: Vòng lặp greedy search cho RNN-Transducer, viết thuần numpy
# ABOUTME: Không phụ thuộc Triton hay ONNX - decoder/joiner được truyền vào dạng hàm

from typing import Callable, List

import numpy as np


def greedy_search(
    encoder_out: np.ndarray,
    num_frames: int,
    run_decoder: Callable[[List[int]], np.ndarray],
    run_joiner: Callable[[np.ndarray, np.ndarray], np.ndarray],
    blank_id: int = 0,
    context_size: int = 2,
) -> List[int]:
    """Giải mã một câu bằng greedy search.

    encoder_out: (T, C) - biểu diễn của cả câu, đã bỏ chiều batch
    num_frames:  số khung thật, phần còn lại của encoder_out là đệm, bỏ qua
    run_decoder: nhận danh sách token ngữ cảnh, trả về (1, C)
    run_joiner:  nhận (1, C) khung audio và (1, C) trạng thái text, trả về (1, vocab)

    Trả về danh sách token id, đã bỏ phần ngữ cảnh khởi tạo.
    """
    # Ngữ cảnh khởi tạo toàn blank - coi như chưa viết được chữ nào
    hyp: List[int] = [blank_id] * context_size
    decoder_out = run_decoder(hyp[-context_size:])

    for t in range(num_frames):
        # encoder_out[t] là (C,), thêm chiều batch thành (1, C) cho khớp joiner
        logits = run_joiner(encoder_out[t : t + 1], decoder_out)
        token = int(np.argmax(logits.reshape(-1)))

        if token != blank_id:
            # Chỉ khi phát ra token thật thì lịch sử text mới đổi,
            # nên chỉ khi đó mới phải chạy lại decoder. Bỏ qua được đa số lần gọi.
            hyp.append(token)
            decoder_out = run_decoder(hyp[-context_size:])

    return hyp[context_size:]
```

- [ ] **Step 4: Chạy test để xác nhận nó xanh**

Run: `.venv/bin/pytest tests/test_greedy_search.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add model_repository/asr_scorer/1/greedy_search.py tests/test_greedy_search.py
git commit -m "feat: hàm greedy search thuần tuý kèm unit test"
```

---

## Task 4: `asr_feature` — Python backend đầu tiên

**Files:**
- Create: `model_repository/asr_feature/config.pbtxt`
- Create: `model_repository/asr_feature/1/model.py`
- Create: `client/common.py`
- Create: `tests/test_asr_feature.py`

**Interfaces:**
- Consumes: không có
- Produces: model Triton `asr_feature`. Vào: `WAV` FP32 `(B, 256000)` đã đệm, `WAV_LEN` INT32 `(B, 1)`. Ra: `SPEECH` FP32 `(B, 1600, 80)`, `SPEECH_LEN` INT64 `(B, 1)`.
- Produces: `client/common.py` với `SAMPLE_RATE`, `MAX_SAMPLES`, `MAX_FRAMES`, `NUM_MEL_BINS`, và `pad_wav(wav: np.ndarray) -> tuple[np.ndarray, int]`

- [ ] **Step 1: Viết `client/common.py`**

```python
# ABOUTME: Hằng số và hàm dùng chung cho client và test
# ABOUTME: Mọi nơi phải dùng cùng bộ hằng số này, lệch một chỗ là batching hỏng

import numpy as np

SAMPLE_RATE = 16000
MAX_DURATION_S = 16
MAX_SAMPLES = SAMPLE_RATE * MAX_DURATION_S   # 256000
FRAME_SHIFT_MS = 10
MAX_FRAMES = MAX_DURATION_S * 1000 // FRAME_SHIFT_MS   # 1600
NUM_MEL_BINS = 80


def pad_wav(wav: np.ndarray) -> tuple[np.ndarray, int]:
    """Đệm waveform về đúng MAX_SAMPLES, trả về (mảng đã đệm, độ dài thật).

    Mọi request phải cùng shape thì dynamic batcher của Triton mới gom được.
    Đây là lý do tồn tại của hàm này - xem mục 7 của spec.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) > MAX_SAMPLES:
        raise ValueError(
            f"Audio dài {len(wav) / SAMPLE_RATE:.1f}s, vượt giới hạn {MAX_DURATION_S}s"
        )
    padded = np.zeros(MAX_SAMPLES, dtype=np.float32)
    padded[: len(wav)] = wav
    return padded, len(wav)
```

- [ ] **Step 2: Viết test thất bại `tests/test_asr_feature.py`**

```python
# ABOUTME: Integration test cho model asr_feature
# ABOUTME: Kiểm tra fbank ra đúng shape cố định và độ dài thật được tính đúng

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.common import MAX_FRAMES, NUM_MEL_BINS, SAMPLE_RATE, pad_wav  # noqa: E402

pytestmark = pytest.mark.integration


def test_feature_luon_ra_shape_co_dinh(triton):
    import tritonclient.grpc as grpcclient

    # 3 giây tiếng ồn - ngắn hơn nhiều so với giới hạn 16 giây
    wav = np.random.randn(3 * SAMPLE_RATE).astype(np.float32) * 0.1
    padded, real_len = pad_wav(wav)

    inputs = [
        grpcclient.InferInput("WAV", [1, len(padded)], "FP32"),
        grpcclient.InferInput("WAV_LEN", [1, 1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(padded.reshape(1, -1))
    inputs[1].set_data_from_numpy(np.array([[real_len]], dtype=np.int32))

    out = triton.infer("asr_feature", inputs)
    speech = out.as_numpy("SPEECH")
    speech_len = out.as_numpy("SPEECH_LEN")

    # Shape phải cố định bất kể audio dài bao nhiêu - đây là điều kiện để encoder batch được
    assert speech.shape == (1, MAX_FRAMES, NUM_MEL_BINS)

    # Độ dài thật phải tương ứng 3 giây, không phải 16 giây
    assert 280 < int(speech_len.reshape(-1)[0]) < 320

    # Phần đệm phía sau phải bằng 0
    real_frames = int(speech_len.reshape(-1)[0])
    assert np.allclose(speech[0, real_frames:], 0.0)
```

- [ ] **Step 3: Chạy test để xác nhận nó đỏ**

Run:
```bash
./scripts/serve.sh asr_feature     # terminal 1 - sẽ báo lỗi không load được
.venv/bin/pytest tests/test_asr_feature.py -v
```
Expected: FAIL hoặc SKIP vì `asr_feature` chưa tồn tại.

- [ ] **Step 4: Viết `model_repository/asr_feature/config.pbtxt`**

```protobuf
name: "asr_feature"
backend: "python"
max_batch_size: 8

input [
  {
    name: "WAV"
    data_type: TYPE_FP32
    dims: [-1]
  },
  {
    name: "WAV_LEN"
    data_type: TYPE_INT32
    dims: [1]
  }
]

output [
  {
    name: "SPEECH"
    data_type: TYPE_FP32
    dims: [-1, 80]
  },
  {
    name: "SPEECH_LEN"
    data_type: TYPE_INT64
    dims: [1]
  }
]

instance_group [
  {
    kind: KIND_CPU
    count: 4
  }
]

parameters {
  key: "max_frames"
  value: { string_value: "1600" }
}
```

Không có `dynamic_batching` — tầng này là DSP thuần trên CPU, gom lại không lợi gì mà còn cộng thêm queue delay. `count: 4` là bốn process Python độc lập, đây mới là cách tăng thông lượng ở tầng này.

- [ ] **Step 5: Viết `model_repository/asr_feature/1/model.py`**

```python
# ABOUTME: Triton Python backend - trích đặc trưng fbank 80 chiều từ waveform 16kHz
# ABOUTME: Luôn xuất ra số khung cố định để tầng encoder phía sau batch được

import json

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # args["model_config"] là chuỗi JSON của chính config.pbtxt
        config = json.loads(args["model_config"])
        params = config.get("parameters", {})
        self.max_frames = int(params["max_frames"]["string_value"])
        self.num_mel_bins = 80
        self.sample_rate = 16000

    def _fbank(self, samples: np.ndarray) -> np.ndarray:
        """Tính fbank cho một câu. samples: (N,) float32 -> (T, 80)"""
        wav = torch.from_numpy(samples).unsqueeze(0)
        feat = kaldi.fbank(
            wav,
            num_mel_bins=self.num_mel_bins,
            frame_length=25.0,
            frame_shift=10.0,
            dither=0.0,
            sample_frequency=self.sample_rate,
            snip_edges=False,
        )
        return feat.numpy()

    def execute(self, requests):
        # requests là một DANH SÁCH. Với backend python, Triton không tự gộp
        # các request lại thành batch - mình tự quyết định xử lý thế nào.
        responses = []

        for request in requests:
            wav = pb_utils.get_input_tensor_by_name(request, "WAV").as_numpy()
            wav_len = pb_utils.get_input_tensor_by_name(request, "WAV_LEN").as_numpy()

            batch = wav.shape[0]
            speech = np.zeros(
                (batch, self.max_frames, self.num_mel_bins), dtype=np.float32
            )
            speech_len = np.zeros((batch, 1), dtype=np.int64)

            for i in range(batch):
                # Chỉ tính trên phần audio thật, bỏ qua phần client đã đệm
                feat = self._fbank(wav[i, : int(wav_len[i, 0])])
                frames = min(feat.shape[0], self.max_frames)
                speech[i, :frames] = feat[:frames]
                speech_len[i, 0] = frames

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("SPEECH", speech),
                        pb_utils.Tensor("SPEECH_LEN", speech_len),
                    ]
                )
            )

        # Bắt buộc: số response phải bằng số request, đúng thứ tự
        return responses
```

- [ ] **Step 6: Khởi động lại server và chạy test**

Run:
```bash
./scripts/serve.sh asr_feature     # terminal 1
.venv/bin/pytest tests/test_asr_feature.py -v
```
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add model_repository/asr_feature client/common.py tests/test_asr_feature.py
git commit -m "feat: model asr_feature trích fbank trên CPU với 4 instance"
```

---

## Task 5: `asr_scorer` — bọc greedy search bằng ONNX thật

**Files:**
- Create: `model_repository/asr_scorer/config.pbtxt`
- Create: `model_repository/asr_scorer/1/model.py`
- Test: `tests/test_asr_scorer.py`

**Interfaces:**
- Consumes: `greedy_search()` từ Task 3; `decoder.onnx`, `joiner.onnx`, `bpe.model` từ Task 1
- Produces: model Triton `asr_scorer`. Vào: `ENCODER_OUT` FP32 `(B, T', C)`, `ENCODER_OUT_LEN` INT64 `(B, 1)`. Ra: `TRANSCRIPT` STRING `(B, 1)`.

- [ ] **Step 1: Viết test thất bại `tests/test_asr_scorer.py`**

```python
# ABOUTME: Integration test cho model asr_scorer
# ABOUTME: Đưa encoder_out ngẫu nhiên vào, chỉ kiểm tra hợp đồng đầu ra chứ không kiểm nội dung

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_scorer_tra_ve_chuoi_utf8(triton):
    import tritonclient.grpc as grpcclient

    # encoder_out ngẫu nhiên - text ra sẽ vô nghĩa, nhưng phải đúng kiểu và không lỗi
    enc = np.random.randn(1, 50, 512).astype(np.float32)
    enc_lens = np.array([[50]], dtype=np.int64)

    inputs = [
        grpcclient.InferInput("ENCODER_OUT", list(enc.shape), "FP32"),
        grpcclient.InferInput("ENCODER_OUT_LEN", list(enc_lens.shape), "INT64"),
    ]
    inputs[0].set_data_from_numpy(enc)
    inputs[1].set_data_from_numpy(enc_lens)

    out = triton.infer("asr_scorer", inputs)
    text = out.as_numpy("TRANSCRIPT")

    assert text.shape == (1, 1)
    assert isinstance(text[0, 0].decode("utf-8"), str)
```

Chiều cuối `512` phải khớp với `encoder_out` thật in ra ở Task 1 Step 7. Nếu khác thì sửa lại con số này.

- [ ] **Step 2: Chạy test để xác nhận nó đỏ**

Run: `.venv/bin/pytest tests/test_asr_scorer.py -v`
Expected: FAIL — model `asr_scorer` chưa tồn tại.

- [ ] **Step 3: Viết `model_repository/asr_scorer/config.pbtxt`**

```protobuf
name: "asr_scorer"
backend: "python"
max_batch_size: 8

input [
  {
    name: "ENCODER_OUT"
    data_type: TYPE_FP32
    dims: [-1, -1]
  },
  {
    name: "ENCODER_OUT_LEN"
    data_type: TYPE_INT64
    dims: [1]
  }
]

output [
  {
    name: "TRANSCRIPT"
    data_type: TYPE_STRING
    dims: [1]
  }
]

instance_group [
  {
    kind: KIND_GPU
    count: 2
  }
]
```

Không có `dynamic_batching`: mỗi câu rẽ nhánh khác nhau tuỳ token nó phát ra, không có cách nào chạy chung một vòng lặp. `count: 2` là cách duy nhất tăng thông lượng ở tầng này.

- [ ] **Step 4: Viết `model_repository/asr_scorer/1/model.py`**

```python
# ABOUTME: Triton Python backend - chạy greedy search rồi dịch token id sang tiếng Việt
# ABOUTME: Logic vòng lặp nằm ở greedy_search.py, file này chỉ nối nó với ONNX và Triton

import os
import sys

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import triton_python_backend_utils as pb_utils

# Triton không tự thêm thư mục model vào sys.path nên phải tự làm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from greedy_search import greedy_search  # noqa: E402

BLANK_ID = 0
CONTEXT_SIZE = 2


class TritonPythonModel:
    def initialize(self, args):
        d = os.path.join(args["model_repository"], args["model_version"])
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Nạp một lần duy nhất, dùng lại cho mọi request
        self.decoder = ort.InferenceSession(
            os.path.join(d, "decoder.onnx"), providers=providers
        )
        self.joiner = ort.InferenceSession(
            os.path.join(d, "joiner.onnx"), providers=providers
        )
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(d, "bpe.model"))

        # Lấy tên cổng từ chính file ONNX thay vì viết cứng - đỡ lệ thuộc phiên bản export
        self.decoder_in = self.decoder.get_inputs()[0].name
        self.joiner_in = [i.name for i in self.joiner.get_inputs()]

    def _run_decoder(self, context):
        y = np.array([context], dtype=np.int64)
        out = self.decoder.run(None, {self.decoder_in: y})[0]
        # Một số bản export trả về (N, 1, C), bỏ chiều giữa cho khớp joiner
        return out[:, 0, :] if out.ndim == 3 else out

    def _run_joiner(self, enc_frame, dec_out):
        feeds = {self.joiner_in[0]: enc_frame, self.joiner_in[1]: dec_out}
        return self.joiner.run(None, feeds)[0]

    def execute(self, requests):
        responses = []

        for request in requests:
            enc = pb_utils.get_input_tensor_by_name(request, "ENCODER_OUT").as_numpy()
            lens = pb_utils.get_input_tensor_by_name(
                request, "ENCODER_OUT_LEN"
            ).as_numpy()

            texts = []
            for i in range(enc.shape[0]):
                tokens = greedy_search(
                    encoder_out=enc[i],
                    num_frames=int(lens[i, 0]),
                    run_decoder=self._run_decoder,
                    run_joiner=self._run_joiner,
                    blank_id=BLANK_ID,
                    context_size=CONTEXT_SIZE,
                )
                texts.append(self.sp.decode(tokens))

            out = np.array([[t.encode("utf-8")] for t in texts], dtype=object)
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("TRANSCRIPT", out)]
                )
            )

        return responses
```

- [ ] **Step 5: Khởi động lại server và chạy test**

Run:
```bash
./scripts/serve.sh asr_scorer     # terminal 1
.venv/bin/pytest tests/test_asr_scorer.py -v
```
Expected: PASS

Nếu lỗi shape ở `joiner.run`, in ra shape thật để đối chiếu:
```bash
.venv/bin/python scripts/inspect_onnx.py model_repository/asr_scorer/1/joiner.onnx
```
Bản export nào nhận `(N, 1, 1, C)` thì sửa `_run_joiner` thành `enc_frame[:, None, None, :]` và `dec_out[:, None, None, :]`.

- [ ] **Step 6: Commit**

```bash
git add model_repository/asr_scorer/config.pbtxt model_repository/asr_scorer/1/model.py tests/test_asr_scorer.py
git commit -m "feat: model asr_scorer chạy greedy search với decoder/joiner ONNX"
```

---

## Task 6: `asr` ensemble và client ASR

**Files:**
- Create: `model_repository/asr/config.pbtxt`
- Create: `client/asr_client.py`
- Create: `tests/assets/sample_vi.wav`
- Test: `tests/test_asr.py`

**Interfaces:**
- Consumes: `asr_feature`, `asr_encoder`, `asr_scorer` từ Task 2/4/5; `pad_wav()` từ Task 4
- Produces: model Triton `asr`. Vào `WAV` FP32 `(B, 256000)` + `WAV_LEN` INT32 `(B, 1)`, ra `TRANSCRIPT` STRING `(B, 1)`.
- Produces: `client/asr_client.py` chạy được dạng `python client/asr_client.py <file.wav>`

- [ ] **Step 1: Chuẩn bị file audio mẫu**

Cần hai file audio tiếng Việt: một câu để test ASR, và một giọng mẫu cho TTS zero-shot ở Task 7. `scripts/prepare_assets.py` tải cả hai từ dataset công khai `doof-ferb/fpt_fosd` rồi chuyển về wav mono 16kHz.

```bash
.venv/bin/python scripts/prepare_assets.py
```

Sinh ra bốn file — mỗi wav đi kèm một txt chứa đúng câu đã nói:

```
tests/assets/sample_vi.wav              + sample_vi.txt
model_repository/tts/1/assets/prompt.wav + prompt.txt
```

Muốn dùng giọng của chính mình thì thu bằng `arecord -f S16_LE -r 16000 -c 1 -d 5 <file>.wav` rồi tự viết file `.txt` tương ứng — cấu trúc giống hệt, không phải sửa code.

Kiểm tra định dạng:
```bash
.venv/bin/python -c "import soundfile as sf; d,r = sf.read('tests/assets/sample_vi.wav'); print(r, d.shape, d.ndim)"
```
Expected: `16000`, mảng một chiều, độ dài vài giây.

- [ ] **Step 2: Viết test thất bại `tests/test_asr.py`**

```python
# ABOUTME: Integration test đầu-cuối cho ensemble asr
# ABOUTME: Gửi file wav tiếng Việt thật, kiểm tra transcript khớp nội dung đã nói

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.common import SAMPLE_RATE, pad_wav  # noqa: E402

pytestmark = pytest.mark.integration

ASSETS = Path(__file__).parent / "assets"


def test_nhan_dang_cau_tieng_viet(triton):
    import tritonclient.grpc as grpcclient

    wav, sr = sf.read(ASSETS / "sample_vi.wav", dtype="float32")
    assert sr == SAMPLE_RATE
    mong_doi = (ASSETS / "sample_vi.txt").read_text(encoding="utf-8").strip().lower()

    padded, real_len = pad_wav(wav)
    inputs = [
        grpcclient.InferInput("WAV", [1, len(padded)], "FP32"),
        grpcclient.InferInput("WAV_LEN", [1, 1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(padded.reshape(1, -1))
    inputs[1].set_data_from_numpy(np.array([[real_len]], dtype=np.int32))

    out = triton.infer("asr", inputs)
    thuc_te = out.as_numpy("TRANSCRIPT")[0, 0].decode("utf-8").strip().lower()

    assert thuc_te, "transcript rỗng"

    # So theo từ thay vì so nguyên chuỗi - cho phép sai vài từ nhưng không được sai hết
    tu_mong_doi = set(mong_doi.split())
    tu_thuc_te = set(thuc_te.split())
    trung = len(tu_mong_doi & tu_thuc_te) / len(tu_mong_doi)
    assert trung >= 0.6, f"mong đợi '{mong_doi}', nhận được '{thuc_te}'"
```

- [ ] **Step 3: Chạy test để xác nhận nó đỏ**

Run: `.venv/bin/pytest tests/test_asr.py -v`
Expected: FAIL — model `asr` chưa tồn tại.

- [ ] **Step 4: Viết `model_repository/asr/config.pbtxt`**

```protobuf
name: "asr"
platform: "ensemble"
max_batch_size: 8

input [
  {
    name: "WAV"
    data_type: TYPE_FP32
    dims: [-1]
  },
  {
    name: "WAV_LEN"
    data_type: TYPE_INT32
    dims: [1]
  }
]

output [
  {
    name: "TRANSCRIPT"
    data_type: TYPE_STRING
    dims: [1]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "asr_feature"
      model_version: -1
      input_map  { key: "WAV"        value: "WAV" }
      input_map  { key: "WAV_LEN"    value: "WAV_LEN" }
      output_map { key: "SPEECH"     value: "speech" }
      output_map { key: "SPEECH_LEN" value: "speech_len" }
    },
    {
      model_name: "asr_encoder"
      model_version: -1
      input_map  { key: "x"                value: "speech" }
      input_map  { key: "x_lens"           value: "speech_len" }
      output_map { key: "encoder_out"      value: "enc" }
      output_map { key: "encoder_out_lens" value: "enc_len" }
    },
    {
      model_name: "asr_scorer"
      model_version: -1
      input_map  { key: "ENCODER_OUT"     value: "enc" }
      input_map  { key: "ENCODER_OUT_LEN" value: "enc_len" }
      output_map { key: "TRANSCRIPT"      value: "TRANSCRIPT" }
    }
  ]
}
```

Cách đọc `input_map` / `output_map` — chỗ này rất dễ nhầm:
- `key` = tên cổng **của model con**, phải khớp `config.pbtxt` của nó
- `value` = tên **sợi dây** trong ensemble, tự đặt
- Nối dây bằng cách đặt trùng tên `value`: `asr_feature` xuất ra dây `speech`, `asr_encoder` nhận vào dây `speech`

Lưu ý kiểu dữ liệu: `speech_len` do `asr_feature` xuất ra là INT64, và `asr_encoder` nhận `x_lens` cũng INT64 — khớp. Nếu lệch kiểu thì Triton từ chối load ensemble ngay lúc khởi động.

- [ ] **Step 5: Chạy toàn bộ 4 model và chạy test**

Run:
```bash
./scripts/serve.sh     # terminal 1 - không truyền tham số nên load hết
.venv/bin/pytest tests/test_asr.py -v
```
Expected: PASS

Nếu ensemble không load, đọc log tìm dòng `ensemble ... input/output mismatch` — luôn là sai tên hoặc sai kiểu ở `input_map`/`output_map`.

- [ ] **Step 6: Viết `client/asr_client.py`**

```python
# ABOUTME: Client dòng lệnh cho ASR - nhận file wav, in ra transcript
# ABOUTME: Chạy: python client/asr_client.py file.wav

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.common import SAMPLE_RATE, pad_wav  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="file wav mono 16kHz")
    ap.add_argument("--url", default="localhost:8001")
    args = ap.parse_args()

    wav, sr = sf.read(args.wav, dtype="float32")
    if sr != SAMPLE_RATE:
        raise SystemExit(f"Cần {SAMPLE_RATE}Hz, file này {sr}Hz")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)   # trộn stereo về mono

    padded, real_len = pad_wav(wav)

    client = grpcclient.InferenceServerClient(args.url)
    inputs = [
        grpcclient.InferInput("WAV", [1, len(padded)], "FP32"),
        grpcclient.InferInput("WAV_LEN", [1, 1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(padded.reshape(1, -1))
    inputs[1].set_data_from_numpy(np.array([[real_len]], dtype=np.int32))

    out = client.infer("asr", inputs)
    print(out.as_numpy("TRANSCRIPT")[0, 0].decode("utf-8"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Chạy thử client**

Run: `.venv/bin/python client/asr_client.py tests/assets/sample_vi.wav`
Expected: in ra câu tiếng Việt đã thu.

- [ ] **Step 8: Commit**

```bash
git add model_repository/asr/config.pbtxt client/asr_client.py tests/test_asr.py tests/assets
git commit -m "feat: ensemble asr nối 3 tầng và client dòng lệnh"
```

---

## Task 7: Chạy được ZipVoice ngoài Triton

Đây là bước gỡ rủi ro lớn nhất của spec (mục 11). **Không viết `tts/1/model.py` trước khi task này xanh.** Nếu checkpoint hoặc tokenizer có vấn đề thì phải biết ngay tại đây, chứ không phải lúc đang debug lồng trong Triton.

**Files:**
- Create: `scripts/smoke_zipvoice.sh`

**Interfaces:**
- Consumes: `model_repository/tts/1/{model.pt,tokens.txt,config.json}` và `vocos/` từ Task 1
- Produces: xác nhận bộ tham số CLI đúng cho checkpoint tiếng Việt, và một file wav nghe được

- [ ] **Step 1: Kiểm tra prompt audio cho zero-shot**

ZipVoice là zero-shot nên bắt buộc phải có giọng mẫu. File này đã được `scripts/prepare_assets.py` sinh ra ở Task 6 Step 1:

```bash
ls -l model_repository/tts/1/assets/prompt.wav model_repository/tts/1/assets/prompt.txt
cat model_repository/tts/1/assets/prompt.txt
```
Expected: wav khoảng 4–5 giây và một câu tiếng Việt khớp nội dung.

- [ ] **Step 2: Viết `scripts/smoke_zipvoice.sh`**

```bash
#!/usr/bin/env bash
# ABOUTME: Chạy thử ZipVoice bằng CLI gốc, bên ngoài Triton
# ABOUTME: Mục đích là xác nhận checkpoint + tokenizer + vocoder hoạt động trước khi bọc vào Triton

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker run --gpus all --rm \
  -v "$ROOT/model_repository/tts/1:/tts" \
  triton-voice \
  python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --model-dir /tts \
    --checkpoint-name model.pt \
    --tokenizer espeak \
    --lang vi \
    --vocoder-path /tts/vocos \
    --prompt-wav /tts/assets/prompt.wav \
    --prompt-text "$(cat "$ROOT/model_repository/tts/1/assets/prompt.txt")" \
    --text "Xin chào, tôi là trợ lý ảo." \
    --res-wav-path /tts/assets/smoke_out.wav \
    --num-step 8
```

- [ ] **Step 3: Chạy và nghe kết quả**

Run:
```bash
chmod +x scripts/smoke_zipvoice.sh && ./scripts/smoke_zipvoice.sh
aplay model_repository/tts/1/assets/smoke_out.wav
```
Expected: file wav 24kHz, nghe ra tiếng Việt, giọng giống prompt.

**Nếu hỏng, xử lý theo lỗi:**

| Lỗi | Xử lý |
|---|---|
| `--lang` không tồn tại | Bỏ `--lang vi`, kiểm tra lại bằng `docker run --rm triton-voice python3 -m zipvoice.bin.infer_zipvoice --help` rồi dùng đúng tên tham số |
| Thiếu key trong `config.json` | Đối chiếu `config.json` của hynt với file mẫu trong `/opt/ZipVoice/egs/`, bổ sung key thiếu |
| `espeak-ng` không có tiếng Việt | Chạy `docker run --rm triton-voice espeak-ng --voices=vi` để xác nhận, thiếu thì thêm gói `espeak-ng-data` vào Dockerfile |
| Checkpoint không load | Thử `--model-name zipvoice_distill` — repo hynt ghi là "compact" nên có thể là bản distill |

- [ ] **Step 4: Ghi lại bộ tham số đã chạy được**

Chép nguyên bộ tham số CLI vừa thành công vào cuối `README.md` dưới mục `## Tham số ZipVoice đã xác nhận`. Task 8 sẽ dùng đúng bộ này.

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_zipvoice.sh README.md
git commit -m "chore: smoke test ZipVoice ngoài Triton, xác nhận tham số cho checkpoint tiếng Việt"
```

---

## Task 8: Model `tts` và client TTS

**Files:**
- Create: `model_repository/tts/config.pbtxt`
- Create: `model_repository/tts/1/model.py`
- Create: `client/tts_client.py`
- Test: `tests/test_tts.py`

**Interfaces:**
- Consumes: bộ tham số đã xác nhận ở Task 7
- Produces: model Triton `tts`. Vào `TEXT` STRING `(1,)`, tuỳ chọn `PROMPT_WAV` FP32 `(-1,)`, `PROMPT_TEXT` STRING `(1,)`, `NUM_STEPS` INT32 `(1,)`. Ra `WAV` FP32 `(-1,)` + `SAMPLE_RATE` INT32 `(1,)`.

- [ ] **Step 1: Viết test thất bại `tests/test_tts.py`**

```python
# ABOUTME: Integration test cho model tts
# ABOUTME: Sinh audio từ câu tiếng Việt, kiểm tra định dạng và độ dài hợp lý

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def _goi_tts(triton, text, num_steps=8):
    import tritonclient.grpc as grpcclient

    text_np = np.array([text.encode("utf-8")], dtype=object)
    steps_np = np.array([num_steps], dtype=np.int32)

    inputs = [
        grpcclient.InferInput("TEXT", [1], "BYTES"),
        grpcclient.InferInput("NUM_STEPS", [1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(text_np)
    inputs[1].set_data_from_numpy(steps_np)

    out = triton.infer("tts", inputs)
    return out.as_numpy("WAV"), int(out.as_numpy("SAMPLE_RATE")[0])


def test_sinh_audio_hop_le(triton):
    wav, sr = _goi_tts(triton, "Xin chào, tôi là trợ lý ảo.")

    assert sr == 24000
    assert wav.ndim == 1 and len(wav) > 0
    assert not np.isnan(wav).any(), "audio có NaN"
    assert np.abs(wav).max() <= 1.0, "audio vượt biên độ"
    assert np.abs(wav).max() > 0.01, "audio gần như im lặng"

    # Câu ~26 ký tự thì độ dài hợp lý nằm trong khoảng 1 đến 8 giây
    assert 1.0 < len(wav) / sr < 8.0


def test_cau_dai_hon_thi_audio_dai_hon(triton):
    ngan, sr = _goi_tts(triton, "Xin chào.")
    dai, _ = _goi_tts(
        triton,
        "Xin chào, hôm nay trời rất đẹp và tôi muốn đi dạo một vòng quanh hồ.",
    )
    assert len(dai) > len(ngan)
```

- [ ] **Step 2: Chạy test để xác nhận nó đỏ**

Run: `.venv/bin/pytest tests/test_tts.py -v`
Expected: FAIL — model `tts` chưa tồn tại.

- [ ] **Step 3: Viết `model_repository/tts/config.pbtxt`**

```protobuf
name: "tts"
backend: "python"
max_batch_size: 0

input [
  {
    name: "TEXT"
    data_type: TYPE_STRING
    dims: [1]
  },
  {
    name: "PROMPT_WAV"
    data_type: TYPE_FP32
    dims: [-1]
    optional: true
  },
  {
    name: "PROMPT_TEXT"
    data_type: TYPE_STRING
    dims: [1]
    optional: true
  },
  {
    name: "NUM_STEPS"
    data_type: TYPE_INT32
    dims: [1]
    optional: true
  }
]

output [
  {
    name: "WAV"
    data_type: TYPE_FP32
    dims: [-1]
  },
  {
    name: "SAMPLE_RATE"
    data_type: TYPE_INT32
    dims: [1]
  }
]

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]

# Tham số của ZipVoice. Bản "zipvoice" mặc định num_step=16/guidance_scale=1.0,
# bản "zipvoice_distill" là 8/3.0 - đổi ở đây chứ không sửa code.
parameters [
  {
    key: "model_name"
    value: { string_value: "zipvoice" }
  },
  {
    key: "lang"
    value: { string_value: "vi" }
  },
  {
    key: "num_step"
    value: { string_value: "16" }
  },
  {
    key: "guidance_scale"
    value: { string_value: "1.0" }
  }
]
```

`max_batch_size: 0` tắt batching hoàn toàn — mỗi câu dài ngắn khác nhau, gom lại phải đệm rất phí. Độ song song lấy qua `instance_group`, và đó chính là biến số của thí nghiệm E2.

Giá trị mặc định đọc từ source ZipVoice (`model_defaults` trong `infer_zipvoice.py`): bản `zipvoice` là `num_step=16, guidance_scale=1.0`; bản `zipvoice_distill` là `8` và `3.0`. Đưa ra `parameters` để Task 7 phát hiện checkpoint là bản nào thì chỉ sửa config, không sửa code.

- [ ] **Step 4: Viết `model_repository/tts/1/model.py`**

Gọi thẳng `generate_sentence` của ZipVoice và ghi vào `/dev/shm` (tmpfs, nằm trong RAM nên không có I/O đĩa). Đây là lựa chọn có chủ ý: chép lại phần tiền/hậu xử lý của thư viện vào đây sẽ nhân đôi logic và vỡ mỗi lần upstream đổi.

```python
# ABOUTME: Triton Python backend - sinh giọng nói tiếng Việt bằng ZipVoice
# ABOUTME: Nạp model một lần trong initialize, mỗi request chạy flow matching rồi vocoder

import json
import os
import uuid

import numpy as np
import soundfile as sf
import torch
import triton_python_backend_utils as pb_utils
from vocos import Vocos
from zipvoice.bin.infer_zipvoice import generate_sentence, get_vocoder
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import EspeakTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank

SAMPLING_RATE = 24000
TMP = "/dev/shm"   # tmpfs trong RAM, không chạm đĩa


class TritonPythonModel:
    def initialize(self, args):
        d = os.path.join(args["model_repository"], args["model_version"])
        self.device = torch.device("cuda")

        with open(os.path.join(d, "config.json")) as f:
            model_config = json.load(f)

        self.tokenizer = EspeakTokenizer(
            token_file=os.path.join(d, "tokens.txt"), lang="vi"
        )
        tokenizer_config = {
            "vocab_size": self.tokenizer.vocab_size,
            "pad_id": self.tokenizer.pad_id,
        }

        self.model = ZipVoice(**model_config["model"], **tokenizer_config)
        load_checkpoint(
            filename=os.path.join(d, "model.pt"), model=self.model, strict=True
        )
        self.model = self.model.to(self.device).eval()

        self.vocoder = get_vocoder(os.path.join(d, "vocos")).to(self.device).eval()
        self.feature_extractor = VocosFbank()

        # Prompt mặc định cho zero-shot, đóng gói sẵn trong repo
        self.prompt_wav = os.path.join(d, "assets", "prompt.wav")
        with open(os.path.join(d, "assets", "prompt.txt"), encoding="utf-8") as f:
            self.prompt_text = f.read().strip()

    @staticmethod
    def _lay(request, ten, mac_dinh=None):
        """Đọc input tuỳ chọn - client không gửi thì trả về giá trị mặc định."""
        t = pb_utils.get_input_tensor_by_name(request, ten)
        return mac_dinh if t is None else t.as_numpy()

    def execute(self, requests):
        responses = []

        for request in requests:
            text = self._lay(request, "TEXT")[0].decode("utf-8")
            num_steps = int(self._lay(request, "NUM_STEPS", np.array([8]))[0])

            prompt_wav_arr = self._lay(request, "PROMPT_WAV")
            prompt_text_arr = self._lay(request, "PROMPT_TEXT")

            tam = []
            if prompt_wav_arr is not None:
                # Client gửi giọng mẫu riêng - ghi ra tmpfs vì ZipVoice nhận đường dẫn
                prompt_wav = f"{TMP}/prompt_{uuid.uuid4().hex}.wav"
                sf.write(prompt_wav, prompt_wav_arr.reshape(-1), 16000)
                tam.append(prompt_wav)
                prompt_text = prompt_text_arr[0].decode("utf-8")
            else:
                prompt_wav = self.prompt_wav
                prompt_text = self.prompt_text

            ket_qua = f"{TMP}/tts_{uuid.uuid4().hex}.wav"
            tam.append(ket_qua)

            try:
                with torch.inference_mode():
                    generate_sentence(
                        save_path=ket_qua,
                        prompt_text=prompt_text,
                        prompt_wav=prompt_wav,
                        text=text,
                        model=self.model,
                        vocoder=self.vocoder,
                        tokenizer=self.tokenizer,
                        feature_extractor=self.feature_extractor,
                        device=self.device,
                        num_step=num_steps,
                        guidance_scale=1.0,
                        speed=1.0,
                        t_shift=0.5,
                        target_rms=0.1,
                        feat_scale=0.1,
                        sampling_rate=SAMPLING_RATE,
                        max_duration=1000,
                        remove_long_sil=False,
                    )
                wav, _ = sf.read(ket_qua, dtype="float32")
            finally:
                for f in tam:
                    if os.path.exists(f):
                        os.remove(f)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("WAV", wav.reshape(-1).astype(np.float32)),
                        pb_utils.Tensor(
                            "SAMPLE_RATE", np.array([SAMPLING_RATE], dtype=np.int32)
                        ),
                    ]
                )
            )

        return responses
```

Các giá trị `guidance_scale`, `t_shift`, `target_rms`, `feat_scale` phải lấy đúng theo giá trị mặc định in ra từ `--help` ở Task 7 Step 3. Nếu khác thì sửa lại cho khớp.

- [ ] **Step 5: Khởi động lại server và chạy test**

Run:
```bash
./scripts/serve.sh     # terminal 1
.venv/bin/pytest tests/test_tts.py -v
```
Expected: PASS

Nếu lỗi import ở `initialize`, xem log Triton — Python backend in nguyên traceback. Đối chiếu tên class với `docker run --rm triton-voice python -c "import zipvoice.models.zipvoice as m; print(dir(m))"`.

- [ ] **Step 6: Viết `client/tts_client.py`**

```python
# ABOUTME: Client dòng lệnh cho TTS - nhận text, ghi ra file wav
# ABOUTME: Chạy: python client/tts_client.py --text "Xin chào" --out ra.wav

import argparse

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="ra.wav")
    ap.add_argument("--prompt", help="file wav giọng mẫu 16kHz cho zero-shot")
    ap.add_argument("--prompt-text", help="nội dung đã nói trong file giọng mẫu")
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--url", default="localhost:8001")
    args = ap.parse_args()

    inputs = [
        grpcclient.InferInput("TEXT", [1], "BYTES"),
        grpcclient.InferInput("NUM_STEPS", [1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(np.array([args.text.encode("utf-8")], dtype=object))
    inputs[1].set_data_from_numpy(np.array([args.num_steps], dtype=np.int32))

    if args.prompt:
        if not args.prompt_text:
            raise SystemExit("Dùng --prompt thì phải kèm --prompt-text")
        wav, sr = sf.read(args.prompt, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        pw = grpcclient.InferInput("PROMPT_WAV", [len(wav)], "FP32")
        pw.set_data_from_numpy(wav.astype(np.float32))
        pt = grpcclient.InferInput("PROMPT_TEXT", [1], "BYTES")
        pt.set_data_from_numpy(
            np.array([args.prompt_text.encode("utf-8")], dtype=object)
        )
        inputs += [pw, pt]

    client = grpcclient.InferenceServerClient(args.url)
    out = client.infer("tts", inputs)

    wav = out.as_numpy("WAV")
    sr = int(out.as_numpy("SAMPLE_RATE")[0])
    sf.write(args.out, wav, sr)
    print(f"đã ghi {args.out} — {len(wav) / sr:.2f}s @ {sr}Hz")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Chạy thử cả hai chế độ**

Run:
```bash
.venv/bin/python client/tts_client.py --text "Xin chào, tôi là trợ lý ảo." --out ra.wav
aplay ra.wav

.venv/bin/python client/tts_client.py --text "Hôm nay trời đẹp quá." \
  --prompt tests/assets/sample_vi.wav \
  --prompt-text "$(cat tests/assets/sample_vi.txt)" --out clone.wav
aplay clone.wav
```
Expected: `ra.wav` dùng giọng mặc định, `clone.wav` nghe giống giọng trong `sample_vi.wav`.

- [ ] **Step 8: Chạy toàn bộ test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: tất cả xanh. Kiểm tra lại phần unit chạy được độc lập: `.venv/bin/pytest tests/ -m "not integration" -v`

- [ ] **Step 9: Commit**

```bash
git add model_repository/tts client/tts_client.py tests/test_tts.py
git commit -m "feat: model tts chạy ZipVoice và client dòng lệnh có clone giọng"
```

---

## Task 9: Benchmark E1–E3

**Files:**
- Create: `bench/bench.py`
- Create: `bench/input_asr.json`
- Create: `bench/README.md`

**Interfaces:**
- Consumes: model `asr` và `tts` đang chạy
- Produces: `bench/results/results.csv` với các cột `experiment,variant,concurrency,throughput_rps,p50_ms,p95_ms,vram_mb`

- [ ] **Step 1: Sinh file input cho `perf_analyzer`**

`perf_analyzer` cần dữ liệu mẫu đúng shape. Viết `bench/gen_input.py`:

```python
# ABOUTME: Sinh file JSON đầu vào cho perf_analyzer từ file wav mẫu
# ABOUTME: perf_analyzer cần dữ liệu thật đúng shape, không tự sinh được

import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.common import pad_wav  # noqa: E402

wav, sr = sf.read("tests/assets/sample_vi.wav", dtype="float32")
padded, real_len = pad_wav(wav)

data = {
    "data": [
        {
            "WAV": padded.tolist(),
            "WAV_LEN": [real_len],
        }
    ]
}
Path("bench/input_asr.json").write_text(json.dumps(data))
print("đã ghi bench/input_asr.json")
```

Run: `.venv/bin/python bench/gen_input.py`

- [ ] **Step 2: Viết `bench/bench.py`**

> **Bản đã triển khai khác đoạn dưới đây.** File thật ở `bench/bench.py` tự động hoá thêm ba việc mà bản phác thảo này còn làm tay: sửa `config.pbtxt` bằng regex, khởi động lại container rồi chờ `/v2/health/ready`, và đọc batch trung bình thật từ `/v2/models/asr_encoder/stats`. Đoạn code dưới đây giữ lại làm tham chiếu cho phần phân tích output của `perf_analyzer`.

```python
# ABOUTME: Điều phối perf_analyzer cho ba thí nghiệm E1-E3, gom kết quả ra CSV
# ABOUTME: Chạy: python bench/bench.py e1  (hoặc e2, e3)

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

RESULTS = Path("bench/results")
RESULTS.mkdir(parents=True, exist_ok=True)
CSV = RESULTS / "results.csv"
COLS = [
    "experiment",
    "variant",
    "concurrency",
    "throughput_rps",
    "p50_ms",
    "p95_ms",
    "vram_mb",
]


def vram_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def perf(model: str, concurrency: int, input_file: str | None) -> dict:
    """Chạy perf_analyzer trong container, trả về throughput và latency."""
    cmd = [
        "docker", "run", "--rm", "--net", "host",
        "-v", f"{Path.cwd()}:/w", "-w", "/w",
        "triton-voice",
        "perf_analyzer", "-m", model, "-u", "localhost:8001", "-i", "grpc",
        "--concurrency-range", f"{concurrency}:{concurrency}",
        "--measurement-interval", "10000",
        "-f", "/tmp/pa.csv",
    ]
    if input_file:
        cmd += ["--input-data", input_file]

    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout

    tp = re.search(r"Throughput:\s+([\d.]+)\s+infer/sec", text)
    p50 = re.search(r"p50 latency:\s+(\d+)\s+usec", text)
    p95 = re.search(r"p95 latency:\s+(\d+)\s+usec", text)
    if not tp:
        print(text, out.stderr, file=sys.stderr)
        raise SystemExit(f"perf_analyzer hỏng cho {model} @ {concurrency}")

    return {
        "throughput_rps": float(tp.group(1)),
        "p50_ms": int(p50.group(1)) / 1000 if p50 else 0,
        "p95_ms": int(p95.group(1)) / 1000 if p95 else 0,
        "vram_mb": vram_mb(),
    }


def ghi(rows):
    moi = not CSV.exists()
    with CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if moi:
            w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


def e1():
    """Dynamic batching: đổi max_queue_delay_microseconds, đo ở concurrency 8.

    Trước mỗi lần chạy phải sửa tay giá trị trong
    model_repository/asr_encoder/config.pbtxt rồi khởi động lại server.
    Script này chỉ đo, không tự sửa config.
    """
    delay = input("max_queue_delay hiện tại đang đặt là bao nhiêu us? ").strip()
    r = perf("asr", 8, "bench/input_asr.json")
    ghi([{"experiment": "e1", "variant": f"delay={delay}us", "concurrency": 8, **r}])


def e2():
    """Model concurrency của TTS: đổi instance_group.count, quét concurrency client."""
    count = input("instance_group.count của tts đang đặt là bao nhiêu? ").strip()
    rows = []
    for c in (1, 2, 4, 8):
        r = perf("tts", c, None)
        rows.append(
            {"experiment": "e2", "variant": f"instances={count}", "concurrency": c, **r}
        )
    ghi(rows)


def e3():
    """Ensemble breakdown: đọc metrics per-model để tìm tầng nghẽn."""
    import urllib.request

    perf("asr", 8, "bench/input_asr.json")   # tạo tải rồi mới đọc metrics

    raw = urllib.request.urlopen("http://localhost:8002/metrics").read().decode()
    ket_qua = {}
    for line in raw.splitlines():
        for metric in ("nv_inference_queue_duration_us",
                       "nv_inference_compute_infer_duration_us",
                       "nv_inference_request_success"):
            if line.startswith(metric) and "asr" in line:
                model = re.search(r'model="([^"]+)"', line).group(1)
                ket_qua.setdefault(model, {})[metric] = float(line.split()[-1])

    print(json.dumps(ket_qua, indent=2, ensure_ascii=False))
    (RESULTS / "e3_metrics.json").write_text(
        json.dumps(ket_qua, indent=2, ensure_ascii=False)
    )

    print("\nThời gian trung bình mỗi request (ms):")
    for model, m in sorted(ket_qua.items()):
        n = m.get("nv_inference_request_success", 0) or 1
        q = m.get("nv_inference_queue_duration_us", 0) / n / 1000
        c = m.get("nv_inference_compute_infer_duration_us", 0) / n / 1000
        print(f"  {model:15s} chờ {q:7.2f}  tính {c:7.2f}")


if __name__ == "__main__":
    {"e1": e1, "e2": e2, "e3": e3}[sys.argv[1]]()
```

- [ ] **Step 3: Chạy E1 — dynamic batching**

Run: `.venv/bin/python bench/bench.py e1`

Script tự làm cả vòng lặp: sửa `max_queue_delay_microseconds` qua {0, 1000, 5000, 20000}, khởi động lại container, chờ `/v2/health/ready`, đo, rồi trả config về mặc định 5000.

Mỗi mức delay đo hai lần — gọi thẳng `asr_encoder` và gọi qua ensemble `asr` — vì hai cột này kể hai câu chuyện khác nhau (xem E3).

Batch trung bình thật lấy từ `/v2/models/asr_encoder/stats`, tính bằng hiệu `inference_count / execution_count` trước và sau mỗi lần đo. Không lấy từ output perf_analyzer: nó chỉ in breakdown khi model là ensemble.

- [ ] **Step 4: Chạy E2 — model concurrency của TTS**

Run: `.venv/bin/python bench/bench.py e2`

Cũng tự động: quét `instance_group.count` qua {1, 2, 4}, mỗi mức đo ở concurrency client {1, 2, 4}.

Expected: throughput tăng theo `count` cho tới khi chạm trần 4GB VRAM. Ghi lại điểm gãy — đó là số liệu có giá trị, không phải thất bại.

- [ ] **Step 5: Chạy E3 — ensemble breakdown**

Run: `.venv/bin/python bench/bench.py e3`
Expected: in ra bảng chờ/tính cho từng model con, chỉ rõ tầng nào là nút cổ chai.

- [ ] **Step 6: Viết `bench/README.md`**

Ghi lại: cách chạy từng thí nghiệm, bảng kết quả thực tế đo được, và một đoạn nhận xét cho mỗi thí nghiệm trả lời đúng câu hỏi *"vặn nút này thì số nào đổi, và vì sao"*.

- [ ] **Step 7: Commit**

```bash
git add bench
git commit -m "feat: benchmark E1-E3 cho batching, concurrency và ensemble breakdown"
```

---

## Self-Review

**Spec coverage** — đối chiếu từng mục của spec:

| Mục spec | Task |
|---|---|
| §3 Môi trường Docker | Task 1 |
| §4 Tài sản model, vocos, espeak-ng | Task 1 |
| §5 ASR ensemble 4 model | Task 2, 4, 5, 6 |
| §5 TTS Python backend | Task 7, 8 |
| §6 Interface `asr` | Task 6 |
| §6 Interface `tts` | Task 8 |
| §7 Bảng cấu hình | Task 2, 4, 5, 8 |
| §7 Đệm độ dài cố định | Task 4 (`client/common.py`) |
| §8 Cấu trúc thư mục | rải đều |
| §9 `test_greedy_search` unit | Task 3 |
| §9 `test_asr` integration | Task 6 |
| §9 `test_tts` integration | Task 8 |
| §10 E1, E2, E3 | Task 9 |
| §11 Rủi ro ZipVoice API | Task 7 (gỡ trước khi viết model.py) |
| §11 Rủi ro tên cổng ONNX | Task 1 Step 7 (bắt buộc, chặn các task sau) |
| §11 Rủi ro 4GB VRAM | Task 9 Step 4 |

Không có mục nào của spec thiếu task.

**Type consistency** — tên và kiểu dùng xuyên suốt:

| Tên | Kiểu | Nơi khai | Nơi dùng |
|---|---|---|---|
| `WAV` | FP32 `(B, 256000)` | Task 4 config | Task 6 ensemble, client, test |
| `WAV_LEN` | INT32 `(B, 1)` | Task 4 config | Task 6 ensemble, client, test |
| `SPEECH` | FP32 `(B, 1600, 80)` | Task 4 config | Task 6 dây `speech` |
| `SPEECH_LEN` | INT64 `(B, 1)` | Task 4 config | Task 6 dây `speech_len` → `x_lens` INT64, khớp |
| `ENCODER_OUT` | FP32 `(B, T', C)` | Task 5 config | Task 6 dây `enc` |
| `ENCODER_OUT_LEN` | INT64 `(B, 1)` | Task 5 config | Task 6 dây `enc_len` |
| `TRANSCRIPT` | STRING `(B, 1)` | Task 5 config | Task 6 ensemble output |
| `greedy_search()` | xem Task 3 Interfaces | Task 3 | Task 5 `model.py` |
| `pad_wav()` | `(np.ndarray) -> (np.ndarray, int)` | Task 4 | Task 6 test + client, Task 9 gen_input |
| `MAX_FRAMES = 1600` | | Task 4 `common.py` | Task 4 `config.pbtxt` parameter, Task 4 test |

**Ghi chú về thứ tự phụ thuộc:** Task 1 Step 7 (in tên cổng ONNX) chặn Task 2 và Task 5. Task 7 chặn Task 8. Các task còn lại theo thứ tự tuyến tính.

**Ngoài phạm vi plan này** — đã ghi ở §12 của spec, không làm ở vòng này: export ZipVoice sang ONNX + BLS, TensorRT, ragged batching, model int8, streaming ASR, port Jetson Thor. Trang Gradio đã bàn nhưng chưa chốt, cũng không nằm trong plan.
