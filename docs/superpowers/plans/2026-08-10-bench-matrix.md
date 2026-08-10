# Benchmark Matrix (E4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Một lệnh `python bench/bench.py e4` sinh ma trận 5 component × CCU 1–4 × 7 metric (P50/P90/P95/P99/min/max/throughput) cho cả ASR và TTS.

**Architecture:** Thêm thí nghiệm `e4` vào `bench/bench.py`, dùng chung `restart_server()`/`vram_mb()`/`run_perf()` đã kiểm chứng qua E1–E3. Bốn percentile và throughput lấy từ stdout của perf_analyzer (nguồn có thẩm quyền); `min`/`max`/`samples` tính từ `--profile-export-file` bằng module thuần `bench/stats.py`. Kết quả ghi ra file riêng (`matrix.csv`, `matrix.md`) nên `results.csv` của E1–E3 không bị ảnh hưởng.

**Tech Stack:** Python 3.12, perf_analyzer 2.60.0 (`.venv/bin/perf_analyzer`), tritonclient 2.53.0 (gRPC), numpy 1.26.4, pytest.

**Spec:** `docs/superpowers/specs/2026-08-10-bench-matrix-design.md`

## Global Constraints

- Mọi file code mới bắt đầu bằng 2 dòng comment `# ABOUTME:`.
- Comment và chuỗi in ra viết tiếng Việt, khớp style `bench/bench.py` hiện có.
- Không sửa `COLUMNS`, `write_csv`, `print_table`, `e1`, `e1b`, `e2`, `e3` — E1–E3 phải chạy y như cũ.
- Không sửa bất kỳ `model_repository/*/config.pbtxt` nào. E4 đo cấu hình mặc định đang commit.
- `--stability-percentage 25` giữ nguyên (máy một GPU dùng chung với desktop).
- `numpy==1.26.4` đã có trong `requirements.txt`; không thêm dependency mới.
- Python dùng để chạy mọi lệnh là `.venv/bin/python`.
- Ngưỡng cảnh báo lệch percentile: **5%**.
- `STABLE_WINDOWS = 3`.
- Mức CCU đo: **1, 2, 3, 4**.

### Trần RAM — vì sao E4 chặn cứng số request

`--profile-export-file` nhúng nguyên `request_inputs` (cả tensor) vào **mỗi** bản ghi và giữ toàn bộ trong RAM của tiến trình client tới cuối run. Số đo thật trên máy này (`asr_encoder`, `/usr/bin/time -v`):

| | giá trị |
|---|---|
| 100 request | export 636 MB, **RSS đỉnh 2,1 GB** |
| suy ra | ~6,4 MB dữ liệu/request, ~21 MB RAM/request |

Cửa sổ 20 giây ở 25 rps = 500 request ≈ 10,5 GB cho **một** cửa sổ, mà vòng ổn định cần ≥ 3 cửa sổ và perf_analyzer không giải phóng request cũ → ≈ 31 GB. Máy có 15 GB. Một lần chạy theo cấu hình cũ đã bị kernel oom-kill ở 6,6 GB RSS.

Đây là **RAM hệ thống, không phải VRAM** — VRAM lúc load cả 6 model chỉ dùng 2,3/4,1 GB.

Vì vậy E4 dùng `--request-count N` (chặn cứng tổng số request) thay cho cửa sổ thời gian, N đặt theo cỡ tensor từng component để RSS đỉnh ~2 GB:

| component | tensor/request | `total_requests` |
|---|---|---|
| `asr` (ensemble) | 1,9 MB | 150 |
| `asr_feature` | 1,9 MB in + 1,5 MB out | 120 |
| `asr_encoder` | 6,4 MB (đo thật) | 100 |
| `asr_scorer` | ~4,3 MB | 120 |
| `tts` | 83 byte | dùng `request_count=16` như cũ, không cần chặn |

**Hai hệ quả phải ghi vào README ở Task 5, không được im lặng:**

1. P99 tính trên 100–150 mẫu chỉ là một hai request chậm nhất, không phải phân vị thật. Cột `samples` tồn tại để người đọc thấy điều đó.
2. `--request-count` tắt vòng lặp ổn định, nên `--stability-percentage 25` không còn tác dụng với E4. Số đo là một lát cắt cố định, không phải giá trị đã hội tụ.

---

### Task 1: `bench/stats.py` — tính min/max/percentile từ profile export

**Files:**
- Create: `bench/stats.py`
- Test: `tests/test_bench_stats.py`

**Interfaces:**
- Consumes: không gì (module thuần, chỉ dùng `json`, `pathlib`, `numpy`)
- Produces:
  - `STABLE_WINDOWS: int = 3`
  - `parse_profile_export(path: str | Path) -> list[float]` — latency ms của request trong 3 cửa sổ cuối; raise `ValueError` nếu rỗng
  - `latency_stats(latencies: Sequence[float]) -> dict` — khoá `p50_ms`, `p90_ms`, `p95_ms`, `p99_ms`, `min_ms`, `max_ms`, `samples`

- [ ] **Step 1: Write the failing tests**

Tạo `tests/test_bench_stats.py`:

```python
# ABOUTME: Unit test cho bench/stats.py - tính latency từ profile export
# ABOUTME: Không cần Triton server, dùng fixture JSON viết tay

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench.stats import latency_stats, parse_profile_export  # noqa: E402


def write_export(tmp_path, requests, boundaries):
    """Ghi file export tối giản theo schema của perf_analyzer 2.60.0."""
    path = tmp_path / "export.json"
    path.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment": {"mode": "concurrency", "value": 1},
                        "requests": requests,
                        "window_boundaries": boundaries,
                    }
                ],
                "version": "2.60.0",
            }
        )
    )
    return path


def test_latency_from_last_response_timestamp(tmp_path):
    # timestamp tính bằng nanosecond, kết quả phải ra millisecond
    path = write_export(
        tmp_path,
        [{"timestamp": 1_000_000, "response_timestamps": [3_000_000]}],
        [],
    )
    assert parse_profile_export(path) == [2.0]


def test_multiple_response_timestamps_uses_last(tmp_path):
    # Request có nhiều response (streaming): latency tính tới mốc cuối cùng
    path = write_export(
        tmp_path,
        [{"timestamp": 0, "response_timestamps": [1_000_000, 5_000_000]}],
        [],
    )
    assert parse_profile_export(path) == [5.0]


def test_requests_outside_stable_windows_excluded(tmp_path):
    # 5 mốc biên, STABLE_WINDOWS=3 -> mốc bắt đầu là boundaries[-4] = 200
    requests = [
        {"timestamp": 150, "response_timestamps": [150 + 9_000_000]},   # loại
        {"timestamp": 250, "response_timestamps": [250 + 1_000_000]},
        {"timestamp": 350, "response_timestamps": [350 + 2_000_000]},
        {"timestamp": 450, "response_timestamps": [450 + 3_000_000]},
    ]
    path = write_export(tmp_path, requests, [100, 200, 300, 400, 500])
    assert parse_profile_export(path) == [1.0, 2.0, 3.0]


def test_few_boundaries_keeps_all_requests(tmp_path):
    # Không đủ mốc biên để cắt 3 cửa sổ thì lấy tất, đừng trả rỗng
    requests = [
        {"timestamp": 0, "response_timestamps": [1_000_000]},
        {"timestamp": 10, "response_timestamps": [10 + 2_000_000]},
    ]
    path = write_export(tmp_path, requests, [100, 200])
    assert parse_profile_export(path) == [1.0, 2.0]


def test_empty_stable_window_raises(tmp_path):
    # Không mẫu nào thì phải nổ rõ ràng, không trả 0 âm thầm
    path = write_export(tmp_path, [], [])
    with pytest.raises(ValueError, match="không có request"):
        parse_profile_export(path)


def test_percentiles_on_known_array():
    # Mảng 1..100: numpy nội suy tuyến tính cho đáp án biết trước
    stats = latency_stats(list(range(1, 101)))
    assert stats["p50_ms"] == 50.5
    assert stats["p90_ms"] == 90.1
    assert stats["p95_ms"] == 95.05
    assert stats["p99_ms"] == 99.01
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 100.0


def test_reports_sample_count():
    # samples cần để đọc p99 trung thực: dưới ~100 mẫu thì p99 không phải phân vị thật
    assert latency_stats([1.0, 2.0, 3.0])["samples"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_bench_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench.stats'` (hoặc ImportError vì `bench/` chưa có `__init__.py`; xử lý ở Step 3)

- [ ] **Step 3: Write the implementation**

Tạo `bench/stats.py`:

```python
# ABOUTME: Tính min/max/percentile latency từ file profile export của perf_analyzer
# ABOUTME: Hàm thuần - không subprocess, không mạng, test được khi server tắt

import json
from pathlib import Path

import numpy as np

# perf_analyzer xét ổn định trên 3 cửa sổ đo cuối. Lấy đúng 3 cửa sổ đó thì
# thống kê tự tính mới so được với số nó in ra stdout (xem cross_check ở bench.py).
STABLE_WINDOWS = 3


def parse_profile_export(path) -> list[float]:
    """Latency (ms) của từng request trong các cửa sổ đo ổn định.

    perf_analyzer ghi timestamp theo nanosecond. Một request có thể có nhiều
    mốc response (streaming), latency tính tới mốc cuối cùng.
    """
    data = json.loads(Path(path).read_text())
    experiment = data["experiments"][0]
    requests = experiment["requests"]
    boundaries = experiment.get("window_boundaries") or []

    # Cần STABLE_WINDOWS+1 mốc để cắt ra STABLE_WINDOWS cửa sổ cuối. Ít hơn thì
    # phép đo chưa chạy đủ lâu, lấy tất còn hơn trả rỗng.
    if len(boundaries) > STABLE_WINDOWS:
        start = boundaries[-(STABLE_WINDOWS + 1)]
        requests = [r for r in requests if r["timestamp"] >= start]

    latencies = [
        (r["response_timestamps"][-1] - r["timestamp"]) / 1e6
        for r in requests
        if r.get("response_timestamps")
    ]
    if not latencies:
        raise ValueError(f"{path}: không có request nào trong cửa sổ ổn định")
    return latencies


def latency_stats(latencies) -> dict:
    """Sáu thống kê latency. samples để biết p99 có ý nghĩa hay không."""
    a = np.asarray(latencies, dtype=float)
    p50, p90, p95, p99 = np.percentile(a, [50, 90, 95, 99])
    return {
        "p50_ms": round(float(p50), 2),
        "p90_ms": round(float(p90), 2),
        "p95_ms": round(float(p95), 2),
        "p99_ms": round(float(p99), 2),
        "min_ms": round(float(a.min()), 2),
        "max_ms": round(float(a.max()), 2),
        "samples": int(a.size),
    }
```

Tạo `bench/__init__.py` rỗng để `from bench.stats import ...` chạy được:

```bash
touch bench/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bench_stats.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Verify E1–E3 vẫn import được**

`bench/__init__.py` mới có thể đổi cách Python phân giải `bench/bench.py`. Kiểm tra:

Run: `.venv/bin/python -c "import ast,sys; ast.parse(open('bench/bench.py').read()); print('ok')"`
Expected: `ok`

Run: `.venv/bin/python bench/bench.py 2>&1 | head -3`
Expected: `IndexError` hoặc `KeyError` vì thiếu `sys.argv[1]` — nghĩa là file vẫn nạp được, chưa hỏng import.

- [ ] **Step 6: Commit**

```bash
git add bench/stats.py bench/__init__.py tests/test_bench_stats.py
git commit -m "Add bench/stats.py for latency percentiles from profile export"
```

---

### Task 2: `gen_input.py` — bắt `ENCODER_OUT` thật cho `asr_scorer`

**Files:**
- Modify: `bench/gen_input.py` (thêm vào cuối file, sau khi 3 file offline đã ghi)
- Generates: `bench/input_scorer.json`

**Interfaces:**
- Consumes: `bench/stats.py` không liên quan. Dùng `client.common.pad_wav`, `MAX_FRAMES`, `NUM_MEL_BINS` đã import sẵn ở đầu file.
- Produces: `bench/input_scorer.json` với schema `{"data": [{"ENCODER_OUT": [<T*512 số>], "ENCODER_OUT_LEN": [T]}]}` — Task 4 đọc `ENCODER_OUT_LEN[0]` để suy `--shape`.

**Bối cảnh cần biết:** shape đã kiểm chứng trong `tests/test_asr_encoder.py` và `tests/test_asr_scorer.py`:
- `asr_feature`: vào `WAV [1, N]` FP32 + `WAV_LEN [1,1]` INT32 → ra `SPEECH [1, F, 80]` FP32 + `SPEECH_LEN` INT64
- `asr_encoder`: vào `x [1, 1600, 80]` FP32 + `x_lens [1,1]` INT64 → ra `encoder_out [1, T, 512]` FP32 (Zipformer subsample ~4 lần nên T ≈ 400)

- [ ] **Step 1: Thêm bước bắt input scorer**

Thêm vào cuối `bench/gen_input.py`:

```python
# Scorer nhận ENCODER_OUT - tensor trung gian, không tồn tại dưới dạng file.
# RNN-T greedy search chạy số bước phụ thuộc nội dung tensor (mỗi bước phát một
# token hoặc blank) nên tensor ngẫu nhiên cho thời gian giải mã không đại diện.
# Phải bắt tensor thật từ server, dùng đúng file wav mà E1-E3 đã dùng.
# Bước này để cuối cùng: ba file trên không cần server, hỏng ở đây vẫn còn dùng được.
import tritonclient.grpc as grpcclient  # noqa: E402

client = grpcclient.InferenceServerClient("localhost:8001")
try:
    ready = client.is_server_ready()
except Exception as e:
    raise SystemExit(
        f"Không kết nối được Triton tại localhost:8001 ({e}).\n"
        "Chạy scripts/serve.sh rồi gọi lại lệnh này để sinh input_scorer.json."
    )
if not ready:
    raise SystemExit(
        "Triton chưa sẵn sàng. Chạy scripts/serve.sh rồi gọi lại lệnh này."
    )

feature_inputs = [
    grpcclient.InferInput("WAV", [1, len(padded)], "FP32"),
    grpcclient.InferInput("WAV_LEN", [1, 1], "INT32"),
]
feature_inputs[0].set_data_from_numpy(padded.reshape(1, -1))
feature_inputs[1].set_data_from_numpy(np.array([[real_len]], dtype=np.int32))
feature_out = client.infer("asr_feature", feature_inputs)
speech = feature_out.as_numpy("SPEECH")
speech_len = feature_out.as_numpy("SPEECH_LEN").reshape(1, 1).astype(np.int64)

encoder_inputs = [
    grpcclient.InferInput("x", list(speech.shape), "FP32"),
    grpcclient.InferInput("x_lens", [1, 1], "INT64"),
]
encoder_inputs[0].set_data_from_numpy(speech)
encoder_inputs[1].set_data_from_numpy(speech_len)
encoder_out = client.infer("asr_encoder", encoder_inputs).as_numpy("encoder_out")

frames = encoder_out.shape[1]
(ROOT / "bench/input_scorer.json").write_text(
    json.dumps(
        {
            "data": [
                {
                    "ENCODER_OUT": encoder_out.reshape(-1).tolist(),
                    "ENCODER_OUT_LEN": [frames],
                }
            ]
        }
    )
)
print(f"đã ghi bench/input_scorer.json (ENCODER_OUT shape {frames}x512)")
```

- [ ] **Step 2: Khởi động server**

Run: `./scripts/serve.sh > /tmp/serve.log 2>&1 &` rồi chờ tới khi ready:

```bash
for i in $(seq 60); do
  curl -sf -o /dev/null http://localhost:8000/v2/health/ready && echo READY && break
  sleep 2
done
```
Expected: `READY`

- [ ] **Step 3: Chạy gen_input và kiểm tra kết quả**

Run: `.venv/bin/python bench/gen_input.py`
Expected: 4 dòng "đã ghi ...", dòng cuối in shape dạng `ENCODER_OUT shape 400x512` (T nằm trong khoảng 300–500)

- [ ] **Step 4: Xác minh file dùng được với perf_analyzer**

Đây là bước quan trọng nhất của task: file có đúng schema nhưng sai shape thì perf_analyzer mới báo lỗi.

```bash
T=$(.venv/bin/python -c "import json; print(json.load(open('bench/input_scorer.json'))['data'][0]['ENCODER_OUT_LEN'][0])")
echo "T=$T"
.venv/bin/perf_analyzer -m asr_scorer -u localhost:8001 -i grpc \
  --concurrency-range 1:1 --measurement-request-count 4 \
  --measurement-mode count_windows --stability-percentage 25 \
  --input-data bench/input_scorer.json --shape "ENCODER_OUT:$T,512" 2>&1 | tail -20
```
Expected: có dòng `Throughput: ... infer/sec` và `p50 latency: ... usec`, không có lỗi shape

- [ ] **Step 5: Commit**

Commit **cả** file input: `bench/input_asr.json` (1.9M) và `bench/input_encoder.json` (2.6M) đã được commit sẵn, nên `input_scorer.json` theo đúng convention đó. Lợi ích thật: chạy lại E4 không cần dựng server để sinh input.

```bash
git add bench/gen_input.py bench/input_scorer.json
git commit -m "Capture real ENCODER_OUT for asr_scorer benchmark input"
```

---

### Task 3: `run_perf()` lấy đủ 4 percentile + min/max, tự kiểm chứng chéo

**Files:**
- Modify: `bench/bench.py` — thay `run_perf()` (dòng 110–157), thêm `parse_summary()` và `cross_check()`

**Interfaces:**
- Consumes: `bench.stats.latency_stats`, `bench.stats.parse_profile_export` từ Task 1
- Produces:
  - `parse_summary(output: str) -> dict` — `{throughput_rps, p50_ms, p90_ms, p95_ms, p99_ms, request_count}`
  - `run_perf(model, concurrency, input_file, shape, interval_ms=20000, request_count=0, export_path=None, warmup=0) -> dict` — thêm khoá `min_ms`, `max_ms`, `samples` khi truyền `export_path`; các khoá cũ (`throughput_rps`, `p50_ms`, `p95_ms`, `vram_mb`, `_raw`) giữ nguyên tên để E1–E3 không hỏng

**Bối cảnh cần biết:** format summary của perf_analyzer, lấy từ `bench/results/e3_perf_analyzer.txt` dòng 14–19:

```
    Request count: 1726
    Throughput: 23.829 infer/sec
    p50 latency: 327724 usec
    p90 latency: 362057 usec
    p95 latency: 434014 usec
    p99 latency: 584353 usec
```

Regex throughput cũ (`throughput:\s+([\d.]+)\s+infer/sec`, chữ thường) khớp dòng tổng kết cuối file — giữ y nguyên để số E1–E3 không đổi.

- [ ] **Step 1: Thêm import**

Sửa phần import ở đầu `bench/bench.py`, thêm sau `from pathlib import Path`:

```python
from bench.stats import latency_stats, parse_profile_export
```

- [ ] **Step 2: Thêm `parse_summary()` và `cross_check()`**

Chèn ngay trước `def run_perf(` trong `bench/bench.py`:

```python
def parse_summary(output: str) -> dict:
    """Bóc throughput và bốn percentile từ stdout của perf_analyzer.

    Đây là nguồn có thẩm quyền cho percentile - không phải suy diễn từ timestamp
    nên không phụ thuộc cách cắt cửa sổ ổn định.
    """
    throughput = re.search(r"throughput:\s+([\d.]+)\s+infer/sec", output)
    if not throughput:
        return {}
    result = {"throughput_rps": float(throughput.group(1))}
    for q in (50, 90, 95, 99):
        m = re.search(rf"p{q} latency:\s+(\d+)\s+usec", output)
        result[f"p{q}_ms"] = round(int(m.group(1)) / 1000, 2) if m else 0
    count = re.search(r"Request count:\s+(\d+)", output)
    result["request_count"] = int(count.group(1)) if count else 0
    return result


def cross_check(computed: dict, reported: dict, label: str) -> None:
    """So percentile tự tính với số perf_analyzer in ra.

    Bốn điểm so trên cùng một phân bố: nếu khớp thì tập mẫu dùng cho min/max
    đúng là tập perf_analyzer đã báo cáo. Lệch nghĩa là cắt cửa sổ ổn định sai,
    lúc đó min/max mới là số cần xem lại - percentile trong ma trận vẫn đúng vì
    lấy từ stdout.
    """
    for q in (50, 90, 95, 99):
        key = f"p{q}_ms"
        mine, theirs = computed.get(key, 0), reported.get(key, 0)
        if not mine or not theirs:
            continue
        drift = abs(mine - theirs) / theirs
        if drift > 0.05:
            print(
                f"  ! {label} {key}: export {mine:.1f} ms vs perf_analyzer "
                f"{theirs:.1f} ms (lệch {drift:.0%}) - min/max có thể không đáng tin"
            )
```

- [ ] **Step 3: Thay thân `run_perf()`**

Thay toàn bộ `def run_perf(...)` hiện tại bằng:

```python
def run_perf(
    model: str,
    concurrency: int,
    input_file: str,
    shape: str | None,
    interval_ms: int = 20000,
    request_count: int = 0,
    export_path: Path | None = None,
    warmup: int = 0,
    total_requests: int = 0,
) -> dict:
    """Chạy perf_analyzer, trả về throughput và latency. Kèm cả output thô.

    stability-percentage nới lên 25%: máy này chỉ có một GPU dùng chung với
    desktop nên nhiễu tự nhiên cao, để mặc định 10% thì không bao giờ hội tụ.

    Bốn percentile lấy từ stdout. min/max phải tự tính từ profile export vì
    perf_analyzer không in hai số đó - truyền export_path để bật.

    total_requests chặn cứng tổng số request. Bắt buộc khi bật export: mỗi bản
    ghi export ôm nguyên tensor input nên RAM phình theo số request, cửa sổ thời
    gian không có trần và đã từng làm kernel oom-kill. Xem mục trần RAM ở plan.
    """
    cmd = [
        PERF_ANALYZER, "-m", model, "-u", "localhost:8001", "-i", "grpc",
        "--concurrency-range", f"{concurrency}:{concurrency}",
        "--measurement-interval", str(interval_ms),
        "--stability-percentage", "25",
        "--input-data", input_file,
        "--percentile", "95",
    ]
    if shape:
        cmd += ["--shape", shape]
    if request_count:
        # Model chậm (TTS ~2.4s/câu) thì cửa sổ theo thời gian có quá ít mẫu để
        # hội tụ. Đo theo số request ổn định hơn nhiều.
        cmd += [
            "--measurement-mode", "count_windows",
            "--measurement-request-count", str(request_count),
        ]
    if warmup:
        # Lượt suy luận đầu gánh chi phí một lần (ONNX dựng plan, CUDA nạp
        # kernel) và rơi trọn vào max. perf_analyzer tự loại request warmup
        # khỏi cả thống kê lẫn profile export.
        cmd += ["--warmup-request-count", str(warmup)]
    if total_requests:
        # Trần RAM, không phải tuỳ chọn: xem mục trần RAM ở đầu plan.
        # Cờ này tắt vòng lặp ổn định nên stability-percentage hết tác dụng.
        cmd += ["--request-count", str(total_requests)]
    if export_path:
        cmd += ["--profile-export-file", str(export_path)]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    output = proc.stdout

    summary = parse_summary(output)
    if not summary:
        print(output, proc.stderr, file=sys.stderr)
        raise SystemExit(f"perf_analyzer hỏng cho {model} @ concurrency {concurrency}")

    result = {**summary, "vram_mb": vram_mb(), "_raw": output}

    if export_path:
        computed = latency_stats(parse_profile_export(export_path))
        cross_check(computed, summary, f"{model}@{concurrency}")
        result |= {
            "min_ms": computed["min_ms"],
            "max_ms": computed["max_ms"],
            "samples": computed["samples"],
        }
    return result
```

- [ ] **Step 4: Kiểm tra E1–E3 không hỏng — chạy thật một lần đo ngắn**

Server phải đang chạy (từ Task 2). Chạy E3 (nhanh nhất, một lần đo):

Run: `.venv/bin/python bench/bench.py e3`
Expected: in bảng breakdown 3 tầng + dòng `tổng: ... rps, p95 ... ms`, không traceback

Xác nhận file cũ vẫn ghi đúng cột:

Run: `head -1 bench/results/results.csv && tail -1 bench/results/results.csv`
Expected: header vẫn là `experiment,variant,concurrency,throughput_rps,p50_ms,p95_ms,vram_mb`, dòng cuối có 7 giá trị

- [ ] **Step 5: Xác minh profile export và phép kiểm chứng chéo hoạt động**

Đây là bước then chốt: xác nhận ngữ nghĩa `window_boundaries` mình đoán là đúng.

Chạy có chặn RAM (`ulimit`) để nếu ước lượng sai thì tiến trình tự chết chứ không kéo sập máy:

```bash
mkdir -p bench/results/export
( ulimit -v 8388608
.venv/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from bench.bench import ENCODER_ARGS, run_perf
from bench.stats import latency_stats, parse_profile_export

export = Path("bench/results/export/probe.json")
r = run_perf("asr_encoder", 2, *ENCODER_ARGS, export_path=export, warmup=10,
             total_requests=100)
c = latency_stats(parse_profile_export(export))
print("perf_analyzer:", {k: r[k] for k in ("p50_ms", "p90_ms", "p95_ms", "p99_ms")})
print("tự tính      :", {k: c[k] for k in ("p50_ms", "p90_ms", "p95_ms", "p99_ms")})
print("min/max/samples:", c["min_ms"], c["max_ms"], c["samples"])
print("request_count báo cáo:", r["request_count"])
PY
)
```

Expected: hai dòng percentile khớp nhau trong 5% và **không** có dòng cảnh báo `!`. `samples` phải đúng bằng `request_count`.

**Kết quả đã đo (2026-08-10, `asr_encoder` @ CCU 2, `--request-count 100`):**

| nguồn | p50 | p90 | p95 | p99 |
|---|---|---|---|---|
| perf_analyzer stdout | 59.22 | 59.61 | 60.01 | 61.20 |
| tự tính từ export | 59.22 | 59.62 | 60.02 | 61.21 |

Lệch < 0,02%. `samples` = 100 = `request_count`. `min/max` = 57.84 / 62.53 ms. RSS đỉnh 2,1 GB, export 636 MB.

Ngữ nghĩa hoá ra đơn giản hơn dự đoán: với `--request-count`, `window_boundaries` chỉ có **2 mốc** nên nhánh `len(boundaries) > STABLE_WINDOWS` là False và `parse_profile_export` lấy toàn bộ request — đúng bằng tập perf_analyzer báo cáo. Phép cắt 3 cửa sổ cuối vì thế **không chạy trong đường đi thật của E4**; nó vẫn được giữ (và vẫn có test đơn vị) cho trường hợp đo bằng cửa sổ thời gian.

**Nếu cảnh báo nổ trên máy khác:** in cấu trúc thật để chẩn đoán:

```bash
.venv/bin/python -c "
import json
d = json.load(open('bench/results/export/probe.json'))
e = d['experiments'][0]
print('keys:', list(e.keys()))
print('n requests:', len(e['requests']))
print('boundaries:', e.get('window_boundaries'))
print('request[0]:', {k: v for k, v in e['requests'][0].items() if k != 'request_inputs'})
"
```

So `len(requests)` với `request_count` perf_analyzer báo cáo để suy ra cửa sổ nào được tính, rồi sửa phép cắt trong `parse_profile_export` cho khớp. Cập nhật test `test_requests_outside_stable_windows_excluded` theo ngữ nghĩa đúng và chạy lại Task 1 Step 4.

- [ ] **Step 6: Commit**

```bash
git add bench/bench.py
git commit -m "Extend run_perf with all four percentiles, min/max and cross-check"
```

---

### Task 4: `e4()` — vòng đo ma trận và hai file output

**Files:**
- Modify: `bench/bench.py` — thêm hằng số, `scorer_args()`, `write_matrix_csv()`, `write_matrix_md()`, `print_matrix()`, `e4()`, và đăng ký `e4` ở `__main__`
- Generates: `bench/results/matrix.csv`, `bench/results/matrix.md`, `bench/results/export/*.json`

**Interfaces:**
- Consumes: `run_perf(..., export_path=, warmup=)` và `parse_summary` từ Task 3; `bench/input_scorer.json` từ Task 2
- Produces: `e4()` gọi được qua `python bench/bench.py e4`

- [ ] **Step 1: Thêm hằng số**

Chèn sau `ENCODER_ARGS = ("bench/input_encoder.json", "x:1600,80")` trong `bench/bench.py`:

```python
SCORER_INPUT = "bench/input_scorer.json"

MATRIX_PATH = RESULTS / "matrix.csv"
MATRIX_MD_PATH = RESULTS / "matrix.md"
MATRIX_COLUMNS = [
    "component",
    "concurrency",
    "throughput_rps",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "min_ms",
    "max_ms",
    "samples",
    "vram_mb",
]
CCU_LEVELS = (1, 2, 3, 4)
```

`samples` trong ma trận sẽ bằng đúng `total_requests` của component, không phải hàng nghìn như E1–E3 — đây là hệ quả của trần RAM, xem mục đầu plan.

- [ ] **Step 2: Thêm `scorer_args()`**

Chèn sau `def avg_batch_size(...)`:

```python
def scorer_args() -> tuple[str, str]:
    """Input và --shape cho asr_scorer.

    Shape suy từ chính file input nên hai chỗ không thể lệch nhau: T do encoder
    quyết định (Zipformer subsample ~4 lần), hardcode là sẽ sai khi đổi audio mẫu.
    """
    path = ROOT / SCORER_INPUT
    if not path.exists():
        raise SystemExit(
            f"Thiếu {SCORER_INPUT}. Chạy .venv/bin/python bench/gen_input.py "
            "khi Triton đang chạy để sinh file này."
        )
    frames = json.loads(path.read_text())["data"][0]["ENCODER_OUT_LEN"][0]
    return SCORER_INPUT, f"ENCODER_OUT:{frames},512"
```

- [ ] **Step 3: Thêm ba hàm output**

Chèn sau `def print_table(rows)`:

```python
def write_matrix_csv(rows) -> None:
    """Ghi đè, không append: E4 là một ma trận trọn vẹn chứ không phải log."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    with MATRIX_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATRIX_COLUMNS)
        writer.writeheader()
        writer.writerows([{k: r.get(k, 0) for k in MATRIX_COLUMNS} for r in rows])
    print(f"đã ghi {MATRIX_PATH}")


def write_matrix_md(rows) -> None:
    """Bảng markdown nhóm theo component, dán trực tiếp vào chat được."""
    lines = [
        "# Ma trận latency và throughput",
        "",
        f"Đo ngày {time.strftime('%Y-%m-%d')}. Số liệu thô: `matrix.csv`.",
        "",
        "P50/P90/P95/P99 và throughput lấy từ perf_analyzer; min/max tính từ "
        "profile export. Cột `mẫu` là số request dùng để tính - dưới ~100 thì "
        "P99 chỉ là một hai request cuối, đọc kèm cảnh giác.",
        "",
    ]
    for component in dict.fromkeys(r["component"] for r in rows):
        lines += [
            f"## {component}",
            "",
            "| CCU | throughput (rps) | P50 | P90 | P95 | P99 | min | max | mẫu |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in (x for x in rows if x["component"] == component):
            lines.append(
                f"| {r['concurrency']} | {r['throughput_rps']:.2f} | "
                f"{r['p50_ms']:.1f} | {r['p90_ms']:.1f} | {r['p95_ms']:.1f} | "
                f"{r['p99_ms']:.1f} | {r.get('min_ms', 0):.1f} | "
                f"{r.get('max_ms', 0):.1f} | {r.get('samples', 0)} |"
            )
        lines.append("")
    MATRIX_MD_PATH.write_text("\n".join(lines))
    print(f"đã ghi {MATRIX_MD_PATH}")


def print_matrix(rows) -> None:
    print(
        f"\n{'component':16s} {'ccu':>4s} {'rps':>8s} {'p50':>8s} {'p90':>8s} "
        f"{'p95':>8s} {'p99':>8s} {'min':>8s} {'max':>8s} {'mẫu':>6s}"
    )
    for r in rows:
        print(
            f"{r['component']:16s} {r['concurrency']:4d} {r['throughput_rps']:8.2f} "
            f"{r['p50_ms']:8.1f} {r['p90_ms']:8.1f} {r['p95_ms']:8.1f} "
            f"{r['p99_ms']:8.1f} {r.get('min_ms', 0):8.1f} "
            f"{r.get('max_ms', 0):8.1f} {r.get('samples', 0):6d}"
        )
```

- [ ] **Step 4: Thêm `e4()`**

Chèn sau `def e3():`:

```python
def e4():
    """Ma trận đầy đủ: 5 component × CCU 1-4 × 7 metric.

    Không sửa config.pbtxt nào nên chỉ cần khởi động server một lần cho cả
    thí nghiệm - khác E1/E2 phải restart giữa mỗi bước.
    """
    export_dir = RESULTS / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    restart_server()

    # (nhãn, model, file input, shape, request_count, warmup, total_requests)
    # TTS ~2.4s/câu nên phải đo theo số request; warmup 2 thay vì 10 cho đỡ tốn.
    # total_requests là trần RAM cho profile export - xem mục trần RAM ở đầu plan.
    # Đặt theo cỡ tensor: encoder 6.4 MB/req nặng nhất nên số nhỏ nhất.
    components = [
        ("asr (ensemble)", "asr", *ASR_ARGS, 0, 10, 150),
        ("asr_feature", "asr_feature", *ASR_ARGS, 0, 10, 120),
        ("asr_encoder", "asr_encoder", *ENCODER_ARGS, 0, 10, 100),
        ("asr_scorer", "asr_scorer", *scorer_args(), 0, 10, 120),
        ("tts", "tts", *TTS_ARGS, 16, 2, 0),
    ]

    rows = []
    for label, model, input_file, shape, request_count, warmup, total in components:
        print(f"\n=== E4 {label} ===")
        for concurrency in CCU_LEVELS:
            export_path = export_dir / f"{model}_ccu{concurrency}.json"
            result = run_perf(
                model,
                concurrency,
                input_file,
                shape,
                request_count=request_count,
                export_path=export_path,
                warmup=warmup,
                total_requests=total,
            )
            rows.append({"component": label, "concurrency": concurrency, **result})
            print_matrix(rows[-1:])
            # File export vài trăm MB mỗi ô, 20 ô là chục GB - xoá ngay sau khi
            # đã bóc min/max ra khỏi nó.
            export_path.unlink(missing_ok=True)

    write_matrix_csv(rows)
    write_matrix_md(rows)
    print_matrix(rows)
```

- [ ] **Step 5: Đăng ký `e4`**

Sửa dòng cuối `bench/bench.py`:

```python
    {"e1": e1, "e1b": e1b, "e2": e2, "e3": e3, "e4": e4}[sys.argv[1]]()
```

- [ ] **Step 6: Kiểm tra khô — cú pháp và `scorer_args()`**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from bench.bench import e4, scorer_args; print(scorer_args())"`
Expected: in tuple dạng `('bench/input_scorer.json', 'ENCODER_OUT:400,512')`

- [ ] **Step 7: Chạy thử một component ở một mức CCU**

Kiểm tra vòng lặp chạy đúng trước khi tốn 30 phút:

```bash
.venv/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from bench.bench import (ENCODER_ARGS, RESULTS, print_matrix, run_perf,
                         write_matrix_csv, write_matrix_md)

export = RESULTS / "export" / "smoke.json"
r = run_perf("asr_encoder", 2, *ENCODER_ARGS, export_path=export, warmup=10)
rows = [{"component": "asr_encoder", "concurrency": 2, **r}]
print_matrix(rows)
write_matrix_csv(rows)
write_matrix_md(rows)
PY
cat bench/results/matrix.md
```
Expected: bảng in ra có đủ 9 cột số, `matrix.md` có bảng markdown hợp lệ, không cảnh báo `!`

- [ ] **Step 8: Commit**

```bash
git add bench/bench.py
git commit -m "Add E4 benchmark matrix: 5 components x CCU 1-4"
```

Thêm `bench/results/export/` vào `.gitignore` nếu chưa có (file JSON export có thể vài MB):

```bash
grep -q "bench/results/export" .gitignore || echo "bench/results/export/" >> .gitignore
git add .gitignore && git commit -m "Ignore raw perf_analyzer profile exports" || true
```

---

### Task 5: Chạy E4 đầy đủ và viết kết quả vào `bench/README.md`

**Files:**
- Generates: `bench/results/matrix.csv`, `bench/results/matrix.md`
- Modify: `bench/README.md` — thêm mục E4 ở đầu phần kết quả

**Interfaces:**
- Consumes: `e4()` từ Task 4
- Produces: không có API mới; deliverable là số liệu và phần viết cho mentor

- [ ] **Step 1: Chạy E4 đầy đủ**

Chạy nền vì mất ~30 phút:

```bash
.venv/bin/python bench/bench.py e4 > bench/results/e4_run.log 2>&1
```
Expected: kết thúc bằng `đã ghi .../matrix.csv` và `đã ghi .../matrix.md`

- [ ] **Step 2: Kiểm tra ma trận đủ 20 dòng**

```bash
.venv/bin/python -c "
import csv
rows = list(csv.DictReader(open('bench/results/matrix.csv')))
print('dòng:', len(rows))
missing = [r for r in rows if not r['p99_ms'] or float(r['p99_ms']) == 0]
print('dòng thiếu p99:', len(missing))
import collections
print(collections.Counter(r['component'] for r in rows))
"
```
Expected: `dòng: 20`, `dòng thiếu p99: 0`, mỗi component đúng 4 dòng

- [ ] **Step 3: Soát cảnh báo lệch percentile**

Run: `grep -c "^  !" bench/results/e4_run.log || echo 0`
Expected: `0`

Nếu có cảnh báo: ghi lại component/CCU nào lệch và nêu rõ trong phần viết ở Step 5 rằng `min`/`max` của những dòng đó không đáng tin. Không được im lặng bỏ qua.

- [ ] **Step 4: Kiểm tra số mẫu của TTS**

Run: `.venv/bin/python -c "
import csv
for r in csv.DictReader(open('bench/results/matrix.csv')):
    if r['component'] == 'tts':
        print(r['concurrency'], 'mẫu:', r['samples'], 'p99:', r['p99_ms'])
"`
Expected: in 4 dòng. Số mẫu TTS sẽ nhỏ (vài chục) — đây là dữ kiện phải nêu ở Step 5, không phải lỗi.

- [ ] **Step 5: Viết mục E4 vào `bench/README.md`**

Chèn mục mới **ngay sau** phần "Cách chạy lại" và **trước** mục "E3", vì E4 giờ là bảng số chính.

Cập nhật khối "Cách chạy lại" thêm dòng:

```
.venv/bin/python bench/bench.py e4      # ma trận đầy đủ 5 component × CCU 1-4
```

Rồi thêm mục E4 gồm: bảng dán từ `matrix.md`, và phần đọc số trả lời ba câu:
1. Component nào latency tăng nhanh nhất theo CCU, và vì sao (đối chiếu kết luận E1b/E2 đã có)
2. `asr_feature` (CPU, count=4) so với `asr_scorer` (GPU, count=2) hành xử khác nhau thế nào khi CCU tăng
3. Khoảng cách giữa P99 và max nói gì về độ ổn định

Nêu rõ **năm** hạn chế:

1. Số mẫu mọi component đều thấp (100–150, TTS còn thấp hơn) vì trần RAM của profile export — P99 chỉ là một hai request chậm nhất, không phải phân vị thật.
2. `--request-count` tắt vòng lặp ổn định, nên số đo là một lát cắt cố định chứ không phải giá trị đã hội tụ; `--stability-percentage 25` không còn tác dụng ở E4. Đây là khác biệt đáng kể so với E1–E3 và phải nói rõ khi đặt hai bảng cạnh nhau.
3. `max` không tái lập được.
4. Toàn bộ là số phía server, chưa tính overhead client.
5. Nghẽn là RAM máy chủ chạy client (15 GB), không phải VRAM (2,3/4,1 GB lúc load cả 6 model) — ai chạy lại trên máy nhiều RAM hơn thì nâng `total_requests` lên để P99 có nghĩa hơn.

- [ ] **Step 6: Commit**

`bench/results/*.csv` bị `.gitignore` chặn (dòng 8) — `results.csv` của E1–E3 cũng chưa từng được commit, nên `matrix.csv` giữ nguyên quy ước đó, số liệu đi vào repo qua `README.md` và `matrix.md`.

```bash
git add bench/README.md bench/results/matrix.md
git commit -m "Add E4 benchmark matrix results for all components at CCU 1-4"
```

---

## Self-Review

**Spec coverage:**

| Spec | Task |
|---|---|
| 3.1 thêm E4 không sửa E1–E3 | Task 3 Step 4, Task 4 Step 1 (file output riêng) |
| 3.2 hai nguồn số liệu | Task 3 Step 2 (`parse_summary`), Step 3 (`export_path`) |
| 3.3 cắt 3 cửa sổ cuối | Task 1 Step 3 (`STABLE_WINDOWS`), Task 3 Step 5 (xác minh thật) |
| 3.4 input scorer bắt từ server | Task 2 |
| 3.5 warmup | Task 3 Step 3 (`--warmup-request-count`), Task 4 Step 4 |
| 3.6 kiểm chứng chéo | Task 3 Step 2 (`cross_check`), Task 5 Step 3 |
| 4 ma trận 5 component | Task 4 Step 4 |
| 4.1 chế độ đo theo component | Task 4 Step 4 (`request_count=16` cho TTS) |
| 5.1 `bench/stats.py` | Task 1 |
| 5.2 `bench/bench.py` | Task 3, Task 4 |
| 5.3 `gen_input.py` | Task 2 |
| 6.1 `matrix.csv` ghi đè | Task 4 Step 3 |
| 6.2 `matrix.md` | Task 4 Step 3 |
| 6.3 export thô giữ lại | Task 4 Step 4 (`export_dir`) |
| 7.1 test đơn vị | Task 1 Step 1 |
| 8 hạn chế phép đo | Task 5 Step 5 |

**Type consistency:** `parse_profile_export` → `list[float]` dùng bởi `latency_stats` (Task 1) và `run_perf` (Task 3). `latency_stats` trả `samples`, và `MATRIX_COLUMNS` (Task 4) có cột `samples`. `scorer_args()` trả `tuple[str, str]` được unpack bằng `*` trong `components` (Task 4 Step 4), khớp cấu trúc `ASR_ARGS`/`ENCODER_ARGS`. `run_perf` giữ nguyên `p50_ms`/`p95_ms`/`throughput_rps`/`vram_mb` nên `COLUMNS` của E1–E3 vẫn khớp.

**Rủi ro đã có bước xử lý:** ngữ nghĩa `window_boundaries` là phỏng đoán duy nhất trong thiết kế — Task 3 Step 5 xác minh bằng dữ liệu thật và có sẵn quy trình chẩn đoán nếu sai; hậu quả xấu nhất bị chặn ở `min`/`max` nhờ 3.2.
