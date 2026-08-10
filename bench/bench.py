# ABOUTME: Chạy các thí nghiệm E1-E3, tự sửa config và khởi động lại server giữa các lần đo
# ABOUTME: Chạy: python bench/bench.py e1  (hoặc e1b, e2, e3)

import csv
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench.stats import latency_stats, parse_profile_export  # noqa: E402

# Thí nghiệm chạy hàng chục phút, cần thấy tiến độ ngay cả khi ghi ra file
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "bench/results"
CSV_PATH = RESULTS / "results.csv"
COLUMNS = [
    "experiment",
    "variant",
    "concurrency",
    "throughput_rps",
    "p50_ms",
    "p95_ms",
    "vram_mb",
]
PERF_ANALYZER = str(ROOT / ".venv/bin/perf_analyzer")

ASR_ARGS = ("bench/input_asr.json", "WAV:256000")
TTS_ARGS = ("bench/input_tts.json", None)
ENCODER_ARGS = ("bench/input_encoder.json", "x:1600,80")


# ---------------------------------------------------------------- hạ tầng đo


def vram_mb() -> int:
    """Bộ nhớ GPU đang dùng, tính theo MB."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def read_stats(model: str) -> tuple[int, int]:
    """Đọc (inference_count, execution_count) của một model từ Triton.

    perf_analyzer chỉ in breakdown khi model là ensemble, gọi model lẻ thì
    không có. Endpoint này dùng được cho mọi model. Tỉ số hai số này chính là
    batch trung bình mà dynamic batcher gom được.
    """
    with urllib.request.urlopen(
        f"http://localhost:8000/v2/models/{model}/stats", timeout=5
    ) as r:
        data = json.load(r)
    stats = data["model_stats"][0]
    return (
        int(stats["inference_stats"]["success"]["count"]),
        int(stats.get("execution_count", 0)),
    )


def avg_batch_size(model: str, before: tuple[int, int]) -> float:
    """Batch trung bình trong khoảng giữa hai lần đọc stats."""
    after = read_stats(model)
    d_inference, d_execution = after[0] - before[0], after[1] - before[1]
    return d_inference / d_execution if d_execution else 0.0


def set_config(path: Path, key: str, value) -> None:
    """Thay giá trị của một khoá trong config.pbtxt. Giữ nguyên phần còn lại."""
    content = path.read_text()
    updated, count = re.subn(
        rf"({re.escape(key)}\s*:\s*)\S+", rf"\g<1>{value}", content, count=1
    )
    if count != 1:
        raise SystemExit(f"Không tìm thấy khoá {key!r} trong {path}")
    path.write_text(updated)
    print(f"  đặt {key} = {value}")


def restart_server() -> None:
    """Dựng lại container Triton rồi chờ tới khi server báo sẵn sàng."""
    subprocess.run(["docker", "rm", "-f", "triton-voice-server"], capture_output=True)

    log = open(RESULTS / "server.log", "w")
    subprocess.Popen(
        [str(ROOT / "scripts/serve.sh")], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT
    )

    for _ in range(120):
        time.sleep(2)
        try:
            with urllib.request.urlopen(
                "http://localhost:8000/v2/health/ready", timeout=2
            ) as r:
                if r.status == 200:
                    time.sleep(2)   # để model ổn định trước khi đo
                    print("  server sẵn sàng")
                    return
        except (urllib.error.URLError, OSError):
            continue
    raise SystemExit("Server không lên sau 4 phút, xem bench/results/server.log")


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


def run_perf(
    model: str,
    concurrency: int,
    input_file: str,
    shape: str | None,
    interval_ms: int = 20000,
    request_count: int = 0,
    export_path: Path | None = None,
    warmup: int = 0,
) -> dict:
    """Chạy perf_analyzer, trả về throughput và latency. Kèm cả output thô.

    stability-percentage nới lên 25%: máy này chỉ có một GPU dùng chung với
    desktop nên nhiễu tự nhiên cao, để mặc định 10% thì không bao giờ hội tụ.

    Bốn percentile lấy từ stdout. min/max phải tự tính từ profile export vì
    perf_analyzer không in hai số đó - truyền export_path để bật.
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


def parse_model_stats(raw: str) -> dict:
    """Bóc thống kê từng model con từ output của perf_analyzer.

    Quét theo dòng thay vì regex nhiều dòng - khối của model này liền ngay
    khối của model kia nên regex tham lam sẽ nuốt cả hai.
    """
    result, name = {}, None
    for line in raw.splitlines():
        if m := re.match(r"\s+(\w+), version: \d+", line):
            name = m.group(1)
            result[name] = {}
        elif name is None:
            continue
        elif m := re.search(r"Inference count:\s+(\d+)", line):
            result[name]["inference"] = int(m.group(1))
        elif m := re.search(r"Execution count:\s+(\d+)", line):
            result[name]["execution"] = int(m.group(1))
        elif "Avg request latency" in line:
            result[name]["queue_us"] = int(re.search(r"queue (\d+) usec", line).group(1))
            result[name]["compute_us"] = int(
                re.search(r"compute infer (\d+) usec", line).group(1)
            )
    return result


def write_csv(rows) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    is_new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerows([{k: r[k] for k in COLUMNS} for r in rows])


def print_table(rows) -> None:
    print(
        f"\n{'variant':22s} {'conc':>5s} {'rps':>8s} "
        f"{'p50 ms':>9s} {'p95 ms':>9s} {'vram':>7s}"
    )
    for r in rows:
        print(
            f"{r['variant']:22s} {r['concurrency']:5d} {r['throughput_rps']:8.2f} "
            f"{r['p50_ms']:9.1f} {r['p95_ms']:9.1f} {r['vram_mb']:7d}"
        )


# ---------------------------------------------------------------- thí nghiệm


def e1():
    """Dynamic batching: quét max_queue_delay của asr_encoder ở concurrency 8.

    Đo hai lần cho mỗi mức delay:
      - gọi thẳng asr_encoder  -> thấy được tác dụng thật của batching
      - gọi qua ensemble asr   -> thấy nó bị tầng scorer che mất như thế nào
    Chênh lệch giữa hai cột này chính là bài học của thí nghiệm.
    """
    config = ROOT / "model_repository/asr_encoder/config.pbtxt"
    rows = []
    for delay in (0, 1000, 5000, 20000):
        print(f"\n=== E1 delay={delay}us ===")
        set_config(config, "max_queue_delay_microseconds", delay)
        restart_server()

        for label, model, args in (
            ("encoder", "asr_encoder", ENCODER_ARGS),
            ("ensemble", "asr", ASR_ARGS),
        ):
            before = read_stats("asr_encoder")
            result = run_perf(model, 8, *args)
            batch = avg_batch_size("asr_encoder", before)
            rows.append(
                {
                    "experiment": "e1",
                    "variant": f"{label} delay={delay}us batch={batch:.2f}",
                    "concurrency": 8,
                    **result,
                }
            )
            print_table(rows[-1:])

    set_config(config, "max_queue_delay_microseconds", 5000)   # trả về mặc định
    write_csv(rows)
    print_table(rows)


def e1b():
    """Quét concurrency client với delay cố định 5ms, gọi thẳng asr_encoder.

    E1 cho thấy vặn max_queue_delay gần như không đổi gì. Lý do: ở concurrency
    cố định, batch bị chặn bởi số request đang bay chứ không bởi thời gian chờ.
    Nút vặn đúng để thấy đường cong batching là concurrency của client.
    """
    print("\n=== E1b quét concurrency, asr_encoder, delay=5000us ===")
    rows = []
    for concurrency in (1, 2, 4, 8, 16):
        before = read_stats("asr_encoder")
        result = run_perf("asr_encoder", concurrency, *ENCODER_ARGS)
        batch = avg_batch_size("asr_encoder", before)
        rows.append(
            {
                "experiment": "e1b",
                "variant": f"encoder batch={batch:.2f}",
                "concurrency": concurrency,
                **result,
            }
        )
        print_table(rows[-1:])

    write_csv(rows)
    print_table(rows)


def e2():
    """Model concurrency: quét instance_group.count của tts."""
    config = ROOT / "model_repository/tts/config.pbtxt"
    rows = []
    for instances in (1, 2, 4):
        print(f"\n=== E2 instances={instances} ===")
        set_config(config, "count", instances)
        restart_server()

        for concurrency in (1, 2, 4):
            result = run_perf("tts", concurrency, *TTS_ARGS, request_count=16)
            rows.append(
                {
                    "experiment": "e2",
                    "variant": f"instances={instances}",
                    "concurrency": concurrency,
                    **result,
                }
            )
            print_table(rows[-1:])

    set_config(config, "count", 1)   # trả về mặc định
    write_csv(rows)
    print_table(rows)


def e3():
    """Ensemble breakdown: tách thời gian từng tầng con ở tải cố định."""
    print("\n=== E3 breakdown của ensemble asr @ concurrency 8 ===")
    result = run_perf("asr", 8, *ASR_ARGS)

    print(
        f"\n{'model':14s} {'infer':>7s} {'exec':>7s} {'batch tb':>9s} "
        f"{'chờ ms':>8s} {'tính ms':>9s}"
    )
    for name, stats in parse_model_stats(result["_raw"]).items():
        if "inference" not in stats:
            continue
        print(
            f"{name:14s} {stats['inference']:7d} {stats['execution']:7d} "
            f"{stats['inference'] / max(stats['execution'], 1):9.2f} "
            f"{stats.get('queue_us', 0) / 1000:8.2f} "
            f"{stats.get('compute_us', 0) / 1000:9.2f}"
        )

    (RESULTS / "e3_perf_analyzer.txt").write_text(result["_raw"])
    write_csv([{"experiment": "e3", "variant": "breakdown", "concurrency": 8, **result}])
    print(f"\ntổng: {result['throughput_rps']:.2f} rps, p95 {result['p95_ms']:.1f} ms")
    print("output thô: bench/results/e3_perf_analyzer.txt")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    {"e1": e1, "e1b": e1b, "e2": e2, "e3": e3}[sys.argv[1]]()
