# ABOUTME: Benchmark asr_streaming - đo latency từng chunk 200ms và RTF theo mức CCU
# ABOUTME: Chạy: python bench/stream_bench.py --ccu 1 2 3 4 (cần Triton đang chạy)

import argparse
import csv
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench.stats import latency_stats  # noqa: E402
from client.common import SAMPLE_RATE, chunk_wav, load_wav_16k  # noqa: E402

MODEL = "asr_streaming"
# Bốn bộ đếm này đủ dựng lại vòng đời server-side của một request. Tách compute
# làm ba vì Triton đo riêng, cộng lại mới là thời gian model thật sự bận.
COMPUTE_FIELDS = ("compute_input", "compute_infer", "compute_output")


# ---------------------------------------------------------------- hàm thuần


def repeat_to_duration(wav, seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Lặp waveform cho đủ `seconds` giây rồi cắt đúng độ dài.

    Lặp thay vì dùng file dài sẵn: p99 chỉ có nghĩa khi đủ mẫu, mà file mẫu
    trong repo chỉ ~18s. Lặp làm audio thiếu tự nhiên nhưng thứ đang đo là
    thời gian xử lý mỗi chunk, không phải độ chính xác transcript.
    """
    if seconds <= 0:
        raise ValueError(f"seconds phải dương, nhận {seconds}")
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    total = int(round(seconds * sample_rate))
    reps = -(-total // len(wav))   # trần của phép chia
    return np.tile(wav, reps)[:total]


def send_deadlines(n_chunks: int, chunk_ms: int, t0: float) -> list[float]:
    """Mốc thời gian tuyệt đối phải gửi từng chunk, mô phỏng micro thật.

    Tính từ t0 chứ không cộng dồn: sleep(chunk_ms) lặp 300 lần sẽ trôi dần vì
    mỗi lần sleep đều dài hơn yêu cầu một chút, và nhịp gửi trôi thì RTF sai.
    """
    return [t0 + i * chunk_ms / 1000 for i in range(n_chunks)]


def split_first_chunk(per_stream) -> tuple[list[float], list[float]]:
    """Tách latency chunk mở phiên khỏi phần còn lại.

    Chunk 0 là chunk duy nhất phải dựng _Stream và cấp cache encoder, nên nó
    đo chi phí mở phiên. Đo thực tế cho thấy chi phí đó không đáng kể: chunk 0
    rơi vào nhóm rẻ vì 200ms audio chưa đủ 45 frame để chạy một bước encoder.
    Vẫn tách ra vì nó là chunk duy nhất khác về bản chất - gộp vào thì cột
    first-chunk không còn gì để nói.
    """
    rest, firsts = [], []
    for i, lat in enumerate(per_stream):
        if 0 not in lat:
            raise ValueError(f"stream {i} không có chunk 0 - partial không về đủ")
        firsts.append(lat[0])
        rest.extend(v for k, v in lat.items() if k != 0)
    return rest, firsts


def _model_entry(payload: dict, model_name: str) -> dict:
    """Rút phần thống kê của một model từ payload get_inference_statistics."""
    for entry in payload.get("model_stats", []):
        if entry.get("name") == model_name:
            return entry
    raise ValueError(f"không thấy model {model_name} trong thống kê server")


def _ns(stats: dict, field: str, key: str = "ns") -> int:
    # MessageToJson bỏ hẳn field mang giá trị 0, nên thiếu khoá là chuyện thường
    return int(stats.get(field, {}).get(key, 0))


def _busy(entry: dict) -> tuple[int, int]:
    """(ns server thật sự bận, số lần execute) gộp mọi cỡ batch.

    Phải lấy từ batch_stats chứ không phải inference_stats: batch 2 chạy 6ms
    được inference_stats ghi 6ms cho CẢ HAI request, thành 12ms. Muốn biết
    server bận bao lâu thì chỉ batch_stats đếm mỗi lần execute đúng một lần.
    """
    ns = executions = 0
    for batch in entry.get("batch_stats", []):
        ns += sum(_ns(batch, f) for f in COMPUTE_FIELDS)
        executions += _ns(batch, "compute_infer", "count")
    return ns, executions


def stats_delta(before: dict, after: dict, model_name: str = MODEL) -> dict:
    """Chênh lệch thống kê server giữa hai mốc đo.

    Đây là thứ tách được "RTF xấu vì xếp hàng" khỏi "RTF xấu vì model chậm":
    queue_ms là chờ, compute_ms là làm việc thật.
    """
    eb, ea = _model_entry(before, model_name), _model_entry(after, model_name)
    b, a = eb.get("inference_stats", {}), ea.get("inference_stats", {})
    requests = _ns(a, "success", "count") - _ns(b, "success", "count")
    if requests <= 0:
        raise ValueError(f"không có request nào chạy cho {model_name} giữa hai mốc đo")

    queue_ns = _ns(a, "queue") - _ns(b, "queue")
    credit_ns = sum(_ns(a, f) - _ns(b, f) for f in COMPUTE_FIELDS)
    busy_ns, executions = (x - y for x, y in zip(_busy(ea), _busy(eb)))
    return {
        "requests": requests,
        "executions": executions,
        "batch_avg": round(requests / executions, 2) if executions else None,
        "queue_ms": round(queue_ns / 1e6 / requests, 2),
        "compute_ms": round(credit_ns / 1e6 / requests, 2),
        "compute_total_s": busy_ns / 1e9,
    }


def summarize_run(ccu: int, chunk_ms: int, per_stream, audio_total_s: float, server) -> dict:
    """Gom một mức CCU thành một hàng của bảng kết quả."""
    rest, firsts = split_first_chunk(per_stream)
    row = {"ccu": ccu, **latency_stats(rest)}
    # RTF per-chunk là latency chia hằng số chunk_ms, nên chỉ giữ một mốc:
    # p95 - ngưỡng đậu/rớt. Mọi phân vị khác đã nằm ở các cột latency.
    row["rtf_p95"] = round(row["p95_ms"] / chunk_ms, 3)
    row["first_ms"] = round(max(firsts), 2)   # tệ nhất trong các phiên, không lấy trung bình
    if server is None:
        row.update(rtf_stream=None, queue_ms=None, compute_ms=None, batch_avg=None)
    else:
        # RTF cả stream theo định nghĩa cổ điển: thời gian server bận / độ dài audio.
        # Khác rtf_p95 ở chỗ đo công suất tiêu thụ, không đo độ trễ cảm nhận.
        row["rtf_stream"] = round(server["compute_total_s"] / audio_total_s, 3)
        row["queue_ms"] = server["queue_ms"]
        row["compute_ms"] = server["compute_ms"]
        # Không có cột này thì compute_ms tăng theo CCU trông như model chậm đi,
        # trong khi thật ra chỉ là mỗi execute phải gánh thêm sequence.
        row["batch_avg"] = server["batch_avg"]
    return row


def sanity_warnings(row: dict) -> list[str]:
    """Mâu thuẫn nội tại chứng tỏ hàng số liệu này không đáng tin.

    Thống kê server và latency client đo hai thứ khác nhau nên không bao giờ
    khớp tuyệt đối, nhưng có một ràng buộc không được phép vi phạm: request
    không thể xếp hàng lâu hơn tổng thời gian nó tồn tại. Vi phạm nghĩa là cửa
    sổ get_inference_statistics dính request ngoài phép đo - đã gặp một lần,
    và không có cảnh báo này thì con số vô lý vẫn nằm im trong bảng kết quả.
    """
    if row["queue_ms"] is not None and row["queue_ms"] > row["max_ms"]:
        return [
            f"CCU {row['ccu']}: queue {row['queue_ms']}ms > max latency {row['max_ms']}ms. "
            "Cửa sổ thống kê server dính request ngoài phép đo - bỏ cột queue/compute/RTF stream "
            "của hàng này và đo lại."
        ]
    return []


COLUMNS = [
    ("ccu", "CCU"),
    ("p50_ms", "p50 ms"),
    ("p90_ms", "p90 ms"),
    ("p95_ms", "p95 ms"),
    ("p99_ms", "p99 ms"),
    ("max_ms", "max ms"),
    ("rtf_p95", "RTF p95"),
    ("first_ms", "first-chunk ms"),
    ("rtf_stream", "RTF stream"),
    ("queue_ms", "queue ms"),
    ("compute_ms", "compute ms"),
    ("batch_avg", "batch avg"),
    ("samples", "samples"),
]


def render_matrix(rows, chunk_ms: int) -> str:
    """Bảng markdown một hàng mỗi mức CCU."""
    head = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    rule = "|" + "---|" * len(COLUMNS)
    body = [
        "| " + " | ".join("-" if row[key] is None else str(row[key]) for key, _ in COLUMNS) + " |"
        for row in rows
    ]
    note = (
        f"\nchunk = {chunk_ms}ms · RTF p95 = p95 ms / {chunk_ms}"
        "\nRTF stream = thời gian server bận (batch_stats) / độ dài audio"
        "\ncompute ms = ghi công mỗi request (inference_stats), nên nhân lên theo batch avg"
    )
    # Cảnh báo đi kèm bảng chứ không chỉ in ra stdout - file kết quả bị đọc lại
    # sau nhiều tuần, lúc đó log đã mất
    alerts = [w for row in rows for w in sanity_warnings(row)]
    if alerts:
        note += "\n\nCẢNH BÁO\n" + "\n".join(f"- {w}" for w in alerts)
    return "\n".join([head, rule, *body]) + "\n" + note


# ------------------------------------------------------------- phần gọi server


def run_stream(url, model, seq_id, chunks, chunk_ms, barrier, timeout_s) -> dict:
    """Gửi trọn một phiên theo nhịp thật, trả {chỉ số chunk: latency ms}.

    Mốc nhận đóng dấu ngay trong callback - nếu đợi vòng lặp chính đọc queue
    thì latency đo được sẽ cộng thêm thời gian client bận, không phải của server.
    """
    inbox = queue.Queue()
    client = grpcclient.InferenceServerClient(url)
    client.start_stream(callback=lambda result, error: inbox.put((time.perf_counter(), result, error)))
    sent, latencies = {}, {}
    try:
        barrier.wait(timeout=timeout_s)   # mọi stream phải chồng lấn nhau mới đúng nghĩa CCU
        deadlines = send_deadlines(len(chunks), chunk_ms, time.perf_counter())
        for i, (part, deadline) in enumerate(zip(chunks, deadlines)):
            wait = deadline - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            rid = f"{seq_id}-{i}"
            inp = grpcclient.InferInput("AUDIO_CHUNK", [1, len(part)], "FP32")
            inp.set_data_from_numpy(part.reshape(1, -1))
            sent[rid] = time.perf_counter()
            client.async_stream_infer(
                model,
                [inp],
                request_id=rid,
                sequence_id=seq_id,
                sequence_start=(i == 0),
                sequence_end=(i == len(chunks) - 1),
            )

        limit = time.perf_counter() + timeout_s
        while len(latencies) < len(chunks):
            left = limit - time.perf_counter()
            if left <= 0:
                raise TimeoutError(f"stream {seq_id}: thiếu {len(chunks) - len(latencies)} partial")
            t_recv, result, error = inbox.get(timeout=left)
            if error:
                raise RuntimeError(f"stream {seq_id}: {error}")
            rid = result.get_response().id
            latencies[int(rid.rsplit("-", 1)[1])] = (t_recv - sent[rid]) * 1000
    finally:
        client.stop_stream()
        client.close()
    return latencies


def run_ccu(url, model, ccu, chunks, chunk_ms, seq_base, timeout_s) -> list[dict]:
    """Chạy `ccu` phiên song song, trả kết quả từng phiên."""
    barrier = threading.Barrier(ccu)
    with ThreadPoolExecutor(max_workers=ccu) as pool:
        futures = [
            pool.submit(run_stream, url, model, seq_base + k, chunks, chunk_ms, barrier, timeout_s)
            for k in range(ccu)
        ]
        return [f.result() for f in futures]


def write_raw(path: Path, per_stream) -> None:
    """Ghi latency thô để vẽ lại biểu đồ mà không phải chạy benchmark lần nữa."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream", "chunk", "latency_ms"])
        for s, lat in enumerate(per_stream):
            for chunk in sorted(lat):
                w.writerow([s, chunk, round(lat[chunk], 3)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="tests/assets/sample_vi_long.wav")
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ccu", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--chunk-ms", type=int, default=200)
    ap.add_argument("--duration", type=float, default=60.0, help="giây audio mỗi phiên")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", default="bench/results/streaming.md")
    args = ap.parse_args()

    wav = repeat_to_duration(load_wav_16k(args.wav), args.duration)
    chunks = chunk_wav(wav, args.chunk_ms)
    audio_per_stream_s = len(chunks) * args.chunk_ms / 1000
    print(f"{len(chunks)} chunk × {args.chunk_ms}ms = {audio_per_stream_s:.1f}s audio mỗi phiên")

    client = grpcclient.InferenceServerClient(args.url)
    seq = int(time.time()) % 2**28 + 1   # tránh trùng corrid với lần chạy trước

    # Làm nóng ONNX/CUDA bằng một phiên ngắn. Không có bước này thì mức CCU chạy
    # đầu tiên gánh trọn chi phí khởi động và bảng kết quả đọc sai hoàn toàn.
    print("warmup...")
    run_ccu(args.url, args.model, 1, chunks[:10], args.chunk_ms, seq, args.timeout)
    seq += 1

    rows = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ccu in args.ccu:
        print(f"CCU {ccu}: chạy {ccu} phiên × {audio_per_stream_s:.0f}s...", flush=True)
        before = client.get_inference_statistics(args.model, as_json=True)
        per_stream = run_ccu(args.url, args.model, ccu, chunks, args.chunk_ms, seq, args.timeout)
        after = client.get_inference_statistics(args.model, as_json=True)
        seq += ccu

        row = summarize_run(
            ccu, args.chunk_ms, per_stream, audio_per_stream_s * ccu, stats_delta(before, after, args.model)
        )
        rows.append(row)
        write_raw(out.parent / f"streaming_ccu{ccu}.csv", per_stream)
        print(f"  p95 {row['p95_ms']}ms  RTF p95 {row['rtf_p95']}  RTF stream {row['rtf_stream']}")
        time.sleep(2)   # để server ráo hàng đợi trước mức kế tiếp

    table = render_matrix(rows, args.chunk_ms)
    out.write_text(f"# asr_streaming - latency mỗi chunk theo CCU\n\n{table}\n")
    print(f"\n{table}\n\nĐã ghi {out}")


if __name__ == "__main__":
    main()
