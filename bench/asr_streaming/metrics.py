# ABOUTME: Đo hai thứ perf_analyzer không thấy được ở asr_streaming: first-chunk latency và WER
# ABOUTME: Latency thường, throughput, GPU util -> dùng scripts/perf.sh

import argparse
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import tritonclient.grpc as grpcclient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.common.stats import p50_p95  # noqa: E402
from bench.asr_streaming.wer import wer  # noqa: E402
from client.common import chunk_wav, load_wav_16k  # noqa: E402

MODEL = "asr_streaming"


@dataclass
class StreamResult:
    """Kết quả một phiên: latency từng chunk và transcript cuối cùng."""

    latencies: dict = field(default_factory=dict)   # chỉ số chunk -> ms
    transcript: str = ""


# ---------------------------------------------------------------- hàm thuần


def send_deadlines(n_chunks: int, chunk_ms: int, t0: float) -> list[float]:
    """Mốc thời gian tuyệt đối phải gửi từng chunk, mô phỏng micro thật.

    Tính từ t0 chứ không cộng dồn: sleep(chunk_ms) lặp nhiều lần sẽ trôi dần vì
    mỗi lần sleep đều dài hơn yêu cầu một chút, và nhịp gửi trôi thì số liệu sai.
    """
    return [t0 + i * chunk_ms / 1000 for i in range(n_chunks)]


def first_chunk_latencies(results) -> list[float]:
    """Latency chunk mở phiên của từng phiên.

    Đây là chỉ số duy nhất perf_analyzer không tách ra được: nó gộp mọi request
    trong một sequence thành một phân phối, mà chunk 0 lại khác hẳn về bản chất -
    nó là chunk duy nhất phải dựng state và cấp cache encoder cho phiên mới.
    """
    firsts = []
    for i, r in enumerate(results):
        if 0 not in r.latencies:
            raise ValueError(f"phiên {i} không có chunk 0 - partial không về đủ")
        firsts.append(r.latencies[0])
    return firsts


def summarize_run(ccu: int, results) -> dict:
    """Gom một mức CCU thành một hàng của bảng kết quả."""
    p50, p95 = p50_p95(first_chunk_latencies(results))
    return {"ccu": ccu, "first_p50_ms": round(p50, 2), "first_p95_ms": round(p95, 2)}


COLUMNS = [
    ("ccu", "CCU"),
    ("first_p50_ms", "first-chunk p50 ms"),
    ("first_p95_ms", "first-chunk p95 ms"),
]


def render_matrix(rows) -> str:
    """Bảng markdown một hàng mỗi mức CCU."""
    head = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    rule = "|" + "---|" * len(COLUMNS)
    body = ["| " + " | ".join(str(row[key]) for key, _ in COLUMNS) + " |" for row in rows]
    return "\n".join([head, rule, *body])


# ------------------------------------------------------------- phần gọi server


def run_stream(url, model, seq_id, chunks, chunk_ms, barrier, timeout_s) -> StreamResult:
    """Gửi trọn một phiên đúng nhịp thời gian thực, trả latency và transcript cuối.

    Mốc nhận đóng dấu ngay trong callback - nếu đợi vòng lặp chính đọc queue
    thì latency đo được sẽ cộng thêm thời gian client bận, không phải của server.
    """
    inbox = queue.Queue()
    client = grpcclient.InferenceServerClient(url)
    client.start_stream(callback=lambda result, error: inbox.put((time.perf_counter(), result, error)))
    sent, latencies, texts = {}, {}, {}
    try:
        barrier.wait(timeout=timeout_s)   # mọi phiên phải chồng lấn nhau mới đúng nghĩa CCU
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
                raise TimeoutError(f"phiên {seq_id}: thiếu {len(chunks) - len(latencies)} partial")
            t_recv, result, error = inbox.get(timeout=left)
            if error:
                raise RuntimeError(f"phiên {seq_id}: {error}")
            rid = result.get_response().id
            idx = int(rid.rsplit("-", 1)[1])
            latencies[idx] = (t_recv - sent[rid]) * 1000
            texts[idx] = result.as_numpy("TRANSCRIPT")[0, 0].decode("utf-8")
    finally:
        client.stop_stream()
        client.close()
    # transcript cộng dồn nên bản của chunk cuối là bản đầy đủ
    return StreamResult(latencies=latencies, transcript=texts[max(texts)])


def run_ccu(url, model, ccu, chunks, chunk_ms, seq_base, timeout_s) -> list[StreamResult]:
    """Chạy `ccu` phiên song song, trả kết quả từng phiên."""
    barrier = threading.Barrier(ccu)
    with ThreadPoolExecutor(max_workers=ccu) as pool:
        futures = [
            pool.submit(run_stream, url, model, seq_base + k, chunks, chunk_ms, barrier, timeout_s)
            for k in range(ccu)
        ]
        return [f.result() for f in futures]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="tests/assets/sample_vi_long.wav")
    ap.add_argument("--ref", default="tests/assets/sample_vi_long.txt",
                    help="transcript tham chiếu để tính WER")
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ccu", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--chunk-ms", type=int, default=200)
    ap.add_argument("--chunks", type=int, default=10,
                    help="số chunk mỗi phiên khi đo first-chunk - chỉ cần đủ để mở phiên")
    ap.add_argument("--rounds", type=int, default=10, help="số lượt mở phiên mỗi mức CCU")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", default="bench/asr_streaming/results/metrics.md")
    args = ap.parse_args()

    wav = load_wav_16k(args.wav)
    full = chunk_wav(wav, args.chunk_ms)
    short = full[: args.chunks]
    seq = int(time.time()) % 2**28 + 1   # tránh trùng corrid với lần chạy trước

    # Làm nóng ONNX/CUDA. Không có bước này thì mức CCU chạy đầu gánh trọn chi
    # phí khởi động và cột first-chunk của nó đọc sai hoàn toàn.
    print("warmup...")
    run_ccu(args.url, args.model, 1, short, args.chunk_ms, seq, args.timeout)
    seq += 1

    # first-chunk cần nhiều lần MỞ phiên chứ không cần phiên dài, nên mỗi lượt
    # chỉ gửi vài chunk rồi đóng - đo được nhiều mẫu trong thời gian ngắn.
    rows = []
    for ccu in args.ccu:
        print(f"CCU {ccu}: {args.rounds} lượt × {ccu} phiên...", flush=True)
        results = []
        for _ in range(args.rounds):
            results += run_ccu(args.url, args.model, ccu, short, args.chunk_ms, seq, args.timeout)
            seq += ccu
        rows.append(summarize_run(ccu, results))
        print(f"  first-chunk p50 {rows[-1]['first_p50_ms']}ms  p95 {rows[-1]['first_p95_ms']}ms")

    # WER chạy một phiên đầy đủ, một mình: đây là phép đo chất lượng, không phải
    # phép đo tải. Chạy dưới tải chỉ làm số liệu nhiễu mà không thêm thông tin.
    print(f"WER: một phiên {len(full) * args.chunk_ms / 1000:.1f}s...", flush=True)
    [result] = run_ccu(args.url, args.model, 1, full, args.chunk_ms, seq, args.timeout)
    score = wer(Path(args.ref).read_text(), result.transcript)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = render_matrix(rows)
    out.write_text(
        f"# asr_streaming - chỉ số riêng của model\n\n{table}\n\n"
        f"WER = **{score:.2%}** trên `{Path(args.wav).name}`\n\n"
        f"chunk = {args.chunk_ms}ms · first-chunk = latency chunk mở phiên, "
        f"{args.rounds} lượt mỗi mức CCU\n"
        "latency thường / throughput / GPU util: xem scripts/perf.sh\n"
    )
    print(f"\n{table}\n\nWER = {score:.2%}\n\nĐã ghi {out}")


if __name__ == "__main__":
    main()
