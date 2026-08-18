# ABOUTME: Đo RTF của tts - thứ perf_analyzer không tính được vì không biết WAV trả về dài bao nhiêu
# ABOUTME: Latency, throughput, GPU util -> dùng scripts/perf.sh

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.common.stats import p50_p95  # noqa: E402

MODEL = "tts"


# ---------------------------------------------------------------- hàm thuần


def rtf(latency_s: float, audio_s: float) -> float:
    """Thời gian sinh / độ dài audio sinh ra. Dưới 1.0 là nhanh hơn thời gian thực.

    perf_analyzer đo được latency nhưng không đọc output tensor, mà độ dài audio
    lại đổi theo từng câu - nên mẫu số chỉ có ở đây.
    """
    if audio_s <= 0:
        raise ValueError(f"audio dài {audio_s}s - model trả WAV rỗng")
    return round(latency_s / audio_s, 3)


def summarize_run(rtfs) -> dict:
    """Gom các lần đo thành một hàng kết quả."""
    if not len(rtfs):
        raise ValueError("không có lần đo nào")
    p50, p95 = p50_p95(rtfs)
    return {"rtf_p50": round(p50, 3), "rtf_p95": round(p95, 3)}


COLUMNS = [("rtf_p50", "RTF p50"), ("rtf_p95", "RTF p95")]


def render_table(row: dict) -> str:
    head = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    rule = "|" + "---|" * len(COLUMNS)
    body = "| " + " | ".join(str(row[key]) for key, _ in COLUMNS) + " |"
    return "\n".join([head, rule, body])


# ------------------------------------------------------------- phần gọi server


def synthesize(client, model: str, text: str) -> tuple[float, float]:
    """Sinh một câu, trả (giây chờ, giây audio nhận được)."""
    inp = grpcclient.InferInput("TEXT", [1], "BYTES")
    inp.set_data_from_numpy(np.array([text.encode("utf-8")], dtype=object))
    t0 = time.perf_counter()
    out = client.infer(model, [inp])
    latency_s = time.perf_counter() - t0
    wav = out.as_numpy("WAV")
    sample_rate = int(out.as_numpy("SAMPLE_RATE")[0])
    return latency_s, len(wav) / sample_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", default="tests/assets/sample_vi.txt")
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--rounds", type=int, default=10, help="số lần sinh mỗi câu")
    ap.add_argument("--out", default="bench/tts/results/metrics.md")
    args = ap.parse_args()

    texts = [ln.strip() for ln in Path(args.text_file).read_text().splitlines() if ln.strip()]
    if not texts:
        raise SystemExit(f"{args.text_file} không có câu nào")
    client = grpcclient.InferenceServerClient(args.url)

    # Lần sinh đầu gánh chi phí nạp vocoder lên GPU, bỏ đi kẻo kéo lệch p50
    print("warmup...")
    synthesize(client, args.model, texts[0])

    rtfs = []
    for r in range(args.rounds):
        for text in texts:
            latency_s, audio_s = synthesize(client, args.model, text)
            rtfs.append(rtf(latency_s, audio_s))
        print(f"  lượt {r + 1}/{args.rounds}: RTF gần nhất {rtfs[-1]}", flush=True)

    row = summarize_run(rtfs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = render_table(row)
    out.write_text(
        f"# tts - chỉ số riêng của model\n\n{table}\n\n"
        f"RTF = giây tính toán / giây audio sinh ra · {len(rtfs)} lần sinh "
        f"từ `{Path(args.text_file).name}`\n"
        "latency / throughput / GPU util: xem scripts/perf.sh\n"
    )
    print(f"\n{table}\n\nĐã ghi {out}")


if __name__ == "__main__":
    main()
