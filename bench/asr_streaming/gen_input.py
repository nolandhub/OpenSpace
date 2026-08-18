# ABOUTME: Sinh file input JSON cho perf_analyzer - audio thật, đúng thứ tự chunk
# ABOUTME: Chạy: python bench/asr_streaming/gen_input.py --streams 4 --chunks 50

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from client.common import chunk_wav, load_wav_16k  # noqa: E402


def payload(chunks, n_streams: int) -> dict:
    """Dạng `data` lồng: mỗi mảng con là MỘT sequence, phần tử là các bước.

    Phẳng một tầng thì perf_analyzer hiểu thành nhiều sequence riêng lẻ mỗi cái
    một bước, encoder không bao giờ tích luỹ state và số đo mất hết ý nghĩa.
    """
    if not len(chunks):
        raise ValueError("không có chunk nào - audio rỗng hoặc ngắn hơn một chunk")
    steps = [{"AUDIO_CHUNK": c.tolist()} for c in chunks]
    return {"data": [steps for _ in range(n_streams)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="tests/assets/sample_vi_long.wav")
    ap.add_argument("--chunk-ms", type=int, default=200)
    ap.add_argument("--streams", type=int, default=4, help="số sequence trong file")
    ap.add_argument("--chunks", type=int, default=50, help="số bước mỗi sequence")
    ap.add_argument("--out", default="bench/asr_streaming/results/input.json")
    args = ap.parse_args()

    chunks = chunk_wav(load_wav_16k(args.wav), args.chunk_ms)[: args.chunks]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload(chunks, args.streams)))
    print(f"đã ghi {out} — {args.streams} stream × {len(chunks)} bước × {args.chunk_ms}ms")


if __name__ == "__main__":
    main()
