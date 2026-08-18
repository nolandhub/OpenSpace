# ABOUTME: Sinh file input JSON cho perf_analyzer - text thật, mỗi câu một request
# ABOUTME: Chạy: python bench/tts/gen_input.py --text-file tests/assets/sample_vi.txt

import argparse
import json
from pathlib import Path


def payload(texts) -> dict:
    """`data` phẳng một tầng - tts không phải sequence model, perf_analyzer xoay
    vòng qua các phần tử cho từng request."""
    if not texts:
        raise ValueError("không có text nào để sinh")
    return {"data": [{"TEXT": [t]} for t in texts]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", default="tests/assets/sample_vi.txt")
    ap.add_argument("--out", default="bench/tts/results/input.json")
    args = ap.parse_args()

    texts = [ln.strip() for ln in Path(args.text_file).read_text().splitlines() if ln.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload(texts)))
    print(f"đã ghi {out} — {len(texts)} câu")


if __name__ == "__main__":
    main()
