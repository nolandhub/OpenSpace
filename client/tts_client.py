# ABOUTME: Client dòng lệnh cho TTS - nhận text, ghi ra file wav
# ABOUTME: Chạy: python client/tts_client.py --text "Xin chào" --out ra.wav

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.common import load_wav_16k  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="ra.wav")
    ap.add_argument("--prompt", help="file wav giọng mẫu, tần số nào cũng được")
    ap.add_argument("--prompt-text", help="nội dung đã nói trong file giọng mẫu")
    ap.add_argument("--num-steps", type=int,
                    help="bỏ trống thì dùng mặc định khai trong config.pbtxt")
    ap.add_argument("--url", default="localhost:8001")
    args = ap.parse_args()

    text_input = grpcclient.InferInput("TEXT", [1], "BYTES")
    text_input.set_data_from_numpy(np.array([args.text.encode("utf-8")], dtype=object))
    inputs = [text_input]

    # Chỉ gửi NUM_STEPS khi người dùng chỉ định. Gửi cứng một giá trị ở đây sẽ
    # ghi đè mặc định của model dù người dùng không yêu cầu.
    if args.num_steps is not None:
        steps_input = grpcclient.InferInput("NUM_STEPS", [1], "INT32")
        steps_input.set_data_from_numpy(np.array([args.num_steps], dtype=np.int32))
        inputs.append(steps_input)

    if args.prompt:
        if not args.prompt_text:
            raise SystemExit("Dùng --prompt thì phải kèm --prompt-text")
        # Server luôn ghi lại prompt ở 16kHz, nên phải hạ tần số ở đây.
        # Gửi thẳng file 24kHz vào thì giọng mẫu bị chậm 1.5 lần và trầm hẳn
        # xuống mà không báo lỗi gì - model sẽ clone đúng cái giọng méo đó.
        wav = load_wav_16k(args.prompt)
        pw = grpcclient.InferInput("PROMPT_WAV", [len(wav)], "FP32")
        pw.set_data_from_numpy(wav)
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
