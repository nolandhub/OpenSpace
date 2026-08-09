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
