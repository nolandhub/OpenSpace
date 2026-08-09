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
