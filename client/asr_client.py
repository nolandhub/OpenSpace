# ABOUTME: Client dòng lệnh cho ASR - nhận file wav, in ra transcript
# ABOUTME: Chạy: python client/asr_client.py file.wav

import argparse
import sys
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.common import load_wav_16k, pad_wav  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="file wav, tần số nào cũng được - tự hạ về 16kHz")
    ap.add_argument("--url", default="localhost:8001")
    args = ap.parse_args()

    wav = load_wav_16k(args.wav)
    padded, real_len = pad_wav(wav)

    client = grpcclient.InferenceServerClient(args.url)
    inputs = [
        grpcclient.InferInput("WAV", [1, len(padded)], "FP32"),
        grpcclient.InferInput("WAV_LEN", [1, 1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(padded.reshape(1, -1))
    inputs[1].set_data_from_numpy(np.array([[real_len]], dtype=np.int32))

    response = client.infer("asr", inputs)
    print(response.as_numpy("TRANSCRIPT")[0, 0].decode("utf-8"))


if __name__ == "__main__":
    main()
