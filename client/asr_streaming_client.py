# ABOUTME: Client streaming ASR - cắt wav thành chunk, gửi qua một gRPC stream
# ABOUTME: Chạy: python client/asr_streaming_client.py file.wav (--fast để gửi dồn không mô phỏng mic)

import argparse
import queue
import sys
import time
from pathlib import Path

import tritonclient.grpc as grpcclient

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.common import SAMPLE_RATE, load_wav_16k  # noqa: E402


def _print_partial(result):
    # in đè dòng hiện tại để partial chạy như phụ đề trực tiếp
    print(f"\r{result.as_numpy('TRANSCRIPT')[0, 0].decode('utf-8')}", end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="file wav, tần số nào cũng được - tự hạ về 16kHz")
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--chunk-ms", type=int, default=200)
    ap.add_argument("--fast", action="store_true", help="gửi dồn, không ngủ giữa các chunk")
    args = ap.parse_args()

    wav = load_wav_16k(args.wav)
    chunk = SAMPLE_RATE * args.chunk_ms // 1000
    parts = [wav[i : i + chunk] for i in range(0, len(wav), chunk)]

    q = queue.Queue()
    client = grpcclient.InferenceServerClient(args.url)
    client.start_stream(callback=lambda result, error: q.put((result, error)))
    seq_id = int(time.time()) % 2**31 + 1   # đủ khác nhau giữa các lần chạy

    received = 0
    for i, part in enumerate(parts):
        inp = grpcclient.InferInput("AUDIO_CHUNK", [1, len(part)], "FP32")
        inp.set_data_from_numpy(part.reshape(1, -1))
        client.async_stream_infer(
            "asr_streaming",
            [inp],
            sequence_id=seq_id,
            sequence_start=(i == 0),
            sequence_end=(i == len(parts) - 1),
        )
        if not args.fast:
            time.sleep(args.chunk_ms / 1000)
        # in mọi partial đã về trong lúc chờ, không chặn vòng gửi
        while True:
            try:
                result, error = q.get_nowait()
            except queue.Empty:
                break
            if error:
                raise SystemExit(f"lỗi từ server: {error}")
            received += 1
            _print_partial(result)

    while received < len(parts):
        result, error = q.get(timeout=30)
        if error:
            raise SystemExit(f"lỗi từ server: {error}")
        received += 1
        _print_partial(result)
    print()
    client.stop_stream()


if __name__ == "__main__":
    main()
