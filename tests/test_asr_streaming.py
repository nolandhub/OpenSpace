# ABOUTME: Integration test cho asr_streaming - gửi wav theo chunk qua gRPC stream
# ABOUTME: Kiểm partial dài dần, final khớp nội dung, 2 stream đan xen không lẫn state

import queue
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.common import SAMPLE_RATE  # noqa: E402

pytestmark = pytest.mark.integration

ASSETS = Path(__file__).parent / "assets"
URL = "localhost:8001"
CHUNK_SAMPLES = 3200   # 200ms @ 16kHz


def _chunks(wav):
    return [wav[i : i + CHUNK_SAMPLES] for i in range(0, len(wav), CHUNK_SAMPLES)]


def _load_sample():
    wav, sample_rate = sf.read(ASSETS / "sample_vi.wav", dtype="float32")
    assert sample_rate == SAMPLE_RATE
    return wav


def _send_chunks(client, grpcclient, parts, seq_id, chunk_index):
    part = parts[chunk_index]
    inp = grpcclient.InferInput("AUDIO_CHUNK", [1, len(part)], "FP32")
    inp.set_data_from_numpy(part.reshape(1, -1))
    client.async_stream_infer(
        "asr_streaming",
        [inp],
        sequence_id=seq_id,
        sequence_start=(chunk_index == 0),
        sequence_end=(chunk_index == len(parts) - 1),
    )


def _stream_transcripts(wav, seq_id):
    """Gửi cả wav qua một stream, trả danh sách transcript theo thứ tự chunk."""
    import tritonclient.grpc as grpcclient

    client = grpcclient.InferenceServerClient(URL)
    q = queue.Queue()
    client.start_stream(callback=lambda result, error: q.put((result, error)))
    parts = _chunks(wav)
    for i in range(len(parts)):
        _send_chunks(client, grpcclient, parts, seq_id, i)
    texts = []
    for _ in parts:
        result, error = q.get(timeout=60)
        assert error is None, error
        texts.append(result.as_numpy("TRANSCRIPT")[0, 0].decode("utf-8"))
    client.stop_stream()
    return texts


def test_partials_grow_and_final_matches_reference(triton):
    wav = _load_sample()
    expected = (ASSETS / "sample_vi.txt").read_text(encoding="utf-8").strip().lower()
    assert expected, "sample_vi.txt rỗng - phải chứa câu được nói trong sample_vi.wav"

    texts = _stream_transcripts(wav, seq_id=101)

    assert len(texts) == len(_chunks(wav))
    lengths = [len(t) for t in texts]
    assert lengths == sorted(lengths), f"partial phải dài dần: {lengths}"

    final = texts[-1].strip().lower()
    assert final, "final transcript rỗng"
    expected_words = set(expected.split())
    overlap = len(expected_words & set(final.split())) / len(expected_words)
    assert overlap >= 0.6, f"mong đợi '{expected}', nhận '{final}'"


def test_two_interleaved_streams_share_no_state(triton):
    """Cùng audio trên 2 stream đan xen từng chunk - greedy tất định nên final phải trùng nhau."""
    import tritonclient.grpc as grpcclient

    wav = _load_sample()
    parts = _chunks(wav)
    clients, queues = {}, {}
    for sid in (201, 202):
        c = grpcclient.InferenceServerClient(URL)
        q = queue.Queue()
        c.start_stream(callback=lambda result, error, q=q: q.put((result, error)))
        clients[sid], queues[sid] = c, q

    for i in range(len(parts)):
        for sid in (201, 202):
            _send_chunks(clients[sid], grpcclient, parts, sid, i)

    finals = {}
    for sid in (201, 202):
        texts = []
        for _ in parts:
            result, error = queues[sid].get(timeout=120)
            assert error is None, error
            texts.append(result.as_numpy("TRANSCRIPT")[0, 0].decode("utf-8"))
        finals[sid] = texts[-1]
        clients[sid].stop_stream()

    assert finals[201] == finals[202]
    assert finals[201].strip()
