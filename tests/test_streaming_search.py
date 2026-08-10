# ABOUTME: Unit test cho greedy search theo chunk - đối chiếu với greedy_search cả câu của asr_scorer
# ABOUTME: decoder/joiner giả cùng kiểu test_greedy_search.py, không cần GPU/ONNX

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "model_repository/asr_scorer/1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "model_repository/asr_streaming/1"))
from greedy_search import greedy_search  # noqa: E402
from streaming_search import emitted_tokens, greedy_search_step, init_search_state  # noqa: E402

DIM = 4
VOCAB_SIZE = 10


def fake_decoder(_context):
    """Decoder giả - luôn trả vector 0, đủ dùng vì joiner giả không nhìn tới nó."""
    return np.zeros((1, DIM), dtype=np.float32)


def fake_joiner(emitted):
    """Joiner giả phát token theo kịch bản, mỗi lần gọi lấy 1 phần tử - giữ state qua các chunk."""
    state = {"step": 0}

    def _joiner(_enc, _dec):
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        logits[0, emitted[state["step"]]] = 1.0
        state["step"] += 1
        return logits

    return _joiner


def test_chunked_matches_full_utterance():
    """Bất biến quan trọng nhất: cắt encoder_out kiểu gì thì kết quả cũng như chạy một lần."""
    script = [0, 7, 0, 3, 0, 0, 9, 0, 2, 0]
    enc = np.zeros((10, DIM), dtype=np.float32)
    want = greedy_search(enc, 10, fake_decoder, fake_joiner(script))

    state = init_search_state(fake_decoder)
    joiner = fake_joiner(script)
    for part in np.split(enc, [3, 4, 8]):   # các chunk 3, 1, 4, 2 khung
        greedy_search_step(part, state, fake_decoder, joiner)
    assert emitted_tokens(state) == want


def test_empty_chunk_changes_nothing():
    state = init_search_state(fake_decoder)
    greedy_search_step(np.zeros((0, DIM), dtype=np.float32), state, fake_decoder, fake_joiner([]))
    assert emitted_tokens(state) == []


def test_context_carries_across_chunks():
    seen = []

    def recording_decoder(context):
        seen.append(list(context))
        return np.zeros((1, DIM), dtype=np.float32)

    enc = np.zeros((4, DIM), dtype=np.float32)
    state = init_search_state(recording_decoder)
    joiner = fake_joiner([2, 0, 6, 0])
    greedy_search_step(enc[:2], state, recording_decoder, joiner)
    greedy_search_step(enc[2:], state, recording_decoder, joiner)

    # khởi tạo [0,0]; sau token 2 → [0,2]; sau token 6 (chunk hai) → [2,6]
    assert seen == [[0, 0], [0, 2], [2, 6]]
