# ABOUTME: Unit test cho vòng lặp greedy search - không cần GPU, server hay ONNX
# ABOUTME: decoder và joiner được thay bằng hàm giả để kiểm tra đúng logic vòng lặp

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "model_repository/asr_scorer/1"))
from greedy_search import greedy_search  # noqa: E402

DIM = 4
VOCAB_SIZE = 10


def fake_decoder(_context):
    """Decoder giả - luôn trả về vector 0, đủ dùng vì joiner giả không nhìn tới nó."""
    return np.zeros((1, DIM), dtype=np.float32)


def fake_joiner(emitted_tokens):
    """Tạo joiner giả phát ra token theo đúng kịch bản, mỗi lần gọi lấy 1 phần tử."""
    state = {"step": 0}

    def _joiner(_encoder_frame, _decoder_out):
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        logits[0, emitted_tokens[state["step"]]] = 1.0
        state["step"] += 1
        return logits

    return _joiner


def test_all_blank_produces_no_tokens():
    encoder_out = np.zeros((5, DIM), dtype=np.float32)
    result = greedy_search(encoder_out, 5, fake_decoder, fake_joiner([0] * 5))
    assert result == []


def test_tokens_come_out_in_order():
    encoder_out = np.zeros((5, DIM), dtype=np.float32)
    # khung 0 blank, khung 1 ra token 7, khung 2-3 blank, khung 4 ra token 3
    result = greedy_search(encoder_out, 5, fake_decoder, fake_joiner([0, 7, 0, 0, 3]))
    assert result == [7, 3]


def test_decoder_only_reruns_on_non_blank():
    """Đây là tối ưu chính của vòng lặp: blank thì lịch sử text không đổi nên bỏ qua decoder."""
    call_count = {"n": 0}

    def counting_decoder(_context):
        call_count["n"] += 1
        return np.zeros((1, DIM), dtype=np.float32)

    encoder_out = np.zeros((6, DIM), dtype=np.float32)
    greedy_search(encoder_out, 6, counting_decoder, fake_joiner([0, 5, 0, 0, 9, 0]))

    # 1 lần khởi tạo + 2 lần vì phát ra 2 token
    assert call_count["n"] == 3


def test_decoder_receives_last_two_tokens():
    seen_contexts = []

    def recording_decoder(context):
        seen_contexts.append(list(context))
        return np.zeros((1, DIM), dtype=np.float32)

    encoder_out = np.zeros((4, DIM), dtype=np.float32)
    greedy_search(encoder_out, 4, recording_decoder, fake_joiner([2, 6, 0, 0]))

    assert seen_contexts[0] == [0, 0]   # khởi tạo: toàn blank
    assert seen_contexts[1] == [0, 2]   # sau khi phát token 2
    assert seen_contexts[2] == [2, 6]   # sau khi phát token 6
    assert all(len(c) == 2 for c in seen_contexts)


def test_only_walks_requested_frames():
    """encoder_out có thể dài hơn num_frames vì phần đệm - không được đụng vào phần đó."""
    encoder_out = np.zeros((10, DIM), dtype=np.float32)
    result = greedy_search(encoder_out, 3, fake_decoder, fake_joiner([1, 2, 3, 4, 5]))
    assert result == [1, 2, 3]
