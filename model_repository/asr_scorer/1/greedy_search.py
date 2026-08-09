# ABOUTME: Vòng lặp greedy search cho RNN-Transducer, viết thuần numpy
# ABOUTME: Không phụ thuộc Triton hay ONNX - decoder/joiner được truyền vào dạng hàm

from typing import Callable, List

import numpy as np


def greedy_search(
    encoder_out: np.ndarray,
    num_frames: int,
    run_decoder: Callable[[List[int]], np.ndarray],
    run_joiner: Callable[[np.ndarray, np.ndarray], np.ndarray],
    blank_id: int = 0,
    context_size: int = 2,
) -> List[int]:
    """Giải mã một câu bằng greedy search.

    encoder_out: (T, C) - biểu diễn của cả câu, đã bỏ chiều batch
    num_frames:  số khung thật, phần còn lại của encoder_out là đệm, bỏ qua
    run_decoder: nhận danh sách token ngữ cảnh, trả về (1, C)
    run_joiner:  nhận (1, C) khung audio và (1, C) trạng thái text, trả về (1, vocab)

    Trả về danh sách token id, đã bỏ phần ngữ cảnh khởi tạo.
    """
    # Ngữ cảnh khởi tạo toàn blank - coi như chưa viết được chữ nào
    hyp: List[int] = [blank_id] * context_size
    decoder_out = run_decoder(hyp[-context_size:])

    for t in range(num_frames):
        # encoder_out[t] là (C,), thêm chiều batch thành (1, C) cho khớp joiner
        logits = run_joiner(encoder_out[t : t + 1], decoder_out)
        token = int(np.argmax(logits.reshape(-1)))

        if token != blank_id:
            # Chỉ khi phát ra token thật thì lịch sử text mới đổi,
            # nên chỉ khi đó mới phải chạy lại decoder. Bỏ qua được đa số lần gọi.
            hyp.append(token)
            decoder_out = run_decoder(hyp[-context_size:])

    return hyp[context_size:]
