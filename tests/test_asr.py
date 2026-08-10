# ABOUTME: Integration test đầu-cuối cho ensemble asr
# ABOUTME: Gửi file wav tiếng Việt thật, kiểm tra transcript khớp nội dung đã nói

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.common import SAMPLE_RATE, pad_wav  # noqa: E402

pytestmark = pytest.mark.integration

ASSETS = Path(__file__).parent / "assets"


def test_recognizes_vietnamese_sentence(triton):
    import tritonclient.grpc as grpcclient

    wav, sample_rate = sf.read(ASSETS / "sample_vi.wav", dtype="float32")
    assert sample_rate == SAMPLE_RATE
    expected = (ASSETS / "sample_vi.txt").read_text(encoding="utf-8").strip().lower()
    # Kiểm tra fixture trước khi gọi server - hỏng ở đây thì lỗi phải nói rõ là
    # do dữ liệu test, đừng để nó nổ thành ZeroDivisionError ở phép chia bên dưới
    assert expected, (
        f"{ASSETS / 'sample_vi.txt'} rỗng — file này phải chứa đúng câu "
        f"được nói trong sample_vi.wav"
    )

    padded, real_len = pad_wav(wav)
    inputs = [
        grpcclient.InferInput("WAV", [1, len(padded)], "FP32"),
        grpcclient.InferInput("WAV_LEN", [1, 1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(padded.reshape(1, -1))
    inputs[1].set_data_from_numpy(np.array([[real_len]], dtype=np.int32))

    response = triton.infer("asr", inputs)
    actual = response.as_numpy("TRANSCRIPT")[0, 0].decode("utf-8").strip().lower()

    assert actual, "transcript rỗng"

    # So theo từ thay vì so nguyên chuỗi - cho phép sai vài từ nhưng không được sai hết
    expected_words = set(expected.split())
    actual_words = set(actual.split())
    overlap = len(expected_words & actual_words) / len(expected_words)
    assert overlap >= 0.6, f"mong đợi '{expected}', nhận được '{actual}'"
