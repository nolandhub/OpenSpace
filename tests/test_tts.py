# ABOUTME: Integration test cho model tts
# ABOUTME: Sinh audio từ câu tiếng Việt, kiểm tra định dạng và độ dài hợp lý

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def _synthesize(triton, text, num_steps=8):
    import tritonclient.grpc as grpcclient

    inputs = [
        grpcclient.InferInput("TEXT", [1], "BYTES"),
        grpcclient.InferInput("NUM_STEPS", [1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(np.array([text.encode("utf-8")], dtype=object))
    inputs[1].set_data_from_numpy(np.array([num_steps], dtype=np.int32))

    response = triton.infer("tts", inputs)
    return response.as_numpy("WAV"), int(response.as_numpy("SAMPLE_RATE")[0])


def test_generates_valid_audio(triton):
    wav, sample_rate = _synthesize(triton, "Xin chào, tôi là trợ lý ảo.")

    assert sample_rate == 24000
    assert wav.ndim == 1 and len(wav) > 0
    assert not np.isnan(wav).any(), "audio có NaN"
    assert np.abs(wav).max() <= 1.0, "audio vượt biên độ"
    assert np.abs(wav).max() > 0.01, "audio gần như im lặng"

    # Câu ~26 ký tự thì độ dài hợp lý nằm trong khoảng 1 đến 8 giây
    assert 1.0 < len(wav) / sample_rate < 8.0


def test_longer_text_gives_longer_audio(triton):
    short_wav, _ = _synthesize(triton, "Xin chào.")
    long_wav, _ = _synthesize(
        triton,
        "Xin chào, hôm nay trời rất đẹp và tôi muốn đi dạo một vòng quanh hồ.",
    )
    assert len(long_wav) > len(short_wav)
