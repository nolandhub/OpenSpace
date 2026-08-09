# ABOUTME: Integration test cho model asr_feature
# ABOUTME: Kiểm tra fbank ra đúng shape cố định và độ dài thật được tính đúng

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.common import MAX_FRAMES, NUM_MEL_BINS, SAMPLE_RATE, pad_wav  # noqa: E402

pytestmark = pytest.mark.integration


def test_feature_always_returns_fixed_shape(triton):
    import tritonclient.grpc as grpcclient

    # 3 giây tiếng ồn - ngắn hơn nhiều so với giới hạn 16 giây
    wav = np.random.randn(3 * SAMPLE_RATE).astype(np.float32) * 0.1
    padded, real_len = pad_wav(wav)

    inputs = [
        grpcclient.InferInput("WAV", [1, len(padded)], "FP32"),
        grpcclient.InferInput("WAV_LEN", [1, 1], "INT32"),
    ]
    inputs[0].set_data_from_numpy(padded.reshape(1, -1))
    inputs[1].set_data_from_numpy(np.array([[real_len]], dtype=np.int32))

    response = triton.infer("asr_feature", inputs)
    speech = response.as_numpy("SPEECH")
    speech_len = response.as_numpy("SPEECH_LEN")

    # Shape phải cố định bất kể audio dài bao nhiêu - đây là điều kiện để encoder batch được
    assert speech.shape == (1, MAX_FRAMES, NUM_MEL_BINS)

    # Độ dài thật phải tương ứng 3 giây, không phải 16 giây
    assert 280 < int(speech_len.reshape(-1)[0]) < 320

    # Phần đệm phía sau phải bằng 0
    real_frames = int(speech_len.reshape(-1)[0])
    assert np.allclose(speech[0, real_frames:], 0.0)
