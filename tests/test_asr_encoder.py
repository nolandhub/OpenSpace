# ABOUTME: Integration test cho model asr_encoder
# ABOUTME: Đưa tensor ngẫu nhiên vào, kiểm tra shape đầu ra hợp lệ

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_encoder_returns_subsampled_frames(triton):
    import tritonclient.grpc as grpcclient

    batch_size, frames = 1, 1600
    x = np.random.randn(batch_size, frames, 80).astype(np.float32)
    x_lens = np.array([[frames]], dtype=np.int64)

    inputs = [
        grpcclient.InferInput("x", list(x.shape), "FP32"),
        grpcclient.InferInput("x_lens", list(x_lens.shape), "INT64"),
    ]
    inputs[0].set_data_from_numpy(x)
    inputs[1].set_data_from_numpy(x_lens)

    response = triton.infer("asr_encoder", inputs)
    encoder_out = response.as_numpy("encoder_out")
    encoder_out_lens = response.as_numpy("encoder_out_lens")

    assert encoder_out.ndim == 3 and encoder_out.shape[0] == batch_size
    # Zipformer subsample khoảng 4 lần nên số khung ra phải nhỏ hơn hẳn đầu vào
    assert 0 < encoder_out.shape[1] < frames
    assert int(encoder_out_lens.reshape(-1)[0]) == encoder_out.shape[1]
