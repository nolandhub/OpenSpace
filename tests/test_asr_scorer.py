# ABOUTME: Integration test cho model asr_scorer
# ABOUTME: Đưa encoder_out ngẫu nhiên vào, chỉ kiểm tra hợp đồng đầu ra chứ không kiểm nội dung

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_scorer_returns_utf8_string(triton):
    import tritonclient.grpc as grpcclient

    # encoder_out ngẫu nhiên - text ra sẽ vô nghĩa, nhưng phải đúng kiểu và không lỗi
    encoder_out = np.random.randn(1, 50, 512).astype(np.float32)
    encoder_out_lens = np.array([[50]], dtype=np.int64)

    inputs = [
        grpcclient.InferInput("ENCODER_OUT", list(encoder_out.shape), "FP32"),
        grpcclient.InferInput("ENCODER_OUT_LEN", list(encoder_out_lens.shape), "INT64"),
    ]
    inputs[0].set_data_from_numpy(encoder_out)
    inputs[1].set_data_from_numpy(encoder_out_lens)

    response = triton.infer("asr_scorer", inputs)
    transcript = response.as_numpy("TRANSCRIPT")

    assert transcript.shape == (1, 1)
    assert isinstance(transcript[0, 0].decode("utf-8"), str)
