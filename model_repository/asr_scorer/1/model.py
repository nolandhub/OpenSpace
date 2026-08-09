# ABOUTME: Triton Python backend - chạy greedy search rồi dịch token id sang tiếng Việt
# ABOUTME: Logic vòng lặp nằm ở greedy_search.py, file này chỉ nối nó với ONNX và Triton

import os
import sys

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import triton_python_backend_utils as pb_utils

# Triton không tự thêm thư mục model vào sys.path nên phải tự làm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from greedy_search import greedy_search  # noqa: E402

BLANK_ID = 0
CONTEXT_SIZE = 2


class TritonPythonModel:
    def initialize(self, args):
        d = os.path.join(args["model_repository"], args["model_version"])
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Nạp một lần duy nhất, dùng lại cho mọi request
        self.decoder = ort.InferenceSession(
            os.path.join(d, "decoder.onnx"), providers=providers
        )
        self.joiner = ort.InferenceSession(
            os.path.join(d, "joiner.onnx"), providers=providers
        )
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(d, "bpe.model"))

        # Lấy tên cổng từ chính file ONNX thay vì viết cứng - đỡ lệ thuộc phiên bản export
        self.decoder_in = self.decoder.get_inputs()[0].name
        self.joiner_in = [i.name for i in self.joiner.get_inputs()]

    def _run_decoder(self, context):
        y = np.array([context], dtype=np.int64)
        out = self.decoder.run(None, {self.decoder_in: y})[0]
        # Một số bản export trả về (N, 1, C), bỏ chiều giữa cho khớp joiner
        return out[:, 0, :] if out.ndim == 3 else out

    def _run_joiner(self, enc_frame, dec_out):
        feeds = {self.joiner_in[0]: enc_frame, self.joiner_in[1]: dec_out}
        return self.joiner.run(None, feeds)[0]

    def execute(self, requests):
        responses = []

        for request in requests:
            enc = pb_utils.get_input_tensor_by_name(request, "ENCODER_OUT").as_numpy()
            lens = pb_utils.get_input_tensor_by_name(
                request, "ENCODER_OUT_LEN"
            ).as_numpy()

            texts = []
            for i in range(enc.shape[0]):
                tokens = greedy_search(
                    encoder_out=enc[i],
                    num_frames=int(lens[i, 0]),
                    run_decoder=self._run_decoder,
                    run_joiner=self._run_joiner,
                    blank_id=BLANK_ID,
                    context_size=CONTEXT_SIZE,
                )
                texts.append(self.sp.decode(tokens))

            out = np.array([[t.encode("utf-8")] for t in texts], dtype=object)
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("TRANSCRIPT", out)]
                )
            )

        return responses
