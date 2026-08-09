# ABOUTME: Triton Python backend - trích đặc trưng fbank 80 chiều từ waveform 16kHz
# ABOUTME: Luôn xuất ra số khung cố định để tầng encoder phía sau batch được

import json

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # args["model_config"] là chuỗi JSON của chính config.pbtxt
        config = json.loads(args["model_config"])
        params = config.get("parameters", {})
        self.max_frames = int(params["max_frames"]["string_value"])
        self.num_mel_bins = 80
        self.sample_rate = 16000

    def _fbank(self, samples: np.ndarray) -> np.ndarray:
        """Tính fbank cho một câu. samples: (N,) float32 -> (T, 80)"""
        wav = torch.from_numpy(samples).unsqueeze(0)
        feat = kaldi.fbank(
            wav,
            num_mel_bins=self.num_mel_bins,
            frame_length=25.0,
            frame_shift=10.0,
            dither=0.0,
            sample_frequency=self.sample_rate,
            snip_edges=False,
        )
        return feat.numpy()

    def execute(self, requests):
        # requests là một DANH SÁCH. Với backend python, Triton không tự gộp
        # các request lại thành batch - mình tự quyết định xử lý thế nào.
        responses = []

        for request in requests:
            wav = pb_utils.get_input_tensor_by_name(request, "WAV").as_numpy()
            wav_len = pb_utils.get_input_tensor_by_name(request, "WAV_LEN").as_numpy()

            batch = wav.shape[0]
            speech = np.zeros(
                (batch, self.max_frames, self.num_mel_bins), dtype=np.float32
            )
            speech_len = np.zeros((batch, 1), dtype=np.int64)

            for i in range(batch):
                # Chỉ tính trên phần audio thật, bỏ qua phần client đã đệm
                feat = self._fbank(wav[i, : int(wav_len[i, 0])])
                frames = min(feat.shape[0], self.max_frames)
                speech[i, :frames] = feat[:frames]
                speech_len[i, 0] = frames

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("SPEECH", speech),
                        pb_utils.Tensor("SPEECH_LEN", speech_len),
                    ]
                )
            )

        # Bắt buộc: số response phải bằng số request, đúng thứ tự
        return responses
