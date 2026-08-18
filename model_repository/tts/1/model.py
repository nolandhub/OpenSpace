# ABOUTME: Triton Python backend - sinh giọng nói tiếng Việt bằng ZipVoice
# ABOUTME: Nạp model một lần trong initialize, mỗi request chạy flow matching rồi vocoder

import json
import os
import uuid

import numpy as np
import soundfile as sf
import torch
import triton_python_backend_utils as pb_utils
from zipvoice.bin.infer_zipvoice import generate_sentence, get_vocoder
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EspeakTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank

SAMPLING_RATE = 24000
PROMPT_SAMPLE_RATE = 16000
TMP_DIR = "/dev/shm"   # tmpfs nằm trong RAM, không chạm đĩa


class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.join(args["model_repository"], args["model_version"])
        self.device = torch.device("cuda")

        params = json.loads(args["model_config"]).get("parameters", {})
        self.model_name = params["model_name"]["string_value"]
        self.default_num_step = int(params["num_step"]["string_value"])
        self.guidance_scale = float(params["guidance_scale"]["string_value"])
        lang = params["lang"]["string_value"]

        # ZipVoice quy ước file cấu hình tên model.json (fetch_models.sh đã đổi tên)
        with open(os.path.join(model_dir, "model.json")) as f:
            model_config = json.load(f)

        self.tokenizer = EspeakTokenizer(
            token_file=os.path.join(model_dir, "tokens.txt"), lang=lang
        )
        tokenizer_config = {
            "vocab_size": self.tokenizer.vocab_size,
            "pad_id": self.tokenizer.pad_id,
        }

        model_class = ZipVoice if self.model_name == "zipvoice" else ZipVoiceDistill
        self.model = model_class(**model_config["model"], **tokenizer_config)
        load_checkpoint(
            filename=os.path.join(model_dir, "model.pt"), model=self.model, strict=True
        )
        self.model = self.model.to(self.device).eval()

        self.vocoder = (
            get_vocoder(os.path.join(model_dir, "vocos")).to(self.device).eval()
        )
        self.feature_extractor = VocosFbank()

        # Giọng mẫu mặc định cho zero-shot, đóng gói sẵn trong repo
        self.default_prompt_wav = os.path.join(model_dir, "assets", "prompt.wav")
        with open(
            os.path.join(model_dir, "assets", "prompt.txt"), encoding="utf-8"
        ) as f:
            self.default_prompt_text = f.read().strip()

    @staticmethod
    def _get_optional(request, name, default=None):
        """Đọc input tuỳ chọn - client không gửi thì trả về giá trị mặc định."""
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        return default if tensor is None else tensor.as_numpy()

    def execute(self, requests):
        responses = []

        for request in requests:
            text = self._get_optional(request, "TEXT")[0].decode("utf-8")
            num_step = int(
                self._get_optional(
                    request, "NUM_STEPS", np.array([self.default_num_step])
                )[0]
            )
            # speed < 1 đọc chậm lại, > 1 đọc nhanh hơn
            speed = float(
                self._get_optional(request, "SPEED", np.array([1.0], dtype=np.float32))[0]
            )
            # guidance_scale cao thì bám ngữ điệu của giọng mẫu chặt hơn
            guidance_scale = float(
                self._get_optional(
                    request,
                    "GUIDANCE_SCALE",
                    np.array([self.guidance_scale], dtype=np.float32),
                )[0]
            )

            prompt_wav_array = self._get_optional(request, "PROMPT_WAV")
            prompt_text_array = self._get_optional(request, "PROMPT_TEXT")

            tmp_files = []
            if prompt_wav_array is not None:
                # Client gửi giọng mẫu riêng - ghi ra tmpfs vì ZipVoice nhận đường dẫn
                prompt_wav = f"{TMP_DIR}/prompt_{uuid.uuid4().hex}.wav"
                sf.write(
                    prompt_wav, prompt_wav_array.reshape(-1), PROMPT_SAMPLE_RATE
                )
                tmp_files.append(prompt_wav)
                prompt_text = prompt_text_array[0].decode("utf-8")
            else:
                prompt_wav = self.default_prompt_wav
                prompt_text = self.default_prompt_text

            out_path = f"{TMP_DIR}/tts_{uuid.uuid4().hex}.wav"
            tmp_files.append(out_path)

            try:
                with torch.inference_mode():
                    generate_sentence(
                        save_path=out_path,
                        prompt_text=prompt_text,
                        prompt_wav=prompt_wav,
                        text=text,
                        model=self.model,
                        vocoder=self.vocoder,
                        tokenizer=self.tokenizer,
                        feature_extractor=self.feature_extractor,
                        device=self.device,
                        num_step=num_step,
                        guidance_scale=guidance_scale,
                        speed=speed,
                        sampling_rate=SAMPLING_RATE,
                    )
                wav, _ = sf.read(out_path, dtype="float32")
            finally:
                for path in tmp_files:
                    if os.path.exists(path):
                        os.remove(path)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("WAV", wav.reshape(-1).astype(np.float32)),
                        pb_utils.Tensor(
                            "SAMPLE_RATE", np.array([SAMPLING_RATE], dtype=np.int32)
                        ),
                    ]
                )
            )
            
        return responses
