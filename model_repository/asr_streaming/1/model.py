# ABOUTME: Triton Python backend - streaming ASR, nhận audio theo chunk qua sequence batcher
# ABOUTME: Skeleton: mới quản state per-stream, chưa nhận dạng - xem plan Task 5

import time

import numpy as np
import triton_python_backend_utils as pb_utils

STATE_TTL_S = 60.0   # soi gương max_sequence_idle_microseconds trong config.pbtxt


class TritonPythonModel:
    def initialize(self, args):
        self.streams = {}   # corrid -> state của stream đang sống

    def _new_stream(self):
        return {"samples": 0, "last_seen": time.monotonic()}

    def _sweep(self):
        """Xoá state của stream chết không gửi END - nếu không dict rò rỉ vĩnh viễn."""
        now = time.monotonic()
        for k in [k for k, s in self.streams.items() if now - s["last_seen"] > STATE_TTL_S]:
            pb_utils.Logger.log_warn(f"asr_streaming: xoá state mồ côi corrid={k}")
            del self.streams[k]

    @staticmethod
    def _flag(request, name):
        t = pb_utils.get_input_tensor_by_name(request, name)
        return t is not None and bool(t.as_numpy().reshape(-1)[0])

    def execute(self, requests):
        responses = []
        self._sweep()
        for request in requests:
            corrid = int(
                pb_utils.get_input_tensor_by_name(request, "CORRID").as_numpy().reshape(-1)[0]
            )
            start = self._flag(request, "START")
            end = self._flag(request, "END")

            if start or corrid not in self.streams:
                if not start:
                    # server restart giữa stream chẳng hạn - khởi tạo lại thay vì crash
                    pb_utils.Logger.log_warn(
                        f"asr_streaming: chunk không có state (corrid={corrid}), khởi tạo lại"
                    )
                self.streams[corrid] = self._new_stream()
            stream = self.streams[corrid]
            stream["last_seen"] = time.monotonic()

            audio = (
                pb_utils.get_input_tensor_by_name(request, "AUDIO_CHUNK")
                .as_numpy()
                .reshape(-1)
            )
            stream["samples"] += len(audio)

            # Skeleton trả "corrid:tổng_mẫu" để smoke test kiểm state không lẫn giữa stream
            text = f"{corrid}:{stream['samples']}"
            if end:
                del self.streams[corrid]

            out = np.array([[text.encode("utf-8")]], dtype=object)
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("TRANSCRIPT", out)])
            )
        return responses
