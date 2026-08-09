# ABOUTME: Tải audio tiếng Việt mẫu từ dataset công khai, chuyển về wav mono 16kHz
# ABOUTME: Sinh ra file test cho ASR và giọng mẫu zero-shot cho TTS, kèm transcript

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).parent.parent
SAMPLE_RATE = 16000

# doof-ferb/fpt_fosd - bộ đọc tiếng Việt công khai, không cần đăng nhập.
# Thay bằng file thu của chính mình lúc nào cũng được, chỉ cần giữ đúng
# cặp <file wav 16kHz mono> + <file txt chứa đúng câu đã nói>.
DATASET = "doof-ferb/fpt_fosd"

# (chỉ số hàng trong dataset, đường dẫn đích không kèm đuôi file)
WANTED = [
    (23, ROOT / "tests/assets/sample_vi"),                     # câu để test ASR
    (7, ROOT / "model_repository/tts/1/assets/prompt"),        # giọng mẫu cho TTS
]


def fetch_rows(offset: int, length: int) -> list:
    """Lấy metadata các hàng của dataset qua datasets-server của HuggingFace."""
    url = (
        "https://datasets-server.huggingface.co/rows?"
        + urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": length,
            }
        )
    )
    with urllib.request.urlopen(url) as response:
        return json.load(response)["rows"]


def main():
    rows = fetch_rows(0, max(index for index, _ in WANTED) + 1)

    for index, dest in WANTED:
        row = rows[index]["row"]
        dest.parent.mkdir(parents=True, exist_ok=True)

        tmp_mp3 = dest.with_suffix(".mp3")
        urllib.request.urlretrieve(row["audio"][0]["src"], tmp_mp3)

        wav, sample_rate = sf.read(tmp_mp3, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            # resample_poly lọc chống aliasing sẵn, an toàn hơn cắt mẫu thô
            wav = resample_poly(wav, SAMPLE_RATE, sample_rate).astype(np.float32)

        sf.write(dest.with_suffix(".wav"), wav, SAMPLE_RATE)
        dest.with_suffix(".txt").write_text(
            row["transcription"].strip() + "\n", encoding="utf-8"
        )
        tmp_mp3.unlink()

        print(
            f"{dest.with_suffix('.wav')}  "
            f"{len(wav) / SAMPLE_RATE:.2f}s  \"{row['transcription']}\""
        )


if __name__ == "__main__":
    main()
