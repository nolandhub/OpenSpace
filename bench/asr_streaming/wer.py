# ABOUTME: Word error rate thuần - chuẩn hoá text rồi đo khoảng cách Levenshtein trên từ
# ABOUTME: Không import Triton/ONNX, test được khi server tắt

import re
import unicodedata

# Bỏ mọi ký tự không phải chữ/số/khoảng trắng. \w trong Python đã bao gồm chữ có
# dấu, nên chuẩn hoá kiểu này giữ nguyên tiếng Việt mà vẫn cắt được dấu câu.
_DROP = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize(text: str) -> list[str]:
    """Đưa text về dãy từ so sánh được: thường hoá, bỏ dấu câu, gộp khoảng trắng.

    ASR không xuất dấu câu còn bản tham chiếu thì có. Không cắt thì mọi từ đứng
    cuối câu đều bị tính sai và WER phồng lên vô nghĩa.
    """
    # NFC để "à" viết rời (a + dấu huyền) và "à" viết liền so ra bằng nhau
    text = unicodedata.normalize("NFC", text).lower()
    return _DROP.sub(" ", text).split()


def wer(reference: str, hypothesis: str) -> float:
    """(thay + xoá + thêm) / số từ tham chiếu. Có thể vượt 1.0 khi model lặp."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError("tham chiếu rỗng - không có mẫu số để chia")

    # Levenshtein trên từ, chỉ giữ hai hàng thay vì cả ma trận
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(
                prev[j] + 1,                      # xoá một từ của tham chiếu
                cur[j - 1] + 1,                   # thêm một từ vào giả thuyết
                prev[j - 1] + (r != h),           # thay, hoặc khớp thì không tốn gì
            ))
        prev = cur
    return prev[-1] / len(ref)
