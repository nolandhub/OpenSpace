# ABOUTME: Unit test cho bench/asr_streaming/wer.py - chuẩn hoá text và word error rate
# ABOUTME: Hàm thuần, không cần Triton server

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from bench.asr_streaming.wer import normalize, wer  # noqa: E402


# ---------- normalize ----------


def test_normalize_lowercases_and_splits_on_whitespace():
    assert normalize("Xin  Chào\tBạn\n") == ["xin", "chào", "bạn"]


def test_normalize_strips_punctuation_but_keeps_diacritics():
    # ASR không xuất dấu câu, tham chiếu thì có - không bỏ thì mọi từ cuối câu đều sai
    assert normalize("Mì tôm, ăn là ghiền!") == ["mì", "tôm", "ăn", "là", "ghiền"]


def test_normalize_keeps_words_that_contain_digits():
    assert normalize("phòng 3a") == ["phòng", "3a"]


# ---------- wer ----------


def test_identical_text_has_zero_error():
    assert wer("xin chào bạn", "Xin chào, bạn!") == 0.0


def test_one_substitution_over_four_words():
    assert wer("một hai ba bốn", "một hai ba năm") == pytest.approx(0.25)


def test_deletion_and_insertion_both_count():
    # tham chiếu 3 từ; hyp thiếu "hai" và thừa "bốn" -> 1 xoá + 1 thêm = 2/3
    assert wer("một hai ba", "một ba bốn") == pytest.approx(2 / 3)


def test_empty_hypothesis_is_total_loss():
    assert wer("một hai ba", "") == 1.0


def test_wer_can_exceed_one_when_hypothesis_rambles():
    # Model lặp vô hạn thì WER phải vượt 100%, không được kẹp về 1.0
    assert wer("một", "một một một") == pytest.approx(2.0)


def test_empty_reference_is_rejected():
    # Chia cho 0 - im lặng trả nan sẽ lọt thẳng vào bảng kết quả
    with pytest.raises(ValueError, match="tham chiếu"):
        wer("", "một hai")
