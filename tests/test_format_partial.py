# ABOUTME: Unit test cho format_partial - gò partial ASR vừa đúng một dòng terminal
# ABOUTME: Partial dài quá width là terminal tự wrap, \r hết lùi được, output thành rác

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.asr_streaming_client import format_partial  # noqa: E402


def test_short_text_is_untouched():
    """Chưa chạm mép dòng thì giữ nguyên - phụ đề đọc từ đầu câu."""
    assert format_partial("xin chào", width=80) == "xin chào"


def test_text_exactly_at_width_is_untouched():
    """Vừa khít width vẫn nằm gọn một dòng, chưa cần cắt."""
    text = "a" * 80
    assert format_partial(text, width=80) == text


def test_long_text_never_exceeds_width():
    """Đây là bất biến duy nhất phải giữ: vượt width là wrap, wrap là \\r vô dụng."""
    text = "x" * 500
    assert len(format_partial(text, width=125)) == 125


def test_long_text_keeps_the_tail():
    """Giữ đuôi chứ không giữ đầu - phụ đề trực tiếp thì chữ mới ra mới đáng xem."""
    text = "cuối tháng hai vừa qua nhà chức trách đã công khai kế hoạch"
    assert format_partial(text, width=20) == text[-20:]


def test_result_is_always_a_suffix_of_input():
    """Chỉ được cắt bớt, tuyệt đối không chèn hay đổi chữ."""
    text = "một hai ba bốn năm sáu bảy tám chín mười"
    for width in range(1, len(text) + 5):
        assert text.endswith(format_partial(text, width))


def test_degenerate_width_does_not_crash():
    """Terminal hẹp bất thường vẫn phải in được, thà rỗng còn hơn nổ giữa stream."""
    assert format_partial("xin chào", width=0) == ""
    assert format_partial("xin chào", width=-5) == ""
