# ABOUTME: Unit test cho client vLLM - bóc SSE và dựng payload, không cần server
# ABOUTME: Chạy được ở chế độ "not integration"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.llm_client import (  # noqa: E402
    build_payload,
    iter_sse_content,
    trim_history,
)


def sse(*objs):
    """Dựng các dòng SSE y như vLLM phát ra: mỗi data một dòng, cách nhau dòng trắng."""
    lines = []
    for o in objs:
        lines += [f"data: {o}", ""]
    return lines


def delta(content):
    return '{"choices":[{"delta":{"content":"%s"}}]}' % content


def test_yields_content_in_order():
    lines = sse(delta("Xin"), delta(" chào"), delta("."))
    assert list(iter_sse_content(lines)) == ["Xin", " chào", "."]


def test_stops_at_done():
    lines = sse(delta("a")) + ["data: [DONE]"] + sse(delta("b"))
    assert list(iter_sse_content(lines)) == ["a"]


def test_skips_chunk_without_content():
    # Chunk đầu của OpenAI API chỉ mang role, chunk cuối chỉ mang finish_reason.
    # Cả hai đều không có "content" - bóc mù sẽ ném KeyError giữa luồng.
    lines = sse(
        '{"choices":[{"delta":{"role":"assistant"}}]}',
        delta("có"),
        '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
    )
    assert list(iter_sse_content(lines)) == ["có"]


def test_skips_usage_chunk_with_empty_choices():
    # stream_options.include_usage bật thì chunk chót có choices rỗng
    lines = sse(delta("x"), '{"choices":[],"usage":{"total_tokens":9}}')
    assert list(iter_sse_content(lines)) == ["x"]


def test_ignores_blank_lines_and_comments():
    # SSE keepalive là dòng bắt đầu bằng ":" - không phải JSON, parse là vỡ
    lines = ["", ": ping", "", "data: " + delta("ừ"), ""]
    assert list(iter_sse_content(lines)) == ["ừ"]


def test_accepts_bytes_lines():
    # urllib trả file-like nhị phân, lặp ra bytes chứ không phải str
    lines = [b"data: " + delta("byte").encode(), b""]
    assert list(iter_sse_content(lines)) == ["byte"]


def test_payload_omits_unset_params():
    # Cùng lý do tts_client chỉ gửi NUM_STEPS khi người dùng chỉ định: gửi cứng
    # một giá trị ở client là ghi đè mặc định của server dù không ai yêu cầu
    p = build_payload("Qwen/Qwen3-0.6B", [{"role": "user", "content": "hi"}])
    assert p == {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }


def test_payload_includes_set_params():
    p = build_payload(
        "m", [], stream=True, max_tokens=64, temperature=0.0
    )
    assert p["stream"] is True
    assert p["max_tokens"] == 64
    assert p["temperature"] == 0.0


def test_payload_keeps_temperature_zero():
    # 0.0 là giá trị hợp lệ và falsy - kiểm tra bằng `if temperature` sẽ nuốt mất
    assert "temperature" in build_payload("m", [], temperature=0.0)


def test_payload_carries_chat_template_kwargs():
    p = build_payload("m", [], chat_template_kwargs={"enable_thinking": False})
    assert p["chat_template_kwargs"] == {"enable_thinking": False}


def test_payload_omits_chat_template_kwargs_by_default():
    assert "chat_template_kwargs" not in build_payload("m", [])


def user(text):
    return {"role": "user", "content": text}


def bot(text):
    return {"role": "assistant", "content": text}


def fixed_count(per_message=10):
    """Đếm giả: mỗi message đúng `per_message` token, để biên test nằm ở số tròn."""
    return lambda msgs: per_message * len(msgs)


def test_trim_leaves_short_history_alone():
    msgs = [user("a"), bot("b"), user("c")]
    assert trim_history(msgs, fixed_count(), budget=100) == msgs


def test_trim_drops_oldest_turn_first():
    # Lượt cũ nhất là thứ rẻ nhất để mất: câu hỏi vừa gõ mới là câu phải trả lời
    msgs = [user("1"), bot("1"), user("2"), bot("2"), user("3")]
    assert trim_history(msgs, fixed_count(), budget=30) == msgs[2:]


def test_trim_keeps_system_prompt():
    # System prompt định hình cả phiên - cắt nó đi là model đổi tính giữa chừng
    msgs = [{"role": "system", "content": "s"}, user("1"), bot("1"), user("2")]
    out = trim_history(msgs, fixed_count(), budget=30)
    assert out[0]["role"] == "system"
    assert out[-1] == user("2")


def test_trim_never_leaves_assistant_leading():
    # Bỏ lẻ một message là lịch sử mở đầu bằng câu trả lời không có câu hỏi -
    # chat template dựng ra đoạn hội thoại vô nghĩa và model bắt chước theo
    msgs = [user("1"), bot("1"), user("2"), bot("2"), user("3")]
    out = trim_history(msgs, fixed_count(), budget=25)
    assert out[0]["role"] == "user"


def test_trim_keeps_latest_question_even_when_over_budget():
    # Không còn gì để bỏ nữa thì cứ gửi: để server báo lỗi rõ ràng còn hơn
    # client tự nuốt mất câu hỏi rồi hỏi model một lịch sử rỗng
    msgs = [{"role": "system", "content": "s"}, user("dài")]
    assert trim_history(msgs, fixed_count(), budget=5) == msgs


def test_trim_does_not_mutate_input():
    msgs = [user("1"), bot("1"), user("2")]
    trim_history(msgs, fixed_count(), budget=10)
    assert len(msgs) == 3
