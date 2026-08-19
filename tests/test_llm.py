# ABOUTME: Integration test cho vLLM - cần ./scripts/serve_llm.sh đang chạy
# ABOUTME: Chỉ khẳng định cấu trúc, không khẳng định nội dung - model 0.6B nói gì là chuyện của nó

import pytest

from client.llm_client import (
    chat,
    chat_stream,
    context_limit,
    count_tokens,
    trim_history,
)

pytestmark = pytest.mark.integration


def test_serves_exactly_the_model_asked_for(llm):
    url, model = llm
    assert isinstance(model, str) and model


def test_answers_in_one_shot(llm):
    url, model = llm
    out = chat(url, model, [{"role": "user", "content": "Thủ đô Việt Nam là gì?"}],
               max_tokens=64, temperature=0.0)
    assert out.strip(), "trả lời rỗng"


def test_streaming_arrives_in_more_than_one_piece(llm):
    # Nếu chỉ về đúng 1 mảnh thì stream không thật sự stream - có thể do proxy
    # đệm hết rồi mới nhả, và demo giọng nói mất sạch cái lợi của streaming
    url, model = llm
    pieces = list(chat_stream(
        url, model, [{"role": "user", "content": "Đếm từ 1 đến 10."}],
        max_tokens=64, temperature=0.0,
    ))
    assert len(pieces) > 1, f"chỉ nhận được {len(pieces)} mảnh"
    assert "".join(pieces).strip()


def test_streaming_and_one_shot_agree_at_temperature_zero(llm):
    url, model = llm
    msgs = [{"role": "user", "content": "Nói đúng một từ: xin chào"}]
    once = chat(url, model, msgs, max_tokens=32, temperature=0.0)
    streamed = "".join(chat_stream(url, model, msgs, max_tokens=32, temperature=0.0))
    assert once == streamed


def test_max_tokens_caps_the_answer(llm):
    url, model = llm
    msgs = [{"role": "user", "content": "Kể một câu chuyện thật dài."}]
    short = chat(url, model, msgs, max_tokens=8, temperature=0.0)
    long = chat(url, model, msgs, max_tokens=128, temperature=0.0)
    assert len(short) < len(long)


def test_context_limit_reports_the_real_window(llm):
    url, model = llm
    assert context_limit(url, model) >= 512


def test_count_tokens_grows_with_history(llm):
    url, model = llm
    one = count_tokens(url, model, [{"role": "user", "content": "Xin chào"}])
    three = count_tokens(url, model, [
        {"role": "user", "content": "Xin chào"},
        {"role": "assistant", "content": "Chào bạn"},
        {"role": "user", "content": "Khoẻ không?"},
    ])
    assert three > one > 0


def test_trim_holds_a_long_history_under_budget(llm):
    # Test đơn vị dùng bộ đếm giả 10 token/message; ở đây là tokenizer thật của
    # model thật, nơi độ dài mỗi message khác nhau và chat template cũng tốn chỗ
    url, model = llm
    counter = lambda msgs: count_tokens(url, model, msgs)  # noqa: E731
    msgs = [{"role": "system", "content": "Trả lời thật ngắn."}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"Câu hỏi số {i} về một chủ đề dài dòng."})
        msgs.append({"role": "assistant", "content": f"Trả lời số {i}, cũng dài không kém."})
    msgs.append({"role": "user", "content": "Câu hỏi cuối cùng."})

    out = trim_history(msgs, counter, budget=200)
    assert counter(out) <= 200
    assert out[0]["role"] == "system", "cắt mất system prompt"
    assert out[-1] == msgs[-1], "cắt mất câu hỏi mới nhất"


def test_model_answers_from_an_earlier_turn(llm):
    # Toàn bộ lý do phải giữ lịch sử: lượt sau chỉ trả lời được nếu thấy lượt trước
    url, model = llm
    no_think = {"chat_template_kwargs": {"enable_thinking": False}}
    msgs = [{"role": "user", "content": "Tôi tên là Nam."}]
    msgs.append({"role": "assistant",
                 "content": chat(url, model, msgs, max_tokens=64, temperature=0.0, **no_think)})
    msgs.append({"role": "user", "content": "Tôi vừa nói tên tôi là gì? Trả lời đúng một từ."})
    assert "Nam" in chat(url, model, msgs, max_tokens=32, temperature=0.0, **no_think)
