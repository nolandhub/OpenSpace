# ABOUTME: Client dòng lệnh cho vLLM qua API OpenAI-compatible, in dần từng token
# ABOUTME: Chạy: python client/llm_client.py --prompt "Xin chào" (--no-stream để chờ trọn câu)

import argparse
import json
import urllib.error
import urllib.request

# Chỉ dùng stdlib: cả file gọi đúng hai endpoint HTTP, thêm `openai` hay `httpx`
# vào requirements.txt là kéo một cây dependency cho việc urllib làm xong.
DEFAULT_URL = "http://localhost:8080"
# Chỗ chừa cho câu trả lời khi người dùng không đặt --max-tokens: lịch sử được
# cắt tới `context_limit - ANSWER_RESERVE`, phần còn lại là chỗ model nói.
ANSWER_RESERVE = 256


def build_payload(model, messages, stream=False, max_tokens=None,
                  temperature=None, chat_template_kwargs=None):
    """Dựng body cho /v1/chat/completions, bỏ hẳn tham số người dùng không đặt.

    Cùng lý do tts_client chỉ gửi NUM_STEPS khi được chỉ định: điền cứng một giá
    trị ở client là ghi đè mặc định của server dù không ai yêu cầu. Phải so với
    None chứ không phải kiểm tra falsy - temperature=0.0 là giá trị hợp lệ.
    """
    payload = {"model": model, "messages": messages, "stream": stream}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = chat_template_kwargs
    return payload


def trim_history(messages, count, budget):
    """Bỏ dần lượt cũ nhất cho tới khi lịch sử vừa `budget` token.

    Cửa sổ ngữ cảnh là trần cứng: vượt là server trả 400 chứ không tự quên giúp.
    Giữ lại system prompt (nó định hình cả phiên) và câu hỏi mới nhất (nó là
    việc đang phải làm); mọi thứ ở giữa là thứ rẻ nhất để mất. `count` nhận cả
    list và trả số token - hỏi server qua /tokenize chứ đừng đoán theo ký tự.
    """
    head = messages[:1] if messages[:1] and messages[0]["role"] == "system" else []
    turns = messages[len(head):]
    while len(turns) > 1 and count(head + turns) > budget:
        turns = turns[1:]
        # Bỏ lẻ một message là lịch sử mở đầu bằng câu trả lời không có câu hỏi.
        while len(turns) > 1 and turns[0]["role"] != "user":
            turns = turns[1:]
    return head + turns

def iter_sse_content(lines):
    """Bóc dần phần content từ luồng SSE, bỏ qua mọi thứ không phải chữ.

    Luồng SSE có bốn loại dòng không mang content và cả bốn đều xuất hiện thật:
    dòng trắng ngăn cách, keepalive mở đầu bằng ':', chunk chỉ mang role hoặc
    finish_reason, và chunk usage có choices rỗng. Đọc mù là vỡ giữa chừng.
    """
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        choices = json.loads(data).get("choices") or []
        if not choices:
            continue
        content = (choices[0].get("delta") or {}).get("content")
        if content:
            yield content


def _post(url, path, payload, timeout):
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def list_models(url=DEFAULT_URL, timeout=10):
    """Tên model server đang phục vụ - phải khớp đúng chuỗi trong trường `model`."""
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=timeout) as r:
        return [m["id"] for m in json.load(r)["data"]]


def context_limit(url, model, timeout=10):
    """max_model_len server đang chạy - budget cắt lịch sử phải lấy từ đây.

    Viết cứng 2048 ở client là sai ngay lần đầu đổi MAX_MODEL_LEN trong
    serve_llm.sh, mà sai kiểu im lặng: lịch sử cắt quá tay hoặc tràn context.
    """
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=timeout) as r:
        for m in json.load(r)["data"]:
            if m["id"] == model:
                return m["max_model_len"]
    raise KeyError(f"server không phục vụ model {model}")


def count_tokens(url, model, messages, timeout=10):
    """Số token vLLM thực sự nạp cho `messages`, đã tính cả chat template.

    Đếm ở client thì phải kéo về đúng tokenizer của model rồi tự dựng lại
    template - hai chỗ dễ lệch. /tokenize là chính server trả lời về chính nó.
    """
    payload = {"model": model, "messages": messages}
    with _post(url, "/tokenize", payload, timeout) as r:
        return json.load(r)["count"]

def chat(url, model, messages, timeout=300, **params):
    """Gọi một lượt, trả về trọn câu trả lời."""
    payload = build_payload(model, messages, stream=False, **params)
    with _post(url, "/v1/chat/completions", payload, timeout) as r:
        body = json.load(r)
    return body["choices"][0]["message"]["content"]


def chat_stream(url, model, messages, timeout=300, **params):
    """Gọi một lượt, sinh dần từng mảnh content ngay khi model phát ra."""
    payload = build_payload(model, messages, stream=True, **params)
    with _post(url, "/v1/chat/completions", payload, timeout) as r:
        yield from iter_sse_content(r)


def say(url, model, messages, no_stream=False, **params):
    """In câu trả lời (dần hoặc trọn) và trả lại nguyên văn để ghi vào lịch sử."""
    if no_stream:
        out = chat(url, model, messages, **params)
        print(out)
        return out
    pieces = []
    for piece in chat_stream(url, model, messages, **params):
        print(piece, end="", flush=True)
        pieces.append(piece)
    print()
    return "".join(pieces)


def run_chat(url, model, base, budget, no_stream=False, **params):
    """Hội thoại nhiều lượt: giữ lịch sử, cắt cho vừa `budget` trước mỗi lượt.

    Lịch sử là toàn bộ khác biệt so với chế độ một lượt - model không nhớ gì
    giữa hai request, cái nó "nhớ" chính là list này được gửi lại mỗi lần.
    """
    messages = list(base)
    print(f"{model} — Ctrl-D để thoát, /reset để quên hội thoại")
    while True:
        try:
            question = input("\nbạn> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question == "/reset":
            messages = list(base)
            print("(đã quên hội thoại)")
            continue

        messages.append({"role": "user", "content": question})
        messages = trim_history(
            messages, lambda m: count_tokens(url, model, m), budget)
        print("llm> ", end="", flush=True)
        try:
            answer = say(url, model, messages, no_stream, **params)
        except urllib.error.HTTPError as e:
            # Bỏ lại câu vừa hỏi, nếu không lượt sau gửi lại đúng thứ vừa lỗi
            messages.pop()
            print(f"\nvLLM trả {e.code}: {e.read().decode('utf-8', 'replace')}")
            continue
        messages.append({"role": "assistant", "content": answer})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", help="câu hỏi một lượt; bỏ trống thì phải có --chat")
    ap.add_argument("--system", help="system prompt, bỏ trống thì không gửi")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", help="bỏ trống thì lấy model đầu tiên server đang phục vụ")
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--chat", action="store_true",
                    help="hội thoại nhiều lượt, giữ ngữ cảnh qua từng câu")
    ap.add_argument("--no-stream", action="store_true", help="chờ trọn câu rồi mới in")
    ap.add_argument("--no-think", action="store_true",
                    help="tắt reasoning của Qwen3 - model khác bỏ qua cờ này")
    args = ap.parse_args()
    if not args.prompt and not args.chat:
        ap.error("cần --prompt hoặc --chat")

    try:
        # Không đặt mặc định model ở đây: hai chỗ khai cùng một tên model thì sớm
        # muộn cũng lệch nhau khi serve_llm.sh đổi MODEL. Hỏi server là hết lệch.
        model = args.model or list_models(args.url)[0]
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(f"không gọi được vLLM tại {args.url}: {e}")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    params = {"max_tokens": args.max_tokens, "temperature": args.temperature}
    if args.no_think:
        # Qwen3 bật reasoning mặc định và nhả cả khối <think> vào content.
        # Với demo giọng nói thì đó là nguyên đoạn độc thoại đi thẳng vào TTS.
        # Key này là quy ước chat template của Qwen3, model khác lờ đi.
        params["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        if args.chat:
            budget = context_limit(args.url, model) - (args.max_tokens or ANSWER_RESERVE)
            run_chat(args.url, model, messages, budget, args.no_stream, **params)
        else:
            messages.append({"role": "user", "content": args.prompt})
            say(args.url, model, messages, args.no_stream, **params)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"vLLM trả {e.code}: {e.read().decode('utf-8', 'replace')}")


if __name__ == "__main__":
    main()
