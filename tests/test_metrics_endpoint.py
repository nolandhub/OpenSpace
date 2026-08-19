# ABOUTME: Integration - khẳng định metric tự phát có thật trên :8002/metrics
# ABOUTME: Cần Triton đang chạy; skip nếu chưa

import urllib.request

import pytest

METRICS_URL = "http://localhost:8002/metrics"


def scrape():
    """Đọc :8002/metrics. urllib chứ không requests - repo giữ quy ước stdlib
    cho HTTP, xem comment đầu client/llm_client.py."""
    with urllib.request.urlopen(METRICS_URL, timeout=3) as r:
        return r.read().decode("utf-8")


@pytest.fixture(scope="session")
def metrics_text():
    try:
        return scrape()
    except Exception as e:
        pytest.skip(f"Không lấy được metrics tại {METRICS_URL}: {e}")


def _sample(text, name, **labels):
    """Trả giá trị float của một sample, hoặc None nếu không có dòng nào khớp."""
    for line in text.splitlines():
        if not line.startswith(name + "{"):
            continue
        if all(f'{k}="{v}"' in line for k, v in labels.items()):
            return float(line.rsplit(" ", 1)[1])
    return None


@pytest.mark.integration
def test_summary_latency_duoc_bat(metrics_text):
    """Cần --metrics-config summary_latencies=true, không có thì chỉ tính ra mean."""
    for q in ("0.5", "0.95", "0.99"):
        assert f'quantile="{q}"' in metrics_text, f"thiếu quantile {q}"


@pytest.mark.integration
def test_asr_co_metric_tu_phat(metrics_text):
    assert "voice_rtf_bucket" in metrics_text
    assert _sample(metrics_text, "voice_ccu", model="asr_streaming") is not None
    assert _sample(metrics_text, "voice_ccu_updated_at", model="asr_streaming") is not None


@pytest.mark.integration
def test_ccu_co_label_model_instance_tren_server_that(metrics_text):
    """asr_streaming có count: 2 - thiếu label model_instance là 2 process ghi đè nhau."""
    lines = [l for l in metrics_text.splitlines() if l.startswith("voice_ccu{")]
    assert lines, "không có sample voice_ccu nào"
    assert all("model_instance=" in l for l in lines), lines


@pytest.mark.integration
def test_rtf_khong_co_label_instance_tren_server_that(metrics_text):
    lines = [l for l in metrics_text.splitlines() if l.startswith("voice_rtf_bucket{")]
    assert lines, "không có sample voice_rtf_bucket nào"
    # "model_instance=" cũng chứa chuỗi con "instance=" nên kiểm cả hai dạng -
    # voice_rtf không được có label instance dưới bất kỳ tên nào.
    assert not any("instance=" in l for l in lines), lines


@pytest.mark.integration
def test_stream_asr_lam_tang_rtf_count(triton, metrics_text):
    """Chạy một stream thật rồi khẳng định histogram nhích lên và CCU về 0."""
    import numpy as np
    import tritonclient.grpc as grpcclient

    before = _sample(metrics_text, "voice_rtf_count", model="asr_streaming") or 0.0

    audio = np.zeros(3200, dtype=np.float32)  # 200ms im lặng ở 16kHz
    corrid = 987654
    for i in range(3):
        # max_batch_size > 0 nên shape phải kèm chiều batch ở đầu - cùng quy ước
        # với tests/test_asr_streaming.py và client/asr_streaming_client.py.
        inp = grpcclient.InferInput("AUDIO_CHUNK", [1, len(audio)], "FP32")
        inp.set_data_from_numpy(audio.reshape(1, -1))
        triton.infer(
            "asr_streaming",
            [inp],
            sequence_id=corrid,
            sequence_start=(i == 0),
            sequence_end=(i == 2),
        )

    after_text = scrape()
    after = _sample(after_text, "voice_rtf_count", model="asr_streaming") or 0.0
    assert after > before, f"voice_rtf_count không tăng: {before} -> {after}"

    ccu = sum(
        float(l.rsplit(" ", 1)[1])
        for l in after_text.splitlines()
        if l.startswith("voice_ccu{") and 'model="asr_streaming"' in l
    )
    assert ccu == 0, f"stream đã END mà CCU vẫn {ccu}"


@pytest.mark.integration
def test_tts_co_metric_tu_phat(metrics_text):
    assert _sample(metrics_text, "voice_ccu", model="tts") is not None
    assert _sample(metrics_text, "voice_rtf_bucket", model="tts", le="1") is not None


@pytest.mark.integration
def test_tts_rtf_tang_sau_mot_cau(triton):
    """RTF đo trong execute() nên KHÔNG có RTT gRPC - thấp hơn bench/tts một chút.

    Ở đây chỉ khẳng định nó có ghi nhận, không so số với bench.
    """
    import numpy as np
    import tritonclient.grpc as grpcclient

    before = _sample(scrape(), "voice_rtf_count", model="tts") or 0.0

    # max_batch_size: 0 trong config.pbtxt của tts nên KHÔNG có chiều batch,
    # khác asr_streaming - cùng quy ước với bench/tts/metrics.py.
    inp = grpcclient.InferInput("TEXT", [1], "BYTES")
    inp.set_data_from_numpy(np.array(["Xin chào".encode("utf-8")], dtype=object))
    triton.infer("tts", [inp])

    after = _sample(scrape(), "voice_rtf_count", model="tts") or 0.0
    assert after > before, f"voice_rtf_count của tts không tăng: {before} -> {after}"


@pytest.mark.integration
def test_tts_ccu_ve_0_khi_ranh(triton, metrics_text):
    """execute() set 0 trong finally - xong request là phải về 0."""
    total = sum(
        float(l.rsplit(" ", 1)[1])
        for l in metrics_text.splitlines()
        if l.startswith("voice_ccu{") and 'model="tts"' in l
    )
    assert total == 0, f"tts rảnh mà CCU vẫn {total}"
