# ABOUTME: Fixture dùng chung cho test - kết nối Triton qua gRPC và vLLM qua HTTP
# ABOUTME: Tự động skip test integration nếu server tương ứng chưa chạy

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

URL = "localhost:8001"
LLM_URL = "http://localhost:8080"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: cần Triton hoặc vLLM đang chạy")


@pytest.fixture(scope="session")
def triton():
    grpc = pytest.importorskip("tritonclient.grpc")
    client = grpc.InferenceServerClient(URL)
    try:
        if not client.is_server_ready():
            pytest.skip(f"Triton chưa sẵn sàng tại {URL}")
    except Exception as e:
        pytest.skip(f"Không kết nối được Triton tại {URL}: {e}")
    return client


@pytest.fixture(scope="session")
def llm():
    """(url, tên model) của vLLM. Fixture riêng vì vLLM là server độc lập.

    Triton có thể đang chạy mà vLLM thì chưa, và ngược lại - gộp chung một
    fixture sẽ skip nhầm cả nhóm test còn lại khi chỉ một trong hai chưa lên.
    Hỏi luôn tên model thay vì viết cứng: serve_llm.sh đổi MODEL là lệch ngay.
    """
    from client.llm_client import list_models

    try:
        models = list_models(LLM_URL, timeout=3)
    except Exception as e:
        pytest.skip(f"vLLM chưa sẵn sàng tại {LLM_URL}: {e}")
    if not models:
        pytest.skip(f"vLLM tại {LLM_URL} không phục vụ model nào")
    return LLM_URL, models[0]
