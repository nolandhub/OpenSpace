# ABOUTME: Fixture dùng chung cho test - kết nối Triton qua gRPC
# ABOUTME: Tự động skip test integration nếu server chưa chạy

import pytest

URL = "localhost:8001"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: cần Triton server đang chạy")


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
