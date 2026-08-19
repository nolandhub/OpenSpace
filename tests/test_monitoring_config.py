# ABOUTME: Unit test cho config monitoring - không cần server nào chạy
# ABOUTME: Nhiệm vụ chính là chống drift giữa CCU_TTL_S, config.pbtxt và dashboard

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
MON = ROOT / "docker" / "monitoring"


@pytest.fixture(scope="module")
def prom_cfg():
    return yaml.safe_load((MON / "prometheus.yml").read_text())


def test_scrape_interval_5s(prom_cfg):
    """CCU và queue depth là gauge tức thời - 15s mặc định là trượt mất spike."""
    assert prom_cfg["global"]["scrape_interval"] == "5s"


def test_dung_hai_job(prom_cfg):
    jobs = {j["job_name"]: j for j in prom_cfg["scrape_configs"]}
    assert set(jobs) == {"triton", "vllm"}
    assert jobs["triton"]["static_configs"][0]["targets"] == ["localhost:8002"]
    assert jobs["vllm"]["static_configs"][0]["targets"] == ["localhost:8080"]


def test_compose_dung_host_network():
    """Triton chạy --net host nên :8002 chỉ thấy từ host namespace."""
    compose = yaml.safe_load((MON / "docker-compose.yml").read_text())
    for name, svc in compose["services"].items():
        assert svc.get("network_mode") == "host", f"{name} không dùng host network"
