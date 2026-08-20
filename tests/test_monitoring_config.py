# ABOUTME: Unit test cho config monitoring - không cần server nào chạy
# ABOUTME: Nhiệm vụ chính là chống drift giữa CCU_TTL_S, config.pbtxt và dashboard

import json
import re
import subprocess
import sys
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


DASHBOARD = MON / "grafana" / "dashboards" / "voice-serving.json"

# Chốt từ dump :8002/metrics và source vLLM 0.27.1 (spec §8). Panel dùng tên
# ngoài danh sách này gần như chắc chắn là gõ nhầm - Grafana sẽ im lặng hiện
# "No data" chứ không báo lỗi.
KNOWN_METRICS = {
    "nv_inference_request_success",
    "nv_inference_request_failure",
    "nv_inference_request_summary_us",
    "nv_inference_pending_request_count",
    "nv_gpu_utilization",
    "nv_gpu_memory_used_bytes",
    "nv_gpu_memory_total_bytes",
    "voice_rtf_bucket",
    "voice_ccu",
    "voice_ccu_updated_at",
    "vllm:request_success_total",
    "vllm:e2e_request_latency_seconds_bucket",
    "vllm:time_to_first_token_seconds_bucket",
    "vllm:inter_token_latency_seconds_bucket",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
}


@pytest.fixture(scope="module")
def dashboard():
    return json.loads(DASHBOARD.read_text())


def _panels(dash):
    for panel in dash["panels"]:
        yield panel
        for sub in panel.get("panels", []):
            yield sub


def _exprs(dash):
    for panel in _panels(dash):
        for target in panel.get("targets", []):
            yield panel["title"], target["expr"]


def test_dashboard_khop_generator():
    """JSON commit kèm phải đúng bằng output của generator - không sửa tay."""
    out = subprocess.run(
        [sys.executable, str(MON / "build_dashboard.py"), "--stdout"],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout
    assert json.loads(out) == json.loads(DASHBOARD.read_text()), (
        "voice-serving.json lệch với build_dashboard.py - chạy lại generator"
    )


def test_moi_panel_co_datasource(dashboard):
    for panel in _panels(dashboard):
        if panel.get("type") == "row":
            continue
        assert panel.get("datasource"), f"panel thiếu datasource: {panel['title']}"


def test_moi_metric_deu_co_that(dashboard):
    for title, expr in _exprs(dashboard):
        used = set(re.findall(r"[a-zA-Z_:][a-zA-Z0-9_:]*", expr))
        unknown = {
            m for m in used
            if (m.startswith(("nv_", "vllm", "voice_")) and m not in KNOWN_METRICS)
        }
        assert not unknown, f"panel {title!r} dùng metric lạ: {unknown}"


def test_moi_phep_tinh_failure_deu_boc_sum(dashboard):
    """nv_inference_request_failure có label `reason`.

    Không bọc sum() thì phép cộng với success lệch label, Prometheus không
    match được cặp nào và trả rỗng - panel "No data" chứ không phải số sai.
    """
    for title, expr in _exprs(dashboard):
        for m in re.finditer(r"nv_inference_request_failure", expr):
            prefix = expr[: m.start()]
            assert "sum(" in prefix, f"panel {title!r}: failure không bọc sum()"


def test_ttl_trong_dashboard_khop_hang_so():
    """Số 60 nằm ở 3 nơi - config.pbtxt, CCU_TTL_S, PromQL. Lệch là CCU sai âm thầm."""
    from serving.metrics import CCU_TTL_S

    exprs = [e for _, e in _exprs(json.loads(DASHBOARD.read_text()))]
    ccu_exprs = [e for e in exprs if "voice_ccu_updated_at" in e]
    assert ccu_exprs, "không panel nào lọc CCU theo tuổi"
    for expr in ccu_exprs:
        found = re.search(r"<\s*bool\s+([\d.]+)", expr)
        assert found, f"query CCU thiếu ngưỡng bool: {expr}"
        assert float(found.group(1)) == CCU_TTL_S


def test_phu_du_14_chi_so(dashboard):
    """Mỗi chỉ số trong spec §2 phải có ít nhất một panel."""
    titles = " | ".join(p["title"] for p in _panels(dashboard)).lower()
    for keyword in ("rps", "success", "error", "p50", "p95", "p99",
                    "gpu util", "gpu mem", "queue", "ccu", "rtf", "ttft", "tpot"):
        assert keyword in titles, f"thiếu panel cho: {keyword}"


# Tag phải cố định. `:latest` là con trỏ di động - image đổi dưới chân mà
# git diff trống, không bisect được. Riêng vLLM còn ràng chặt hơn: KNOWN_METRICS
# ở trên chốt theo source 0.27.1, vLLM tự nâng cấp thì panel LLM trống lặng lẽ.
PINNED_IMAGES = {
    "prom/prometheus": "v3.14.0",
    "grafana/grafana": "13.2.0",
    "vllm/vllm-openai": "v0.27.1",
}


def test_compose_pin_image_tag():
    compose = yaml.safe_load((MON / "docker-compose.yml").read_text())
    for name, svc in compose["services"].items():
        repo, _, tag = svc["image"].rpartition(":")
        assert repo in PINNED_IMAGES, f"{name}: image lạ {svc['image']!r}"
        assert tag == PINNED_IMAGES[repo], (
            f"{name}: tag {tag!r} lệch PINNED_IMAGES ({PINNED_IMAGES[repo]!r})"
        )


def test_serve_llm_pin_image_tag():
    """Mặc định của $IMAGE phải là tag chốt, không phải :latest."""
    src = (ROOT / "scripts" / "serve_llm.sh").read_text()
    found = re.search(r"^IMAGE=\$\{IMAGE:-(\S+)\}", src, re.M)
    assert found, "không tìm thấy dòng gán mặc định IMAGE trong serve_llm.sh"
    repo, _, tag = found.group(1).rpartition(":")
    assert tag == PINNED_IMAGES[repo], (
        f"serve_llm.sh dùng {tag!r}, PINNED_IMAGES chốt {PINNED_IMAGES[repo]!r}"
    )
