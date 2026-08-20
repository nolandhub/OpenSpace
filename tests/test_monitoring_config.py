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
TRITON_DASHBOARD = MON / "grafana" / "dashboards" / "triton.json"

# Hai board, một generator. Board "voice" là view sản phẩm (ASR/TTS/LLM cạnh
# nhau), board "triton" là view nội tại server. Tách ra vì hai mức trừu tượng
# khác nhau, không phải vì board cũ dài quá.
BOARDS = {"voice": DASHBOARD, "triton": TRITON_DASHBOARD}

# Chốt từ dump :8002/metrics và source vLLM 0.27.1 (spec §8). Panel dùng tên
# ngoài danh sách này gần như chắc chắn là gõ nhầm - Grafana sẽ im lặng hiện
# "No data" chứ không báo lỗi.
KNOWN_METRICS = {
    "nv_inference_request_success",
    "nv_inference_request_failure",
    "nv_inference_request_summary_us",
    "nv_inference_pending_request_count",
    "nv_inference_count",
    "nv_inference_exec_count",
    "nv_model_load_duration_secs",
    # _sum/_count của summary là counter thường, không bao giờ Nan - dùng cho
    # panel breakdown. Còn quantile thì Nan lúc rảnh, xem docs/observability.md.
    "nv_inference_queue_summary_us",
    "nv_inference_queue_summary_us_sum",
    "nv_inference_queue_summary_us_count",
    "nv_inference_compute_input_summary_us_sum",
    "nv_inference_compute_input_summary_us_count",
    "nv_inference_compute_infer_summary_us_sum",
    "nv_inference_compute_infer_summary_us_count",
    "nv_inference_compute_output_summary_us_sum",
    "nv_inference_compute_output_summary_us_count",
    "nv_gpu_utilization",
    "nv_gpu_memory_used_bytes",
    "nv_gpu_memory_total_bytes",
    "nv_gpu_power_usage",
    # Có thật trên :8002 nhưng đọc ra 0 W trên GPU laptop này - không panel nào
    # dùng, giữ tên ở đây để lần sau khỏi thử lại.
    "nv_gpu_power_limit",
    "nv_cpu_utilization",
    "nv_cpu_memory_used_bytes",
    "nv_cpu_memory_total_bytes",
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


@pytest.fixture(params=sorted(BOARDS), scope="module")
def any_dashboard(request):
    """Test nào đúng cho mọi board thì nhận fixture này, không phải `dashboard`."""
    return json.loads(BOARDS[request.param].read_text())


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
    for board, path in BOARDS.items():
        out = subprocess.run(
            [sys.executable, str(MON / "build_dashboard.py"), "--stdout", "--board", board],
            capture_output=True, text=True, check=True, cwd=ROOT,
        ).stdout
        assert json.loads(out) == json.loads(path.read_text()), (
            f"{path.name} lệch với build_dashboard.py - chạy lại generator"
        )


def test_moi_panel_co_datasource(any_dashboard):
    for panel in _panels(any_dashboard):
        if panel.get("type") == "row":
            continue
        assert panel.get("datasource"), f"panel thiếu datasource: {panel['title']}"


def test_moi_metric_deu_co_that(any_dashboard):
    for title, expr in _exprs(any_dashboard):
        used = set(re.findall(r"[a-zA-Z_:][a-zA-Z0-9_:]*", expr))
        unknown = {
            m for m in used
            if (m.startswith(("nv_", "vllm", "voice_")) and m not in KNOWN_METRICS)
        }
        assert not unknown, f"panel {title!r} dùng metric lạ: {unknown}"


def test_moi_phep_tinh_failure_deu_boc_sum(any_dashboard):
    """nv_inference_request_failure có label `reason`.

    Không bọc sum() thì phép cộng với success lệch label, Prometheus không
    match được cặp nào và trả rỗng - panel "No data" chứ không phải số sai.
    """
    for title, expr in _exprs(any_dashboard):
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


@pytest.fixture(scope="module")
def triton_dashboard():
    return json.loads(TRITON_DASHBOARD.read_text())


def test_hai_board_khac_uid():
    """Trùng uid thì Grafana nạp board sau đè board trước, im lặng, không báo."""
    uids = [json.loads(p.read_text())["uid"] for p in BOARDS.values()]
    assert len(set(uids)) == len(uids), f"uid trùng giữa các board: {uids}"


def test_id_panel_khong_trung_trong_mot_board(any_dashboard):
    """Grafana dùng id để định vị panel; trùng id là link/annotation trỏ sai chỗ."""
    ids = [p["id"] for p in _panels(any_dashboard)]
    assert len(set(ids)) == len(ids), f"id panel trùng: {ids}"


def test_triton_board_du_chi_so_server(triton_dashboard):
    """Những thứ board voice không có, và là lý do board này tồn tại.

    `up` để phân biệt "hệ rảnh" với "hệ chết" - thiếu nó thì mọi panel cùng
    hiện No data mà không chỗ nào nói vì sao.
    """
    titles = " | ".join(p["title"] for p in _panels(triton_dashboard)).lower()
    for keyword in ("up", "queue", "breakdown", "batch", "cpu", "power"):
        assert keyword in titles, f"board triton thiếu panel cho: {keyword}"

    exprs = " ".join(e for _, e in _exprs(triton_dashboard))
    assert 'up{job="triton"}' in exprs, "không panel nào theo dõi target sống/chết"


def test_breakdown_du_bon_thanh_phan(triton_dashboard):
    """request = queue + compute_input + compute_infer + compute_output.

    Thiếu một thành phần thì tổng không khớp p50 request và panel nói dối về
    chỗ thời gian trôi đi.
    """
    for panel in _panels(triton_dashboard):
        if "breakdown" not in panel["title"].lower():
            continue
        exprs = " ".join(t["expr"] for t in panel["targets"])
        for part in ("queue", "compute_input", "compute_infer", "compute_output"):
            assert f"nv_inference_{part}_summary_us_sum" in exprs, (
                f"panel {panel['title']!r} thiếu thành phần {part}"
            )


def test_breakdown_dung_sum_chia_count(triton_dashboard):
    """Quantile của summary trả Nan lúc rảnh, _sum/_count thì không.

    Breakdown phải đọc được cả khi không có traffic, nên bắt buộc dùng cặp
    _sum/_count chứ không phải nhánh quantile.
    """
    for panel in _panels(triton_dashboard):
        if "breakdown" not in panel["title"].lower():
            continue
        for target in panel["targets"]:
            assert "quantile=" not in target["expr"], (
                f"panel {panel['title']!r} dùng quantile - sẽ trống lúc hệ rảnh"
            )
            assert "_sum" in target["expr"] and "_count" in target["expr"]


def test_moi_phep_chia_deu_chan_mau_so(any_dashboard):
    """Mẫu số là rate() thì lúc không có traffic nó bằng 0 - chia ra +Inf.

    clamp_min giữ panel về 0 thay vì vọt lên vô cực rồi kéo hỏng trục y.
    """
    for title, expr in _exprs(any_dashboard):
        for m in re.finditer(r"/\s*(?!\s*clamp_min)", expr):
            tail = expr[m.end():].lstrip()
            if tail.startswith(("1000", "clamp_min")):
                continue
            assert "clamp_min" in tail[:40] or not tail.startswith(("sum(", "rate(")), (
                f"panel {title!r}: phép chia không chặn mẫu số - {expr}"
            )


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
