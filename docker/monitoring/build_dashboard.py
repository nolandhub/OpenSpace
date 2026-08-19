#!/usr/bin/env python3
# ABOUTME: Sinh voice-serving.json từ bảng panel - JSON viết tay ~1200 dòng không review nổi
# ABOUTME: Chạy: python3 docker/monitoring/build_dashboard.py [--stdout]

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serving.metrics import CCU_TTL_S  # noqa: E402

DS = {"type": "prometheus", "uid": "prometheus"}
OUT = Path(__file__).parent / "grafana" / "dashboards" / "voice-serving.json"

# Instance nào không được chạm trong CCU_TTL_S giây thì nhân 0. Mô phỏng đúng
# cái _sweep() sẽ làm nếu nó được chạy - xem spec §6.
#
# Join theo exported_instance chứ không phải instance: label gốc của
# voice_ccu/voice_ccu_updated_at tên là "instance" (spec §5.3), nhưng
# Prometheus scrape target cũng gắn sẵn label "instance" = địa chỉ target
# (localhost:8002). Đụng tên, honor_labels mặc định false (prometheus.yml
# không bật) nên Prometheus đổi tên label gốc thành "exported_instance" và
# ghi đè "instance" bằng địa chỉ target. Join trên "instance" thì mọi
# instance model cùng chung một giá trị "localhost:8002" -> many-to-many,
# Prometheus trả lỗi "duplicate series" (đã xác nhận qua :9090/api/v1/query).
def ccu(selector: str) -> str:
    return (
        f"sum(voice_ccu{selector} * on(model, exported_instance) "
        f"(time() - voice_ccu_updated_at < bool {CCU_TTL_S:g}))"
    )


# sum() bắt buộc quanh failure: nó có label `reason`, không bọc thì phép cộng
# với success lệch label và Prometheus trả rỗng.
def error_rate(model: str) -> str:
    f = f'sum(rate(nv_inference_request_failure{{model="{model}"}}[1m]))'
    s = f'sum(rate(nv_inference_request_success{{model="{model}"}}[1m]))'
    return f"{f} / clamp_min({s} + {f}, 1e-9)"


def triton_rows(model: str) -> list:
    sel = f'{{model="{model}"}}'
    q = lambda p: f'nv_inference_request_summary_us{{model="{model}",quantile="{p}"}} / 1000'
    return [
        (f"{model} · RPS", [f"rate(nv_inference_request_success{sel}[1m])"], "reqps"),
        (f"{model} · Latency p50/p95/p99", [q("0.5"), q("0.95"), q("0.99")], "ms"),
        (f"{model} · CCU", [ccu(sel)], "short"),
        (f"{model} · Queue depth", [f"nv_inference_pending_request_count{sel}"], "short"),
        (
            f"{model} · RTF p50/p95/p99",
            [
                f"histogram_quantile({p}, sum by(le) (rate(voice_rtf_bucket{sel}[5m])))"
                for p in ("0.5", "0.95", "0.99")
            ],
            "none",
        ),
        (f"{model} · Error rate", [error_rate(model)], "percentunit"),
    ]


LLM_E2E = "rate(vllm:e2e_request_latency_seconds_bucket[5m])"
ROWS = [
    (
        "OVERVIEW",
        [
            (
                "RPS tổng",
                [
                    "sum(rate(nv_inference_request_success[1m]))"
                    " + sum(rate(vllm:request_success_total[1m]))"
                ],
                "reqps",
            ),
            (
                "Success rate (Triton)",
                [
                    "sum(rate(nv_inference_request_success[1m]))"
                    " / clamp_min(sum(rate(nv_inference_request_success[1m]))"
                    " + sum(rate(nv_inference_request_failure[1m])), 1e-9)"
                ],
                "percentunit",
            ),
            (
                "Error rate (Triton)",
                [
                    "sum(rate(nv_inference_request_failure[1m]))"
                    " / clamp_min(sum(rate(nv_inference_request_success[1m]))"
                    " + sum(rate(nv_inference_request_failure[1m])), 1e-9)"
                ],
                "percentunit",
            ),
            ("CCU tổng", [f"{ccu('')} + sum(vllm:num_requests_running)"], "short"),
        ],
    ),
    ("ASR_STREAMING", triton_rows("asr_streaming")),
    ("TTS", triton_rows("tts")),
    (
        "LLM (vLLM)",
        [
            ("llm · RPS", ["sum(rate(vllm:request_success_total[1m]))"], "reqps"),
            (
                "llm · Latency p50/p95/p99",
                [f"histogram_quantile({p}, sum by(le) ({LLM_E2E}))" for p in ("0.5", "0.95", "0.99")],
                "s",
            ),
            ("llm · CCU", ["vllm:num_requests_running"], "short"),
            ("llm · Queue depth", ["vllm:num_requests_waiting"], "short"),
            (
                "llm · TTFT p50/p95/p99",
                [
                    f"histogram_quantile({p}, sum by(le) "
                    f"(rate(vllm:time_to_first_token_seconds_bucket[5m])))"
                    for p in ("0.5", "0.95", "0.99")
                ],
                "s",
            ),
            (
                "llm · TPOT p50/p95/p99",
                [
                    f"histogram_quantile({p}, sum by(le) "
                    f"(rate(vllm:inter_token_latency_seconds_bucket[5m])))"
                    for p in ("0.5", "0.95", "0.99")
                ],
                "s",
            ),
            (
                "llm · Error rate (proxy: abort)",
                [
                    'sum(rate(vllm:request_success_total{finished_reason="abort"}[1m]))'
                    " / clamp_min(sum(rate(vllm:request_success_total[1m])), 1e-9)"
                ],
                "percentunit",
            ),
        ],
    ),
    (
        "GPU (chung cả máy)",
        [
            ("GPU Utilization", ["nv_gpu_utilization * 100"], "percent"),
            (
                "GPU Memory usage",
                ["nv_gpu_memory_used_bytes / nv_gpu_memory_total_bytes * 100"],
                "percent",
            ),
            ("GPU Memory used", ["nv_gpu_memory_used_bytes"], "bytes"),
        ],
    ),
]


def build() -> dict:
    panels, pid, y = [], 1, 0
    for row_title, specs in ROWS:
        panels.append(
            {
                "type": "row", "title": row_title, "id": pid,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
                "collapsed": False, "panels": [],
            }
        )
        pid += 1
        y += 1
        for i, (title, exprs, unit) in enumerate(specs):
            panels.append(
                {
                    "type": "timeseries", "title": title, "id": pid,
                    "datasource": DS,
                    "gridPos": {"h": 8, "w": 8, "x": (i % 3) * 8, "y": y + (i // 3) * 8},
                    "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
                    "targets": [
                        {"datasource": DS, "expr": e, "refId": chr(65 + n),
                         "legendFormat": "{{model}}{{instance}}{{quantile}}"}
                        for n, e in enumerate(exprs)
                    ],
                }
            )
            pid += 1
        y += ((len(specs) + 2) // 3) * 8
    return {
        "uid": "voice-serving",
        "title": "Voice Serving",
        "tags": ["triton", "vllm", "voice"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "10s",
        "time": {"from": "now-30m", "to": "now"},
        "panels": panels,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="in ra thay vì ghi file")
    args = ap.parse_args()
    text = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    if args.stdout:
        print(text, end="")
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text)
        print(f"đã ghi {OUT}")
