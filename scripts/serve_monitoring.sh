#!/usr/bin/env bash
# ABOUTME: Dựng Prometheus (9090) + Grafana (3000) scrape Triton và vLLM
# ABOUTME: Dùng: ./scripts/serve_monitoring.sh | ./scripts/serve_monitoring.sh down

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE="$ROOT/docker/monitoring/docker-compose.yml"

if [ "${1:-up}" = "down" ]; then
  docker compose -f "$COMPOSE" down
  exit 0
fi

# Cả hai container dùng host network nên cổng đụng là chết im lặng trong log
# container. Chặn ở đây để báo cho rõ.
for port in 9090 3000; do
  if ss -ltn "sport = :$port" | grep -q LISTEN; then
    echo "dừng: cổng $port đang bị chiếm." >&2
    echo "      ss -ltnp 'sport = :$port'  để xem tiến trình nào." >&2
    exit 1
  fi
done

docker compose -f "$COMPOSE" up -d

echo
echo "Prometheus  http://localhost:9090/targets"
echo "Grafana     http://localhost:3000"
echo
echo "Triton và vLLM phải chạy sẵn thì target mới UP:"
echo "  ./scripts/serve_triton.sh  &&  ./scripts/serve_llm.sh"
