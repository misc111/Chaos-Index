#!/bin/zsh
set -euo pipefail

cd "/Users/davidiruegas/Library/Application Support/SportsModeling"

dashboard_url="http://127.0.0.1:3000"

existing_page="$(curl -fsS "$dashboard_url" 2>/dev/null || true)"

if [[ "$existing_page" == *"/Chaos-Index/_next/static/"* ]]; then
  echo "Stopping stale GitHub Pages build on 127.0.0.1:3000..."
  existing_pids=("${(@f)$(lsof -nP -iTCP:3000 -sTCP:LISTEN -t 2>/dev/null)}")
  if (( ${#existing_pids[@]} > 0 )); then
    kill "${existing_pids[@]}" 2>/dev/null || true
    sleep 1
  fi
elif [[ "$existing_page" == *"<title>Chaos Index</title>"* ]]; then
  open "$dashboard_url"
  exit 0
elif [[ -n "$existing_page" ]]; then
  echo "Port 3000 is already serving a different app. Stop it, then rerun launch_dashboard.command."
  exit 1
fi

make dashboard &
dashboard_pid=$!

cleanup() {
  if kill -0 "$dashboard_pid" 2>/dev/null; then
    kill "$dashboard_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for _ in {1..120}; do
  if page="$(curl -fsS "$dashboard_url" 2>/dev/null)" && [[ "$page" == *"<title>Chaos Index</title>"* ]]; then
    open "$dashboard_url"
    break
  fi
  if ! kill -0 "$dashboard_pid" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

wait "$dashboard_pid"
