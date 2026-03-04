#!/bin/zsh
set -euo pipefail

cd "/Users/davidiruegas/Library/Application Support/SportsModeling"

make dashboard &
dashboard_pid=$!

cleanup() {
  if kill -0 "$dashboard_pid" 2>/dev/null; then
    kill "$dashboard_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for _ in {1..120}; do
  if curl -fsS "http://localhost:3000" >/dev/null 2>&1; then
    open "http://localhost:3000"
    break
  fi
  if ! kill -0 "$dashboard_pid" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

wait "$dashboard_pid"
