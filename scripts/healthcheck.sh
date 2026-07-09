#!/usr/bin/env bash
# Phase 1, step 2 — sanity-check every role's endpoint with curl.
# Prints OK/DOWN per role and a one-line chat completion where reachable.
#
#   ./scripts/healthcheck.sh                    # check local servers
#   HOST=192.168.1.50 ./scripts/healthcheck.sh  # check a remote serving machine
set -uo pipefail

cd "$(dirname "$0")/.."
HOST="${HOST:-127.0.0.1}"

check_role() {
  local role="$1" port base
  port="$(python3 scripts/_config.py field "$role" port)"
  base="http://${HOST}:${port}/v1"
  local model
  model="$(python3 scripts/_config.py field "$role" model)"

  if ! curl -sf "${base}/models" >/dev/null 2>&1; then
    printf '  %-12s :%s  DOWN\n' "$role" "$port"
    return
  fi
  # Fire a tiny completion to confirm the model actually responds.
  local reply
  reply="$(curl -sf "${base}/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${model}\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}" \
      2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"][:40])' 2>/dev/null || true)"
  printf '  %-12s :%s  OK   %s\n' "$role" "$port" "${reply:-<no content>}"
}

echo "endpoint health ($HOST):"
# Portable role iteration (no bash-4 `mapfile`).
python3 scripts/_config.py roles | while IFS= read -r role; do
  [ -n "$role" ] && check_role "$role"
done
