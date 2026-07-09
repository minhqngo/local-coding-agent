#!/usr/bin/env bash
# Phase 1, step 1 — download the GGUF weights for every configured role.
#
# Requires the Hugging Face CLI:  pip install -U "huggingface_hub[cli]"
# Weights land in ./models/ (gitignored). Re-running is safe; hf skips existing.
#
#   ./scripts/download_models.sh            # all roles in config/models.yaml
#   ./scripts/download_models.sh coder planner   # only these roles
set -euo pipefail

cd "$(dirname "$0")/.."
MODELS_DIR="${MODELS_DIR:-./models}"
mkdir -p "$MODELS_DIR"

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "error: install the HF CLI first:  pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi
HF="hf"; command -v hf >/dev/null 2>&1 || HF="huggingface-cli"

download() {  # repo, filename
  local repo="$1" file="$2"
  [ -z "$repo" ] || [ -z "$file" ] && return 0
  echo ">> $repo :: $file"
  "$HF" download "$repo" "$file" --local-dir "$MODELS_DIR" || \
    echo "   (skipped: check the repo/filename in config/models.yaml)" >&2
}

download_role() {
  local role="$1"
  echo "== role: $role =="
  download "$(python3 scripts/_config.py field "$role" hf_repo)" \
           "$(python3 scripts/_config.py field "$role" gguf)"
  # optional speculative-decoding draft weights (coder only)
  download "$(python3 scripts/_config.py field "$role" draft_hf_repo)" \
           "$(python3 scripts/_config.py field "$role" draft_gguf)"
}

if [ "$#" -gt 0 ]; then
  for role in "$@"; do download_role "$role"; done
else
  # Portable role iteration (no bash-4 `mapfile`).
  python3 scripts/_config.py roles | while IFS= read -r role; do
    [ -n "$role" ] && download_role "$role"
  done
fi

echo "done. weights in $MODELS_DIR"
