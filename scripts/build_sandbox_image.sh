#!/usr/bin/env bash
# Build the docker sandbox base image (pytest baked in; see Dockerfile.sandbox).
# Required before using sandbox.backend: docker with the default test command.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-coding-agent-sandbox:latest}"
docker build -f scripts/Dockerfile.sandbox -t "$TAG" .
echo "built $TAG — set sandbox.docker_image: $TAG (and backend: docker) in config/agent.yaml"
