#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-plonk-api}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or is not available in PATH." >&2
    exit 1
fi

echo "Building Docker image: ${IMAGE_NAME}"
docker build --tag "${IMAGE_NAME}" "${SCRIPT_DIR}"
echo "Image built successfully: ${IMAGE_NAME}"
