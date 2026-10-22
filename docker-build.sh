#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-plonk-api}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or is not available in PATH." >&2
    exit 1
fi

echo "Building Docker image: ${IMAGE_NAME}"
docker_args=(build --tag "${IMAGE_NAME}")

# Forward the host proxy to BuildKit without storing it in the image.
for proxy_variable in \
    http_proxy https_proxy no_proxy \
    HTTP_PROXY HTTPS_PROXY NO_PROXY; do
    if [[ -n "${!proxy_variable:-}" ]]; then
        docker_args+=(--build-arg "${proxy_variable}=${!proxy_variable}")
    fi
done

docker_args+=("${SCRIPT_DIR}")
docker "${docker_args[@]}"
echo "Image built successfully: ${IMAGE_NAME}"
