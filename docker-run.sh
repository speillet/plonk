#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-plonk-api}"
CONTAINER_NAME="${CONTAINER_NAME:-plonk-api}"
API_PORT="${API_PORT:-8000}"
CACHE_VOLUME="${CACHE_VOLUME:-plonk-cache}"
USE_GPU="${USE_GPU:-0}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed or is not available in PATH." >&2
    exit 1
fi

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Error: image '${IMAGE_NAME}' does not exist." >&2
    echo "Build it first with: ./docker-build.sh" >&2
    exit 1
fi

docker_args=(
    run
    --rm
    --name "${CONTAINER_NAME}"
    --publish "${API_PORT}:8000"
    --volume "${CACHE_VOLUME}:/home/plonk/.cache"
)

if [[ "${USE_GPU}" == "1" ]]; then
    docker_args+=(--gpus all)
fi

# The API needs network access when Hugging Face and Torch models are downloaded.
for proxy_variable in \
    http_proxy https_proxy no_proxy \
    HTTP_PROXY HTTPS_PROXY NO_PROXY; do
    if [[ -n "${!proxy_variable:-}" ]]; then
        docker_args+=(--env "${proxy_variable}=${!proxy_variable}")
    fi
done

docker_args+=("${IMAGE_NAME}")

echo "Starting ${IMAGE_NAME} on http://localhost:${API_PORT}"
echo "API documentation: http://localhost:${API_PORT}/docs"
docker "${docker_args[@]}"
