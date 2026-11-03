#!/usr/bin/env bash
# Builds locally; never pushes to a registry or changes cloud configuration.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BASE_IMAGE=${BASE_IMAGE:-rescopebench-portable:reviewed18-20261015}
IMAGE=${IMAGE:-rescopebench-harness:local}
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  docker load --input "$ROOT/rescopebench/vendor/hermes-runtime-portable-reviewed18-v2-20261015.tar.gz"
fi
docker build --platform "${PLATFORM:-linux/arm64}" --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --tag "$IMAGE" "$ROOT/hermes_image"
printf 'Use before prepare: export RESCOPE_SANDBOX_IMAGE=%s\n' "$IMAGE"
