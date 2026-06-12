#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <image-tag>"
  echo "example: $0 gabeengineers/diffusers-runpod-bonsai:latest"
  exit 1
fi

IMAGE_TAG="$1"
OMNI_IMAGE="${OMNI_IMAGE:-vllm/vllm-omni:v0.22.0}"
TRANSFORMERS_REPO="${TRANSFORMERS_REPO:-https://github.com/gabe-engineers/transformers.git}"
TRANSFORMERS_REF="${TRANSFORMERS_REF:-fix/hqq-nested-checkpoint-load}"

docker buildx build \
  --platform linux/amd64 \
  --build-arg "OMNI_IMAGE=${OMNI_IMAGE}" \
  --build-arg "TRANSFORMERS_REPO=${TRANSFORMERS_REPO}" \
  --build-arg "TRANSFORMERS_REF=${TRANSFORMERS_REF}" \
  -f tmp/Dockerfile.runpod-bonsai \
  -t "$IMAGE_TAG" \
  --push \
  .
