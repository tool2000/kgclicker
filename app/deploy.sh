#!/usr/bin/env bash
set -euo pipefail

GCP_PROJECT_ID=${GCP_PROJECT_ID:-kggen-ai}
CLOUD_RUN_SERVICE=${CLOUD_RUN_SERVICE:-app-test}
CLOUD_RUN_REGION=${CLOUD_RUN_REGION:-us-central1}
ARTIFACT_REGISTRY_REPO=${ARTIFACT_REGISTRY_REPO:-app}
IMAGE_NAME=${IMAGE_NAME:-cloud-run-app-test}
IMAGE_TAG=${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%s)}
ARTIFACT_REGISTRY_LOCATION=${ARTIFACT_REGISTRY_LOCATION:-$CLOUD_RUN_REGION}
IMAGE_URI="${ARTIFACT_REGISTRY_LOCATION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

printf "[deploy] Building and pushing image %s\n" "$IMAGE_URI"

# Change to project root for build context
cd "$(dirname "$0")/.."

# Ensure Dockerfile exists
if [[ ! -f "app/Dockerfile" ]]; then
  echo "ERROR: app/Dockerfile not found!"
  exit 1
fi

# Copy Dockerfile to root for Cloud Build (most reliable approach)
cp app/Dockerfile Dockerfile
cp app/.dockerignore .dockerignore 2>/dev/null || true

# Submit build with Dockerfile at root
gcloud builds submit --tag "$IMAGE_URI" --project "$GCP_PROJECT_ID" .

# Clean up
rm -f Dockerfile .dockerignore

deploy_args=(
  --image "$IMAGE_URI"
  --region "$CLOUD_RUN_REGION"
  --project "$GCP_PROJECT_ID"
  --platform managed
  --allow-unauthenticated
  --quiet
)

if [[ -n "${CLOUD_RUN_SERVICE_ACCOUNT:-}" ]]; then
  deploy_args+=(--service-account "$CLOUD_RUN_SERVICE_ACCOUNT")
fi

if [[ -n "${CLOUD_RUN_MAX_INSTANCES:-}" ]]; then
  deploy_args+=(--max-instances "$CLOUD_RUN_MAX_INSTANCES")
fi

if [[ -n "${CLOUD_RUN_MIN_INSTANCES:-}" ]]; then
  deploy_args+=(--min-instances "$CLOUD_RUN_MIN_INSTANCES")
fi

if [[ -n "${CLOUD_RUN_CPU:-}" ]]; then
  deploy_args+=(--cpu "$CLOUD_RUN_CPU")
fi

if [[ -n "${CLOUD_RUN_MEMORY:-}" ]]; then
  deploy_args+=(--memory "$CLOUD_RUN_MEMORY")
fi

if [[ -n "${CLOUD_RUN_CONCURRENCY:-}" ]]; then
  deploy_args+=(--concurrency "$CLOUD_RUN_CONCURRENCY")
fi

printf "[deploy] Deploying service %s to Cloud Run\n" "$CLOUD_RUN_SERVICE"

gcloud run deploy "$CLOUD_RUN_SERVICE" "${deploy_args[@]}"

echo "[deploy] Deployment complete"
