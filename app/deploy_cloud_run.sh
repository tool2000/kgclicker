#!/usr/bin/env bash
set -euo pipefail

# Helper script to build, push, and deploy the kg-gen app to Cloud Run.
# Required environment variables:
#   GCP_PROJECT_ID              - Google Cloud project id
#   CLOUD_RUN_SERVICE           - Cloud Run service name
#   CLOUD_RUN_REGION            - Cloud Run region (e.g. us-central1)
#   ARTIFACT_REGISTRY_REPO      - Artifact Registry repository name
# Optional environment variables:
#   ARTIFACT_REGISTRY_LOCATION  - Artifact Registry location (defaults to CLOUD_RUN_REGION)
#   IMAGE_NAME                   - Image name (defaults to cloud-run-app)
#   IMAGE_TAG                    - Image tag (defaults to current git commit or timestamp)
#   CLOUD_RUN_SERVICE_ACCOUNT    - Service account email to run the service as
#   CLOUD_RUN_MAX_INSTANCES      - Maximum number of instances
#   CLOUD_RUN_MIN_INSTANCES      - Minimum number of instances
#   CLOUD_RUN_CPU                - Cloud Run CPU allocation (e.g. 1, 2)
#   CLOUD_RUN_MEMORY             - Cloud Run memory allocation (e.g. 512Mi, 1Gi)
#   CLOUD_RUN_CONCURRENCY        - Max concurrent requests per instance

for var in GCP_PROJECT_ID CLOUD_RUN_SERVICE CLOUD_RUN_REGION ARTIFACT_REGISTRY_REPO; do
  if [[ -z "${!var:-}" ]]; then
    echo "[deploy] Missing required environment variable: $var" >&2
    exit 1
  fi
done

IMAGE_NAME=${IMAGE_NAME:-cloud-run-app}
IMAGE_TAG=${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%s)}
ARTIFACT_REGISTRY_LOCATION=${ARTIFACT_REGISTRY_LOCATION:-$CLOUD_RUN_REGION}
IMAGE_URI="${ARTIFACT_REGISTRY_LOCATION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

printf "[deploy] Building and pushing image %s\n" "$IMAGE_URI"

gcloud builds submit --tag "$IMAGE_URI" --project "$GCP_PROJECT_ID"

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
