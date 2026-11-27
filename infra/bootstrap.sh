#!/usr/bin/env bash
# One-time Azure setup for SafeSite.
#
# Run from the repository root, in Azure Cloud Shell (Bash) or any shell where
# `az login` has been done:
#
#     ./infra/bootstrap.sh <github-owner>/<repo>
#
# Creates the resource group and everything except the app itself - container
# registry, Container Apps environment, Log Analytics, managed identities - and
# an OIDC identity that lets .github/workflows/deploy.yml deploy with no stored
# password. Afterwards, every push to main builds and ships a new revision.
#
# Needs Owner (or User Access Administrator) on the subscription, because it
# creates role assignments. Optional overrides: NAME, RESOURCE_GROUP, LOCATION.
set -euo pipefail

REPO="${1:-}"
if [[ ! "$REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo "usage: $0 <github-owner>/<repo>" >&2
  exit 1
fi

NAME="${NAME:-safesite}"
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-${NAME}}"
LOCATION="${LOCATION:-centralindia}"
DEPLOYMENT="${NAME}-bootstrap"

cd "$(dirname "$0")/.."

echo "==> Signed in as:"
az account show --query "{subscription: name, id: id, user: user.name}" --output table

echo "==> Registering resource providers (first time only, may take a minute)"
for namespace in Microsoft.App Microsoft.ContainerRegistry Microsoft.OperationalInsights Microsoft.ManagedIdentity; do
  az provider register --namespace "$namespace" --wait --only-show-errors
done

echo "==> Resource group ${RESOURCE_GROUP} (${LOCATION})"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

echo "==> Registry, environment and identities"
az deployment group create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$DEPLOYMENT" \
  --template-file infra/main.bicep \
  --parameters name="$NAME" deployApp=false githubRepo="$REPO" \
  --output none

output() {
  az deployment group show --resource-group "$RESOURCE_GROUP" --name "$DEPLOYMENT" \
    --query "properties.outputs.$1.value" --output tsv
}

# Identifiers, not secrets: OIDC needs no password, so these can be plain
# repository variables.
VARIABLES=(
  "AZURE_CLIENT_ID=$(output githubClientId)"
  "AZURE_TENANT_ID=$(output tenantId)"
  "AZURE_SUBSCRIPTION_ID=$(output subscriptionId)"
  "AZURE_RESOURCE_GROUP=${RESOURCE_GROUP}"
  "AZURE_REGISTRY=$(output registryName)"
  "AZURE_APP_NAME=${NAME}"
)

echo
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  echo "==> Setting GitHub repository variables on ${REPO}"
  for pair in "${VARIABLES[@]}"; do
    gh variable set "${pair%%=*}" --body "${pair#*=}" --repo "$REPO"
  done
else
  echo "==> Add these on GitHub: ${REPO} -> Settings -> Secrets and variables -> Actions -> Variables"
  printf '    %s\n' "${VARIABLES[@]}"
fi

cat <<EOF

Done. Next:
  1. Optional: add a repository *secret* PPE_ADMIN_PASSWORD to enable the admin login.
  2. Push to main, or run the workflow by hand:
         gh workflow run deploy.yml --repo ${REPO}
     It builds the image, pushes it to the registry, deploys the Container App
     and prints the public URL in the run summary.
EOF
