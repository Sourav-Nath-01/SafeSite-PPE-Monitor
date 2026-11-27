// SafeSite on Azure Container Apps.
//
//   Log Analytics  <-  Container Apps environment  <-  Container App (1 replica)
//                                                         |  pulls image via managed identity
//                                                         v
//                                            Azure Container Registry
//
// With persistence=true the app also gets durable storage, both reached with
// a second managed identity (Microsoft Entra tokens, no passwords or keys):
//   PostgreSQL flexible server  - violation records
//   Blob Storage                - snapshot images (private container)
//
// Deployed in two phases because the app needs an image that lives in the
// registry this same template creates:
//   1. deployApp=false  -> registry, identities, environment   (infra/bootstrap.sh)
//   2. deployApp=true   -> the Container App on a pushed image  (.github/workflows/deploy.yml)
//
// No passwords anywhere: the app pulls with a managed identity (AcrPull) and
// GitHub Actions signs in through OIDC federation on a second identity.

targetScope = 'resourceGroup'

@description('Short lowercase prefix for every resource name.')
@minLength(3)
@maxLength(16)
param name string = 'safesite'

param location string = resourceGroup().location

@description('Create or update the Container App. False on the first run, before an image exists.')
param deployApp bool = true

@description('Create role assignments. Needs Owner / User Access Administrator, so the CI identity (Contributor) passes false.')
param assignRoles bool = true

@description('Image tag in the registry, e.g. the git commit SHA.')
param imageTag string = 'latest'

@description('vCPU per replica; memory is set to the matching Container Apps size (2 GiB per vCPU).')
@allowed([
  '0.5'
  '1.0'
  '1.5'
  '2.0'
])
param cpu string = '1.0'

@description('0 scales to zero when idle (near-free, ~30-60 s cold start). 1 keeps the demo warm.')
@minValue(0)
@maxValue(1)
param minReplicas int = 0

@description('Delete violation snapshots older than this many hours (0 keeps them).')
param retentionHours int = 24

@description('Concurrent browser video sessions; each holds its own model and tracker.')
@minValue(1)
@maxValue(8)
param maxSessions int = 3

@description('GitHub repository (owner/name) allowed to deploy via OIDC. Empty skips the CI identity.')
param githubRepo string = ''

@description('Store records in PostgreSQL and snapshots in Blob Storage instead of the container\'s temporary disk.')
param persistence bool = false

@description('Region for the PostgreSQL server; empty uses the main location. Some subscriptions restrict PostgreSQL SKUs by region.')
param databaseLocation string = ''

@description('Numeric GitHub owner ID. With githubRepoId, also trusts the immutable subject form repo:owner@id/name@id.')
param githubOwnerId string = ''

@description('Numeric GitHub repository ID (see githubOwnerId).')
param githubRepoId string = ''

@description('Flask session signing key. Defaults to a fresh value per deployment.')
@secure()
param secretKey string = newGuid()

@description('Password for the "admin" user (settings, all visitors\' violations). Empty disables admin.')
@secure()
param adminPassword string = ''

var suffix = uniqueString(resourceGroup().id)
var registryName = take('${replace(name, '-', '')}${suffix}', 50)
var appName = 'ca-${name}'
var port = 3333
var memoryByCpu = {
  '0.5': '1Gi'
  '1.0': '2Gi'
  '1.5': '3Gi'
  '2.0': '4Gi'
}
// torch otherwise sizes its thread pool to the host's cores, not the replica's
// share, and oversubscribes the CPU it actually has.
var torchThreads = string(max(1, int(split(cpu, '.')[0])))

var dbLocation = empty(databaseLocation) ? location : databaseLocation
var storageName = take('st${replace(name, '-', '')}${suffix}', 24)
// The region is part of the name: a server that fails to provision (e.g. a
// SKU the subscription may not use in that region) can linger in ARM's records
// under its name, and a name cannot move regions - so a retry elsewhere needs
// a different name.
var postgresName = 'psql-${name}-${uniqueString(resourceGroup().id, dbLocation)}'
var databaseName = 'safesite'
var blobContainer = 'snapshots'
// Blob lifecycle works in days; it backs up the app's own hourly retention.
var retentionDays = max(1, (retentionHours + 23) / 24)

var blobDataContributorRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
var acrPullRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
var contributorRole = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b24988ac-6180-42a0-ab88-20f7382dd24c')

// ---------------------------------------------------------------------------
// Shared infrastructure
// ---------------------------------------------------------------------------

resource logs 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: 'log-${name}'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: registryName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

// Identity the Container App uses to pull its image.
resource pullIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${name}-pull'
  location: location
}

resource pullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (assignRoles) {
  name: guid(registry.id, pullIdentity.id, acrPullRole)
  scope: registry
  properties: {
    roleDefinitionId: acrPullRole
    principalId: pullIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${name}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Persistent storage (persistence=true)
// ---------------------------------------------------------------------------

// Identity the app uses for its data: PostgreSQL (as Entra admin) and Blob.
resource appIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${name}-app'
  location: location
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = if (persistence) {
  name: storageName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false // identity-only: no account key works
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = if (persistence) {
  parent: storage
  name: 'default'
}

resource snapshotContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = if (persistence) {
  parent: blobService
  name: blobContainer
  properties: {
    publicAccess: 'None'
  }
}

resource snapshotLifecycle 'Microsoft.Storage/storageAccounts/managementPolicies@2023-05-01' = if (persistence && retentionHours > 0) {
  parent: storage
  name: 'default'
  properties: {
    policy: {
      rules: [
        {
          name: 'expire-snapshots'
          enabled: true
          type: 'Lifecycle'
          definition: {
            filters: {
              blobTypes: [
                'blockBlob'
              ]
              prefixMatch: [
                '${blobContainer}/'
              ]
            }
            actions: {
              baseBlob: {
                delete: {
                  daysAfterModificationGreaterThan: retentionDays
                }
              }
            }
          }
        }
      ]
    }
  }
}

resource blobRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (persistence && assignRoles) {
  name: guid(resourceGroup().id, storageName, 'id-${name}-app', blobDataContributorRole)
  scope: storage
  properties: {
    roleDefinitionId: blobDataContributorRole
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Burstable B1ms with 32 GB: the size covered by the 12-month free tier.
resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = if (persistence) {
  name: postgresName
  location: dbLocation
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    version: '16'
    storage: {
      storageSizeGB: 32
      autoGrow: 'Disabled'
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
    network: {
      publicNetworkAccess: 'Enabled'
    }
    // Entra-only: password sign-in is switched off entirely.
    authConfig: {
      activeDirectoryAuth: 'Enabled'
      passwordAuth: 'Disabled'
      tenantId: tenant().tenantId
    }
  }
}

// Consumption-plan Container Apps have no fixed outbound IP; 0.0.0.0 is
// Azure's convention for "connections from Azure services only".
resource postgresFirewall 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2024-08-01' = if (persistence) {
  parent: postgres
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// The server accepts one change at a time, hence the explicit chain below.
resource postgresDatabase 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = if (persistence) {
  parent: postgres
  name: databaseName
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
  dependsOn: [
    postgresFirewall
  ]
}

module postgresAdmin 'modules/postgres-admin.bicep' = if (persistence) {
  name: 'postgres-admin'
  params: {
    serverName: postgresName
    principalId: appIdentity.properties.principalId
    principalName: appIdentity.name
  }
  dependsOn: [
    postgres
    postgresDatabase
  ]
}

// ---------------------------------------------------------------------------
// GitHub Actions deploy identity (OIDC, no stored secret)
// ---------------------------------------------------------------------------

resource githubIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = if (!empty(githubRepo)) {
  name: 'id-${name}-github'
  location: location
}

// Tokens are only accepted from jobs running in the repo's "production"
// environment, which can carry its own protection rules.
//
// GitHub issues the subject claim in one of two forms: by name
// (repo:owner/name:...) or with immutable numeric IDs
// (repo:owner@123/name@456:...), which survives renames and blocks
// re-registration of a deleted name. Trust both, so either works.
var githubParts = split(githubRepo, '/')
var githubSubjects = concat(
  [
    {
      name: 'github-production'
      subject: 'repo:${githubRepo}:environment:production'
    }
  ],
  empty(githubRepoId) || empty(githubOwnerId) ? [] : [
    {
      name: 'github-production-ids'
      subject: 'repo:${githubParts[0]}@${githubOwnerId}/${last(githubParts)}@${githubRepoId}:environment:production'
    }
  ]
)

// One at a time: Azure rejects concurrent writes of federated credentials
// on the same identity.
@batchSize(1)
resource githubFederation 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = [for credential in githubSubjects: if (!empty(githubRepo)) {
  parent: githubIdentity
  name: credential.name
  properties: {
    issuer: 'https://token.actions.githubusercontent.com'
    subject: credential.subject
    audiences: [
      'api://AzureADTokenExchange'
    ]
  }
}]

// Scoped to this resource group only.
resource githubRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(githubRepo) && assignRoles) {
  name: guid(resourceGroup().id, 'id-${name}-github', contributorRole)
  properties: {
    roleDefinitionId: contributorRole
    principalId: githubIdentity!.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ---------------------------------------------------------------------------
// The app
// ---------------------------------------------------------------------------

var probe = {
  path: '/healthz'
  port: port
}

resource app 'Microsoft.App/containerApps@2024-03-01' = if (deployApp) {
  name: appName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${pullIdentity.id}': {}
      '${appIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: port
        transport: 'auto' // HTTP/1.1 + WebSocket upgrade for Socket.IO
        allowInsecure: false
      }
      registries: [
        {
          server: registry.properties.loginServer
          identity: pullIdentity.id
        }
      ]
      secrets: concat(
        [
          {
            name: 'secret-key'
            value: secretKey
          }
        ],
        empty(adminPassword) ? [] : [
          {
            name: 'admin-password'
            value: adminPassword
          }
        ]
      )
    }
    template: {
      containers: [
        {
          name: 'safesite'
          image: '${registry.properties.loginServer}/safesite:${imageTag}'
          resources: {
            cpu: json(cpu)
            memory: memoryByCpu[cpu]
          }
          env: concat(
            [
              {
                name: 'PPE_SECRET_KEY'
                secretRef: 'secret-key'
              }
              {
                name: 'PPE_PUBLIC_DEMO'
                value: 'true'
              }
              {
                name: 'PPE_SECURE_COOKIES'
                value: 'true'
              }
              {
                name: 'PPE_RETENTION_HOURS'
                value: string(retentionHours)
              }
              {
                name: 'PPE_MAX_SESSIONS'
                value: string(maxSessions)
              }
              {
                name: 'OMP_NUM_THREADS'
                value: torchThreads
              }
            ],
            empty(adminPassword) ? [] : [
              {
                name: 'PPE_ADMIN_PASSWORD'
                secretRef: 'admin-password'
              }
            ],
            persistence ? [
              {
                // Which of the app's identities DefaultAzureCredential uses.
                name: 'AZURE_CLIENT_ID'
                value: appIdentity.properties.clientId
              }
              {
                // No password: the app signs in with an Entra token (PPE_DB_AUTH).
                name: 'PPE_DATABASE_URL'
                value: 'host=${postgres!.properties.fullyQualifiedDomainName} port=5432 dbname=${databaseName} user=${appIdentity.name} sslmode=require'
              }
              {
                name: 'PPE_DB_AUTH'
                value: 'entra'
              }
              {
                name: 'PPE_BLOB_ACCOUNT_URL'
                value: storage!.properties.primaryEndpoints.blob
              }
              {
                name: 'PPE_BLOB_CONTAINER'
                value: blobContainer
              }
            ] : []
          )
          probes: [
            {
              // Up to ~110 s for torch import, model load and warm-up.
              type: 'Startup'
              httpGet: probe
              initialDelaySeconds: 10
              periodSeconds: 10
              failureThreshold: 10
              timeoutSeconds: 5
            }
            {
              type: 'Readiness'
              httpGet: probe
              periodSeconds: 10
              failureThreshold: 3
              timeoutSeconds: 5
            }
            {
              type: 'Liveness'
              httpGet: probe
              periodSeconds: 30
              failureThreshold: 3
              timeoutSeconds: 5
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        // Socket.IO sessions and tracker state are in process memory, so a
        // second replica would receive frames for sessions it never saw.
        maxReplicas: 1
      }
    }
  }
  dependsOn: [
    pullRole
    postgresAdmin
    snapshotContainer
  ]
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

output registryName string = registry.name
output registryLoginServer string = registry.properties.loginServer
output appName string = appName
output appUrl string = deployApp ? 'https://${app!.properties.configuration.ingress.fqdn}' : ''
output githubClientId string = empty(githubRepo) ? '' : githubIdentity!.properties.clientId
output databaseServer string = persistence ? postgresName : ''
output storageAccount string = persistence ? storageName : ''
output tenantId string = tenant().tenantId
output subscriptionId string = subscription().subscriptionId
