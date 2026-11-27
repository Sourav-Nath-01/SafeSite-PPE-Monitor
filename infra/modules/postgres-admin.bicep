// Makes a managed identity the Microsoft Entra administrator of a PostgreSQL
// flexible server, so the app can sign in with a token instead of a password.
//
// A module because the administrator resource is named after the identity's
// object ID, which is only known once the identity exists; resource names in
// the parent template must be computable before deployment starts.

param serverName string
param principalId string
param principalName string

resource server 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' existing = {
  name: serverName
}

resource administrator 'Microsoft.DBforPostgreSQL/flexibleServers/administrators@2024-08-01' = {
  parent: server
  name: principalId
  properties: {
    principalType: 'ServicePrincipal'
    principalName: principalName
    tenantId: tenant().tenantId
  }
}
