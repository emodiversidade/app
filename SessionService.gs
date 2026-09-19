// SessionService.gs
/**
 * @overview LEGADO - Gerencia sessões usando getUserCache (INSEGURO em "Execute as: Me")
 *           MIGRADO PARA AuthHelpers.gs com rotinas baseadas em token
 * @module SessionService
 * @deprecated Use AuthHelpers.gs com getSessionUser(token) e isAuthenticatedByToken(token)
 * 
 * MOTIVO DA DEPRECAÇÃO:
 * - getUserCache herda sessão do dono do script em deployments "Execute as: Me"
 * - Cria vulnerabilidade de autenticação compartilhada
 * - Substituído por ScriptProperties + token opaco por sessão
 */

const CURRENT_SESSION_KEY_sgteLegacy = "SGTE_CURRENT_SESSION";
const SESSION_TTL_SECONDS_sgteLegacy = 21600;

/**
 * LEGADO/INSEGURO: Nao use em novos fluxos.
 * Mantido apenas para compatibilidade com codigo antigo que ainda chama.
 * @deprecated Use loginWithToken do AuthHelpers.gs
 */
function createSession_sgteLegacy(userId) {
  // Nao criar sessao sem token
  Logger.log('AVISO: createSession_sgteLegacy foi chamado. Use loginWithToken ao inves.');
  return null;
}

/**
 * LEGADO/INSEGURO: Retorna null sem token.
 * @deprecated Use getSessionUser(token) do AuthHelpers.gs
 */
function getSession_sgteLegacy() {
  Logger.log('AVISO: getSession_sgteLegacy foi chamado sem token. Use getSessionUser(token).');
  return null;
}

/**
 * LEGADO: Nao faz nada sem token.
 * @deprecated Use logoutWithToken(token) do AuthHelpers.gs
 */
function invalidateSession_sgteLegacy() {
  Logger.log('AVISO: invalidateSession_sgteLegacy foi chamado. Use logoutWithToken(token).');
}

/**
 * LEGADO/INSEGURO: Retorna null sem token.
 * @deprecated Use requireAuthenticatedPrincipal_(token) do AuthHelpers.gs
 */
function getCurrentSessionUser_sgteLegacy() {
  Logger.log('AVISO: getCurrentSessionUser_sgteLegacy foi chamado sem token. Use getSessionUser(token).');
  return null;
}
