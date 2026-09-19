/**
 * @file ApiEndpoints.gs
 * Endpoints públicos chamáveis via google.script.run no frontend.
 * Gerado por codex_integrate_frontend_backend.py — pode ser customizado.
 */

/**
 * Carrega dados iniciais do dashboard.
 * Aceita o token via parâmetro (frota usa loginWithToken → ?page=app#tok= na URL).
 * @param {string} [tok] Token da sessão.
 * @return {{success:boolean, currentUser?:Object, appData?:Object, message?:string}}
 */
function getInitialAppData(tok) {
  var session = null;
  if (tok && typeof isAuthenticatedByToken === 'function' && isAuthenticatedByToken(tok)) {
    session = (typeof getSessionUser === 'function') ? getSessionUser(tok) : null;
  }
  if (!session) {
    return { success: false, message: 'Sessão inválida. Faça login novamente.' };
  }
  var currentUser = {
    id:       session.userId   || session.id       || session.username || 'unknown',
    username: session.username || session.name     || 'Usuário',
    name:     session.nome     || session.name     || session.username || 'Usuário',
    role:     session.role     || 'USER',
    email:    session.email    || ''
  };
  return { success: true, currentUser: currentUser, appData: {} };
}

/**
 * Valida se a sessão corrente ainda é válida (heartbeat do frontend).
 * @param {string} tok
 */
function pingSession(tok) {
  if (!tok) return { ok: false };
  try {
    return { ok: typeof isAuthenticatedByToken === 'function' && isAuthenticatedByToken(tok) };
  } catch (e) {
    return { ok: false };
  }
}
