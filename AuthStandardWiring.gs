/**
 * AuthStandardWiring.gs — Conexao do AuthStandardService (rollout FROTA-17).
 *
 * Replica o piloto "Way To Go As Is" (AUTH_PILOT_PLAN.md). Configura o servico
 * padronizado de autenticacao por token contra a aba 'Usuarios' real e expoe
 * wrappers de sessao por token. ADITIVO, NAO-DESTRUTIVO e INERTE: nao altera o
 * login por sessao em producao; nada roda no load.
 *
 * Os wrappers usam nomes prefixados `AuthStd_*` para NAO colidirem com a cadeia
 * de descoberta do gateway consolidado (ApiGateway.gs / gw_login_). O cutover
 * para o caminho por token e um passo deliberado: apontar o dispatch de login
 * para `AuthStd_loginWithToken_` e o gate para `AuthStd_isAuthenticatedByToken_`
 * apos validar `AuthStandardContractTest.gs` no editor Apps Script.
 *
 * Credencial em TEXTO PLANO (decisao de frota) — zero hash. Aceita coluna
 * 'PasswordHash' opcional (apenas SHA-256 hex valido) sem exigi-la.
 */

/** Resolve a planilha principal reutilizando os helpers nativos do projeto. */
function AuthStd_getSpreadsheet_() {
  try { if (typeof _getSpreadsheet === 'function')          { var a = _getSpreadsheet();          if (a) return a; } } catch (e1) {}
  try { if (typeof getBoundSpreadsheet_ === 'function')     { var b = getBoundSpreadsheet_();     if (b) return b; } } catch (e2) {}
  try { if (typeof Auth_getSpreadsheet_ === 'function')     { var c = Auth_getSpreadsheet_();     if (c) return c; } } catch (e3) {}
  try { if (typeof SheetsDB_getSpreadsheet === 'function')  { var d = SheetsDB_getSpreadsheet();  if (d) return d; } } catch (e4) {}
  try { if (typeof authResolveSpreadsheet_ === 'function')  { var e = authResolveSpreadsheet_();  if (e) return e; } } catch (e5) {}
  try { var f = SpreadsheetApp.getActiveSpreadsheet(); if (f) return f; } catch (e6) {}
  return SpreadsheetApp.getActive();
}

/** Aceita apenas hashes SHA-256 hex (64 chars); caso contrario, vazio. */
function AuthStd_normalizarHash_(valor) {
  var v = String(valor == null ? '' : valor).trim();
  return /^[0-9a-fA-F]{64}$/.test(v) ? v : '';
}

/**
 * Adaptador findUser: localiza um usuario na aba 'Usuarios' por username OU email
 * (case-insensitive) e o devolve no formato esperado pelo AuthStandardService
 * ({ id, username, role, active, password, passwordHash }).
 */
function AuthStd_findUser_(username) {
  try {
    try {
      var ss = AuthStd_getSpreadsheet_();
      var sheet = ss.getSheetByName('Usuarios') ||
                  ss.getSheetByName('Users') ||
                  ss.getSheetByName('Usuários');
      if (!sheet || sheet.getLastRow() < 2) return null;
      var values = sheet.getDataRange().getValues();
      var h = values[0].map(function (x) { return String(x || '').trim().toLowerCase(); });
      var iUser = h.indexOf('username');
      var iPass = h.indexOf('password');
      var iHash = h.indexOf('passwordhash');
      var iRole = h.indexOf('role');
      var iNome = h.indexOf('nome');
      var iEmail = h.indexOf('email');
      var iId = h.indexOf('id');
      var iStatus = h.indexOf('status');
      if (iUser < 0 || (iPass < 0 && iHash < 0)) return null;

      var u = String(username || '').trim().toLowerCase();
      for (var r = 1; r < values.length; r++) {
        var row = values[r];
        var rowUser = String(row[iUser] || '').trim().toLowerCase();
        var rowEmail = iEmail >= 0 ? String(row[iEmail] || '').trim().toLowerCase() : '';
        if (rowUser !== u && rowEmail !== u) continue;
        return {
          id:           iId >= 0 && row[iId] ? row[iId] : rowUser,
          username:     row[iUser],
          name:         iNome >= 0 ? row[iNome] : row[iUser],
          email:        iEmail >= 0 ? row[iEmail] : '',
          role:         iRole >= 0 && row[iRole] ? row[iRole] : 'professor',
          active:       iStatus < 0 || String(row[iStatus]).trim().toLowerCase() !== 'inativo',
          password:     iPass >= 0 ? String(row[iPass] || '') : '',
          passwordHash: AuthStd_normalizarHash_(iHash >= 0 ? row[iHash] : '')
        };
      }
      return null;
    } catch (error) {
      Logger.log("Erro em AuthStd_findUser_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em AuthStd_findUser_: " + error.message);
    throw error;
  }
}

/** Instancia o AuthStandardService configurado para este projeto. */
function AuthStd_service_() {
  return AuthStandardService.configure({
    sessionKey: 'EMODIVERSA_AUTH_SESSION',
    sessionTtlSeconds: 21600,
    adapters: { findUser: AuthStd_findUser_ }
  });
}

/**
 * Valida credenciais via AuthStandardService e emite um token de sessao.
 * (Inerte ate o cutover; o servico gerencia o token internamente.)
 * @return {{ success:boolean, token?:string, user?:Object, message?:string }}
 */
function AuthStd_loginWithToken_(username, password) {
  try {
    if (!String(username || '').trim() || !String(password || '')) {
      return { success: false, message: 'Informe usuario e senha.' };
    }
    var result = AuthStd_service_().login(String(username).trim(), String(password));
    if (!result || !result.ok) return { success: false, message: 'Credenciais invalidas.' };
    return { success: true, token: result.token, user: result.user };
  } catch (error) {
    Logger.log("Erro em AuthStd_loginWithToken_: " + error.message);
    throw error;
  }
}

/** Indica se um token de sessao e valido e nao expirou. */
function AuthStd_isAuthenticatedByToken_(tok) {
  return AuthStd_service_().isAuthenticatedByToken(tok);
}

/** Revoga (logout) um token de sessao. */
function AuthStd_logoutWithToken_(tok) {
  return AuthStd_service_().logout(tok);
}
