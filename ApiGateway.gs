/**
 * ApiGateway.gs — Gateway RPC unico da frota (consolidado em 2026-06-21).
 *
 * O ApiClient.html chama `google.script.run.apiCall(service, method, payload)`.
 * Este e o UNICO ponto de entrada `apiCall` do projeto (a colisao com
 * ApiGatewayStandard.gs foi eliminada).
 *
 * Login e sessao sao resolvidos por DESCOBERTA EM RUNTIME (typeof), espelhando
 * o FleetLoginCheck.gs (harness verde) — assim o gateway funciona com a funcao
 * de auth real de cada projeto sem hardcode. Texto plano no login (decisao de
 * frota): apenas delega; nao verifica nem gera hash.
 *
 * Convencao: switch(`${service}.${method}`); envelope de sucesso { ok:true, data },
 * de falha { ok:false, error:{ message } }. Unica rota publica: AuthService.login.
 */
function apiCall(service, method, payload) {
  try {
    var request = payload || {};
    var operation = String(service) + '.' + String(method);
    var publicCall = operation === 'AuthService.login';

    try {
      if (!publicCall && !gw_currentUser_()) {
        throw new Error('Sessao expirada. Faca login novamente.');
      }

      var data;
      switch (operation) {
        case 'AuthService.login':
          data = gw_login_(
            String(request.username || '').trim(),
            String(request.password || '').trim()
          );
          if (!gw_isLoginOk_(data)) throw new Error('Usuario ou senha invalidos.');
          data = gw_normalizeLogin_(data);
          break;
        case 'AuthService.logout':
          data = gw_logout_();
          break;
        default:
          throw new Error('Operacao de API nao permitida: ' + operation);
      }

      return { ok: true, data: toClientSafe_(data) };
    } catch (apiError) {
      return {
        ok: false,
        error: { message: (apiError && apiError.message) || 'Erro interno do servidor.' }
      };
    }
  } catch (error) {
    Logger.log("Erro em apiCall: " + error.message);
    throw error;
  }
}

/**
 * Resolve o login na MESMA ordem do FleetLoginCheck.gs: a primeira funcao de
 * login existente vence. Tenta a forma posicional (u, p) e, se falhar, a forma
 * de objeto ({ username, password, senha, email }). Cobre as duas convencoes da frota.
 */
function gw_login_(username, password) {
  try {
    var entries = [];
    if (typeof AuthService !== 'undefined' && AuthService && typeof AuthService.login === 'function') {
      entries.push(function (form) { return AuthService.login.apply(AuthService, form); });
    }
    if (typeof doLogin === 'function')             entries.push(function (form) { return doLogin.apply(null, form); });
    if (typeof processLoginRequest === 'function') entries.push(function (form) { return processLoginRequest.apply(null, form); });
    if (typeof loginWithPassword === 'function')   entries.push(function (form) { return loginWithPassword.apply(null, form); });
    if (typeof loginWithToken === 'function')      entries.push(function (form) { return loginWithToken.apply(null, form); });
    if (typeof login === 'function')               entries.push(function (form) { return login.apply(null, form); });
    if (typeof authenticate === 'function')        entries.push(function (form) { return authenticate.apply(null, form); });

    var positional = [username, password];
    var objectForm = [{ username: username, password: password, senha: password, email: username }];
    var last = null;
    for (var i = 0; i < entries.length; i++) {
      try {
        var r = entries[i](positional);
        if (gw_isLoginOk_(r)) return r;
        last = r;
        try {
          var r2 = entries[i](objectForm);
          if (gw_isLoginOk_(r2)) return r2;
          last = last || r2;
        } catch (ignoredObj) {}
      } catch (err) {
        last = last || { success: false, message: (err && err.message) || String(err) };
      }
    }
    return last;
  } catch (error) {
    Logger.log("Erro em gw_login_: " + error.message);
    throw error;
  }
}

/** Resolve o logout pela primeira funcao existente; nunca lanca. */
function gw_logout_() {
  try {
    if (typeof doLogout === 'function') return doLogout();
    if (typeof logout === 'function') return logout();
  } catch (ignored) {}
  return { success: true };
}

/**
 * Verificador de sessao da frota (fail-closed): primeira funcao existente vence.
 * Cobre as variantes nativas (sgteLegacy, AuthService, getCurrentUser*).
 */
function gw_currentUser_() {
  try {
    if (typeof getCurrentSessionUser === 'function')         { var a = getCurrentSessionUser();          if (a) return a; }
    if (typeof getCurrentSessionUser_sgteLegacy === 'function') { var b = getCurrentSessionUser_sgteLegacy(); if (b) return b; }
    if (typeof AuthService !== 'undefined' && AuthService && typeof AuthService.getSessionUser === 'function') { var c = AuthService.getSessionUser(); if (c) return c; }
    if (typeof getCurrentUser_ === 'function')               { var d = getCurrentUser_();                if (d) return d; }
    if (typeof getCurrentUser === 'function')                { var e = getCurrentUser();                 if (e) return e; }
  } catch (ignored) {}
  return null;
}

/** Normaliza o resultado do login para um booleano de sucesso (igual ao harness). */
function gw_isLoginOk_(r) {
  if (r === null || r === undefined || r === false) return false;
  if (typeof r === 'object') {
    if (r.success === false || r.ok === false) return false;
    if (r.success === true || r.ok === true) return true;
    if (r.token || r.sessionToken || r.redirectUrl || r.session) return true;
    if (r.user || r.id || r.username || r.role || r.perfil) return true;
    return false;
  }
  return !!r;
}


/**
 * Normaliza qualquer resultado de login aceito por gw_isLoginOk_ para o
 * envelope { success:true, user?:{}, token?:string } esperado pelo Login.html.
 * Sem esta etapa, funcoes que retornam { ok:true } ou o objeto de usuario
 * diretamente passam a validacao mas chegam ao cliente sem .success=true.
 */
function gw_normalizeLogin_(r) {
  if (!r || typeof r !== 'object') return { success: true };
  if ('success' in r) return r;
  if ('ok' in r) {
    var out = { success: !!r.ok };
    if (r.user)         out.user         = r.user;
    if (r.principal)    out.user         = r.principal;
    if (r.token)        out.token        = r.token;
    if (r.sessionToken) out.sessionToken = r.sessionToken;
    if (r.message)      out.message      = r.message;
    return out;
  }
  if (r.id || r.username || r.role || r.perfil) return { success: true, user: r };
  if (r.token || r.sessionToken) return { success: true, token: r.token || r.sessionToken };
  return { success: true };
}

/** Sanitiza dados para o cliente: remove credenciais, serializa Date, recursivo. */
function toClientSafe_(value) {
  try {
    if (value instanceof Date) return value.toISOString();
    if (Array.isArray(value)) return value.map(toClientSafe_);
    if (value && typeof value === 'object') {
      var safe = {};
      Object.keys(value).forEach(function (key) {
        if (key === 'password' || key === 'passwordHash' || key === 'senha' || key === 'senha_hash') return;
        safe[key] = toClientSafe_(value[key]);
      });
      return safe;
    }
    return value;
  } catch (error) {
    Logger.log("Erro em toClientSafe_: " + error.message);
    throw error;
  }
}
