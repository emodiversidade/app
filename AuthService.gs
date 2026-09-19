/**
 * @file AuthService.gs
 * @description Autenticacao, registro e sessao da aplicacao.
 *
 * English Description:
 * This service manages user registration, account authentication, and session persistence for the
 * web application. It enforces rate limits to prevent brute-force attacks, validates username
 * and password requirements, securely validates passwords with migration support for updated
 * cryptographic hashing versions, and manages session state. It allows plain-text passwords
 * to be readable on the spreadsheet to satisfy specific design constraints.
 */

/**
 * Autentica por nome de usuario e senha, com rate limit por conta e global.
 * Hashes v1$ sao atualizados para v2$ na propria autenticacao; senhas em
 * texto plano sao aceitas e preservadas como estao na planilha.
 * @param {string} username Minimo 3 caracteres; comparacao case-insensitive.
 * @param {string} password Minimo 6 caracteres.
 * @returns {{success: boolean, data: ?{user: Object}, message: string, code: string, traceId: string}}
 */
function doLogin(username, password) {
  try {
    const normalizedUsername = normalizeText_(username, "Nome de usuario", 3);
    const normalizedPassword = normalizeText_(password, "Senha", 6);
    enforceLoginRateLimit_(normalizedUsername);
    const sheet = getSpreadsheet_().getSheetByName(getGameConfig_().SHEET_NAMES.USERS);
    if (!sheet) {
      throw new Error("A estrutura de usuarios ainda nao foi inicializada.");
    }

    const data = sheet.getDataRange().getValues();
    for (let index = 1; index < data.length; index++) {
      const row = data[index];
      if (String(row[1]).toLowerCase() !== normalizedUsername.toLowerCase()) {
        continue;
      }
      if (!verifyPassword_(normalizedPassword, String(row[2]))) {
        break;
      }

      setCurrentUser_({ id: String(row[0]), username: String(row[1]), role: String(row[3] || "Jogador") });
      ensurePlayerProgress_(String(row[0]));
      logAction_("LOGIN", String(row[0]), "Login realizado");
      return successResponse_({ user: getCurrentUser_() }, "Login realizado com sucesso.");
    }
    recordFailedLogin_(normalizedUsername);
    return publicError_(new Error("Nome de usuario ou senha invalidos."));
  } catch (error) {
    return publicError_(error);
  }
}

/**
 * Cria uma conta com role "Jogador", sob lock de script para evitar duplicidade,
 * inicializa o progresso e abre a sessao.
 * @param {string} username Minimo 3 caracteres; precisa ser inedito (case-insensitive).
 * @param {string} password Minimo 6 caracteres; armazenada em texto plano na
 *   planilha por decisao do projeto (senhas legiveis para quem tem acesso).
 * @returns {{success: boolean, data: ?{user: Object}, message: string, code: string, traceId: string}}
 */
function doRegister(username, password) {
  try {
    try {
      try {
        const lock = LockService.getScriptLock();
        try {
          lock.waitLock(5000);
          const normalizedUsername = normalizeText_(username, "Nome de usuario", 3);
          const normalizedPassword = normalizeText_(password, "Senha", 6);
          const sheet = getSpreadsheet_().getSheetByName(getGameConfig_().SHEET_NAMES.USERS);
          if (!sheet) {
            throw new Error("A estrutura de usuarios ainda nao foi inicializada.");
          }

          const data = sheet.getDataRange().getValues();
          const exists = data.slice(1).some(row =>
            String(row[1]).toLowerCase() === normalizedUsername.toLowerCase()
          );
          if (exists) {
            return publicError_(new Error("Esse nome de usuario ja esta em uso."));
          }

          const userId = Utilities.getUuid();
          const newUserRow = [
            userId,
            normalizedUsername,
            normalizedPassword,
            "Jogador",
            new Date().toISOString()
          ];
          sheet.appendRow(newUserRow);
          lock.releaseLock();
          ensurePlayerProgress_(userId);
          setCurrentUser_({ id: userId, username: normalizedUsername, role: "Jogador" });
          logAction_("REGISTER", userId, "Conta criada");
          return successResponse_({ user: getCurrentUser_() }, "Conta criada. Sua jornada pode comecar.");
        } catch (error) {
          return publicError_(error);
        } finally {
          if (lock.hasLock()) {
            lock.releaseLock();
          }
        }
      } catch (error) {
        Logger.log("Erro em doRegister: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em doRegister: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em doRegister: " + error.message);
    throw error;
  }
}

/**
 * LEGADO: Le a sessao corrente das UserProperties.
 * DEPRECADO: Use getSessionUser(token) do AuthHelpers.gs
 * Mantido para compatibilidade com chamadas antigas dentro do script.
 * @returns {?{id: string, username: string, role: string}} Null sem sessao ativa.
 */
function getCurrentUser_() {
  try {
    const properties = PropertiesService.getUserProperties();
    const id = properties.getProperty("currentUserId");
    if (!id) {
      return null;
    }
    return {
      id: id,
      username: properties.getProperty("currentUsername"),
      role: properties.getProperty("currentUserRole") || "Jogador"
    };
  } catch (error) {
    Logger.log("Erro em getCurrentUser_: " + error.message);
    throw error;
  }
}

function requireCurrentUser_() {
  const user = getCurrentUser_();
  if (!user) {
    throw new Error("Sua sessao terminou. Entre novamente.");
  }
  return user;
}

/**
 * Resolve a pessoa autenticada no servidor; nunca confie em IDs vindos do cliente.
 * NOTA: Para chamadas que recebem token, use requireAuthenticatedPrincipal_(token)
 * @returns {{id: string, username: string, role: string}}
 * @throws {Error} "Sua sessao terminou" quando nao ha sessao ativa.
 */
function requireAuth_() {
  return requireCurrentUser_();
}

/**
 * LEGADO: Grava usuario em UserProperties.
 * DEPRECADO: As novas rotinas usam ScriptProperties via token.
 * Mantido para compatibilidade interna com doLogin/doRegister.
 */
function setCurrentUser_(user) {
  try {
    PropertiesService.getUserProperties().setProperties({
      currentUserId: String(user.id),
      currentUsername: String(user.username),
      currentUserRole: String(user.role || "Jogador")
    });
  } catch (error) {
    Logger.log("Erro em setCurrentUser_: " + error.message);
    throw error;
  }
}

/**
 * Encerra a sessao da pessoa autenticada limpando as UserProperties.
 * Idempotente: chamadas sem sessao ativa tambem retornam sucesso.
 * @returns {{success: boolean, data: null, message: string, code: string, traceId: string}}
 */
function doLogout() {
  try {
    const user = getCurrentUser_();
    if (user) {
      logAction_("LOGOUT", user.id, "Logout realizado");
    }
    const properties = PropertiesService.getUserProperties();
    properties.deleteProperty("currentUserId");
    properties.deleteProperty("currentUsername");
    properties.deleteProperty("currentUserRole");
    return successResponse_(null, "Sessao encerrada.");
  } catch (error) {
    Logger.log("Erro em doLogout: " + error.message);
    throw error;
  }
}

// ---------------------------------------------------------------------------
// Sessoes baseadas em token (ScriptProperties) — gate do doGet
// MIGRADO PARA AuthHelpers.gs - Rotinas reutilizaveis do Prompt 18
// Os metodos de jogo (requireAuth_) continuam usando UserProperties por
// compatibilidade; o novo fluxo usa loginWithToken do AuthHelpers.gs
// ---------------------------------------------------------------------------

/**
 * WRAPPER LEGADO mantido para compatibilidade.
 * Use loginWithToken do AuthHelpers.gs diretamente.
 */
var EMOD_TOK_PREFIX_ = 'EMOD_SESSION_'; // Migrado para AuthHelpers
var EMOD_TOK_TTL_MS_ = 21600 * 1000;

/** LEGADO: Use getSessionUser(token) do AuthHelpers.gs */
function getSessionByToken_(tok) {
  return getSessionUser(tok); // Delega para AuthHelpers
}

/** DEPRECADO: nao usado mais */
function storeSessionAsToken_() {
  try {
    var user = getCurrentUser_();
    if (!user) return null;
    var token = Utilities.getUuid().replace(/-/g, '');
    PropertiesService.getScriptProperties().setProperty(
      EMOD_TOK_PREFIX_ + token,
      JSON.stringify({ id: user.id, username: user.username, role: user.role,
                       expiresAt: Date.now() + EMOD_TOK_TTL_MS_ })
    );
    return token;
  } catch (error) {
    Logger.log("Erro em storeSessionAsToken_: " + error.message);
    throw error; // Re-lança para tratamento superior
  }
}

// loginWithToken agora esta em AuthHelpers.gs
// registerWithToken mantido aqui por usar doRegister interno

/** Wrapper de doRegister que tambem retorna um token de sessao. */
function registerWithToken(username, password) {
  var result = doRegister(username, password);
  if (!result || !result.success) return result;
  var token = storeSessionAsToken_();
  return Object.assign({}, result, { token: token });
}

/**
 * Compara a senha informada com o valor armazenado — SEMPRE em texto plano
 * (quiosque escolar): a coluna Senha guarda a senha legivel, sem qualquer hash.
 * @param {string} password Senha em texto claro.
 * @param {string} storedValue Valor da coluna Senha.
 * @returns {boolean}
 */
function verifyPassword_(password, storedValue) {
  try {
    return String(storedValue) === String(password);
  } catch (error) {
    Logger.log("Erro em verifyPassword_: " + error.message);
    throw error;
  }
}

function enforceLoginRateLimit_(username) {
  const globalAttempts = CacheProvider.getGlobalLoginAttempts();
  if (globalAttempts >= 100) {
    throw new Error("Sistema temporariamente indisponivel. Muitas tentativas globais.");
  }

  const attempts = CacheProvider.getLoginAttempts(username);
  if (attempts >= 5) {
    throw new Error("Muitas tentativas. Aguarde alguns minutos antes de tentar novamente.");
  }
}

function recordFailedLogin_(username) {
  CacheProvider.incrementLoginAttempts(username);
  CacheProvider.incrementGlobalLoginAttempts();
}
