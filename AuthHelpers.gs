/**
 * AuthHelpers.gs
 * Rotinas Reutilizáveis de Autenticação - Prompt 18
 * 
 * Implementa comportamentos padrão sem descaracterizar o fluxo do Emodiversa.
 * Remove dependência de getUserProperties/getUserCache para sessões.
 */

// ============================================================================
// Configuração
// ============================================================================

const EMOD_AUTH_CONFIG_ = {
  SESSION_KEY_PREFIX: 'EMOD_SESSION_',
  SESSION_TTL_SECONDS: 21600, // 6 horas
  TOKEN_LENGTH: 32
};

// ============================================================================
// Rotina 1: ensureUsuariosSheet_
// ============================================================================

/**
 * Garante que a aba Usuarios existe com os cabecalhos corretos.
 * Idempotente: nao sobrescreve se ja existir.
 * @returns {GoogleAppsScript.Spreadsheet.Sheet}
 */
function ensureUsuariosSheet_() {
  try {
    const ss = getSpreadsheet_();
    const config = getGameConfig_();
    let sheet = ss.getSheetByName(config.SHEET_NAMES.USERS);
  
    if (!sheet) {
      sheet = ss.insertSheet(config.SHEET_NAMES.USERS);
      sheet.getRange(1, 1, 1, 5).setValues([
        ['ID', 'Username', 'Password', 'Role', 'CreatedAt']
      ]);
      sheet.getRange(1, 1, 1, 5).setFontWeight('bold');
    }
  
    return sheet;
  } catch (error) {
    Logger.log("Erro em ensureUsuariosSheet_: " + error.message);
    throw error; // Re-lança para tratamento superior
  }
}

// ============================================================================
// Rotina 2: seedSyntheticAdminUsers_
// ============================================================================

/**
 * Cria admins sinteticos admin01 a admin15, senha admin123.
 * Idempotente: nao duplica usuarios existentes por ID ou username.
 */
function seedSyntheticAdminUsers_() {
  try {
    try {
      try {
        const sheet = ensureUsuariosSheet_();
        const data = sheet.getDataRange().getValues();
        const existingUsernames = data.slice(1).map(row => 
          String(row[1]).toLowerCase()
        );
  
        const toAdd = [];
        for (let i = 1; i <= 15; i++) {
          const username = `admin${String(i).padStart(2, '0')}`;
          if (!existingUsernames.includes(username)) {
            toAdd.push([
              Utilities.getUuid(),
              username,
              'admin123',
              'Admin',
              new Date().toISOString()
            ]);
          }
        }
  
        if (toAdd.length > 0) {
          sheet.getRange(sheet.getLastRow() + 1, 1, toAdd.length, 5).setValues(toAdd);
        }
  
        return { added: toAdd.length, total: 15 };
      } catch (error) {
        Logger.log("Erro em seedSyntheticAdminUsers_: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em seedSyntheticAdminUsers_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em seedSyntheticAdminUsers_: " + error.message);
    throw error;
  }
}

// ============================================================================
// Rotina 3: readUsuariosRows_
// ============================================================================

/**
 * Le cabecalhos e linhas da aba real de usuarios.
 * @returns {{headers: string[], rows: any[][]}}
 */
function readUsuariosRows_() {
  try {
    try {
      const sheet = ensureUsuariosSheet_();
      const data = sheet.getDataRange().getValues();
  
      return {
        headers: data[0],
        rows: data.slice(1)
      };
    } catch (error) {
      Logger.log("Erro em readUsuariosRows_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em readUsuariosRows_: " + error.message);
    throw error;
  }
}

// ============================================================================
// Rotina 4: findPlaintextUser_
// ============================================================================

/**
 * Encontra usuario por username, normalizado case-insensitive.
 * Retorna usuario publico mais password para comparacao interna.
 * @param {string} username
 * @returns {?{id: string, username: string, password: string, role: string, active: boolean}}
 */
function findPlaintextUser_(username) {
  try {
    const normalized = String(username).toLowerCase().trim();
    const { rows } = readUsuariosRows_();
  
    for (const row of rows) {
      if (String(row[1]).toLowerCase().trim() === normalized) {
        return {
          id: String(row[0]),
          username: String(row[1]),
          password: String(row[2]), // Texto plano por decisao do projeto
          role: String(row[3] || 'Jogador'),
          active: true // Emodiversa nao tem coluna Status
        };
      }
    }
  
    return null;
  } catch (error) {
    Logger.log("Erro em findPlaintextUser_: " + error.message);
    throw error;
  }
}

// ============================================================================
// Rotina 5: loginWithToken
// ============================================================================

/**
 * Valida senha texto plano, cria token opaco e retorna envelope padr ao.
 * Mantem compatibilidade com doLogin do projeto.
 * @param {string} username
 * @param {string} password
 * @returns {{success: boolean, token?: string, user?: Object, message: string}}
 */
function loginWithToken(username, password) {
  try {
    try {
      try {
        try {
          // Validacoes minimas
          if (!username || username.length < 3) {
            return { success: false, message: 'Nome de usuario deve ter pelo menos 3 caracteres.' };
          }
          if (!password || password.length < 6) {
            return { success: false, message: 'Senha deve ter pelo menos 6 caracteres.' };
          }
    
          // Buscar usuario
          const user = findPlaintextUser_(username);
          if (!user) {
            return { success: false, message: 'Nome de usuario ou senha invalidos.' };
          }
    
          // Validar senha
          if (user.password !== password) {
            return { success: false, message: 'Nome de usuario ou senha invalidos.' };
          }
    
          // Criar token
          const token = Utilities.getUuid().replace(/-/g, '');
    
          // Gravar sessao em ScriptProperties
          const session = {
            userId: user.id,
            username: user.username,
            role: user.role,
            issuedAt: Date.now(),
            expiresAt: Date.now() + (EMOD_AUTH_CONFIG_.SESSION_TTL_SECONDS * 1000)
          };
    
          PropertiesService.getScriptProperties().setProperty(
            EMOD_AUTH_CONFIG_.SESSION_KEY_PREFIX + token,
            JSON.stringify(session)
          );
    
          // Retornar envelope
          return {
            success: true,
            token: token,
            user: { id: user.id, username: user.username, role: user.role },
            redirectUrl: ScriptApp.getService().getUrl() + '?page=app#tok=' + encodeURIComponent(token),
            message: 'Login realizado com sucesso.'
          };
    
        } catch (error) {
          return { success: false, message: String(error.message || error) };
        }
      } catch (error) {
        Logger.log("Erro em loginWithToken: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em loginWithToken: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em loginWithToken: " + error.message);
    throw error;
  }
}

// ============================================================================
// Rotina 6: isAuthenticatedByToken
// ============================================================================

/**
 * Valida token em ScriptProperties e remove sessoes vencidas.
 * @param {string} tok
 * @returns {boolean}
 */
function isAuthenticatedByToken(tok) {
  try {
    try {
      if (!tok) return false;
  
      const key = EMOD_AUTH_CONFIG_.SESSION_KEY_PREFIX + tok;
      const raw = PropertiesService.getScriptProperties().getProperty(key);
  
      if (!raw) return false;
  
      try {
        const session = JSON.parse(raw);
    
        // Verificar expiracao
        if (!session || Array.isArray(session) ||
            typeof session.expiresAt !== 'number' || !isFinite(session.expiresAt) ||
            session.expiresAt <= Date.now() ||
            typeof (session.userId || session.id) !== 'string' || !(session.userId || session.id).trim()) {
          PropertiesService.getScriptProperties().deleteProperty(key);
          return false;
        }
    
        return true;
      } catch (e) {
        PropertiesService.getScriptProperties().deleteProperty(key);
        return false;
      }
    } catch (error) {
      Logger.log("Erro em isAuthenticatedByToken: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em isAuthenticatedByToken: " + error.message);
    throw error;
  }
}

// ============================================================================
// Rotina 7: getSessionUser
// ============================================================================

/**
 * Retorna principal publico da sessao (sem senha).
 * @param {string} tok
 * @returns {?{userId: string, username: string, role: string}}
 */
function getSessionUser(tok) {
  try {
    try {
      if (!tok) return null;
  
      const key = EMOD_AUTH_CONFIG_.SESSION_KEY_PREFIX + tok;
      const raw = PropertiesService.getScriptProperties().getProperty(key);
  
      if (!raw) return null;
  
      try {
        const session = JSON.parse(raw);
    
        // Verificar expiracao
        if (!session || Array.isArray(session) ||
            typeof session.expiresAt !== 'number' || !isFinite(session.expiresAt) ||
            session.expiresAt <= Date.now() ||
            typeof (session.userId || session.id) !== 'string' || !(session.userId || session.id).trim()) {
          PropertiesService.getScriptProperties().deleteProperty(key);
          return null;
        }
    
        return {
          userId: session.userId || session.id,
          username: session.username,
          role: session.role
        };
      } catch (e) {
        PropertiesService.getScriptProperties().deleteProperty(key);
        return null;
      }
    } catch (error) {
      Logger.log("Erro em getSessionUser: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em getSessionUser: " + error.message);
    throw error;
  }
}

// ============================================================================
// Rotina 8: logoutWithToken
// ============================================================================

/**
 * Remove sessao por token.
 * Idempotente: retorna sucesso mesmo se token nao existir.
 * @param {string} tok
 * @returns {{success: boolean, message: string}}
 */
function logoutWithToken(tok) {
  try {
    try {
      if (tok) {
        const key = EMOD_AUTH_CONFIG_.SESSION_KEY_PREFIX + tok;
        PropertiesService.getScriptProperties().deleteProperty(key);
      }
  
      return { success: true, message: 'Sessao encerrada.' };
    } catch (error) {
      Logger.log("Erro em logoutWithToken: " + error.message);
      throw error; // Re-lança para tratamento superior
    }
  } catch (error) {
    Logger.log("Erro em logoutWithToken: " + error.message);
    throw error;
  }
}

// ============================================================================
// Rotina 9: buildAuthenticatedRedirectUrl_
// ============================================================================

/**
 * Monta URL absoluta com token na query string.
 * @param {string} tok
 * @returns {string}
 */
function buildAuthenticatedRedirectUrl_(tok) {
  return ScriptApp.getService().getUrl() + '?page=app#tok=' + encodeURIComponent(tok);
}

// ============================================================================
// Rotina 10: resolveAuthTokenFromPayload_
// ============================================================================

/**
 * Aceita string, {_authToken} ou {tok}.
 * @param {string|Object} payloadOrToken
 * @returns {?string}
 */
function resolveAuthTokenFromPayload_(payloadOrToken) {
  if (typeof payloadOrToken === 'string') {
    return payloadOrToken || null;
  }
  
  if (typeof payloadOrToken === 'object' && payloadOrToken !== null) {
    return payloadOrToken._authToken || payloadOrToken.tok || null;
  }
  
  return null;
}

// ============================================================================
// Rotina 11: requireAuthenticatedPrincipal_
// ============================================================================

/**
 * Resolve sessao e falha com erro normalizado se ausente.
 * @param {string|Object} payloadOrToken
 * @returns {{userId: string, username: string, role: string}}
 * @throws {Error} Quando sessao nao existe ou expirou
 */
function requireAuthenticatedPrincipal_(payloadOrToken) {
  const token = resolveAuthTokenFromPayload_(payloadOrToken);
  
  if (!token) {
    throw new Error('Sua sessao terminou. Entre novamente.');
  }
  
  const session = getSessionUser(token);
  
  if (!session) {
    throw new Error('Sua sessao terminou. Entre novamente.');
  }
  
  return session;
}
