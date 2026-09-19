/**
 * @file AppService.gs
 * @description Roteamento web e fachada segura consumida pelo frontend.
 *
 * English Description:
 * This service serves as the central web routing entrypoint and security facade for the Google Apps Script
 * web application. It intercepts HTTP GET requests via the doGet function, determines the current user
 * session status, validates user authorization for the requested screen, and safely outputs the appropriate
 * HTML interface template while embedding configuration variables, styling, and client-side logic dynamically.
 */

const APP_SCREENS = Object.freeze({
  login: "Login",
  LoginScreen: "Login",
  dashboard: "DashboardScreen",
  situation: "SituationScreen",
  result: "ResultScreen",
  achievements: "AchievementsScreen",
  settings: "SettingsScreen",
  admin: "AdminDashboard"
});

/**
 * Entrypoint web. Renderiza a tela pedida em `?screen=` se ela for conhecida
 * e permitida para o estado da sessao; senao cai em login ou dashboard.
 * @param {GoogleAppsScript.Events.DoGet} event Evento com `parameter.screen`.
 * @returns {GoogleAppsScript.HTML.HtmlOutput} Tela renderizada.
 */
function doGet(event) {
  try {
    const params    = (event && event.parameter) || {};
    // FLEET_FRAGMENT_BOOTSTRAP: #tok= não chega ao Apps Script. Sirva o
    // dashboard como shell; o cliente lê o fragmento e autentica as chamadas.
    if (String(params.page || '') === 'app' && !params.tok) {
      const bootstrap = HtmlService.createTemplateFromFile(APP_SCREENS.dashboard);
      bootstrap.appUrl = ScriptApp.getService().getUrl();
      bootstrap.tok = '';
      return bootstrap.evaluate()
        .setTitle(getGameConfig_().PROJECT_NAME)
        .addMetaTag('viewport', 'width=device-width, initial-scale=1');
    }
    const requested = String(params.screen || "");
    // Token em ScriptProperties evita o bug de UserProperties compartilhado em
    // deployments "Execute as: Me" (sessao do dono vazava para todos os visitantes).
    const tok = String(params.tok || "");
    // Usa isAuthenticatedByToken do AuthHelpers.gs (Prompt 18)
    const authenticated = isAuthenticatedByToken(tok);
    const currentUser = authenticated ? getSessionUser(tok) : null;
  
    let screen = APP_SCREENS[requested] ? requested : (authenticated ? "dashboard" : "login");
    if (!authenticated && screen !== "login") {
      screen = "login";
    }
    if (authenticated && screen === "login") {
      screen = "dashboard";
    }
    if (
      screen === "admin" &&
      String(currentUser && currentUser.role || "").toLowerCase() !== "admin"
    ) {
      screen = "dashboard";
    }
    const template = HtmlService.createTemplateFromFile(APP_SCREENS[screen]);
    template.appUrl = ScriptApp.getService().getUrl();
    template.tok = tok;
    return template.evaluate()
      .setTitle(getGameConfig_().PROJECT_NAME)
      .addMetaTag("viewport", "width=device-width, initial-scale=1");
  } catch (error) {
    Logger.log("Erro em doGet: " + error.message);
    throw error;
  }
}

/**
 * Avalia um parcial HTML (estilos ou client compartilhado) com `appUrl` no escopo.
 * @param {string} filename Nome do arquivo HTML do projeto, sem extensao.
 * @returns {string} Conteudo HTML avaliado.
 */
function include_(filename) {
  try {
    const template = HtmlService.createTemplateFromFile(filename);
    template.appUrl = ScriptApp.getService().getUrl();
    return template.evaluate().getContent();
  } catch (error) {
    Logger.log("Erro em include_: " + error.message);
    throw error;
  }
}

/**
 * Produz a URL de navegacao de uma rota conhecida; rotas invalidas viram dashboard.
 * @param {string} screen Chave de APP_SCREENS (login, dashboard, situation...).
 * @returns {string} URL absoluta do web app com `?screen=`.
 */
function getNavigationUrl(screen) {
  try {
    const normalized = APP_SCREENS[screen] ? screen : "dashboard";
    return `${ScriptApp.getService().getUrl()}?screen=${encodeURIComponent(normalized)}`;
  } catch (error) {
    Logger.log("Erro em getNavigationUrl: " + error.message);
    throw error;
  }
}

/**
 * Dados do painel da pessoa autenticada: usuario, progresso e total de conquistas.
 * @returns {{success: boolean, data: ?Object, message: string, code: string, traceId: string}}
 */
function getDashboardData() {
  try {
    const user = requireAuth_();
    const progress = ensurePlayerProgress_(user.id);
    const autonomyProfile = getPlayerAutonomyProfile_(user.id);
    return successResponse_({
      user: user,
      progress: progress,
      achievementsCount: getPlayerAchievements_(user.id).length,
      stage: autonomyProfile.stage,
      autonomyProfile: autonomyProfile
    });
  } catch (error) {
    return publicError_(error);
  }
}

/**
 * Perfil de autonomia da pessoa autenticada (totais por pilar, estagio e selo).
 * @returns {{success: boolean, data: ?Object, message: string, code: string, traceId: string}}
 */
function getAutonomyProfileData() {
  try {
    const user = requireAuth_();
    return successResponse_(getPlayerAutonomyProfile_(user.id));
  } catch (error) {
    return publicError_(error);
  }
}

/**
 * Proxima situacao disponivel para a pessoa autenticada; `data` e null ao fim do jogo.
 * @returns {{success: boolean, data: ?Object, message: string, code: string, traceId: string}}
 */
function getSituationData() {
  try {
    const user = requireAuth_();
    return successResponse_(getNextSituation_(user.id));
  } catch (error) {
    return publicError_(error);
  }
}

/**
 * Registra a escolha (1 ou 2) da pessoa autenticada para uma situacao ainda nao respondida.
 * Aceita indice numerico (1 ou 2) OU texto canonico da opcao (opcao1/opcao2 da situacao).
 * @param {string} situationId ID da situacao na aba Situacoes.
 * @param {number|string} choiceIndex 1, 2 ou texto da opcao; outros valores retornam erro.
 * @returns {{success: boolean, data: ?Object, message: string, code: string, traceId: string}}
 *   `data` traz id do resultado, description, impact, score e level atualizados.
 */
function submitChoice(situationId, choiceIndex) {
  try {
    const user = requireAuth_();
    const normalizedSituationId = normalizeText_(situationId, "Situacao", 1);
    
    // Resolve indice numerico OU texto canonico
    const resolvedIndex = resolveChoiceIndex_(normalizedSituationId, choiceIndex);
    
    return successResponse_(makeChoice_(user.id, normalizedSituationId, resolvedIndex, choiceIndex));
  } catch (error) {
    return publicError_(error);
  }
}

/**
 * Resultado mais recente da pessoa autenticada; `data` e null se nada foi respondido.
 * @returns {{success: boolean, data: ?Object, message: string, code: string, traceId: string}}
 */
function getLastResultData() {
  try {
    const user = requireAuth_();
    const rows = readRecords_(getGameConfig_().SHEET_NAMES.RESULTS);
    for (let i = rows.length - 1; i >= 1; i--) {
      const row = rows[i];
      if (String(row[1]) === String(user.id)) {
        const impact = Number(row[4] || 0);
        const dimensions = parseImpactVector_(row[6]) || {};
        return successResponse_({
          id: String(row[0]),
          description: String(row[3]),
          impact: impact,
          createdAt: row[5],
          dimensions: dimensions,
          reflection: summarizeReflection_(dimensions, impact),
          stage: resolveLevelStage_(Number(ensurePlayerProgress_(user.id).NivelAtual || 1))
        });
      }
    }
    return successResponse_(null);
  } catch (error) {
    return publicError_(error);
  }
}

/**
 * Conquistas derivadas do historico e da pontuacao da pessoa autenticada.
 * @returns {{success: boolean, data: ?Array<Object>, message: string, code: string, traceId: string}}
 */
function getAchievementsData() {
  try {
    const user = requireAuth_();
    return successResponse_(getPlayerAchievements_(user.id));
  } catch (error) {
    return publicError_(error);
  }
}

/**
 * Preferencias persistidas em UserProperties da pessoa autenticada.
 * @returns {{success: boolean, data: ?Object, message: string, code: string, traceId: string}}
 */
function getSettingsData() {
  try {
    requireAuth_();
    return successResponse_(getPlayerSettings_());
  } catch (error) {
    return publicError_(error);
  }
}

/**
 * Persiste preferencias da pessoa autenticada. Somente chaves da whitelist
 * (soundEnabled, notificationsEnabled) sao aceitas; valores viram boolean.
 * @param {Object} settings Objeto plano vindo do cliente.
 * @returns {{success: boolean, data: ?Object, message: string, code: string, traceId: string}}
 */
function saveSettingsData(settings) {
  try {
    const user = requireAuth_();
    if (!settings || typeof settings !== 'object' || Array.isArray(settings)) {
        throw new Error("Configuracoes invalidas. Esperado um objeto.");
    }
    const whitelistedSettings = {
        soundEnabled: Boolean(settings.soundEnabled),
        notificationsEnabled: Boolean(settings.notificationsEnabled)
    };
    const saved = savePlayerSettings_(whitelistedSettings);
    logAction_("SETTINGS", user.id, JSON.stringify(saved));
    return successResponse_(saved, "Configuracoes salvas.");
  } catch (error) {
    return publicError_(error);
  }
}

/**
 * Verifica versao e alcance da planilha sem expor dados de pessoas.
 * @returns {{success: boolean, data: ?Object, message: string, code: string, traceId: string}}
 */
function healthCheck() {
  try {
    const spreadsheet = getSpreadsheet_();
    return successResponse_({
      status: "healthy",
      version: getGameConfig_().VERSION,
      spreadsheetReachable: Boolean(spreadsheet.getId())
    });
  } catch (error) {
    return publicError_(error);
  }
}

/**
 * Converte escolha (indice numerico ou texto canonico) para indice 1 ou 2.
 * @param {string} situationId ID da situacao na aba Situacoes.
 * @param {number|string} choice 1, 2 ou texto da opcao.
 * @returns {number} 1 ou 2.
 * @throws {Error} Situacao nao encontrada ou escolha invalida.
 * @private
 */
function resolveChoiceIndex_(situationId, choice) {
  // Se ja e numero valido, retorna direto
  const asNumber = Number(choice);
  if (asNumber === 1 || asNumber === 2) {
    return asNumber;
  }
  
  // Se e texto, busca a situacao e compara com opcao1/opcao2
  const situations = readRecords_(getGameConfig_().SHEET_NAMES.SITUATIONS);
  const situation = situations.slice(1).find(row => String(row[0]) === String(situationId));
  
  if (!situation) {
    throw new Error("Situacao nao encontrada.");
  }
  
  const choiceText = String(choice).trim();
  const opcao1 = String(situation[3] || "").trim();
  const opcao2 = String(situation[4] || "").trim();
  
  if (choiceText === opcao1) {
    return 1;
  }
  if (choiceText === opcao2) {
    return 2;
  }
  
  throw new Error("A escolha deve ser 1, 2 ou o texto exato de uma das opcoes.");
}
