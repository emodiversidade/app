/**
 * @file ConfigService.gs
 * @description Centraliza configuracoes, acesso a planilha e contratos comuns.
 *
 * English Description:
 * This service manages global configurations, provides helper methods to access the spreadsheet instance,
 * and defines common interface contracts. It exposes immutable configurations like project settings,
 * sheet names, starting points, developmental stages, and the five pillars of the autonomy model. It acts
 * as the primary adapter for retrieving spreadsheet data, applying caching, and mapping spreadsheet cells
 * to typed records.
 */

const GAME_CONFIG = Object.freeze({
  PROJECT_NAME: "Desprincesamento: A Jornada da Autonomia",
  VERSION: "1.2.0",
  TARGET_AUDIENCE: "Meninas de 11-12 anos",
  SHEET_NAMES: Object.freeze({
    USERS: "Usuarios",
    GAME_PROGRESS: "ProgressoJogo",
    SITUATIONS: "Situacoes",
    CHOICES: "Escolhas",
    RESULTS: "Resultados",
    AUDIT_LOG: "Auditoria"
  }),
  INITIAL_LEVEL: 1,
  STARTING_SCORE: 0,
  // Pontos por nivel: usado para derivar nivel a partir da pontuacao total.
  POINTS_PER_LEVEL: 100,
  // Os cinco pilares da autonomia ("emodiversidade"): cada escolha movimenta
  // a jogadora por estes eixos, em vez de um unico placar de acerto/erro.
  // A ordem e estavel — telas e relatorios iteram nesta sequencia.
  DIMENSIONS: Object.freeze([
    Object.freeze({ key: "autoconhecimento", label: "Autoconhecimento", emoji: "🪞",
      short: "Reconhecer e nomear o que sinto." }),
    Object.freeze({ key: "voz", label: "Voz", emoji: "🗣️",
      short: "Dizer o que penso e o que preciso." }),
    Object.freeze({ key: "coragem", label: "Coragem", emoji: "🌱",
      short: "Tentar mesmo sentindo medo." }),
    Object.freeze({ key: "empatia", label: "Empatia", emoji: "🤝",
      short: "Cuidar da outra sem me apagar." }),
    Object.freeze({ key: "criterio", label: "Pensamento Critico", emoji: "🧭",
      short: "Questionar regras e estereotipos." })
  ]),
  // Estagios de desenvolvimento: dao sentido ao numero do nivel. resolveLevelStage_()
  // escolhe o ultimo estagio cujo minLevel <= nivel atual.
  LEVEL_STAGES: Object.freeze([
    Object.freeze({ key: "despertar", minLevel: 1, name: "Despertar",
      description: "Voce esta comecando a perceber que cada escolha e sua." }),
    Object.freeze({ key: "questionando", minLevel: 2, name: "Questionando",
      description: "Voce comeca a perguntar: por que tem que ser assim?" }),
    Object.freeze({ key: "voz", minLevel: 3, name: "Encontrando a Voz",
      description: "Voce ja consegue dizer o que pensa, mesmo quando e dificil." }),
    Object.freeze({ key: "decidindo", minLevel: 4, name: "Decidindo por Si",
      description: "Suas decisoes nascem do que voce sente e pensa, nao do que esperam." }),
    Object.freeze({ key: "inspirando", minLevel: 5, name: "Inspirando Outras",
      description: "Sua autonomia abre caminho para outras meninas tambem." })
  ])
});

function getGameConfig_() {
  return GAME_CONFIG;
}

/**
 * Catalogo ordenado dos cinco pilares da autonomia.
 * @returns {Array<{key: string, label: string, emoji: string, short: string}>}
 */
function getAutonomyDimensions_() {
  return getGameConfig_().DIMENSIONS;
}

/**
 * Conjunto de chaves de dimensao validas, para validar vetores vindos da planilha.
 * @returns {Object<string, boolean>}
 */
function getDimensionKeySet_() {
  try {
    const set = {};
    getAutonomyDimensions_().forEach(dimension => { set[dimension.key] = true; });
    return set;
  } catch (error) {
    Logger.log("Erro em getDimensionKeySet_: " + error.message);
    throw error;
  }
}

function getSpreadsheet_() {
  try {
    const spreadsheetId = CoreConfig.getSpreadsheetId();
    return SpreadsheetApp.openById(spreadsheetId);
  } catch (error) {
    Logger.log("Erro em getSpreadsheet_: " + error.message);
    throw error;
  }
}

/**
 * Converte para string, apara espacos e valida o tamanho minimo.
 * @param {*} value Valor vindo da fronteira publica.
 * @param {string} fieldName Nome usado na mensagem de erro.
 * @param {number} [minimumLength=1] Tamanho minimo apos trim.
 * @returns {string} Valor normalizado.
 * @throws {Error} Quando o valor e curto demais.
 */
function normalizeText_(value, fieldName, minimumLength) {
  const normalized = String(value || "").trim();
  if (normalized.length < (minimumLength || 1)) {
    throw new Error(`${fieldName} deve ter ao menos ${minimumLength || 1} caractere(s).`);
  }
  return normalized;
}

function generateTraceId_() {
  try {
    return Utilities.getUuid().split('-')[0];
  } catch (error) {
    Logger.log("Erro em generateTraceId_: " + error.message);
    throw error;
  }
}

/**
 * Converte qualquer erro no envelope padrao da API.
 * @param {Error} error Erro capturado na fronteira publica.
 * @returns {Object} Envelope StandardReturn de erro.
 */
function publicError_(error) {
  const errorMsg = error && error.message ? error.message : "Nao foi possivel concluir a operacao.";
  return ErrorHandler.handleError(error, "Fronteira Pública", errorMsg);
}

/**
 * Embala dados de sucesso no envelope padrao da API.
 * @param {*} data Carga util; undefined vira null.
 * @param {string} [message] Mensagem opcional para a interface.
 * @returns {Object} Envelope StandardReturn de sucesso.
 */
function successResponse_(data, message) {
  return StandardReturn.ok(data, { message: message || "", code: "OK", traceId: generateTraceId_() });
}

