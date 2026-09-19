/**
 * @file PlayerProgressService.gs
 * @description Consulta e atualizacao do progresso individual.
 *
 * English Description:
 * This service handles querying, initialization, and updates of individual player progress records.
 * It reads current level and score information from the spreadsheet database, ensures progress rows
 * exist for newly registered users, and provides atomic methods to increment score points and calculate
 * level upgrades. This component maintains the persistent progression state necessary to guide players
 * through the autonomy game.
 */

/**
 * Busca o progresso de uma pessoa como objeto cabecalho->valor.
 * @param {string} userId ID resolvido no servidor.
 * @returns {?Object} Progresso (ID, ID_Usuario, NivelAtual, Pontuacao...) ou null.
 */
function getPlayerProgress_(userId) {
  try {
    const normalizedUserId = normalizeText_(userId, "Usuario", 1);
    const rows = readRecords_(getGameConfig_().SHEET_NAMES.GAME_PROGRESS);
    const headers = rows[0] || [];
    for (let index = 1; index < rows.length; index++) {
      if (String(rows[index][1]) === normalizedUserId) {
        const progress = {};
        headers.forEach((header, column) => progress[header] = rows[index][column]);
        return progress;
      }
    }
    return null;
  } catch (error) {
    Logger.log("Erro em getPlayerProgress_: " + error.message);
    throw error;
  }
}

/**
 * Garante que exista uma linha de progresso, criando-a com nivel e pontuacao iniciais.
 * @param {string} userId ID resolvido no servidor.
 * @returns {Object} Progresso existente ou recem-criado.
 */
function ensurePlayerProgress_(userId) {
  try {
    const existing = getPlayerProgress_(userId);
    if (existing) {
      return existing;
    }
    const config = getGameConfig_();
    createRecord_(config.SHEET_NAMES.GAME_PROGRESS, [
      Utilities.getUuid(),
      userId,
      config.INITIAL_LEVEL,
      config.STARTING_SCORE,
      new Date().toISOString()
    ]);
    return getPlayerProgress_(userId);
  } catch (error) {
    Logger.log("Erro em ensurePlayerProgress_: " + error.message);
    throw error;
  }
}

/**
 * Soma a variacao a pontuacao e recalcula o nivel (1 nivel a cada 100 pontos).
 * @param {string} userId ID resolvido no servidor.
 * @param {number} scoreChange Variacao positiva ou negativa.
 * @returns {Object} Progresso atualizado.
 * @throws {Error} Variacao nao numerica.
 */
function updatePlayerScore_(userId, scoreChange) {
  try {
    const change = Number(scoreChange);
    if (!Number.isFinite(change)) {
      throw new Error("A variacao de pontuacao deve ser numerica.");
    }
    const progress = ensurePlayerProgress_(userId);
    const newScore = Number(progress.Pontuacao || 0) + change;
    const pointsPerLevel = Number(getGameConfig_().POINTS_PER_LEVEL) || 100;
    const newLevel = Math.max(1, Math.floor(newScore / pointsPerLevel) + 1);
    updateRecord_(getGameConfig_().SHEET_NAMES.GAME_PROGRESS, progress.ID, {
      Pontuacao: newScore,
      NivelAtual: newLevel,
      UltimaAtualizacao: new Date().toISOString()
    });
    return getPlayerProgress_(userId);
  } catch (error) {
    Logger.log("Erro em updatePlayerScore_: " + error.message);
    throw error;
  }
}

