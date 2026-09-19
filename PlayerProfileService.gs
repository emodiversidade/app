/**
 * @file PlayerProfileService.gs
 * @description Perfil, conquistas derivadas e preferencias da jogadora.
 *
 * English Description:
 * This service aggregates and analyzes individual player profiles, achievements, and preferences.
 * It reads results from the spreadsheet database, computes progress metrics across autonomy dimensions,
 * and maps scores to developmental stages. Additionally, it dynamically evaluates and unlocks specific
 * achievements based on historical game choices and diversity of responses, without persisting achievement
 * state directly in the database.
 */

/**
 * Le os resultados de uma jogadora (sem cabecalho).
 * @param {string} userId ID resolvido no servidor.
 * @returns {Array<Array>} Linhas da aba Resultados pertencentes a jogadora.
 */
function getPlayerResultRows_(userId) {
  try {
    return readRecords_(getGameConfig_().SHEET_NAMES.RESULTS)
      .slice(1)
      .filter(row => String(row[1]) === String(userId));
  } catch (error) {
    Logger.log("Erro em getPlayerResultRows_: " + error.message);
    throw error;
  }
}

/**
 * Perfil de autonomia agregado: totais por pilar, mais forte, a desenvolver,
 * estagio de desenvolvimento e o selo de equilibrio "Emodiversa".
 * @param {string} userId ID resolvido no servidor.
 * @returns {Object} Perfil de computeAutonomyProfile_ acrescido de stage.
 */
function getPlayerAutonomyProfile_(userId) {
  try {
    const progress = ensurePlayerProgress_(userId);
    const profile = computeAutonomyProfile_(getPlayerResultRows_(userId));
    profile.stage = resolveLevelStage_(Number(progress.NivelAtual || 1));
    profile.score = Number(progress.Pontuacao || 0);
    return profile;
  } catch (error) {
    Logger.log("Erro em getPlayerAutonomyProfile_: " + error.message);
    throw error;
  }
}

/**
 * Deriva conquistas do historico, do perfil por dimensao e do estagio.
 * Conquistas premiam diversidade de competencias (nao "pontos"), coerente com a
 * proposta do jogo. Nada e persistido — sempre recalculado do historico.
 * @param {string} userId ID resolvido no servidor.
 * @returns {Array<{name: string, description: string, emoji: string, unlocked: boolean}>}
 */
function getPlayerAchievements_(userId) {
  try {
    const results = getPlayerResultRows_(userId);
    const profile = getPlayerAutonomyProfile_(userId);
    const DIMENSION_MASTERY = 20; // pontos acumulados num pilar para "dominio"

    const catalog = [];
    catalog.push({
      name: "Primeiro Passo",
      emoji: "🌟",
      description: "Concluiu sua primeira situacao de autonomia.",
      unlocked: results.length >= 1
    });
    catalog.push({
      name: "Caminhante",
      emoji: "🚶‍♀️",
      description: "Refletiu sobre cinco situacoes diferentes.",
      unlocked: results.length >= 5
    });
    // Uma conquista por pilar, destravada ao acumular dominio naquele eixo.
    getAutonomyDimensions_().forEach(dimension => {
      const dim = profile.dimensions.filter(d => d.key === dimension.key)[0] || { value: 0 };
      catalog.push({
        name: dimension.label,
        emoji: dimension.emoji,
        description: `Desenvolveu o pilar ${dimension.label.toLowerCase()}: ${dimension.short}`,
        unlocked: dim.value >= DIMENSION_MASTERY
      });
    });
    catalog.push({
      name: "Emodiversa",
      emoji: "🌈",
      description: "Equilibrou todos os cinco pilares da autonomia — uma jornada diversa, nao so forte num ponto.",
      unlocked: profile.balanced
    });
    catalog.push({
      name: profile.stage.name,
      emoji: "🧭",
      description: profile.stage.description,
      unlocked: profile.stage.level >= 2
    });

    return catalog.filter(achievement => achievement.unlocked);
  } catch (error) {
    Logger.log("Erro em getPlayerAchievements_: " + error.message);
    throw error;
  }
}

/**
 * Le as preferencias da pessoa autenticada; valores ausentes ou corrompidos
 * caem no padrao { soundEnabled: true, notificationsEnabled: false }.
 * @returns {{soundEnabled: boolean, notificationsEnabled: boolean}}
 */
function getPlayerSettings_() {
  try {
    try {
      const raw = PropertiesService.getUserProperties().getProperty("playerSettings");
      if (!raw) {
        return { soundEnabled: true, notificationsEnabled: false };
      }
      try {
        return JSON.parse(raw);
      } catch (error) {
        return { soundEnabled: true, notificationsEnabled: false };
      }
    } catch (error) {
      Logger.log("Erro em getPlayerSettings_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em getPlayerSettings_: " + error.message);
    throw error;
  }
}

/**
 * Normaliza e persiste as preferencias em UserProperties da pessoa autenticada.
 * @param {Object} settings Objeto com soundEnabled e notificationsEnabled.
 * @returns {{soundEnabled: boolean, notificationsEnabled: boolean}} Valores salvos.
 * @throws {Error} Configuracoes invalidas.
 */
function savePlayerSettings_(settings) {
  try {
    try {
      if (!settings || typeof settings !== "object") {
        throw new Error("Configuracoes invalidas.");
      }
      const normalized = {
        soundEnabled: Boolean(settings.soundEnabled),
        notificationsEnabled: Boolean(settings.notificationsEnabled)
      };
      PropertiesService.getUserProperties().setProperty("playerSettings", JSON.stringify(normalized));
      return normalized;
    } catch (error) {
      Logger.log("Erro em savePlayerSettings_: " + error.message);
      throw error; // Re-lança para tratamento superior
    }
  } catch (error) {
    Logger.log("Erro em savePlayerSettings_: " + error.message);
    throw error;
  }
}

