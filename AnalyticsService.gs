/**
 * @file AnalyticsService.gs
 * @description Analises agregadas do modelo de autonomia para administradores.
 *
 * English Description:
 * This service provides aggregated analytics and key statistics of the player autonomy model exclusively
 * for administrators. It processes database records to compute summaries of the five core developmental
 * dimensions, progression trends, and participation rates, ensuring that no individual user identifiers
 * are exposed. It enables administrators to inspect the collective growth and developmental stages of
 * players in a completely anonymized format.
 */

/**
 * Agrega resultados e progresso sem retornar identificadores individuais.
 * @returns {Object} Envelope StandardReturn com pilares, estagios e participacao.
 */
function getAdminAnalyticsData() {
  try {
    try {
      requireAdmin_();
      var config = getGameConfig_();
      var users = dataRows_(config.SHEET_NAMES.USERS);
      var progress = dataRows_(config.SHEET_NAMES.GAME_PROGRESS);
      var results = dataRows_(config.SHEET_NAMES.RESULTS);
      var choices = dataRows_(config.SHEET_NAMES.CHOICES);
      var dimensions = getAutonomyDimensions_();
      var totals = {};

      dimensions.forEach(function(dimension) {
        totals[dimension.key] = 0;
      });
      results.forEach(function(row) {
        var vector = parseImpactVector_(row[6]);
        if (!vector) return;
        Object.keys(vector).forEach(function(key) {
          if (totals[key] !== undefined) {
            totals[key] += Number(vector[key] || 0);
          }
        });
      });

      var dimensionSummary = dimensions.map(function(dimension) {
        var total = totals[dimension.key];
        return {
          key: dimension.key,
          label: dimension.label,
          emoji: dimension.emoji,
          total: total,
          averagePerResult: results.length
            ? Math.round((total / results.length) * 10) / 10
            : 0
        };
      });

      var stageCounts = {};
      config.LEVEL_STAGES.forEach(function(stage) {
        stageCounts[stage.key] = {
          key: stage.key,
          name: stage.name,
          minLevel: stage.minLevel,
          count: 0
        };
      });
      progress.forEach(function(row) {
        var stage = resolveLevelStage_(Number(row[2] || 1));
        if (stageCounts[stage.key]) {
          stageCounts[stage.key].count += 1;
        }
      });

      var playerIds = {};
      users.forEach(function(row) {
        if (String(row[3] || "").toLowerCase() !== "admin") {
          playerIds[String(row[0])] = true;
        }
      });
      var activePlayerIds = {};
      choices.forEach(function(row) {
        var userId = String(row[1] || "");
        if (playerIds[userId]) activePlayerIds[userId] = true;
      });
      var playerCount = Object.keys(playerIds).length;
      var activePlayers = Object.keys(activePlayerIds).length;

      return successResponse_({
        dimensions: dimensionSummary,
        stages: config.LEVEL_STAGES.map(function(stage) {
          return stageCounts[stage.key];
        }),
        participation: {
          players: playerCount,
          activePlayers: activePlayers,
          rate: playerCount
            ? Math.round((activePlayers / playerCount) * 100)
            : 0,
          choices: choices.length,
          results: results.length
        }
      });
    } catch (error) {
      return publicError_(error);
    }
  } catch (error) {
    Logger.log("Erro em getAdminAnalyticsData: " + error.message);
    throw error;
  }
}
