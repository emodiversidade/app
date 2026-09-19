/**
 * @file AdminService.gs
 * @description Operacoes administrativas protegidas por perfil Admin.
 *
 * English Description:
 * This service provides essential administrative operations protected by the Admin user profile.
 * It validates session credentials to ensure that only authorized administrators can access
 * system configurations, retrieve aggregated dashboard statistics, reset the database tabs,
 * modify default game settings, view user actions in the audit log, and perform maintenance
 * tasks securely within the application's ecosystem.
 */

/**
 * Exige uma sessao administrativa.
 * @returns {{id: string, username: string, role: string}}
 * @throws {Error} Sem sessao ou quando o perfil nao e Admin.
 */
function requireAdmin_() {
  try {
    var user = requireAuth_();
    if (String(user.role || "").toLowerCase() !== "admin") {
      throw new Error("Acesso restrito a administradores.");
    }
    return user;
  } catch (error) {
    Logger.log("Erro em requireAdmin_: " + error.message);
    throw error;
  }
}

/**
 * Retorna metricas agregadas e atividade recente, sem senhas ou detalhes livres.
 * @returns {Object}
 */
function getAdminDashboardData() {
  try {
    try {
      var admin = requireAdmin_();
      var config = getGameConfig_();
      var users = dataRows_(config.SHEET_NAMES.USERS);
      var progress = dataRows_(config.SHEET_NAMES.GAME_PROGRESS);
      var situations = dataRows_(config.SHEET_NAMES.SITUATIONS);
      var choices = dataRows_(config.SHEET_NAMES.CHOICES);
      var results = dataRows_(config.SHEET_NAMES.RESULTS);
      var audit = dataRows_(config.SHEET_NAMES.AUDIT_LOG);

      var administrators = users.filter(function(row) {
        return String(row[3] || "").toLowerCase() === "admin";
      }).length;
      var totalScore = progress.reduce(function(sum, row) {
        var score = Number(row[3]);
        return sum + (Number.isFinite(score) ? score : 0);
      }, 0);

      return successResponse_({
        admin: {
          id: admin.id,
          username: admin.username,
          role: admin.role
        },
        metrics: {
          users: users.length,
          administrators: administrators,
          players: Math.max(users.length - administrators, 0),
          situations: situations.length,
          choices: choices.length,
          results: results.length,
          auditEvents: audit.length,
          averageScore: progress.length
            ? Math.round((totalScore / progress.length) * 10) / 10
            : 0
        },
        recentActivity: audit.slice(-10).reverse().map(function(row) {
          return {
            action: String(row[1] || ""),
            actor: String(row[2] || ""),
            occurredAt: String(row[4] || "")
          };
        })
      });
    } catch (error) {
      return publicError_(error);
    }
  } catch (error) {
    Logger.log("Erro em getAdminDashboardData: " + error.message);
    throw error;
  }
}

/**
 * Executa apenas seeds idempotentes conhecidos.
 * @param {string} action "situations" ou "admins".
 * @returns {Object} Envelope StandardReturn sem credenciais.
 */
function runAdminSeedAction(action) {
  try {
    try {
      try {
        var admin = requireAdmin_();
        var normalizedAction = String(action || "").trim().toLowerCase();
        var result;

        if (normalizedAction === "situations") {
          result = seedSituations_();
        } else if (normalizedAction === "admins") {
          result = seedSyntheticAdminUsers();
        } else {
          throw new Error("Acao administrativa invalida.");
        }

        var safeResult = {
          action: String(result.action || normalizedAction),
          inserted: Number(result.inserted || 0),
          skipped: Number(result.skipped || 0),
          totalUsers: Number(result.totalUsers || 0)
        };
        logAction_(
          "ADMIN_SEED",
          admin.id,
          normalizedAction + ": " + JSON.stringify(safeResult)
        );
        return successResponse_(safeResult, "Seed administrativo concluido.");
      } catch (error) {
        return publicError_(error);
      }
    } catch (error) {
      Logger.log("Erro em runAdminSeedAction: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em runAdminSeedAction: " + error.message);
    throw error;
  }
}

function dataRows_(sheetName) {
  try {
    var rows = readRecords_(sheetName);
    return rows.length > 1 ? rows.slice(1) : [];
  } catch (error) {
    Logger.log("Erro em dataRows_: " + error.message);
    throw error;
  }
}
