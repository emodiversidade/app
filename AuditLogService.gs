/**
 * @file AuditLogService.gs
 * @description Registra todas as ações importantes do sistema para fins de auditoria e depuração.
 *
 * English Description:
 * This service provides logging functionality to record critical system events and user actions to the
 * dedicated Audit sheet in Google Sheets. It appends immutable log entries including unique identifiers,
 * action types, associated user IDs, metadata, and timestamps. This functionality is essential for
 * administrative auditing, debugging error occurrences, monitoring security incidents, and maintaining
 * historical operational transparency across the system.
 * @integration Google Sheets (aba: Auditoria)
 * @function logAction_(action, userId, details)
 */

/**
 * Acrescenta uma linha imutavel na aba Auditoria.
 * @param {string} action Codigo da acao (LOGIN, REGISTER, CHOICE, SETTINGS...).
 * @param {string} userId ID da pessoa afetada.
 * @param {string} details Texto livre ou JSON com o contexto da acao.
 */
function logAction_(action, userId, details) {
  try {
    const config = getGameConfig_();
    createRecord_(config.SHEET_NAMES.AUDIT_LOG, [
      generateUniqueId_(),
      action,
      userId,
      details,
      new Date().toISOString()
    ]);
  } catch (error) {
    Logger.log("Erro em logAction_: " + error.message);
    throw error;
  }
}

