/** Exclusao da jornada pessoal, sem remover a conta nem consentimentos legais. */
var JourneyPrivacyService = (function() {
  var CONFIRMATION = 'APAGAR MINHA JORNADA';

  function deleteRows_(sheetName, userColumn, userId) {
    var sheet = getSheet_(sheetName);
    var rows = sheet.getDataRange().getValues();
    var removed = 0;
    for (var index = rows.length - 1; index >= 1; index--) {
      if (String(rows[index][userColumn] || '') === String(userId)) {
        sheet.deleteRow(index + 1);
        removed++;
      }
    }
    return removed;
  }

  function deleteJourney(userId, confirmation) {
    if (String(confirmation || '').trim() !== CONFIRMATION) {
      throw new Error('Confirmacao incorreta. Digite exatamente: ' + CONFIRMATION);
    }
    var lock = LockService.getScriptLock();
    if (!lock.tryLock(10000)) throw new Error('Outra operacao esta em andamento. Tente novamente.');
    try {
      var cfg = getGameConfig_().SHEET_NAMES;
      var removed = {
        choices: deleteRows_(cfg.CHOICES, 1, userId),
        results: deleteRows_(cfg.RESULTS, 1, userId),
        progress: deleteRows_(cfg.GAME_PROGRESS, 1, userId),
        audit: deleteRows_(cfg.AUDIT_LOG, 2, userId)
      };
      PropertiesService.getUserProperties().deleteProperty('playerSettings');
      return { removed: removed, accountPreserved: true, consentRecordsPreserved: true };
    } finally {
      lock.releaseLock();
    }
  }

  return { deleteJourney: deleteJourney, confirmation: CONFIRMATION };
})();

function deleteMyJourneyData(confirmation) {
  try {
    var user = requireAuth_();
    return successResponse_(JourneyPrivacyService.deleteJourney(user.id, confirmation), 'Jornada apagada.');
  } catch (error) {
    return publicError_(error);
  }
}
