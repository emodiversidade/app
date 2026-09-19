/**
 * BackupFolderService.gs - Backup simples para a pasta BACKUP_FOLDER_ID.
 * @integration DriveFolderConfig.gs — getConfiguredBackupFolder()
 */

function createConfiguredSpreadsheetBackup(spreadsheetId, prefix) {
  try {
    var ssId = spreadsheetId || SpreadsheetApp.getActiveSpreadsheet().getId();
    var file = DriveApp.getFileById(ssId);
    var timestamp = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'yyyyMMdd_HHmmss');
    var copyName = (prefix || 'Backup') + '_' + timestamp;
    var copy = file.makeCopy(copyName, getConfiguredBackupFolder());
    return {
      ok: true,
      id: copy.getId(),
      name: copy.getName(),
      url: copy.getUrl(),
      folderId: getConfiguredBackupFolderId(),
      generatedAt: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em createConfiguredSpreadsheetBackup: " + error.message);
    throw error;
  }
}

function listConfiguredBackups(limit) {
  try {
    var folder = getConfiguredBackupFolder();
    var files = folder.getFiles();
    var rows = [];
    var max = Number(limit || 50);
    while (files.hasNext() && rows.length < max) {
      var file = files.next();
      rows.push({
        id: file.getId(),
        name: file.getName(),
        url: file.getUrl(),
        createdAt: file.getDateCreated(),
        updatedAt: file.getLastUpdated()
      });
    }
    return {
      ok: true,
      folderId: folder.getId(),
      backups: rows,
      generatedAt: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em listConfiguredBackups: " + error.message);
    throw error;
  }
}
