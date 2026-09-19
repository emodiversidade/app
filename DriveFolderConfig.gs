/**
 * DriveFolderConfig.gs - Configuracao de pastas Drive do projeto.
 *
 * Variaveis esperadas nas propriedades do script:
 *   INPUT_FOLDER_ID  - pasta para arquivos de entrada/importacao.
 *   OUTPUT_FOLDER_ID - pasta para relatorios, PDFs, CSVs e artefatos gerados.
 *   BACKUP_FOLDER_ID - pasta para backups e snapshots.
 */

var DRIVE_FOLDER_REQUIRED_KEYS = ['OUTPUT_FOLDER_ID', 'BACKUP_FOLDER_ID'];

var DRIVE_FOLDER_ALIASES = Object.freeze({
  INPUT_FOLDER_ID:  ['DRIVE_INPUT_FOLDER_ID',  'SOURCE_FOLDER_ID',  'PASTA_ORIGEM_ID'],
  OUTPUT_FOLDER_ID: ['DRIVE_OUTPUT_FOLDER_ID', 'REPORT_FOLDER_ID',  'DRIVE_FOLDER_ID', 'FOLDER_ID', 'PASTA_DESTINO_ID'],
  BACKUP_FOLDER_ID: ['DRIVE_BACKUP_FOLDER_ID', 'BACKUPS_FOLDER_ID', 'DRIVE_FOLDER_ID', 'FOLDER_ID']
});

function getDriveFolderConfig() {
  try {
    var props = PropertiesService.getScriptProperties();
    var config = {};
    ['INPUT_FOLDER_ID', 'OUTPUT_FOLDER_ID', 'BACKUP_FOLDER_ID'].forEach(function(key) {
      config[key] = getDriveFolderProperty_(props, key);
    });
    return config;
  } catch (error) {
    Logger.log("Erro em getDriveFolderConfig: " + error.message);
    throw error;
  }
}

function validateDriveFolderConfig() {
  try {
    var config = getDriveFolderConfig();
    var folders = DRIVE_FOLDER_REQUIRED_KEYS.map(function(key) {
      return validateDriveFolder_(key, config[key]);
    });
    return {
      ok: folders.every(function(f) { return f.ok; }),
      required: DRIVE_FOLDER_REQUIRED_KEYS.slice(),
      missing: folders.filter(function(f) { return !f.configured; }).map(function(f) { return f.key; }),
      invalid: folders.filter(function(f) { return f.configured && !f.ok; }),
      folders: folders,
      generatedAt: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em validateDriveFolderConfig: " + error.message);
    throw error;
  }
}

function requireDriveFolderConfig() {
  try {
    var v = validateDriveFolderConfig();
    if (!v.ok) throw new Error('Pastas Drive pendentes ou invalidas: ' + v.required.join(', '));
    return v;
  } catch (error) {
    Logger.log("Erro em requireDriveFolderConfig: " + error.message);
    throw error;
  }
}

function getConfiguredInputFolder()    { return getConfiguredDriveFolderByKey_('INPUT_FOLDER_ID'); }
function getConfiguredOutputFolder()   { return getConfiguredDriveFolderByKey_('OUTPUT_FOLDER_ID'); }
function getConfiguredBackupFolder()   { return getConfiguredDriveFolderByKey_('BACKUP_FOLDER_ID'); }
function getConfiguredInputFolderId()  { return getDriveFolderConfig().INPUT_FOLDER_ID; }
function getConfiguredOutputFolderId() { return getDriveFolderConfig().OUTPUT_FOLDER_ID; }
function getConfiguredBackupFolderId() { return getDriveFolderConfig().BACKUP_FOLDER_ID; }

function createFileInConfiguredOutputFolder(fileName, content, mimeType) {
  return createFileInConfiguredFolder_(getConfiguredOutputFolder(), fileName, content, mimeType);
}
function createFileInConfiguredBackupFolder(fileName, content, mimeType) {
  return createFileInConfiguredFolder_(getConfiguredBackupFolder(), fileName, content, mimeType);
}
function createFileInConfiguredInputFolder(fileName, content, mimeType) {
  return createFileInConfiguredFolder_(getConfiguredInputFolder(), fileName, content, mimeType);
}

function getDriveFolderProperty_(props, key) {
  try {
    var direct = props.getProperty(key);
    if (direct) return direct;
    var aliases = DRIVE_FOLDER_ALIASES[key] || [];
    for (var i = 0; i < aliases.length; i++) {
      var v = props.getProperty(aliases[i]);
      if (v) return v;
    }
    return '';
  } catch (error) {
    Logger.log("Erro em getDriveFolderProperty_: " + error.message);
    throw error;
  }
}

function getConfiguredDriveFolderByKey_(key) {
  try {
    var folderId = getDriveFolderConfig()[key];
    if (!folderId) throw new Error(key + ' nao configurado nas propriedades do script.');
    return DriveApp.getFolderById(folderId);
  } catch (error) {
    Logger.log("Erro em getConfiguredDriveFolderByKey_: " + error.message);
    throw error;
  }
}

function validateDriveFolder_(key, folderId) {
  var result = { key: key, folderId: folderId || '', configured: !!folderId, ok: false, name: '', url: '', error: '' };
  if (!folderId) return result;
  try {
    var folder = DriveApp.getFolderById(folderId);
    result.ok   = true;
    result.name = folder.getName();
    result.url  = folder.getUrl();
  } catch (e) {
    result.error = e.message;
  }
  return result;
}

function createFileInConfiguredFolder_(folder, fileName, content, mimeType) {
  try {
    var safeName = String(fileName || ('arquivo_' + new Date().getTime()));
    var file;
    if (content && typeof content.copyBlob === 'function') {
      file = folder.createFile(content.copyBlob()).setName(safeName);
    } else if (content && typeof content.getBlob === 'function') {
      file = folder.createFile(content.getBlob()).setName(safeName);
    } else {
      file = folder.createFile(safeName, String(content || ''), mimeType || MimeType.PLAIN_TEXT);
    }
    return {
      ok: true,
      id: file.getId(),
      name: file.getName(),
      url: file.getUrl(),
      folderId: folder.getId(),
      generatedAt: new Date().toISOString()
    };
  } catch (error) {
    Logger.log("Erro em createFileInConfiguredFolder_: " + error.message);
    throw error;
  }
}
