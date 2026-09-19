/**
 * NotebookGraphService.gs
 * Lista imagens PNG/JPG geradas pelos notebooks na pasta configurada em FOLDER_ID.
 */
function getNotebookGraphImages(options) {
  try {
    options = options || {};
    var folderId = resolveNotebookGraphFolderId_();
    if (!folderId) {
      return { ok: false, images: [], message: 'FOLDER_ID nao configurado nas Script Properties.' };
    }

    var folder = DriveApp.getFolderById(folderId);
    var files = folder.getFiles();
    var images = [];
    var limit = Number(options.limit || 0);
    var allowed = /\.(png|jpg|jpeg)$/i;

    while (files.hasNext()) {
      if (limit && images.length >= limit) break;
      var file = files.next();
      var name = file.getName();
      if (!allowed.test(name)) continue;
      var blob = file.getBlob();
      var contentType = blob.getContentType() || (/\.png$/i.test(name) ? 'image/png' : 'image/jpeg');
      images.push({
        id: file.getId(),
        name: name,
        contentType: contentType,
        dataUrl: 'data:' + contentType + ';base64,' + Utilities.base64Encode(blob.getBytes()),
        updatedAt: file.getLastUpdated() ? file.getLastUpdated().toISOString() : ''
      });
    }

    images.sort(function(a, b) { return String(a.name).localeCompare(String(b.name)); });
    return { ok: true, images: images, folderId: folderId };
  } catch (error) {
    Logger.log("Erro em getNotebookGraphImages: " + error.message);
    throw error;
  }
}

function resolveNotebookGraphFolderId_() {
  try {
    var props = PropertiesService.getScriptProperties();
    var keys = ['FOLDER_ID', 'OUTPUT_FOLDER_ID', 'DRIVE_OUTPUT_FOLDER_ID', 'DRIVE_FOLDER_ID'];
    for (var i = 0; i < keys.length; i++) {
      var value = props.getProperty(keys[i]);
      if (value) return value;
    }
    return '';
  } catch (error) {
    Logger.log("Erro em resolveNotebookGraphFolderId_: " + error.message);
    throw error;
  }
}

