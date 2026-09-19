/**
 * Contrato de imagens do Emodiversa.
 *
 * Quando presentes, os arquivos locais em webapp/assets servem somente como
 * espelho de autoria e conferência. Em produção, o webapp resolve cada imagem
 * pelo nome exato dentro da pasta indicada pela Script Property FOLDER_ID.
 */
var GAME_ASSET_PROJECT = 'emodiversa';
var GAME_ASSET_FILES = [
  'jornada_autonomia_hero.webp',
  'mapa_cinco_pilares.svg',
  'avatar_exploradora_emodiversa.png',
  'pilares_simbolos.svg',
  'cartas_dilemas.webp',
  'equilibrio_em_movimento.svg',
  'selo_jornada_propria.svg'
];

var GAME_ASSET_MIME_TYPES = Object.freeze({
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp'
});

/**
 * RPC público consumido pelas telas HTML.
 * @returns {Object} Manifesto filename -> URL do Drive, sem expor FOLDER_ID.
 */
function getGameAssetManifest() {
  return GameAssetService_getManifest_(GAME_ASSET_FILES, GAME_ASSET_PROJECT);
}

/** FOLDER_ID é o único contrato de configuração dos assets do jogo. */
function GameAssetService_resolveFolder_() {
  var value = String(PropertiesService.getScriptProperties().getProperty('FOLDER_ID') || '').trim();
  return { id: value, configuredBy: value ? 'FOLDER_ID' : '' };
}

/**
 * Procura cada elemento pelo nome, em vez de construir caminho local ou URL
 * presumida. Nomes duplicados são recusados para não escolher arte ao acaso.
 */
function GameAssetService_getManifest_(expectedFiles, project) {
  expectedFiles = (expectedFiles || []).slice();
  var manifest = {
    project: project || '',
    folderProperty: 'FOLDER_ID',
    configuredBy: '',
    expectedFiles: expectedFiles,
    assets: {},
    assetItems: [],
    missingFiles: [],
    invalidFiles: [],
    duplicateFiles: [],
    ok: false
  };

  try {
    var folderConfig = GameAssetService_resolveFolder_();
    manifest.configuredBy = folderConfig.configuredBy;
    if (!folderConfig.id) {
      manifest.missingFiles = expectedFiles.slice();
      manifest.error = 'Configure a Script Property FOLDER_ID com a pasta de assets do Emodiversa.';
      return manifest;
    }

    var folder = DriveApp.getFolderById(folderConfig.id);
    expectedFiles.forEach(function(name) {
      var files = folder.getFilesByName(name);
      if (!files.hasNext()) {
        manifest.missingFiles.push(name);
        return;
      }

      var file = files.next();
      if (files.hasNext()) {
        manifest.duplicateFiles.push(name);
        return;
      }

      var extensionMatch = /\.[^.]+$/.exec(name.toLowerCase());
      var extension = extensionMatch ? extensionMatch[0] : '';
      var expectedMime = GAME_ASSET_MIME_TYPES[extension] || '';
      var actualMime = String(file.getMimeType() || '').toLowerCase();
      if (!expectedMime || actualMime !== expectedMime) {
        manifest.invalidFiles.push({
          name: name,
          expectedMimeType: expectedMime,
          actualMimeType: actualMime
        });
        return;
      }

      var url = 'https://drive.google.com/uc?export=view&id=' + encodeURIComponent(file.getId());
      manifest.assets[name] = url;
      manifest.assetItems.push({ name: name, url: url, mimeType: actualMime });
    });

    manifest.ok = manifest.missingFiles.length === 0 &&
      manifest.invalidFiles.length === 0 &&
      manifest.duplicateFiles.length === 0;
    if (!manifest.ok) {
      manifest.error = 'A pasta FOLDER_ID não contém um conjunto único e válido de todos os assets esperados.';
    }
    return manifest;
  } catch (error) {
    Logger.log('[GameAssetService] ' + error.message);
    manifest.assets = {};
    manifest.assetItems = [];
    manifest.error = 'Não foi possível ler a pasta de assets configurada em FOLDER_ID.';
    return manifest;
  }
}
