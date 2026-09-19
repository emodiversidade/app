/**
 * SchemaService.gs — Emodiversa
 * Gerado por audit_schema_setup.py v2.
 * Implementacao padrao de frota: getSpreadsheet_ robusto,
 * ensureAllSheets, runSchemaServiceSetup, seed de admins sinteticos
 * com senha em texto plano (admin123).
 */

var PROJECT_SCHEMA_DEFINITIONS = {
  USERS: {
    entity: 'USERS', sheetName: 'Usuarios',
    headers: ['ID','NomeUsuario','Senha','Perfil','DataCriacao'],
    required: ['ID'], identifier: 'ID'
  },
  PROGRESS: {
    entity: 'PROGRESS', sheetName: 'ProgressoJogo',
    headers: ['ID','ID_Usuario','NivelAtual','Pontuacao','UltimaAtualizacao'],
    required: ['ID'], identifier: 'ID'
  },
  SITUACOES: {
    entity: 'SITUACOES', sheetName: 'Situacoes',
    headers: ['ID','Titulo','Descricao','Opcao1','Opcao2','Resultado1','Resultado2',
              'NivelPrincesamento','ImpactoOpcao1','ImpactoOpcao2'],
    required: ['ID'], identifier: 'ID'
  },
  ESCOLHAS: {
    entity: 'ESCOLHAS', sheetName: 'Escolhas',
    headers: ['ID','ID_Usuario','ID_Situacao','EscolhaFeita','DataEscolha'],
    required: ['ID'], identifier: 'ID'
  },
  RESULTADOS: {
    entity: 'RESULTADOS', sheetName: 'Resultados',
    headers: ['ID','ID_Usuario','ID_Situacao','ResultadoFinal','ImpactoPontuacao',
              'DataResultado','Dimensoes'],
    required: ['ID'], identifier: 'ID'
  },
  AUDITORIA: {
    entity: 'AUDITORIA', sheetName: 'Auditoria',
    headers: ['ID','Acao','Usuario','Detalhes','DataHora'],
    required: ['ID'], identifier: 'ID'
  },
  SETTINGS: {
    entity: 'SETTINGS', sheetName: 'Settings',
    headers: ['Key','Value','Description','UpdatedAt'],
    required: ['Key'], identifier: 'Key'
  }
};

// Login usa a aba Usuarios com senha em texto plano
var SYNTHETIC_USERS_LOGIN_SCHEMAS = [
  { sheetName: 'Usuarios',
    headers: ['ID','NomeUsuario','Senha','Perfil','DataCriacao'],
    constants: {} }
];

var SchemaService = (function() {
  function clone_(value) {
    try {
      return JSON.parse(JSON.stringify(value));
    } catch (error) {
      Logger.log("Erro em clone_: " + error.message);
      throw error;
    }
  }

  function getSpreadsheet_(options) {
    try {
      options = options || {};
      if (options.spreadsheet) return options.spreadsheet;
      if (options.spreadsheetId) return SpreadsheetApp.openById(options.spreadsheetId);
      if (typeof getSpreadsheetId === 'function') {
        var id = getSpreadsheetId();
        if (id) return SpreadsheetApp.openById(id);
      }
      if (typeof CONFIG !== 'undefined' && CONFIG.SPREADSHEET_ID)
        return SpreadsheetApp.openById(CONFIG.SPREADSHEET_ID);
      if (typeof CONFIG !== 'undefined' && CONFIG.SPREADSHEETS_ID)
        return SpreadsheetApp.openById(CONFIG.SPREADSHEETS_ID);
      if (typeof Config !== 'undefined' && typeof Config.getSpreadsheetId === 'function') {
        var id2 = Config.getSpreadsheetId();
        if (id2) return SpreadsheetApp.openById(id2);
      }
      if (typeof Config !== 'undefined' && Config.SPREADSHEET_ID)
        return SpreadsheetApp.openById(Config.SPREADSHEET_ID);
      if (typeof Config !== 'undefined' && Config.SPREADSHEETS_ID)
        return SpreadsheetApp.openById(Config.SPREADSHEETS_ID);
      var props = PropertiesService.getScriptProperties();
      var pid = props.getProperty('SPREADSHEETS_ID') || props.getProperty('SPREADSHEET_ID');
      if (pid) return SpreadsheetApp.openById(pid);
      return getBoundSpreadsheet_();
    } catch (error) {
      Logger.log("Erro em getSpreadsheet_: " + error.message);
      throw error;
    }
  }

  function normalizeEntityKey_(value) {
    try {
      return String(value || '')
        .replace(/^DB_/, '')
        .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
        .replace(/[^A-Za-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '')
        .toUpperCase();
    } catch (error) {
      Logger.log("Erro em normalizeEntityKey_: " + error.message);
      throw error;
    }
  }

  function headersFromStructure_(structure) {
    if (!structure) return [];
    if (Array.isArray(structure)) return structure;
    if (Array.isArray(structure.headers)) return structure.headers;
    if (Array.isArray(structure.columns)) return structure.columns;
    return [];
  }

  function upsertRuntimeSchema_(schemas, entityName, sheetName, headers, source) {
    try {
      var clean = (headers || []).filter(function(h, i, a) {
        return h && a.indexOf(h) === i;
      });
      if (!sheetName || clean.length === 0) return;
      var key = normalizeEntityKey_(entityName || sheetName);
      var ex = schemas[key] || {};
      schemas[key] = {
        entity: ex.entity || key,
        sheetName: sheetName,
        headers: clean,
        required: ex.required && ex.required.length ? ex.required : [clean[0]],
        identifier: ex.identifier || clean[0],
        source: source || ex.source || 'runtime'
      };
    } catch (error) {
      Logger.log("Erro em upsertRuntimeSchema_: " + error.message);
      throw error;
    }
  }

  function collectRuntimeSchemas_() {
    var schemas = clone_(PROJECT_SCHEMA_DEFINITIONS);
    try {
      if (typeof CODEX_SCHEMA_HEADER_OVERRIDES !== 'undefined') {
        Object.keys(CODEX_SCHEMA_HEADER_OVERRIDES).forEach(function(key) {
          var s = schemas[key];
          if (s) upsertRuntimeSchema_(schemas, key, s.sheetName,
            CODEX_SCHEMA_HEADER_OVERRIDES[key], 'CODEX_SCHEMA_HEADER_OVERRIDES');
        });
      }
    } catch(e) {}
    return schemas;
  }

  function getSchema(entityName) {
    var key = normalizeEntityKey_(entityName);
    var schemas = collectRuntimeSchemas_();
    var schema = schemas[key];
    if (!schema) {
      Object.keys(schemas).some(function(k) {
        if (normalizeEntityKey_(schemas[k].sheetName) === key) {
          schema = schemas[k]; return true;
        }
      });
    }
    if (!schema) throw new Error('Schema nao encontrado para: ' + entityName);
    return clone_(schema);
  }

  function getSchemas(options) {
    try {
      options = options || {};
      var schemas = collectRuntimeSchemas_();
      if (!options.asArray) return schemas;
      return Object.keys(schemas).sort().map(function(k) { return schemas[k]; });
    } catch (error) {
      Logger.log("Erro em getSchemas: " + error.message);
      throw error;
    }
  }

  function readHeaders_(sheet) {
    try {
      if (!sheet || sheet.getLastColumn() === 0) return [];
      return sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0]
        .map(function(h) { return String(h || '').trim(); }).filter(Boolean);
    } catch (error) {
      Logger.log("Erro em readHeaders_: " + error.message);
      throw error;
    }
  }

  function ensureHeaders_(sheet, headers) {
    try {
      var current = readHeaders_(sheet);
      var missing = headers.filter(function(h) { return current.indexOf(h) === -1; });
      if (current.length === 0) {
        sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
        sheet.setFrozenRows(1);
        return headers;
      }
      if (missing.length > 0) {
        sheet.getRange(1, current.length + 1, 1, missing.length).setValues([missing]);
      }
      return current.concat(missing);
    } catch (error) {
      Logger.log("Erro em ensureHeaders_: " + error.message);
      throw error; // Re-lança para tratamento superior
    }
  }

  function ensureSheet(entityName, options) {
    options = options || {};
    var schema = typeof entityName === 'object' ? entityName : getSchema(entityName);
    var ss = getSpreadsheet_(options);
    var sheet = ss.getSheetByName(schema.sheetName);
    var created = false;
    if (!sheet) { sheet = ss.insertSheet(schema.sheetName); created = true; }
    var headers = ensureHeaders_(sheet, schema.headers || []);
    return { success: true, entity: schema.entity, sheetName: schema.sheetName,
             created: created, headers: headers };
  }

  function ensureAllSheets(options) {
    try {
      options = options || {};
      var schemas = getSchemas({ asArray: true });
      var result = schemas.map(function(schema) { return ensureSheet(schema, options); });
      if (options.seedAdminUsers !== false) {
        try { seedSyntheticUsers(options); } catch(e) {}
      }
      return result;
    } catch (error) {
      Logger.log("Erro em ensureAllSheets: " + error.message);
      throw error;
    }
  }

  function validateSpreadsheet(options) {
    options = options || {};
    var ss = getSpreadsheet_(options);
    return getSchemas({ asArray: true }).map(function(schema) {
      var sheet = ss.getSheetByName(schema.sheetName);
      if (!sheet) return { success: false, entity: schema.entity,
        sheetName: schema.sheetName, missingSheet: true };
      var headers = readHeaders_(sheet);
      var missing = schema.headers.filter(function(h) { return headers.indexOf(h) === -1; });
      return { success: missing.length === 0, entity: schema.entity,
        sheetName: schema.sheetName, missingHeaders: missing };
    });
  }

  // --- Usuarios sinteticos (senha em texto plano: admin123) ---
  var SYNTHETIC_ADMIN_USERS_PASSWORD = 'admin123';
  var SYNTHETIC_PREFIX   = 'SYN-ADMIN-';
  var SYNTHETIC_SEED = [
    'Ana Beatriz Carvalho','Bruno Henrique Alves','Carla Regina Souza',
    'Diego Martins Lima','Eduarda Nunes Pereira','Felipe Augusto Rocha',
    'Gabriela Santos Dias','Henrique Oliveira Costa','Isabela Fernandes Melo',
    'Joao Pedro Ribeiro','Larissa Gomes Barros','Marcelo Tavares Pinto',
    'Natalia Cardoso Freitas','Otavio Ramos Teixeira','Patricia Lopes Moreira'
  ];

  function buildAdmins_() {
    try {
      return SYNTHETIC_SEED.map(function(name, i) {
        var pad = (i + 1 < 10 ? '0' : '') + (i + 1);
        return { id: SYNTHETIC_PREFIX + pad, name: name,
                 email: 'admin' + pad + '@synthetic.local',
                 username: 'admin' + pad, role: 'admin',
                 status: 'ativo', password: SYNTHETIC_ADMIN_USERS_PASSWORD };
      });
    } catch (error) {
      Logger.log("Erro em buildAdmins_: " + error.message);
      throw error;
    }
  }

  function isUserSchema_(schema) {
    try {
      var h = (schema.headers || []).map(function(x) {
        return String(x || '').toLowerCase();
      });
      var hasSecret = h.some(function(x) {
        return x.indexOf('password') >= 0 || x.indexOf('senha') >= 0 || x === 'hash';
      });
      var hasId = h.some(function(x) {
        return x.indexOf('email') >= 0 || x.indexOf('username') >= 0 ||
               x.indexOf('usuario') >= 0 || x.indexOf('login') >= 0;
      });
      return hasSecret && hasId;
    } catch (error) {
      Logger.log("Erro em isUserSchema_: " + error.message);
      throw error;
    }
  }

  function userValue_(header, user, now) {
    try {
      var n = String(header || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
      if (!n) return '';
      if (n.indexOf('password') >= 0 || n.indexOf('senha') >= 0 || n === 'hash') return user.password;
      if (n.indexOf('username') >= 0 || n.indexOf('usuario') >= 0 || n.indexOf('login') >= 0) return user.username;
      if (n === 'id' || n.indexOf('userid') >= 0) return user.id;
      if (n.indexOf('email') >= 0) return user.email;
      if (n.indexOf('name') >= 0 || n.indexOf('nome') >= 0) return user.name;
      if (n.indexOf('role') >= 0 || n.indexOf('papel') >= 0 || n.indexOf('perfil') >= 0) return user.role;
      if (n.indexOf('status') >= 0 || n.indexOf('ativo') >= 0 || n.indexOf('active') >= 0) return user.status;
      if (n.indexOf('created') >= 0 || n.indexOf('criado') >= 0 || n.indexOf('timestamp') >= 0) return now;
      if (n.indexOf('updated') >= 0) return now;
      return '';
    } catch (error) {
      Logger.log("Erro em userValue_: " + error.message);
      throw error;
    }
  }

  function seedSyntheticUsers(options) {
    try {
      try {
        try {
          options = options || {};
          var targets = [];
          try {
            if (typeof SYNTHETIC_USERS_LOGIN_SCHEMAS !== 'undefined' && SYNTHETIC_USERS_LOGIN_SCHEMAS.length) {
              targets = SYNTHETIC_USERS_LOGIN_SCHEMAS;
            }
          } catch(e) {}
          if (!targets.length) {
            targets = getSchemas({ asArray: true }).filter(isUserSchema_);
          }
          var now = new Date().toISOString();
          var users = buildAdmins_();
          var summary = [];
          targets.forEach(function(schema) {
            ensureSheet(schema, options);
            var ss = getSpreadsheet_(options);
            var sheet = ss.getSheetByName(schema.sheetName);
            var headers = readHeaders_(sheet);
            var existing = {};
            if (sheet.getLastRow() > 1) {
              sheet.getRange(2, 1, sheet.getLastRow() - 1, Math.max(headers.length, 1))
                .getValues().forEach(function(row) {
                  row.forEach(function(v) {
                    var k = String(v || '').trim().toLowerCase();
                    if (k) existing[k] = true;
                  });
                });
            }
            var inserted = 0, skipped = 0;
            users.forEach(function(user) {
              if (existing[user.id.toLowerCase()] || existing[user.username.toLowerCase()]) {
                skipped++; return;
              }
              var constants = schema.constants || {};
              var record = {};
              headers.forEach(function(h) { record[h] = userValue_(h, user, now); });
              Object.keys(constants).forEach(function(h) { record[h] = constants[h]; });
              sheet.appendRow(headers.map(function(h) {
                return record[h] !== undefined ? record[h] : '';
              }));
              inserted++;
            });
            summary.push({ sheetName: schema.sheetName, inserted: inserted, skipped: skipped });
          });
          return { ok: true, action: 'seed-synthetic-users',
                   password: SYNTHETIC_ADMIN_USERS_PASSWORD, sheets: summary };
        } catch (error) {
          Logger.log("Erro em seedSyntheticUsers: " + error.message);
          throw error; // Re-lança para tratamento superior
        }
      } catch (error) {
        Logger.log("Erro em seedSyntheticUsers: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em seedSyntheticUsers: " + error.message);
      throw error;
    }
  }

  function getSyntheticAdminUsers() { return buildAdmins_(); }

  function clearSyntheticUsers(options) {
    try {
      try {
        options = options || {};
        var targets = [];
        try {
          if (typeof SYNTHETIC_USERS_LOGIN_SCHEMAS !== 'undefined' && SYNTHETIC_USERS_LOGIN_SCHEMAS.length)
            targets = SYNTHETIC_USERS_LOGIN_SCHEMAS;
        } catch(e) {}
        if (!targets.length) targets = getSchemas({ asArray: true }).filter(isUserSchema_);
        var summary = [];
        targets.forEach(function(schema) {
          var ss = getSpreadsheet_(options);
          var sheet = ss.getSheetByName(schema.sheetName);
          if (!sheet || sheet.getLastRow() < 2) { summary.push({ sheetName: schema.sheetName, removed: 0 }); return; }
          var lastCol = sheet.getLastColumn();
          var values = sheet.getRange(2, 1, sheet.getLastRow() - 1, lastCol).getValues();
          var keep = values.filter(function(row) {
            return !row.some(function(v) { return String(v || '').indexOf(SYNTHETIC_PREFIX) === 0; });
          });
          var removed = values.length - keep.length;
          if (removed > 0) {
            sheet.getRange(2, 1, values.length, lastCol).clearContent();
            if (keep.length > 0) sheet.getRange(2, 1, keep.length, lastCol).setValues(keep);
          }
          summary.push({ sheetName: schema.sheetName, removed: removed });
        });
        return { ok: true, action: 'clear-synthetic-users', sheets: summary };
      } catch (error) {
        Logger.log("Erro em clearSyntheticUsers: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em clearSyntheticUsers: " + error.message);
      throw error;
    }
  }

  return {
    getSchema: getSchema, getSchemas: getSchemas, getSheet: function(e, o) {
      try {
        return getSpreadsheet_(o || {}).getSheetByName(ensureSheet(e, o || {}).sheetName);
      } catch (error) {
        Logger.log("Erro em getSheet: " + error.message);
        throw error;
      }
    },
    ensureSheet: ensureSheet, ensureAllSheets: ensureAllSheets,
    validateSpreadsheet: validateSpreadsheet,
    getSyntheticAdminUsers: getSyntheticAdminUsers,
    seedSyntheticUsers: seedSyntheticUsers, clearSyntheticUsers: clearSyntheticUsers
  };
})();

function seedSyntheticAdminUsers(options) { return SchemaService.seedSyntheticUsers(options || {}); }
function clearSyntheticAdminUsers(options) { return SchemaService.clearSyntheticUsers(options || {}); }
function listSyntheticAdminUsers() { return SchemaService.getSyntheticAdminUsers(); }

function runSchemaServiceSetup(options) {
  return SchemaService.ensureAllSheets(options || {});
}

function runSchemaServiceValidation(options) {
  return SchemaService.validateSpreadsheet(options || {});
}

/* FLEET_AI_SCHEMA_V2: estrutura real + fixtures semanticos explicitamente catalogados. */
function restoreAiCrudStructure(options) {
  options = options || {};
  if (typeof SchemaService === 'undefined') throw new Error('SchemaService indisponivel.');
  if (typeof SchemaService.ensureAllSheets === 'function') return SchemaService.ensureAllSheets(options);
  if (typeof SchemaService.ensureAll === 'function') return SchemaService.ensureAll(options);
  if (typeof SchemaService.restore === 'function') return SchemaService.restore(options);
  throw new Error('SchemaService nao expoe restauracao de estrutura.');
}

function seedSyntheticAiData(options) {
  try {
    try {
      options = options || {};
      var ss = options.spreadsheet || getBoundSpreadsheet_();
      var catalog = typeof AI_FIXTURE_CATALOG !== 'undefined' ? AI_FIXTURE_CATALOG : [];
      var summary = [];
      catalog.forEach(function(fixture) {
        var sheet = ss.getSheetByName(fixture.sheetName);
        if (!sheet) sheet = ss.insertSheet(fixture.sheetName);
        var currentHeaders = sheet.getLastColumn()
          ? sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0] : [];
        if (!currentHeaders.length || currentHeaders.every(function(v) { return !v; })) {
          sheet.getRange(1, 1, 1, fixture.headers.length).setValues([fixture.headers]);
          currentHeaders = fixture.headers.slice();
        }
        var sameHeaders = fixture.headers.length === currentHeaders.length &&
          fixture.headers.every(function(h, i) { return String(currentHeaders[i]) === String(h); });
        if (!sameHeaders) throw new Error('Fixture incompativel com os cabecalhos de ' + fixture.sheetName);
        var inserted = 0;
        if (options.reset === true && sheet.getLastRow() > 1)
          sheet.getRange(2, 1, sheet.getLastRow() - 1, sheet.getLastColumn()).clearContent();
        if (sheet.getLastRow() < 2 || options.append === true) {
          sheet.getRange(sheet.getLastRow() + 1, 1, fixture.rows.length, fixture.headers.length)
            .setValues(fixture.rows);
          inserted = fixture.rows.length;
        }
        summary.push({ sheetName: fixture.sheetName, purpose: fixture.purpose, inserted: inserted });
      });
      return { ok: true, action: 'seed-semantic-ai-fixtures', entities: summary };
    } catch (error) {
      Logger.log("Erro em seedSyntheticAiData: " + error.message);
      throw error; // Re-lança para tratamento superior
    }
  } catch (error) {
    Logger.log("Erro em seedSyntheticAiData: " + error.message);
    throw error;
  }
}

function prepareAiIntegrationFixtures(options) {
  options = options || {};
  return {
    structure: restoreAiCrudStructure(options),
    synthetic: seedSyntheticAiData(options),
    visualization: seedSyntheticVisualizationData(options)
  };
}
/* CODEX_VISUALIZATION_FIXTURES_V1: dados pequenos para validar os notebook.py. */
function _codexVisualizationSpreadsheet_(options) {
  try {
    try {
      options = options || {};
      if (options.spreadsheet) return options.spreadsheet;
      if (typeof getBoundSpreadsheet_ === 'function') return getBoundSpreadsheet_();
      if (typeof getSS === 'function') return getSS();
      var active = SpreadsheetApp.getActiveSpreadsheet();
      if (active) return active;
      var props = PropertiesService.getScriptProperties();
      var spreadsheetId = props.getProperty('SPREADSHEETS_ID') || props.getProperty('SPREADSHEET_ID');
      if (!spreadsheetId) throw new Error('SPREADSHEETS_ID nao configurado para seed de visualizacoes.');
      return SpreadsheetApp.openById(spreadsheetId);
    } catch (error) {
      Logger.log("Erro em _codexVisualizationSpreadsheet_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em _codexVisualizationSpreadsheet_: " + error.message);
    throw error;
  }
}

function _codexVisualizationRows_(projectName) {
  try {
    var today = new Date();
    var categories = ['Leitura', 'Matematica', 'Ciencias', 'Artes'];
    var statuses = ['Planejado', 'Em andamento', 'Concluido'];
    var rows = [];
    for (var i = 0; i < 16; i++) {
      var d = new Date(today.getFullYear(), today.getMonth(), today.getDate() - (15 - i));
      var categoria = categories[i % categories.length];
      var status = statuses[i % statuses.length];
      var ciclo = Math.floor(i / categories.length);
      var metricaA = 48 + (i * 3) + (ciclo * 2);
      var metricaB = 32 + ((i % 5) * 7) + ciclo;
      var valor = 120 + (i * 11) + ((i % 3) * 17);
      rows.push([
        'VIS_SYN_' + Utilities.formatString('%02d', i + 1),
        Utilities.formatDate(d, Session.getScriptTimeZone(), 'yyyy-MM-dd'),
        categoria,
        status,
        metricaA,
        metricaB,
        valor,
        'Amostra sintetica para validar graficos do notebook.py',
        projectName
      ]);
    }
    return rows;
  } catch (error) {
    Logger.log("Erro em _codexVisualizationRows_: " + error.message);
    throw error;
  }
}

function seedSyntheticVisualizationData(options) {
  try {
    try {
      try {
        options = options || {};
        var ss = _codexVisualizationSpreadsheet_(options);
        var sheetName = options.sheetName || 'Visualizacoes_Sinteticas';
        var projectName = options.projectName || (typeof PROJECT_NAME !== 'undefined' ? PROJECT_NAME : ss.getName());
        var headers = ['ID', 'Data', 'Categoria', 'Status', 'Metrica_A', 'Metrica_B', 'Valor', 'Observacao', 'Projeto'];
        var rows = _codexVisualizationRows_(projectName);
        var sheet = ss.getSheetByName(sheetName) || ss.insertSheet(sheetName);

        if (sheet.getLastRow() === 0) {
          sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
        } else {
          sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
        }

        if (options.reset === true && sheet.getLastRow() > 1) {
          sheet.getRange(2, 1, sheet.getLastRow() - 1, Math.max(sheet.getLastColumn(), headers.length)).clearContent();
        }

        var existing = {};
        if (sheet.getLastRow() > 1) {
          sheet.getRange(2, 1, sheet.getLastRow() - 1, 1).getValues().forEach(function(row) {
            existing[String(row[0] || '')] = true;
          });
        }

        var toInsert = rows.filter(function(row) { return options.append === true || !existing[row[0]]; });
        if (toInsert.length) {
          sheet.getRange(sheet.getLastRow() + 1, 1, toInsert.length, headers.length).setValues(toInsert);
        }

        return {
          ok: true,
          action: 'seed-synthetic-visualization-data',
          sheetName: sheetName,
          inserted: toInsert.length,
          totalRows: sheet.getLastRow() - 1
        };
      } catch (error) {
        Logger.log("Erro em seedSyntheticVisualizationData: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em seedSyntheticVisualizationData: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em seedSyntheticVisualizationData: " + error.message);
    throw error;
  }
}




