/**
 * FleetLoginCheck.gs — Harness único de verificação de login da frota.
 *
 * Objetivo: numa execução, (1) resolver/garantir a planilha, (2) rodar o seed
 * padrão (runSchemaServiceSetup -> ensureAllSheets -> auto-seed dos 15 admins
 * sinteticos admin01..admin15 / 'admin123' em TEXTO PLANO) e (3) exercitar o
 * MESMO caminho de login que o frontend usa (AuthService.login / doLogin /
 * login / loginWithPassword / processLoginRequest), devolvendo um relatorio
 * verde/vermelho. Roda dentro do projeto Apps Script (Editor -> selecionar
 * runFleetLoginCheck -> Executar; autorize os escopos na 1a vez).
 *
 * Este arquivo e BYTE-IDENTICO nos 27 webapps: nao define nada que ja exista
 * (apenas CHAMA getBoundSpreadsheet_/runSchemaServiceSetup), e todos os helpers
 * sao prefixados com flc_ para evitar colisao no escopo global do Apps Script.
 *
 * Nenhum hash em nenhum ponto: credenciais comparadas em texto plano.
 */

'use strict';

var FLC_ADMIN_USERNAME = 'admin01';
var FLC_ADMIN_EMAIL    = 'admin01@synthetic.local';
var FLC_ADMIN_PASSWORD = 'admin123';

/**
 * Entry point. Execute esta funcao no Editor do Apps Script.
 * @returns {Object} relatorio estruturado (tambem impresso via Logger.log).
 */
function runFleetLoginCheck() {
  var report = {
    project: flc_projectName_(),
    ok: false,
    steps: {
      spreadsheet: { ok: false },
      setup:       { ok: false },
      login:       { ok: false }
    },
    errors: []
  };

  // 1) Planilha
  var ss = null;
  try {
    ss = (typeof getBoundSpreadsheet_ === 'function')
      ? getBoundSpreadsheet_()
      : SpreadsheetApp.getActiveSpreadsheet();
    report.steps.spreadsheet.ok = !!ss;
    report.steps.spreadsheet.id = ss && ss.getId ? ss.getId() : '';
  } catch (e1) {
    flc_addError_(report, 'spreadsheet', e1);
    flc_print_(report);
    return report; // sem planilha, nada mais roda
  }

  // 2) Setup + seed (idempotente)
  try {
    if (typeof runSchemaServiceSetup === 'function') {
      runSchemaServiceSetup({ spreadsheet: ss });
      report.steps.setup.via = 'runSchemaServiceSetup';
    } else if (typeof seedSyntheticAdminUsers === 'function') {
      seedSyntheticAdminUsers({ spreadsheet: ss });
      report.steps.setup.via = 'seedSyntheticAdminUsers';
    } else {
      throw new Error('Nenhum setup/seed encontrado (runSchemaServiceSetup/seedSyntheticAdminUsers).');
    }
    report.steps.setup.ok = true;
  } catch (e2) {
    flc_addError_(report, 'setup', e2);
    // segue para o login: a planilha pode ja ter usuarios de execucoes anteriores
  }

  // 3) Login real (mesmo caminho do frontend)
  try {
    var login = flc_attemptLogin_();
    report.steps.login.ok        = login.ok;
    report.steps.login.via       = login.via;
    report.steps.login.identifier = login.identifier;
    if (!login.ok) {
      report.steps.login.detail = login.detail;
      flc_addError_(report, 'login', login.detail || 'login nao retornou sucesso');
    }
  } catch (e3) {
    flc_addError_(report, 'login', e3);
  }

  report.ok = report.steps.spreadsheet.ok && report.steps.setup.ok && report.steps.login.ok;
  flc_print_(report);
  return report;
}

/**
 * Tenta o login pelo MESMO ponto de entrada que o Login.html usa, com fallback
 * para as assinaturas conhecidas. Testa username e e-mail sinteticos.
 * @returns {{ ok:boolean, via:string, identifier:string, detail:string }}
 */
function flc_attemptLogin_() {
  var identifiers = [FLC_ADMIN_USERNAME, FLC_ADMIN_EMAIL];
  var entries = flc_loginEntries_();
  var lastDetail = 'nenhum ponto de entrada de login encontrado';

  for (var i = 0; i < entries.length; i++) {
    var entry = entries[i];
    for (var j = 0; j < identifiers.length; j++) {
      var id = identifiers[j];
      try {
        var result = entry.call(id, FLC_ADMIN_PASSWORD);
        if (flc_isLoginOk_(result)) {
          return { ok: true, via: entry.name, identifier: id, detail: '' };
        }
        lastDetail = entry.name + ' rejeitou "' + id + '": ' + flc_describe_(result);
      } catch (err) {
        lastDetail = entry.name + ' lancou em "' + id + '": ' + (err && err.message ? err.message : String(err));
      }
    }
  }
  return { ok: false, via: '', identifier: '', detail: lastDetail };
}

/**
 * Lista ordenada de pontos de entrada de login disponiveis no escopo global.
 * Cada um aceita (username, password) e tambem testa a forma de objeto.
 */
function flc_loginEntries_() {
  var entries = [];

  if (typeof AuthService !== 'undefined' && AuthService && typeof AuthService.login === 'function') {
    entries.push({ name: 'AuthService.login', call: function (u, p) {
      try {
        return flc_callFlexible_(AuthService.login, AuthService, u, p);
      } catch (error) {
        Logger.log("Erro em call: " + error.message);
        throw error;
      }
    }});
  }
  if (typeof doLogin === 'function') {
    entries.push({ name: 'doLogin', call: function (u, p) { return flc_callFlexible_(doLogin, null, u, p); } });
  }
  if (typeof processLoginRequest === 'function') {
    entries.push({ name: 'processLoginRequest', call: function (u, p) { return flc_callFlexible_(processLoginRequest, null, u, p); } });
  }
  if (typeof loginWithPassword === 'function') {
    entries.push({ name: 'loginWithPassword', call: function (u, p) { return flc_callFlexible_(loginWithPassword, null, u, p); } });
  }
  if (typeof login === 'function') {
    entries.push({ name: 'login', call: function (u, p) { return flc_callFlexible_(login, null, u, p); } });
  }
  if (typeof authenticate === 'function') {
    entries.push({ name: 'authenticate', call: function (u, p) { return flc_callFlexible_(authenticate, null, u, p); } });
  }
  return entries;
}

/**
 * Chama fn como (username, password); se devolver resultado de falha, tenta a
 * forma de objeto ({ username, password }). Cobre as duas convencoes da frota.
 */
function flc_callFlexible_(fn, thisArg, username, password) {
  var positional = fn.call(thisArg, username, password);
  if (flc_isLoginOk_(positional)) return positional;
  try {
    var asObject = fn.call(thisArg, { username: username, password: password, senha: password, email: username });
    if (flc_isLoginOk_(asObject)) return asObject;
    return positional || asObject;
  } catch (ignored) {
    return positional;
  }
}

/** Normaliza o resultado do login para um booleano de sucesso. */
function flc_isLoginOk_(r) {
  if (r === null || r === undefined || r === false) return false;
  if (typeof r === 'object') {
    if (r.success === false || r.ok === false) return false;
    if (r.success === true || r.ok === true) return true;
    if (r.token || r.sessionToken || r.redirectUrl || r.session) return true;
    if (r.user || r.id || r.username || r.role || r.perfil) return true; // devolveu o usuario
    return false;
  }
  return !!r; // primitivo truthy (ex.: token em string)
}

function flc_describe_(r) {
  try {
    if (r === null || r === undefined) return String(r);
    if (typeof r === 'object') {
      if (r.message) return String(r.message);
      return JSON.stringify(r).slice(0, 160);
    }
    return String(r);
  } catch (e) { return '[obj]'; }
}

function flc_projectName_() {
  try {
    if (typeof CONFIG !== 'undefined' && CONFIG && CONFIG.PROJECT_NAME) return CONFIG.PROJECT_NAME;
    if (typeof Config !== 'undefined' && Config && Config.PROJECT_NAME) return Config.PROJECT_NAME;
  } catch (ignored) {}
  try { return DriveApp.getFileById(ScriptApp.getScriptId()).getName(); } catch (ignored2) {}
  return 'projeto';
}

function flc_addError_(report, stage, error) {
  report.errors.push({
    stage: stage,
    message: error && error.message ? error.message : String(error)
  });
}

function flc_print_(report) {
  var mark = report.ok ? 'VERDE OK' : 'VERMELHO FALHOU';
  Logger.log('=== FleetLoginCheck [' + report.project + '] => ' + mark + ' ===');
  Logger.log('  planilha: ' + (report.steps.spreadsheet.ok ? 'OK (' + (report.steps.spreadsheet.id || '') + ')' : 'FALHOU'));
  Logger.log('  setup/seed: ' + (report.steps.setup.ok ? 'OK (' + (report.steps.setup.via || '') + ')' : 'FALHOU'));
  Logger.log('  login: ' + (report.steps.login.ok
    ? 'OK via ' + report.steps.login.via + ' (' + report.steps.login.identifier + ')'
    : 'FALHOU — ' + (report.steps.login.detail || '')));
  if (report.errors.length) {
    Logger.log('  erros:');
    report.errors.forEach(function (e) { Logger.log('   - [' + e.stage + '] ' + e.message); });
  }
}
