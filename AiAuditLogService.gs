/**
 * AiAuditLogService.gs — FROTA-06: Auditoria mínima da geração de IA
 *
 * Registra cada geração com os campos técnicos mínimos e vincula à decisão
 * humana do HumanReviewService, sem reproduzir conteúdo sensível.
 *
 * Campos gravados por linha:
 *   genId        — identificador único da geração (não vincula a pessoa)
 *   userId       — chave anônima do Apps Script (nunca o email)
 *   useCase      — nome da função / caso de uso
 *   model        — modelo do provedor (ex.: 'gemini-2.0-flash')
 *   date         — ISO 8601 UTC
 *   durationMs   — tempo de resposta do provedor em ms
 *   status       — 'ok' | 'fail' | 'fallback'
 *   fallback     — TRUE/FALSE (se houve degradação para resposta local)
 *   errorCode    — código HTTP ou nome do erro; vazio em sucesso
 *   reviewId     — ID do rascunho no HumanReviewService (opcional)
 *   reviewDecision — 'draft' | 'edited' | 'rejected' | 'approved' (opcional)
 *
 * Campos NUNCA gravados: prompt pessoal completo, resposta do modelo, email,
 *   tokens de API, dados biométricos, raciocínio interno.
 *
 * Armazenamento: Google Spreadsheet auto-provisionada (aba "AI_Log").
 *   Configuração via Script Properties:
 *     AI_AUDIT_SPREADSHEET_ID  — ID da planilha; auto-criada se ausente
 *     AI_AUDIT_RETENTION_DAYS  — dias de retenção (padrão 90)
 *
 * Acesso: getAiAuditLog(limit) retorna as N últimas linhas para administração.
 */

'use strict';

var AiAuditLogService = (function () {

  var SHEET_NAME = 'AI_Log';
  var HEADERS    = ['genId','userId','useCase','model','date','durationMs','status','fallback','errorCode','reviewId','reviewDecision'];
  var PROP_SS    = 'AI_AUDIT_SPREADSHEET_ID';
  var PROP_RET   = 'AI_AUDIT_RETENTION_DAYS';
  var DEFAULT_RETENTION = 90;

  // ── Utilitários ──────────────────────────────────────────────────────────

  function _genId() {
    try {
      // ID compacto: timestamp36 + 4 chars aleatórios
      return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    } catch (error) {
      Logger.log("Erro em _genId: " + error.message);
      throw error;
    }
  }

  function _userId() {
    try {
      // Identificador anonimo do Apps Script (sem hash, nunca o e-mail).
      return Session.getTemporaryActiveUserKey() || 'anon';
    } catch (_) { return 'anon'; }
  }

  function _today() {
    try { return new Date().toISOString(); } catch (_) { return String(Date.now()); }
  }

  function _retention() {
    try {
      var v = PropertiesService.getScriptProperties().getProperty(PROP_RET);
      return v ? parseInt(v, 10) : DEFAULT_RETENTION;
    } catch (_) { return DEFAULT_RETENTION; }
  }

  // ── Planilha de auditoria ────────────────────────────────────────────────

  function _getSpreadsheet() {
    try {
      var props = PropertiesService.getScriptProperties();
      var id    = props.getProperty(PROP_SS);
      if (id) {
        try { return SpreadsheetApp.openById(id); } catch (_) { id = null; }
      }
      // Auto-provisão
      var ss = SpreadsheetApp.create('AI_Audit_Log');
      props.setProperty(PROP_SS, ss.getId());
      Logger.log('[AiAuditLogService] Planilha criada: ' + ss.getId());
      return ss;
    } catch (error) {
      Logger.log("Erro em _getSpreadsheet: " + error.message);
      throw error;
    }
  }

  function _getSheet() {
    var ss    = _getSpreadsheet();
    var sheet = ss.getSheetByName(SHEET_NAME);
    if (!sheet) {
      sheet = ss.insertSheet(SHEET_NAME);
      sheet.getRange(1, 1, 1, HEADERS.length).setValues([HEADERS]);
      sheet.setFrozenRows(1);
    } else if (sheet.getLastRow() === 0) {
      sheet.getRange(1, 1, 1, HEADERS.length).setValues([HEADERS]);
      sheet.setFrozenRows(1);
    }
    return sheet;
  }

  // ── Poda por retenção ────────────────────────────────────────────────────

  function _pruneOldRows(sheet) {
    try {
      try {
        var retDays = _retention();
        var cutoff  = new Date(Date.now() - retDays * 86400000).toISOString();
        var last    = sheet.getLastRow();
        if (last < 2) return;
        var dates = sheet.getRange(2, 5, last - 1, 1).getValues(); // coluna 'date'
        var toDelete = [];
        for (var i = 0; i < dates.length; i++) {
          if (String(dates[i][0]) < cutoff) toDelete.push(i + 2);
        }
        // Remove de baixo para cima para preservar índices
        for (var j = toDelete.length - 1; j >= 0; j--) {
          sheet.deleteRow(toDelete[j]);
        }
      } catch (_) {}
    } catch (error) {
      Logger.log("Erro em _pruneOldRows: " + error.message);
      throw error;
    }
  }

  // ── API pública ──────────────────────────────────────────────────────────

  /**
   * Registra uma geração de IA.
   *
   * @param {object} opt
   *   useCase      {string}  Nome da função / caso de uso.
   *   model        {string}  Identificador do modelo (ex.: 'gemini-2.0-flash').
   *   durationMs   {number}  Tempo de resposta em ms.
   *   status       {string}  'ok' | 'fail' | 'fallback'.
   *   fallback     {boolean} true se houve degradação para resposta local.
   *   errorCode    {string}  Código HTTP ou nome do erro (vazio em sucesso).
   *   reviewId     {string}  (opcional) ID do rascunho no HumanReviewService.
   * @returns {string} genId gerado.
   */
  function record(opt) {
    opt = opt || {};
    var id  = _genId();
    var row = [
      id,
      _userId(),
      opt.useCase      || '',
      opt.model        || '',
      _today(),
      opt.durationMs   || 0,
      opt.status       || 'ok',
      opt.fallback      ? 'TRUE' : 'FALSE',
      opt.errorCode    || '',
      opt.reviewId     || '',
      opt.reviewDecision || 'draft'
    ];
    try {
      var sheet = _getSheet();
      sheet.appendRow(row);
      _pruneOldRows(sheet);
    } catch (e) {
      // Log local se planilha indisponível — não quebra o fluxo principal
      Logger.log('[AiAuditLogService] falha ao gravar: ' + e.message + ' | row=' + JSON.stringify(row));
    }
    return id;
  }

  /**
   * Atualiza a decisão humana numa linha existente pelo genId.
   * Chamado após HumanReviewService.approve / reject / editDraft.
   *
   * @param {string} genId      ID retornado por record().
   * @param {string} decision   'edited' | 'rejected' | 'approved'.
   * @param {string} [reviewId] ID do rascunho no HumanReviewService.
   */
  function recordDecision(genId, decision, reviewId) {
    try {
      try {
        var sheet = _getSheet();
        var last  = sheet.getLastRow();
        if (last < 2) return;
        var ids = sheet.getRange(2, 1, last - 1, 1).getValues();
        for (var i = 0; i < ids.length; i++) {
          if (String(ids[i][0]) === genId) {
            var row = i + 2;
            if (reviewId) sheet.getRange(row, 10).setValue(reviewId);   // reviewId col
            sheet.getRange(row, 11).setValue(decision);                  // reviewDecision col
            return;
          }
        }
      } catch (e) {
        Logger.log('[AiAuditLogService] recordDecision falhou: ' + e.message);
      }
    } catch (error) {
      Logger.log("Erro em recordDecision: " + error.message);
      throw error;
    }
  }

  /**
   * Retorna as N linhas mais recentes para administração.
   * Não retorna conteúdo de prompts — apenas os campos do cabeçalho.
   *
   * @param {number} [limit=50]
   * @returns {Array<Object>}
   */
  function getLog(limit) {
    try {
      try {
        limit = limit || 50;
        try {
          var sheet = _getSheet();
          var last  = sheet.getLastRow();
          if (last < 2) return [];
          var start  = Math.max(2, last - limit + 1);
          var count  = last - start + 1;
          var values = sheet.getRange(start, 1, count, HEADERS.length).getValues();
          return values.map(function (row) {
            var obj = {};
            HEADERS.forEach(function (h, i) { obj[h] = row[i]; });
            return obj;
          }).reverse();
        } catch (e) {
          Logger.log('[AiAuditLogService] getLog falhou: ' + e.message);
          return [];
        }
      } catch (error) {
        Logger.log("Erro em getLog: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em getLog: " + error.message);
      throw error;
    }
  }

  return { record: record, recordDecision: recordDecision, getLog: getLog };
})();

/** Atalho global para painel de administração. */
function getAiAuditLog(limit) {
  return AiAuditLogService.getLog(limit || 50);
}
