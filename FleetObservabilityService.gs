/**
 * FleetObservabilityService.gs — FROTA-10: Relatório periódico por projeto
 *
 * Agrega métricas técnicas e de governança do projeto e as persiste em uma
 * aba "Fleet_Metrics" na mesma planilha de auditoria (FROTA-06). Separa
 * telemetria técnica de conteúdo e dados pessoais.
 *
 * Funções principais:
 *   generateReport()     — lê o log de auditoria, computa KPIs, grava na aba
 *   getMetricsSummary()  — retorna resumo JSON sem gravar (para polling/painel)
 *
 * Acionamento sugerido: gatilho de tempo diário (Editar → Acionadores).
 * Exemplo:  ScriptApp.newTrigger('runFleetObservabilityReport').timeBased()
 *             .everyDays(1).atHour(6).create();
 *
 * Dados NUNCA incluídos: prompts, respostas, e-mail, tokens, biometria.
 * Dados incluídos: contagens, taxas, latências, status de camadas, alertas.
 *
 * Colunas da aba Fleet_Metrics:
 *   reportId | project | timestamp | period_days | total_calls | success_rate |
 *   fallback_rate | error_rate | avg_latency_ms | pending_reviews | approved |
 *   rejected | quota_pct | frota_layers_active | alert_count | alert_codes
 */

'use strict';

var FleetObservabilityService = (function () {

  var METRICS_SHEET  = 'Fleet_Metrics';
  var PROP_SS_ID     = 'AI_AUDIT_SPREADSHEET_ID'; // reusa a planilha do FROTA-06
  var REPORT_HEADERS = [
    'reportId', 'project', 'timestamp', 'period_days',
    'total_calls', 'success_rate', 'fallback_rate', 'error_rate', 'avg_latency_ms',
    'pending_reviews', 'approved', 'rejected',
    'quota_used', 'quota_limit', 'quota_pct',
    'frota_layers_active', 'frota_layers_total',
    'alert_count', 'alert_codes'
  ];
  var DEFAULT_PERIOD_DAYS = 7; // janela padrão de análise

  // ── Infraestrutura de planilha ─────────────────────────────────────────────

  function _getMetricsSheet() {
    try {
      var ssId = PropertiesService.getScriptProperties().getProperty(PROP_SS_ID);
      if (!ssId) return null;
      var ss    = SpreadsheetApp.openById(ssId);
      var sheet = ss.getSheetByName(METRICS_SHEET);
      if (!sheet) {
        sheet = ss.insertSheet(METRICS_SHEET);
        sheet.appendRow(REPORT_HEADERS);
        sheet.setFrozenRows(1);
      }
      return sheet;
    } catch (e) {
      Logger.log('[FleetObservabilityService] planilha indisponível: ' + e.message);
      return null;
    }
  }

  function _genId() {
    return 'frpt-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 5);
  }

  // ── Cálculo de KPIs ────────────────────────────────────────────────────────

  function _computeKpis(log, periodDays) {
    try {
      var cutoff   = new Date(Date.now() - periodDays * 86400000);
      var filtered = log.filter(function(r) {
        try { return new Date(r.date) >= cutoff; } catch (_) { return true; }
      });

      var kpi = {
        total: filtered.length,
        success: 0, fallback: 0, fail: 0,
        approved: 0, rejected: 0, draft: 0,
        totalMs: 0
      };

      filtered.forEach(function(r) {
        var s = String(r.status || '');
        if (s === 'ok')       kpi.success++;
        if (s === 'fallback') kpi.fallback++;
        if (s === 'fail')     kpi.fail++;
        var d = String(r.reviewDecision || '');
        if (d === 'approved') kpi.approved++;
        if (d === 'rejected') kpi.rejected++;
        if (d === 'draft')    kpi.draft++;
        kpi.totalMs += Number(r.durationMs) || 0;
      });

      var t = kpi.total || 1; // evita divisão por zero
      return {
        total_calls:    kpi.total,
        success_rate:   Math.round(kpi.success  / t * 1000) / 1000,
        fallback_rate:  Math.round(kpi.fallback / t * 1000) / 1000,
        error_rate:     Math.round(kpi.fail     / t * 1000) / 1000,
        avg_latency_ms: kpi.total ? Math.round(kpi.totalMs / kpi.total) : 0,
        pending_reviews: kpi.draft,
        approved:        kpi.approved,
        rejected:        kpi.rejected
      };
    } catch (error) {
      Logger.log("Erro em _computeKpis: " + error.message);
      throw error;
    }
  }

  // ── API pública ────────────────────────────────────────────────────────────

  /**
   * Retorna resumo de métricas sem gravar — usado por polling ou painel live.
   * @param {{ periodDays?: number }} opts
   * @returns {Object}
   */
  function getMetricsSummary(opts) {
    opts = opts || {};
    var periodDays = opts.periodDays || DEFAULT_PERIOD_DAYS;
    var log = [];
    try {
      if (typeof AiAuditLogService !== 'undefined') log = AiAuditLogService.getLog(500);
    } catch (_) {}

    var kpis = _computeKpis(log, periodDays);

    var rl = {};
    try {
      if (typeof AiRateLimitService !== 'undefined') rl = AiRateLimitService.getMetrics() || {};
    } catch (_) {}

    var health = {};
    try {
      if (typeof FleetHealthCheck !== 'undefined') health = FleetHealthCheck.getStatus();
    } catch (_) {}

    var quotaUsed  = Number(rl.calls        || 0);
    var quotaLimit = Number(rl.dailyQuota   || 0);
    var quotaPct   = quotaLimit > 0 ? Math.round(quotaUsed / quotaLimit * 1000) / 1000 : 0;

    var alerts     = (health.alerts || []);
    var alertCodes = alerts.map(function(a) { return a.code; }).join(';');

    return {
      project:             health.project || 'unknown',
      timestamp:           new Date().toISOString(),
      period_days:         periodDays,
      kpis:                kpis,
      quota:               { used: quotaUsed, limit: quotaLimit, pct: quotaPct },
      frota_layers:        health.frota_layers || {},
      alert_count:         alerts.length,
      alert_codes:         alertCodes,
      alerts:              alerts
    };
  }

  /**
   * Gera relatório, grava na aba Fleet_Metrics e retorna o resumo.
   * Chamada ideal: gatilho de tempo diário.
   * @param {{ periodDays?: number }} opts
   * @returns {Object}
   */
  function generateReport(opts) {
    try {
      try {
        try {
          var summary = getMetricsSummary(opts);
          var id      = _genId();
          var fl      = summary.frota_layers;
          var k       = summary.kpis;
          var q       = summary.quota;

          var row = [
            id,
            summary.project,
            summary.timestamp,
            summary.period_days,
            k.total_calls,
            k.success_rate,
            k.fallback_rate,
            k.error_rate,
            k.avg_latency_ms,
            k.pending_reviews,
            k.approved,
            k.rejected,
            q.used,
            q.limit,
            q.pct,
            fl.active  || 0,
            fl.total   || 0,
            summary.alert_count,
            summary.alert_codes
          ];

          var sheet = _getMetricsSheet();
          if (sheet) {
            sheet.appendRow(row);
            Logger.log('[FleetObservabilityService] relatório gravado id=' + id + ' project=' + summary.project);
          } else {
            Logger.log('[FleetObservabilityService] planilha indisponível — relatório não gravado.');
          }

          return { reportId: id, summary: summary };
        } catch (error) {
          Logger.log("Erro em generateReport: " + error.message);
          throw error; // Re-lança para tratamento superior
        }
      } catch (error) {
        Logger.log("Erro em generateReport: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em generateReport: " + error.message);
      throw error;
    }
  }

  return { generateReport: generateReport, getMetricsSummary: getMetricsSummary };
})();

/** Acionado por gatilho de tempo ou execução manual. */
function runFleetObservabilityReport() {
  var result = FleetObservabilityService.generateReport({ periodDays: 7 });
  var k = result.summary.kpis;
  Logger.log('=== Fleet Observability Report — ' + result.summary.project + ' ===');
  Logger.log('  Chamadas (7d): ' + k.total_calls);
  Logger.log('  Taxa sucesso:  ' + Math.round(k.success_rate  * 100) + '%');
  Logger.log('  Taxa fallback: ' + Math.round(k.fallback_rate * 100) + '%');
  Logger.log('  Taxa falha:    ' + Math.round(k.error_rate    * 100) + '%');
  Logger.log('  Latência média:' + k.avg_latency_ms + 'ms');
  Logger.log('  Rascunhos pendentes: ' + k.pending_reviews);
  Logger.log('  Quota: ' + result.summary.quota.used + '/' + result.summary.quota.limit +
    ' (' + Math.round(result.summary.quota.pct * 100) + '%)');
  if (result.summary.alert_count > 0) {
    Logger.log('  Alertas: ' + result.summary.alert_codes);
  }
  Logger.log('  Report ID: ' + result.reportId);
  return result;
}
