/**
 * FleetHealthCheck.gs — FROTA-10: Health check por projeto
 *
 * Expõe o estado de cada camada de governança do projeto sem revelar segredos,
 * conteúdo gerado, dados pessoais ou tokens de API.
 *
 * Uso:
 *   runFleetHealthCheck()         — executa e exibe no Logger
 *   FleetHealthCheck.getStatus()  — retorna objeto estruturado para painel
 *
 * Nenhum dado sensível é exposto: apenas presença de configuração (boolean),
 * contagens numéricas e timestamps.
 *
 * Campos do resultado:
 *   project       — nome inferido da Script Property FLEET_PROJECT_NAME
 *   timestamp     — ISO 8601 UTC
 *   checks        — boolean por camada de governança
 *   metrics       — contagens do dia (sem conteúdo)
 *   alerts        — lista de alertas com severidade 'warn' | 'error'
 *   frota_layers  — mapa de quais FROTAs estão ativas neste projeto
 */

'use strict';

var FleetHealthCheck = (function () {

  // ── Limiares configuráveis via Script Properties ────────────────────────────
  var DEFAULT_ERROR_RATE_WARN   = 0.20;  // 20% de falhas
  var DEFAULT_QUOTA_PCT_WARN    = 0.80;  // 80% da quota diária
  var DEFAULT_PENDING_REV_WARN  = 10;    // rascunhos sem revisão

  function _cfg() {
    try {
      var p = PropertiesService.getScriptProperties().getProperties();
      return {
        errorRateWarn:  parseFloat(p.FLEET_ERROR_RATE_ALERT  || String(DEFAULT_ERROR_RATE_WARN)),
        quotaPctWarn:   parseFloat(p.FLEET_QUOTA_PCT_ALERT   || String(DEFAULT_QUOTA_PCT_WARN)),
        pendingRevWarn: parseInt(  p.FLEET_PENDING_REV_ALERT || String(DEFAULT_PENDING_REV_WARN), 10)
      };
    } catch (_) {
      return { errorRateWarn: DEFAULT_ERROR_RATE_WARN, quotaPctWarn: DEFAULT_QUOTA_PCT_WARN, pendingRevWarn: DEFAULT_PENDING_REV_WARN };
    }
  }

  function _projectName() {
    try { return PropertiesService.getScriptProperties().getProperty('FLEET_PROJECT_NAME') || 'unknown'; }
    catch (_) { return 'unknown'; }
  }

  // ── Verificações de presença de camadas ────────────────────────────────────

  function _checks() {
    return {
      gemini_key_configured:       _hasKey('GEMINI_API_KEY'),
      frota03_human_review:        _svcAvail('HumanReviewService', 'createDraft'),
      frota05_rate_limit:          _svcAvail('AiRateLimitService', 'check'),
      frota06_audit_log:           _svcAvail('AiAuditLogService',  'record'),
      frota07_model_config:        _svcAvail('GeminiModelConfig',  'getModel'),
      frota08_contract_test:       typeof runAiContractTest === 'function',
      frota09_consent:             _svcAvail('ConsentService',     'check'),
      frota10_observability:       true  // este arquivo está presente
    };
  }

  function _hasKey(propName) {
    try {
      try {
        var v = PropertiesService.getScriptProperties().getProperty(propName);
        return !!(v && v.trim().length > 0);
      } catch (_) { return false; }
    } catch (error) {
      Logger.log("Erro em _hasKey: " + error.message);
      throw error;
    }
  }

  function _svcAvail(svcName, methodName) {
    try {
      var svc = (typeof this !== 'undefined' && this[svcName]) || (typeof globalThis !== 'undefined' && globalThis[svcName]);
      // Apps Script: globals are accessible via eval in a pinch, but typeof is safer
      var g = (function(n) {
        try { return eval(n); } catch (_) { return undefined; } // eslint-disable-line no-eval
      })(svcName);
      return typeof g !== 'undefined' && typeof g[methodName] === 'function';
    } catch (_) { return false; }
  }

  // ── Métricas do dia (sem conteúdo) ────────────────────────────────────────

  function _metricsFromAuditLog() {
    try {
      var m = { total: 0, success: 0, fallback: 0, fail: 0,
                approved: 0, rejected: 0, draft: 0, avg_latency_ms: 0 };
      try {
        if (typeof AiAuditLogService === 'undefined') return m;
        var log = AiAuditLogService.getLog(200);
        if (!log || !log.length) return m;
        m.total = log.length;
        var totalMs = 0;
        log.forEach(function (r) {
          var s = String(r.status || '');
          if (s === 'ok')       m.success++;
          if (s === 'fallback') m.fallback++;
          if (s === 'fail')     m.fail++;
          var d = String(r.reviewDecision || '');
          if (d === 'approved') m.approved++;
          if (d === 'rejected') m.rejected++;
          if (d === 'draft')    m.draft++;
          totalMs += Number(r.durationMs) || 0;
        });
        m.avg_latency_ms = m.total ? Math.round(totalMs / m.total) : 0;
      } catch (_) {}
      return m;
    } catch (error) {
      Logger.log("Erro em _metricsFromAuditLog: " + error.message);
      throw error;
    }
  }

  function _metricsFromRateLimit() {
    try {
      if (typeof AiRateLimitService === 'undefined') return {};
      return AiRateLimitService.getMetrics();
    } catch (_) { return {}; }
  }

  // ── Geração de alertas ─────────────────────────────────────────────────────

  function _buildAlerts(chk, auditM, rlM, cfg) {
    try {
      var alerts = [];

      function warn(code, msg)  { alerts.push({ severity: 'warn',  code: code, message: msg }); }
      function error_(code, msg) { alerts.push({ severity: 'error', code: code, message: msg }); }

      if (!chk.gemini_key_configured)  warn('NO_API_KEY',       'GEMINI_API_KEY não configurada.');
      if (!chk.frota03_human_review)   warn('NO_HUMAN_REVIEW',  'FROTA-03: HumanReviewService ausente.');
      if (!chk.frota05_rate_limit)     warn('NO_RATE_LIMIT',    'FROTA-05: AiRateLimitService ausente.');
      if (!chk.frota06_audit_log)      warn('NO_AUDIT_LOG',     'FROTA-06: AiAuditLogService ausente.');
      if (!chk.frota07_model_config)   warn('NO_MODEL_CONFIG',  'FROTA-07: GeminiModelConfig ausente.');
      if (!chk.frota09_consent)        warn('NO_CONSENT',       'FROTA-09: ConsentService ausente.');

      if (auditM.total > 0) {
        var errorRate = auditM.fail / auditM.total;
        if (errorRate >= cfg.errorRateWarn) {
          error_('HIGH_ERROR_RATE',
            'Taxa de falha ' + Math.round(errorRate * 100) + '% ≥ ' + Math.round(cfg.errorRateWarn * 100) + '% (limiar).');
        }
        if (auditM.draft >= cfg.pendingRevWarn) {
          warn('PENDING_REVIEWS',
            auditM.draft + ' rascunhos aguardando revisão humana (limiar: ' + cfg.pendingRevWarn + ').');
        }
      }

      if (rlM && typeof rlM.dailyQuota === 'number' && typeof rlM.calls === 'number' && rlM.dailyQuota > 0) {
        var quotaPct = rlM.calls / rlM.dailyQuota;
        if (quotaPct >= cfg.quotaPctWarn) {
          warn('QUOTA_HIGH',
            Math.round(quotaPct * 100) + '% da quota diária consumida (' + rlM.calls + '/' + rlM.dailyQuota + ').');
        }
      }

      return alerts;
    } catch (error) {
      Logger.log("Erro em _buildAlerts: " + error.message);
      throw error;
    }
  }

  // ── API pública ────────────────────────────────────────────────────────────

  /**
   * Retorna o status de saúde do projeto — nenhum dado sensível incluído.
   * @returns {{ project, timestamp, ok, checks, metrics, alerts, frota_layers }}
   */
  function getStatus() {
    try {
      var cfg    = _cfg();
      var chk    = _checks();
      var auditM = _metricsFromAuditLog();
      var rlM    = _metricsFromRateLimit();
      var alerts = _buildAlerts(chk, auditM, rlM, cfg);

      var layerCount = Object.keys(chk).filter(function(k) { return k.startsWith('frota') && chk[k]; }).length;

      return {
        project:     _projectName(),
        timestamp:   new Date().toISOString(),
        ok:          alerts.filter(function(a) { return a.severity === 'error'; }).length === 0,
        checks:      chk,
        metrics: {
          audit:      auditM,
          rate_limit: rlM
        },
        alerts:      alerts,
        frota_layers: {
          active:  layerCount,
          total:   Object.keys(chk).filter(function(k) { return k.startsWith('frota'); }).length,
          details: chk
        }
      };
    } catch (error) {
      Logger.log("Erro em getStatus: " + error.message);
      throw error;
    }
  }

  return { getStatus: getStatus };
})();

/**
 * Ponto de entrada para execução manual (Run → runFleetHealthCheck).
 * Imprime no Logger sem expor conteúdo sensível.
 */
function runFleetHealthCheck() {
  try {
    var status = FleetHealthCheck.getStatus();
    Logger.log('=== Fleet Health Check — ' + status.project + ' ===');
    Logger.log('Timestamp: ' + status.timestamp);
    Logger.log('Status geral: ' + (status.ok ? 'OK' : 'ALERTAS'));

    Logger.log('\n-- Camadas de governança (' + status.frota_layers.active + '/' + status.frota_layers.total + ' ativas) --');
    Object.keys(status.checks).forEach(function(k) {
      Logger.log('  ' + (status.checks[k] ? '[OK]  ' : '[---] ') + k);
    });

    Logger.log('\n-- Métricas do log de auditoria --');
    var a = status.metrics.audit;
    Logger.log('  Total chamadas: ' + a.total);
    if (a.total > 0) {
      Logger.log('  Sucesso:   ' + a.success + ' (' + Math.round(a.success / a.total * 100) + '%)');
      Logger.log('  Fallback:  ' + a.fallback);
      Logger.log('  Falha:     ' + a.fail);
      Logger.log('  Latência média: ' + a.avg_latency_ms + 'ms');
      Logger.log('  Aprovados: ' + a.approved + ' | Rejeitados: ' + a.rejected + ' | Rascunhos: ' + a.draft);
    }

    var rl = status.metrics.rate_limit;
    if (rl && typeof rl.calls === 'number') {
      Logger.log('\n-- Rate limit (hoje) --');
      Logger.log('  Chamadas: ' + rl.calls + ' / ' + (rl.dailyQuota || '?'));
      if (rl.rateLimitHits) Logger.log('  Bloqueios por rate limit: ' + rl.rateLimitHits);
    }

    if (status.alerts.length > 0) {
      Logger.log('\n-- Alertas --');
      status.alerts.forEach(function(al) {
        Logger.log('  [' + al.severity.toUpperCase() + '] ' + al.code + ': ' + al.message);
      });
    } else {
      Logger.log('\nNenhum alerta ativo.');
    }

    Logger.log('\n=== Fim do health check ===');
    return status;
  } catch (error) {
    Logger.log("Erro em runFleetHealthCheck: " + error.message);
    throw error;
  }
}
