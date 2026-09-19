/**
 * AiRateLimitService.gs — FROTA-05: Rate limit, quota e controle de custo
 *
 * Controla chamadas ao provedor de IA antes de atingir a API:
 *   - Limite por usuário (sliding window, 1 min, configurável)
 *   - Limite por função/useCase (sliding window, 1 min, configurável)
 *   - Teto diário global configurável; resposta específica para quota excedida
 *   - Deduplicação por hash do contexto anonimizado (sem conteúdo pessoal)
 *   - Sem retry em erros permanentes; backoff exponencial para 429 / 5xx
 *   - Métricas agregadas de uso e falha (sem conteúdo pessoal)
 *
 * Uso mínimo (acrescenta 1 linha no topo do ponto de entrada Gemini):
 *   AiRateLimitService.check('nomeUseCase', prompt);
 *
 * Uso completo (guarda + dedup + métricas automáticas):
 *   return AiRateLimitService.wrap(function() {
 *     return <chamada existente ao Gemini>;
 *   }, 'nomeUseCase', prompt);
 *
 * Configuração via PropertiesService (Script Properties):
 *   AI_RATE_USER_PER_MIN   — máx chamadas por usuário/useCase por min (padrão 10)
 *   AI_RATE_FN_PER_MIN     — máx chamadas por useCase por min (padrão 30)
 *   AI_QUOTA_DAILY         — teto diário global de chamadas (padrão 200)
 */

'use strict';

var AiRateLimitService = (function () {

  // ── Configuração (sobrescrível via Script Properties) ────────────────────
  function _cfg() {
    var raw = {};
    try {
      var p = PropertiesService.getScriptProperties().getProperties();
      if (p.AI_RATE_USER_PER_MIN) raw.userPerMin  = parseInt(p.AI_RATE_USER_PER_MIN,  10);
      if (p.AI_RATE_FN_PER_MIN)   raw.fnPerMin    = parseInt(p.AI_RATE_FN_PER_MIN,    10);
      if (p.AI_QUOTA_DAILY)       raw.dailyQuota  = parseInt(p.AI_QUOTA_DAILY,        10);
    } catch (_) {}
    return {
      userPerMin:  raw.userPerMin  || 10,
      fnPerMin:    raw.fnPerMin    || 30,
      dailyQuota:  raw.dailyQuota  || 200,
      dedupTtlSec: 300,
      retryMax:    3,
      retryBaseMs: 800,
      retryMaxMs:  20000
    };
  }

  // ── Cache de curta duração e estado diário persistente ──────────────────
  var _MAX_CACHE_TTL_SECONDS = 21600;
  var _STATE_PROPERTY_KEY = 'ai5_state_v1';

  function _stateKind_(key) {
    var value = String(key || '');
    if (value.indexOf('ai5_day_') === 0) return 'daily';
    if (value.indexOf('ai5_met_') === 0) return 'metrics';
    return '';
  }

  function _stateRead_(key) {
    var kind = _stateKind_(key);
    if (!kind || typeof PropertiesService === 'undefined') return null;
    try {
      var raw = PropertiesService.getScriptProperties()
        .getProperty(_STATE_PROPERTY_KEY + ':' + kind);
      if (!raw) return null;
      var state = JSON.parse(raw);
      var date = String(key).slice(8);
      return state && Object.prototype.hasOwnProperty.call(state, date)
        ? String(state[date])
        : null;
    } catch (_) {
      return null;
    }
  }

  function _stateWrite_(key, value) {
    var kind = _stateKind_(key);
    if (!kind || typeof PropertiesService === 'undefined') return false;
    try {
      var properties = PropertiesService.getScriptProperties();
      var propertyKey = _STATE_PROPERTY_KEY + ':' + kind;
      var raw = properties.getProperty(propertyKey);
      var state = {};
      if (raw) {
        try {
          var parsed = JSON.parse(raw);
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) state = parsed;
        } catch (_) {}
      }
      state[String(key).slice(8)] = String(value);
      var dates = Object.keys(state).sort();
      dates.slice(0, -8).forEach(function (oldDate) { delete state[oldDate]; });
      properties.setProperty(propertyKey, JSON.stringify(state));
      return true;
    } catch (_) {
      return false;
    }
  }

  function _cacheTtl_(ttl) {
    var seconds = Number(ttl);
    if (!isFinite(seconds) || seconds <= 0) seconds = 600;
    return Math.min(Math.floor(seconds), _MAX_CACHE_TTL_SECONDS);
  }
  function _get(k) {
    try {
      var persisted = _stateRead_(k);
      if (persisted !== null) return persisted;
      return CacheService.getScriptCache().get(k);
    } catch (_) {
      return null;
    }
  }
  function _put(k, v, ttl) {
    try {
      if (_stateWrite_(k, v)) return;
      CacheService.getScriptCache().put(k, String(v), _cacheTtl_(ttl));
    } catch (_) {}
  }
  function _inc(k, ttl)    { var v = parseInt(_get(k) || '0', 10) + 1; _put(k, v, ttl); return v; }

  // ── Identificador anônimo do usuário (hash; nunca persiste o email) ──────
  function _uid() {
    try {
      try {
        var e = Session.getActiveUser().getEmail() || 'anon';
        var h = 0;
        for (var i = 0; i < e.length; i++) h = (Math.imul(31, h) + e.charCodeAt(i)) | 0;
        return 'u' + Math.abs(h).toString(36);
      } catch (_) { return 'anon'; }
    } catch (error) {
      Logger.log("Erro em _uid: " + error.message);
      throw error;
    }
  }

  // ── Sliding-window check (incrementa e verifica) ─────────────────────────
  function _window(key, max, winSec) {
    var n = _inc(key, winSec);
    return n <= max ? { ok: true } : { ok: false, count: n, max: max };
  }

  // ── Data key ─────────────────────────────────────────────────────────────
  function _today() {
    try { return Utilities.formatDate(new Date(), 'America/Sao_Paulo', 'yyyy-MM-dd'); }
    catch (_) { return new Date().toISOString().slice(0, 10); }
  }
  function _dailyKey() { return 'ai5_day_' + _today(); }

  function _checkDaily(limit) {
    try {
      var k = _dailyKey();
      var used = parseInt(_get(k) || '0', 10);
      if (used >= limit) return { ok: false, used: used, limit: limit };
      _put(k, used + 1, _MAX_CACHE_TTL_SECONDS);
      return { ok: true, used: used + 1, limit: limit };
    } catch (error) {
      Logger.log("Erro em _checkDaily: " + error.message);
      throw error;
    }
  }

  function _dailyUsed() { return parseInt(_get(_dailyKey()) || '0', 10); }

  // ── Hash de contexto (para dedup; não armazena o conteúdo real) ──────────
  function _hash(ctx) {
    try {
      var s = String(ctx || '');
      var h = 0;
      for (var i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
      return 'ai5_dd_' + Math.abs(h).toString(36);
    } catch (error) {
      Logger.log("Erro em _hash: " + error.message);
      throw error;
    }
  }

  // ── Métricas agregadas (sem conteúdo pessoal) ────────────────────────────
  function _metricsKey() { return 'ai5_met_' + _today(); }

  function _record(useCase, ok, latMs, rl) {
    try {
      var k = _metricsKey();
      var raw = _get(k);
      var m = raw ? JSON.parse(raw) : { calls: 0, ok: 0, fail: 0, rl: 0, latMs: 0, uc: {} };
      m.calls++; if (ok) m.ok++; else m.fail++; if (rl) m.rl++;
      m.latMs += latMs || 0;
      m.uc[useCase] = (m.uc[useCase] || 0) + 1;
      _put(k, JSON.stringify(m), _MAX_CACHE_TTL_SECONDS);
    } catch (_) {}
  }

  // ── Erros nomeados ───────────────────────────────────────────────────────
  function AiRateLimitError(msg) { this.message = msg; this.name = 'AiRateLimitError'; this.isAiLimit = true; }
  AiRateLimitError.prototype = Object.create(Error.prototype);

  function AiQuotaError(msg) { this.message = msg; this.name = 'AiQuotaError'; this.isAiLimit = true; }
  AiQuotaError.prototype = Object.create(Error.prototype);

  // ── Backoff ──────────────────────────────────────────────────────────────
  var PERM = ['permission', 'not found', 'invalid', 'unauthorized', 'forbidden', 'bad request'];
  var TRANS = [429, 500, 503];

  function _isPerm(msg) {
    var m = (msg || '').toLowerCase();
    return PERM.some(function(e) { return m.indexOf(e) !== -1; });
  }

  function _isTransient(code) { return TRANS.indexOf(code) !== -1; }

  function _sleep(ms) { try { Utilities.sleep(ms); } catch (_) {} }

  function _backoff(attempt, baseMs, maxMs) {
    var ms = Math.min(baseMs * Math.pow(2, attempt), maxMs);
    return Math.floor(ms + ms * 0.3 * Math.random());
  }

  // ── check(): barreira antes do provedor (não faz retry) ──────────────────
  /**
   * Verifica limites e lança AiRateLimitError ou AiQuotaError se excedido.
   * Chame no topo da função pública que dispara a chamada ao Gemini.
   *
   * @param {string} useCase  Identificador da função/caso de uso (ex.: 'gerarConto').
   * @param {string} [ctx]    Texto do contexto (usado apenas para dedup hash).
   * @returns {{ dedupKey: string|null, dedupHit: boolean, cached: string|null }}
   */
  function check(useCase, ctx) {
    var c   = _cfg();
    var uid = _uid();

    // Dedup (só verifica, não bloqueia)
    var dKey = _hash(ctx || '');
    var cached = _get(dKey);
    if (cached) {
      _record(useCase, true, 0, false);
      return { dedupKey: dKey, dedupHit: true, cached: cached };
    }

    // Limite por usuário×useCase
    var ur = _window('ai5_u_' + uid + '_' + useCase, c.userPerMin, 60);
    if (!ur.ok) {
      _record(useCase, false, 0, true);
      throw new AiRateLimitError(
        'Limite de uso pessoal atingido (' + ur.count + '/' + ur.max + ' por min). Aguarde e tente novamente.');
    }

    // Limite por useCase global
    var fr = _window('ai5_fn_' + useCase, c.fnPerMin, 60);
    if (!fr.ok) {
      _record(useCase, false, 0, true);
      throw new AiRateLimitError(
        'Limite de função atingido (' + fr.count + '/' + fr.max + ' por min). Tente em instantes.');
    }

    // Cota diária
    var dr = _checkDaily(c.dailyQuota);
    if (!dr.ok) {
      _record(useCase, false, 0, true);
      throw new AiQuotaError(
        'Cota diária de IA atingida (' + dr.used + '/' + dr.limit + ' chamadas). Retomada amanhã.');
    }

    return { dedupKey: dKey, dedupHit: false, cached: null };
  }

  // ── recordResult(): salva dedup e métrica após a chamada ─────────────────
  /**
   * Registra o resultado de uma chamada bem-sucedida (dedup + métrica).
   *
   * @param {string} dedupKey  Valor retornado por check().dedupKey.
   * @param {string} result    Texto da resposta do modelo.
   * @param {string} useCase   Mesmo useCase passado a check().
   * @param {number} latMs     Latência em ms.
   */
  function recordResult(dedupKey, result, useCase, latMs) {
    _record(useCase, true, latMs, false);
    if (dedupKey && typeof result === 'string' && result.length < 90000) {
      _put(dedupKey, result, _cfg().dedupTtlSec);
    }
  }

  // ── recordError(): registra falha na métrica ──────────────────────────────
  function recordError(useCase, latMs) {
    _record(useCase, false, latMs, false);
  }

  // ── wrap(): check + call com retry + dedup + métricas ────────────────────
  /**
   * Executa fn() com todos os controles: limites, dedup, retry exponencial
   * (apenas erros transitórios) e métricas.
   *
   * @param {function} fn       Chamada ao Gemini (deve retornar string).
   * @param {string}   useCase  Identificador do caso de uso.
   * @param {string}   [ctx]    Contexto (para dedup hash).
   * @returns {string}
   */
  function wrap(fn, useCase, ctx) {
    var t0  = Date.now();
    var ck  = check(useCase, ctx);
    if (ck.dedupHit) return ck.cached;

    var c = _cfg();
    var lastErr = null;

    for (var attempt = 0; attempt <= c.retryMax; attempt++) {
      try {
        var result = fn();
        recordResult(ck.dedupKey, result, useCase, Date.now() - t0);
        return result;
      } catch (e) {
        lastErr = e;
        var msg = e.message || '';
        if (_isPerm(msg)) break; // erro permanente: não retenta
        var m = msg.match(/HTTP\s+(\d+)/);
        var code = m ? parseInt(m[1], 10) : 0;
        if (code && !_isTransient(code)) break;
        if (attempt >= c.retryMax) break;
        _sleep(_backoff(attempt, c.retryBaseMs, c.retryMaxMs));
      }
    }

    recordError(useCase, Date.now() - t0);
    throw lastErr || new Error('Chamada ao provedor de IA falhou após ' + (c.retryMax + 1) + ' tentativas.');
  }

  // ── getMetrics(): métricas do dia, sem conteúdo pessoal ──────────────────
  /**
   * Retorna métricas agregadas do dia para administração.
   * Não expõe conteúdo de prompts nem identificadores pessoais.
   */
  function getMetrics() {
    try {
      try {
        try {
          var raw = _get(_metricsKey());
          var m   = raw ? JSON.parse(raw) : { calls: 0, ok: 0, fail: 0, rl: 0, latMs: 0, uc: {} };
          return {
            date:        _today(),
            calls:       m.calls,
            ok:          m.ok,
            fail:        m.fail,
            rateLimited: m.rl,
            avgLatencyMs: m.calls > 0 ? Math.round(m.latMs / m.calls) : 0,
            useCases:    m.uc,
            dailyUsed:   _dailyUsed(),
            dailyLimit:  _cfg().dailyQuota
          };
        } catch (_) {
          return { date: _today(), calls: 0, ok: 0, fail: 0, rateLimited: 0, avgLatencyMs: 0, useCases: {}, dailyUsed: 0 };
        }
      } catch (error) {
        Logger.log("Erro em getMetrics: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em getMetrics: " + error.message);
      throw error;
    }
  }

  // ── API pública ───────────────────────────────────────────────────────────
  return {
    check:            check,
    recordResult:     recordResult,
    recordError:      recordError,
    wrap:             wrap,
    getMetrics:       getMetrics,
    AiRateLimitError: AiRateLimitError,
    AiQuotaError:     AiQuotaError,
    // internos expostos para testes
    _uid:    _uid,
    _hash:   _hash,
    _dailyUsed: _dailyUsed
  };
})();

/** Atalho global — administrador chama via doGet/Server Actions para painel. */
function getAiRateLimitMetrics() {
  return AiRateLimitService.getMetrics();
}
