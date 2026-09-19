/**
 * FROTA-14: gateway unico para APIs externas.
 */
var ExternalApiClient = (function () {
  'use strict';
  var adapters_ = {};
  var defaults_ = {
    maxAttempts: 3,
    baseDelayMs: 500,
    maxDelayMs: 8000,
    cacheSeconds: 300
  };

  function configure(options) {
    try {
      options = options || {};
      adapters_ = options.adapters || adapters_;
      Object.keys(defaults_).forEach(function (key) {
        if (options[key] != null) defaults_[key] = options[key];
      });
      return api;
    } catch (error) {
      Logger.log("Erro em configure: " + error.message);
      throw error;
    }
  }

  function request(options) {
    options = options || {};
    var started = now_();
    var operation = options.operation || 'external.request';
    var cached = readCache_(options.cacheKey);
    if (cached) return cached;
    enforceRateLimit_(options);

    var lastError;
    for (var attempt = 1; attempt <= (options.maxAttempts || defaults_.maxAttempts); attempt++) {
      try {
        var response = fetch_(options.url, buildFetchOptions_(options));
        var status = Number(response.getResponseCode());
        var body = response.getContentText();
        if (status >= 200 && status < 300) {
          var normalized = normalize_(status, body, options.normalize);
          writeCache_(options.cacheKey, normalized, options.cacheSeconds);
          monitor_(operation, status, started, attempt, null);
          return normalized;
        }
        lastError = { status: status, code: 'UPSTREAM_HTTP_' + status };
        if (!isRetryable_(status) || attempt === (options.maxAttempts || defaults_.maxAttempts)) break;
      } catch (error) {
        lastError = { status: 503, code: 'UPSTREAM_NETWORK_ERROR' };
        if (attempt === (options.maxAttempts || defaults_.maxAttempts)) break;
      }
      sleep_(backoff_(attempt, options));
    }

    monitor_(operation, lastError.status, started, null, lastError.code);
    if (typeof options.fallback === 'function') {
      return { ok: true, status: 200, source: 'fallback', data: options.fallback(lastError) };
    }
    return ApiError.create(503, 'Servico externo temporariamente indisponivel.',
      lastError.code);
  }

  function buildFetchOptions_(options) {
    try {
      try {
        var headers = {};
        Object.keys(options.headers || {}).forEach(function (key) {
          headers[key] = options.headers[key];
        });
        if (options.secretProperty) {
          var secret = property_(options.secretProperty);
          if (!secret) throw new Error('EXTERNAL_API_SECRET_NOT_CONFIGURED');
          headers[options.secretHeader || 'Authorization'] =
            (options.secretPrefix == null ? 'Bearer ' : options.secretPrefix) + secret;
        }
        return {
          method: options.method || 'post',
          contentType: options.contentType || 'application/json',
          headers: headers,
          payload: options.payload == null ? undefined :
            (typeof options.payload === 'string' ? options.payload : JSON.stringify(options.payload)),
          muteHttpExceptions: true
        };
      } catch (error) {
        Logger.log("Erro em buildFetchOptions_: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em buildFetchOptions_: " + error.message);
      throw error;
    }
  }

  function normalize_(status, body, normalizer) {
    try {
      try {
        var parsed;
        try { parsed = JSON.parse(body); } catch (ignored) { parsed = body; }
        var data = typeof normalizer === 'function' ? normalizer(parsed) : parsed;
        return { ok: true, status: status, source: 'external', data: data };
      } catch (error) {
        Logger.log("Erro em normalize_: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em normalize_: " + error.message);
      throw error;
    }
  }

  function isRetryable_(status) {
    return status === 408 || status === 429 || status >= 500;
  }

  function backoff_(attempt, options) {
    var base = options.baseDelayMs || defaults_.baseDelayMs;
    var cap = options.maxDelayMs || defaults_.maxDelayMs;
    var exponential = Math.min(cap, base * Math.pow(2, attempt - 1));
    return Math.floor(exponential / 2 + random_() * exponential / 2);
  }

  function monitor_(operation, status, started, attempt, errorCode) {
    if (typeof StructuredLogService === 'undefined') return;
    StructuredLogService.write(errorCode ? 'ERROR' : 'INFO', operation, {
      providerStatus: status,
      durationMs: now_() - started,
      attempt: attempt,
      errorCode: errorCode
    });
  }

  function enforceRateLimit_(options) {
    if (!options.rateLimitKey || typeof AiRateLimitService === 'undefined') return;
    var decision = AiRateLimitService.check(options.rateLimitKey, options.rateLimit || {});
    if (decision && decision.allowed === false) throw new Error('EXTERNAL_API_RATE_LIMITED');
  }

  function readCache_(key) {
    try {
      if (!key) return null;
      var value = cache_().get(key);
      if (!value) return null;
      try { return JSON.parse(value); } catch (ignored) { return null; }
    } catch (error) {
      Logger.log("Erro em readCache_: " + error.message);
      throw error;
    }
  }

  function writeCache_(key, value, seconds) {
    try {
      try {
        if (key) cache_().put(key, JSON.stringify(value), seconds || defaults_.cacheSeconds);
      } catch (error) {
        Logger.log("Erro em writeCache_: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em writeCache_: " + error.message);
      throw error;
    }
  }

  function fetch_(url, options) {
    try {
      return adapters_.fetch ? adapters_.fetch(url, options) : UrlFetchApp.fetch(url, options);
    } catch (error) {
      Logger.log("Erro em fetch_: " + error.message);
      throw error; // Re-lança para tratamento superior
    }
  }
  function property_(name) {
    try {
      return adapters_.getProperty ? adapters_.getProperty(name) :
        PropertiesService.getScriptProperties().getProperty(name);
    } catch (error) {
      Logger.log("Erro em property_: " + error.message);
      throw error;
    }
  }
  function cache_() {
    try {
      return adapters_.cache || CacheService.getScriptCache();
    } catch (error) {
      Logger.log("Erro em cache_: " + error.message);
      throw error;
    }
  }
  function sleep_(ms) {
    return adapters_.sleep ? adapters_.sleep(ms) : Utilities.sleep(ms);
  }
  function random_() { return adapters_.random ? adapters_.random() : Math.random(); }
  function now_() { return adapters_.now ? adapters_.now() : new Date().getTime(); }

  var api = { configure: configure, request: request };
  return api;
}());

