/**
 * codex-backend-standard
 * Camada pequena e aditiva para respostas, erros, validacao e healthcheck.
 * Nao substitui servicos existentes; apenas oferece um contrato comum para novas chamadas.
 */
var CodexBackend = (function () {
  'use strict';

  function now_() {
    try {
      return new Date().toISOString();
    } catch (error) {
      Logger.log("Erro em now_: " + error.message);
      throw error;
    }
  }

  function hasOwn_(obj, key) {
    return Object.prototype.hasOwnProperty.call(obj || {}, key);
  }

  function safeMessage_(error, fallback) {
    try {
      var message = fallback || 'Nao foi possivel concluir. Tente novamente.';
      if (!error) return message;
      var raw = error.message ? String(error.message) : String(error);
      if (/(api[_-]?key|authorization|bearer\s|password|senha|secret|token\s*[:=]|stack trace|scriptapp|spreadsheetapp|urlfetchapp)/i.test(raw)) {
        return message;
      }
      return raw || message;
    } catch (error) {
      Logger.log("Erro em safeMessage_: " + error.message);
      throw error;
    }
  }

  function ok(data, meta) {
    if (typeof StandardReturn !== 'undefined' && StandardReturn.ok) {
      return StandardReturn.ok(data, meta);
    }
    return {
      success: true,
      data: data === undefined ? null : data,
      error: null,
      message: 'Operacao concluida.',
      code: 'OK',
      meta: meta || { generatedAt: now_() }
    };
  }

  function fail(error, code, meta) {
    var message = safeMessage_(error);
    if (typeof StandardReturn !== 'undefined' && StandardReturn.fail) {
      var result = StandardReturn.fail(message, null, meta || { code: code || 'BACKEND_ERROR' });
      result.message = message;
      result.code = code || 'BACKEND_ERROR';
      return result;
    }
    return {
      success: false,
      data: null,
      error: message,
      message: message,
      code: code || 'BACKEND_ERROR',
      meta: meta || { generatedAt: now_() }
    };
  }

  function normalize(value) {
    if (typeof StandardReturn !== 'undefined' && StandardReturn.normalize) {
      return StandardReturn.normalize(value);
    }
    if (value && typeof value === 'object' && typeof value.success === 'boolean') return value;
    if (value && typeof value === 'object' && typeof value.ok === 'boolean') {
      return value.ok ? ok(hasOwn_(value, 'data') ? value.data : value) : fail(value.error || value.message, value.code || 'BACKEND_ERROR');
    }
    return ok(value === undefined ? null : value);
  }

  function wrap(fn, options) {
    options = options || {};
    try {
      return normalize(fn());
    } catch (error) {
      logError_(error, options);
      if (typeof ApiError !== 'undefined' && ApiError.fromException && options.apiError === true) {
        return ApiError.fromException(error, {
          code: options.code || 'BACKEND_ERROR',
          publicMessage: options.publicMessage || 'Nao foi possivel concluir. Tente novamente.'
        });
      }
      return fail(error, options.code || 'BACKEND_ERROR', { generatedAt: now_() });
    }
  }

  function validate(input, schema) {
    if (typeof RequestValidator !== 'undefined' && RequestValidator.validate) {
      return RequestValidator.validate(input, schema || {});
    }
    var errors = [];
    var required = schema && schema.required;
    if (Object.prototype.toString.call(required) === '[object Array]') {
      required.forEach(function (field) {
        if (!input || input[field] === undefined || input[field] === null || input[field] === '') {
          errors.push({ field: '$.' + field, rule: 'required', message: 'Campo obrigatorio.' });
        }
      });
    }
    return { valid: errors.length === 0, errors: errors };
  }

  function validateOrThrow(input, schema) {
    var result = validate(input, schema);
    if (!result.valid) {
      var error = new Error('Dados invalidos.');
      error.code = 'VALIDATION_ERROR';
      error.details = result.errors;
      throw error;
    }
    return input;
  }

  function health(extra) {
    var base = {
      status: 'ok',
      service: 'backend',
      standard: 'codex-backend-standard',
      timestamp: now_(),
      hasStandardReturn: typeof StandardReturn !== 'undefined',
      hasRequestValidator: typeof RequestValidator !== 'undefined',
      hasApiError: typeof ApiError !== 'undefined',
      hasFleetHealthCheck: typeof FleetHealthCheck !== 'undefined'
    };
    if (typeof FleetHealthCheck !== 'undefined' && FleetHealthCheck.getStatus) {
      try { base.fleet = FleetHealthCheck.getStatus(); } catch (ignored) {}
    }
    Object.keys(extra || {}).forEach(function (key) { base[key] = extra[key]; });
    return ok(base, { code: 'HEALTH_OK', generatedAt: now_() });
  }

  function logError_(error, options) {
    try {
      var record = {
        code: options.code || 'BACKEND_ERROR',
        message: error && error.message ? String(error.message) : String(error),
        timestamp: now_()
      };
      try {
        if (typeof StructuredLogService !== 'undefined' && StructuredLogService.error) {
          StructuredLogService.error('codex.backend.exception', record);
        } else if (typeof Logger !== 'undefined' && Logger.log) {
          Logger.log('[codex.backend.exception] ' + JSON.stringify(record));
        }
      } catch (ignored) {}
    } catch (error) {
      Logger.log("Erro em logError_: " + error.message);
      throw error;
    }
  }

  return {
    ok: ok,
    fail: fail,
    normalize: normalize,
    wrap: wrap,
    validate: validate,
    validateOrThrow: validateOrThrow,
    health: health
  };
}());

function codexBackendOk(data, meta) {
  return CodexBackend.ok(data, meta);
}

function codexBackendFail(error, code, meta) {
  return CodexBackend.fail(error, code, meta);
}

function codexBackendWrap(fn, options) {
  return CodexBackend.wrap(fn, options);
}

function codexBackendValidate(input, schema) {
  return CodexBackend.validate(input, schema);
}

function codexBackendHealthCheck() {
  return CodexBackend.health();
}

function codexBackendPing() {
  return CodexBackend.health({ ping: true });
}
