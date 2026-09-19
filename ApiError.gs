/**
 * FROTA-12: contrato unico para erros publicos de API.
 */
var ApiError = (function () {
  'use strict';

  var SAFE_MESSAGES = {
    400: 'Requisicao invalida.',
    401: 'Autenticacao necessaria.',
    403: 'Permissao insuficiente.',
    404: 'Recurso nao encontrado.',
    409: 'Conflito ao processar a requisicao.',
    422: 'Dados semanticamente invalidos.',
    429: 'Limite de requisicoes excedido.',
    500: 'Erro interno ao processar a requisicao.',
    503: 'Servico temporariamente indisponivel.'
  };

  function create(status, message, code, details, requestId) {
    try {
      status = normalizeStatus_(status);
      var response = {
        ok: false,
        status: status,
        statusCode: status,
        error: {
          code: code || defaultCode_(status),
          message: safeMessage_(status, message)
        }
      };
      if (isSafeDetails_(details)) response.error.details = details;
      if (requestId) response.requestId = String(requestId);
      return response;
    } catch (error) {
      Logger.log("Erro em create: " + error.message);
      throw error;
    }
  }

  function validation(details, message, requestId) {
    return create(
      400,
      message || 'Um ou mais campos sao invalidos.',
      'VALIDATION_ERROR',
      details,
      requestId
    );
  }

  function fromException(error, options) {
    options = options || {};
    logInternal_(error, options);
    return create(
      options.status || 500,
      options.publicMessage || SAFE_MESSAGES[options.status || 500],
      options.code || 'INTERNAL_ERROR',
      null,
      options.requestId
    );
  }

  function safeMessage_(status, message) {
    try {
      if (status >= 500) return SAFE_MESSAGES[status] || SAFE_MESSAGES[500];
      var text = message == null ? SAFE_MESSAGES[status] : String(message);
      if (containsSecret_(text) || containsInternalDetail_(text)) {
        return SAFE_MESSAGES[status] || 'Nao foi possivel processar a requisicao.';
      }
      return text;
    } catch (error) {
      Logger.log("Erro em safeMessage_: " + error.message);
      throw error;
    }
  }

  function isSafeDetails_(details) {
    if (!details || typeof details !== 'object') return false;
    try {
      var serialized = JSON.stringify(details);
      return !containsSecret_(serialized) && !containsInternalDetail_(serialized);
    } catch (ignored) {
      return false;
    }
  }

  function containsSecret_(text) {
    return /(api[_-]?key|authorization|bearer\s|password|senha|secret|token\s*[:=])/i.test(text);
  }

  function containsInternalDetail_(text) {
    return /(\bat\s+\S+\s*\(|stack trace|scriptapp|spreadsheetapp|utilities\.|urlfetchapp)/i.test(text);
  }

  function logInternal_(error, options) {
    var record = {
      code: options.code || 'INTERNAL_ERROR',
      requestId: options.requestId || null,
      message: error && error.message ? String(error.message) : String(error),
      stack: error && error.stack ? String(error.stack) : null
    };
    if (typeof options.logger === 'function') {
      options.logger(record);
    } else if (typeof StructuredLogService !== 'undefined' &&
               StructuredLogService.error) {
      StructuredLogService.error('api.exception', record);
    }
  }

  function normalizeStatus_(status) {
    try {
      status = Number(status);
      return status >= 400 && status <= 599 ? status : 500;
    } catch (error) {
      Logger.log("Erro em normalizeStatus_: " + error.message);
      throw error;
    }
  }

  function defaultCode_(status) {
    var codes = {
      400: 'BAD_REQUEST',
      401: 'UNAUTHORIZED',
      403: 'FORBIDDEN',
      404: 'NOT_FOUND',
      409: 'CONFLICT',
      422: 'UNPROCESSABLE_ENTITY',
      429: 'RATE_LIMITED',
      500: 'INTERNAL_ERROR',
      503: 'SERVICE_UNAVAILABLE'
    };
    return codes[status] || 'API_ERROR';
  }

  return {
    create: create,
    validation: validation,
    fromException: fromException
  };
}());
