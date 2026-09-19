/**
 * @file StandardReturn.gs
 * @description Define o contrato canônico de retorno de API para a comunicação entre frontend e backend.
 * Toda função de backend exposta ao frontend deve retornar este envelope.
 *
 * English Description:
 * This module defines the canonical API return envelope used for communication between the frontend client
 * and the Google Apps Script backend. Every public backend function exposed to the client wraps its output
 * in this structure, providing uniform fields for status flags, data payloads, error details, and metadata.
 * This pattern ensures robust client-side error handling and response parsing.
 */

var StandardReturn = (function() {
  function buildMeta_(meta) {
    try {
      var baseMeta = {
        generatedAt: new Date().toISOString()
      };
      if (meta && typeof meta === 'object') {
        return Object.assign(baseMeta, meta);
      }
      return baseMeta;
    } catch (error) {
      Logger.log("Erro em buildMeta_: " + error.message);
      throw error;
    }
  }

  function errorMessage_(error) {
    try {
      if (!error) return "Erro desconhecido.";
      if (error instanceof Error) return error.message;
      if (typeof error === 'string') return error;
      if (typeof error === 'object') {
        if (error.message) return String(error.message);
        if (error.error) return errorMessage_(error.error);
        return JSON.stringify(error);
      }
      return String(error);
    } catch (error) {
      Logger.log("Erro em errorMessage_: " + error.message);
      throw error;
    }
  }

  function ok(data, meta) {
    return {
      success: true,
      data: data === undefined ? null : data,
      error: null,
      meta: buildMeta_(meta)
    };
  }

  function fail(error, data, meta) {
    return {
      success: false,
      data: data === undefined ? null : data,
      error: errorMessage_(error),
      meta: buildMeta_(meta)
    };
  }

  function isEnvelope(value) {
    if (!value || typeof value !== 'object') return false;
    return ('success' in value) && ('data' in value) && ('error' in value) && ('meta' in value);
  }

  function normalize(value) {
    if (isEnvelope(value)) {
      return value;
    }

    if (!value || typeof value !== 'object') {
      return ok(value);
    }

    if ('success' in value && ('message' in value || 'code' in value || 'data' in value)) {
      if (value.success) {
        return ok(value.data, { code: value.code, message: value.message, traceId: value.traceId });
      } else {
        return fail(value.message || value.code || "Erro no processamento.", value.data, { code: value.code, traceId: value.traceId });
      }
    }

    if ('ok' in value) {
      if (value.ok) {
        return ok(value.payload || value.data || value);
      } else {
        return fail(value.error || value.message || "Erro no processamento.");
      }
    }

    if ('statusCode' in value || 'status' in value) {
      var isSuccess = (value.statusCode >= 200 && value.statusCode < 300) || value.status === 'success' || value.status === 'ok' || value.success === true;
      if (isSuccess) {
        return ok(value.payload || value.data || value);
      } else {
        return fail(value.error || value.message || "Erro no processamento: " + (value.status || value.statusCode));
      }
    }

    return ok(value);
  }

  return {
    ok: ok,
    fail: fail,
    isEnvelope: isEnvelope,
    normalize: normalize
  };
})();
