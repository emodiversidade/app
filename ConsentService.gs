/**
 * ConsentService.gs — FROTA-09: Consentimento e finalidade para dados sensíveis
 *
 * Verifica consentimento válido antes de qualquer coleta ou uso em IA de dados
 * biométricos ou clínicos. Bloqueia captura e envio à IA quando consentimento
 * estiver ausente, expirado ou revogado.
 *
 * Finalidades suportadas (independentes e cumulativas):
 *   'collect'    — coleta e armazenamento do dado sensível
 *   'pedagogy'   — uso pedagógico/clínico interno (relatórios, acompanhamento)
 *   'generative' — envio à IA gerativa (Gemini) para geração de conteúdo
 *
 * Armazenamento:
 *   CacheService  — estado rápido (TTL 6h) sem PII
 *   Spreadsheet   — registro persistente (aba "Consentimentos")
 *
 * Nunca armazena: e-mail, nome, CPF, dados biométricos, tokens, prompts.
 * O subjectId deve ser passado já anonimizado (hash) pelo chamador.
 *
 * Critério de aceite (FROTA-09):
 *   ✓ check() lança ConsentError para ausente / expirado / revogado.
 *   ✓ grant() registra propósito, responsável e validade.
 *   ✓ revoke() invalida imediatamente e registra motivo.
 *   ✓ Cada finalidade é verificada independentemente.
 */

'use strict';

// ── Erro de consentimento ─────────────────────────────────────────────────────

function ConsentError(message, subjectHash, purpose, status) {
  this.name    = 'ConsentError';
  this.message = message;
  this.subjectHash = subjectHash;
  this.purpose = purpose;
  this.status  = status; // 'absent' | 'expired' | 'revoked'
  this.isConsentError = true;
  if (Error.captureStackTrace) Error.captureStackTrace(this, ConsentError);
}
ConsentError.prototype = Object.create(Error.prototype);

// ── Módulo principal ──────────────────────────────────────────────────────────

var ConsentService = (function () {

  var SHEET_NAME    = 'Consentimentos';
  var CACHE_TTL_SEC = 6 * 3600; // 6 horas
  var PROP_SS_ID    = 'CONSENT_SPREADSHEET_ID';

  var VALID_PURPOSES = ['collect', 'pedagogy', 'generative'];

  var HEADERS = [
    'consentId', 'subjectHash', 'purpose', 'responsible',
    'grantedAt', 'validUntil', 'status', 'revokedAt', 'revokeReason'
  ];

  // ── Hash anônimo (sem PII) ──────────────────────────────────────────────────

  function _hash(subjectId) {
    try {
      var s = String(subjectId || '').trim();
      if (!s) throw new Error('[ConsentService] Identificador do sujeito obrigatório.');
      var bytes = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, s, Utilities.Charset.UTF_8);
      return 'cs' + bytes.map(function(byte) {
        var value = byte < 0 ? byte + 256 : byte;
        return ('0' + value.toString(16)).slice(-2);
      }).join('').slice(0, 32);
    } catch (error) {
      Logger.log("Erro em _hash: " + error.message);
      throw error;
    }
  }

  // ── Cache ──────────────────────────────────────────────────────────────────

  function _cacheKey(subjectHash, purpose) {
    return 'consent09:' + subjectHash + ':' + purpose;
  }

  function _getCache(subjectHash, purpose) {
    try {
      var raw = CacheService.getScriptCache().get(_cacheKey(subjectHash, purpose));
      return raw ? JSON.parse(raw) : null;
    } catch (_) { return null; }
  }

  function _setCache(subjectHash, purpose, status) {
    try {
      try {
        CacheService.getScriptCache().put(
          _cacheKey(subjectHash, purpose),
          JSON.stringify({ status: status, ts: Date.now() }),
          CACHE_TTL_SEC
        );
      } catch (_) {}
    } catch (error) {
      Logger.log("Erro em _setCache: " + error.message);
      throw error;
    }
  }

  function _clearCache(subjectHash, purpose) {
    try { CacheService.getScriptCache().remove(_cacheKey(subjectHash, purpose)); } catch (_) {}
  }

  // ── Spreadsheet ────────────────────────────────────────────────────────────

  function _getSheet() {
    try {
      try {
        var props = PropertiesService.getScriptProperties();
        var ssId  = props.getProperty(PROP_SS_ID);
        var ss;
        if (ssId) {
          try { ss = SpreadsheetApp.openById(ssId); } catch (_) { ss = null; }
        }
        if (!ss) {
          ss = SpreadsheetApp.create('Consent_Log');
          props.setProperty(PROP_SS_ID, ss.getId());
        }
        var sheet = ss.getSheetByName(SHEET_NAME);
        if (!sheet) {
          sheet = ss.insertSheet(SHEET_NAME);
          sheet.appendRow(HEADERS);
          sheet.setFrozenRows(1);
        }
        return sheet;
      } catch (error) {
        Logger.log("Erro em _getSheet: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em _getSheet: " + error.message);
      throw error;
    }
  }

  function _genId() {
    return 'cst-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 6);
  }

  // ── Leitura persistente (linha mais recente para subjectHash + purpose) ─────

  function _readRecord(subjectHash, purpose) {
    try {
      try {
        try {
          var sheet = _getSheet();
          var data  = sheet.getDataRange().getValues();
          var found = null;
          for (var i = data.length - 1; i >= 1; i--) {
            var row = data[i];
            if (String(row[1]) === subjectHash && String(row[2]) === purpose) {
              found = {
                consentId:   String(row[0]),
                subjectHash: String(row[1]),
                purpose:     String(row[2]),
                responsible: String(row[3]),
                grantedAt:   row[4] ? new Date(row[4]) : null,
                validUntil:  row[5] ? new Date(row[5]) : null,
                status:      row[7] ? 'revoked' : String(row[6] || 'absent'),
                revokedAt:   row[7] ? new Date(row[7]) : null,
                revokeReason:String(row[8] || '')
              };
              // Recalcula status baseado em datas
              if (found.status === 'active' && found.validUntil && found.validUntil < new Date()) {
                found.status = 'expired';
              }
              break;
            }
          }
          return found;
        } catch (e) {
          Logger.log('[ConsentService._readRecord] erro: ' + e.message);
          return null;
        }
      } catch (error) {
        Logger.log("Erro em _readRecord: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em _readRecord: " + error.message);
      throw error;
    }
  }

  // ── API pública ─────────────────────────────────────────────────────────────

  /**
   * Verifica se o sujeito tem consentimento ativo para a finalidade.
   * Lança ConsentError se ausente, expirado ou revogado.
   *
   * @param {string} subjectId  Identificador do sujeito (será anonimizado).
   * @param {string} purpose    'collect' | 'pedagogy' | 'generative'
   * @throws {ConsentError}
   */
  function check(subjectId, purpose) {
    if (VALID_PURPOSES.indexOf(purpose) === -1) {
      throw new Error('[ConsentService] Finalidade inválida: "' + purpose + '".');
    }
    var sh = _hash(subjectId);

    // Fast path: cache
    var cached = _getCache(sh, purpose);
    if (cached) {
      if (cached.status === 'active') return;
      throw new ConsentError(
        '[ConsentService] Consentimento ' + cached.status + ' para finalidade "' + purpose + '".',
        sh, purpose, cached.status
      );
    }

    // Slow path: planilha
    var rec = _readRecord(sh, purpose);
    var status = rec ? rec.status : 'absent';
    _setCache(sh, purpose, status);

    if (status === 'active') return;
    throw new ConsentError(
      '[ConsentService] Consentimento ' + status + ' para finalidade "' + purpose + '".',
      sh, purpose, status
    );
  }

  /**
   * Registra consentimento ativo para o sujeito e finalidade.
   *
   * @param {string} subjectId
   * @param {string} purpose
   * @param {{ validDays?: number, responsible?: string }} opts
   * @returns {{ consentId: string, subjectHash: string, validUntil: Date }}
   */
  function grant(subjectId, purpose, opts) {
    if (VALID_PURPOSES.indexOf(purpose) === -1) {
      throw new Error('[ConsentService] Finalidade inválida: "' + purpose + '".');
    }
    opts = opts || {};
    var responsible = String(opts.responsible || '').trim();
    if (responsible.length < 3 || responsible.length > 120) {
      throw new Error('[ConsentService] Referência do responsável obrigatória (3 a 120 caracteres).');
    }
    var sh         = _hash(subjectId);
    var now        = new Date();
    var days       = Number(opts.validDays == null ? 365 : opts.validDays);
    if (!isFinite(days) || days < 1 || days > 365) {
      throw new Error('[ConsentService] Validade deve ficar entre 1 e 365 dias.');
    }
    var validUntil = new Date(now.getTime() + days * 86400000);
    var id         = _genId();

    var sheet = _getSheet();
    sheet.appendRow([
      id, sh, purpose, responsible,
      now.toISOString(), validUntil.toISOString(),
      'active', '', ''
    ]);

    _setCache(sh, purpose, 'active');
    Logger.log('[ConsentService] grant id=' + id + ' purpose=' + purpose);
    return { consentId: id, subjectHash: sh, validUntil: validUntil };
  }

  /**
   * Revoga consentimento. Invalida cache imediatamente.
   *
   * @param {string} subjectId
   * @param {string} purpose
   * @param {{ reason?: string, responsible?: string }} opts
   */
  function revoke(subjectId, purpose, opts) {
    try {
      opts = opts || {};
      if (VALID_PURPOSES.indexOf(purpose) === -1) {
        throw new Error('[ConsentService] Finalidade inválida: "' + purpose + '".');
      }
      var sh  = _hash(subjectId);
      var now = new Date();
      var persisted = false;

      try {
        var sheet = _getSheet();
        var data  = sheet.getDataRange().getValues();
        for (var i = data.length - 1; i >= 1; i--) {
          if (String(data[i][1]) === sh && String(data[i][2]) === purpose && data[i][6] === 'active') {
            sheet.getRange(i + 1, 7).setValue('revoked');   // status
            sheet.getRange(i + 1, 8).setValue(now.toISOString()); // revokedAt
            sheet.getRange(i + 1, 9).setValue(opts.reason || ''); // revokeReason
            persisted = true;
            break;
          }
        }
      } catch (e) {
        Logger.log('[ConsentService.revoke] erro: ' + e.message);
      }

      _clearCache(sh, purpose);
      _setCache(sh, purpose, 'revoked');
      if (!persisted) {
        throw new Error('[ConsentService] Revogação bloqueada no cache, mas não persistida; verifique o registro.');
      }
      Logger.log('[ConsentService] revoke purpose=' + purpose + ' reason=' + (opts.reason || ''));
    } catch (error) {
      Logger.log("Erro em revoke: " + error.message);
      throw error;
    }
  }

  /**
   * Retorna o status atual do consentimento sem lançar.
   *
   * @param {string} subjectId
   * @param {string} purpose
   * @returns {{ status: string, subjectHash: string, record: Object|null }}
   */
  function getStatus(subjectId, purpose) {
    var sh     = _hash(subjectId);
    var cached = _getCache(sh, purpose);
    if (cached) return { status: cached.status, subjectHash: sh, record: null };
    var rec    = _readRecord(sh, purpose);
    var status = rec ? rec.status : 'absent';
    _setCache(sh, purpose, status);
    return { status: status, subjectHash: sh, record: rec };
  }

  return {
    check:     check,
    grant:     grant,
    revoke:    revoke,
    getStatus: getStatus,
    _hash:     _hash  // exposto apenas para testes
  };
})();
