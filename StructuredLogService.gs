/**
 * FROTA-13: logging estruturado e privado em JSONL diario no Google Drive.
 *
 * Configure FOLDER_ID nas Script Properties.
 */
var StructuredLogService = (function () {
  'use strict';

  var config_ = {
    folderProperty: 'FOLDER_ID',
    saltProperty: 'LOG_PSEUDONYM_SALT',
    filePrefix: '_SYSTEM_LOGS_',
    maxStringLength: 2000,
    allowPublicFolder: true
  };
  var adapters_ = {};

  var SECRET_KEYS = /(password|senha|secret|api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|cookie)/i;
  var PII_KEYS = /(actor[_-]?id|user[_-]?id|student[_-]?id|name|nome|email|cpf|cnpj|phone|telefone|address|endereco|birth|nascimento)/i;

  function configure(options) {
    try {
      options = options || {};
      Object.keys(options).forEach(function (key) {
        if (key === 'adapters') adapters_ = options.adapters || {};
        else config_[key] = options[key];
      });
      return api;
    } catch (error) {
      Logger.log("Erro em configure: " + error.message);
      throw error;
    }
  }

  function write(level, operation, metadata) {
    var event = buildEvent_(level, operation, metadata || {});
    append_(event);
    return event;
  }

  function info(operation, metadata) {
    return write('INFO', operation, metadata);
  }

  function warn(operation, metadata) {
    return write('WARN', operation, metadata);
  }

  function error(operation, metadata) {
    return write('ERROR', operation, metadata);
  }

  function auditAi(operation, metadata) {
    metadata = metadata || {};
    return write(metadata.errorCode ? 'ERROR' : 'AUDIT', operation, {
      sessionId: metadata.sessionId,
      actorId: metadata.actorId,
      ageRange: metadata.ageRange || null,
      ethicalSubstitution: metadata.ethicalSubstitution === true,
      errorCode: metadata.errorCode || null,
      model: metadata.model || null,
      source: metadata.source || null,
      durationMs: metadata.durationMs,
      outcome: metadata.outcome || (metadata.errorCode ? 'failure' : 'success'),
      details: metadata.details || null
    });
  }

  function buildEvent_(level, operation, metadata) {
    try {
      var sessionId = metadata.sessionId;
      var event = {
        timestamp: now_().toISOString(),
        level: normalizeLevel_(level),
        operation: String(operation || 'unknown'),
        sessionId: sessionId ? pseudonymize_(sessionId) : null,
        metadata: sanitize_(metadata, '')
      };
      delete event.metadata.sessionId;
      return event;
    } catch (error) {
      Logger.log("Erro em buildEvent_: " + error.message);
      throw error;
    }
  }

  function append_(event) {
    try {
      try {
        var folder = getPrivateFolder_();
        var fileName = config_.filePrefix + event.timestamp.slice(0, 10) + '.jsonl';
        var line = JSON.stringify(event);
        var file = findFile_(folder, fileName);
        if (file) {
          var current = file.getBlob().getDataAsString('UTF-8');
          file.setContent(current + (current ? '\n' : '') + line);
        } else {
          folder.createFile(fileName, line, 'text/plain');
        }
      } catch (error) {
        Logger.log("Erro em append_: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em append_: " + error.message);
      throw error;
    }
  }

  function getPrivateFolder_() {
    var folderId = property_(config_.folderProperty);
    if (!folderId) throw new Error('FOLDER_ID_NOT_CONFIGURED');
    var folder = drive_().getFolderById(folderId);
    assertPrivate_(folder);
    return folder;
  }

  function assertPrivate_(folder) {
    try {
      if (config_.allowPublicFolder === true) return;
      if (typeof folder.getSharingAccess !== 'function') {
        throw new Error('LOG_FOLDER_PRIVACY_UNVERIFIABLE');
      }
      var access = String(folder.getSharingAccess());
      if (/ANYONE|DOMAIN_WITH_LINK|PUBLIC/i.test(access)) {
        throw new Error('LOG_FOLDER_MUST_BE_PRIVATE');
      }
    } catch (error) {
      Logger.log("Erro em assertPrivate_: " + error.message);
      throw error;
    }
  }

  function findFile_(folder, name) {
    var files = folder.getFilesByName(name);
    return files.hasNext() ? files.next() : null;
  }

  function sanitize_(value, key) {
    try {
      if (value == null) return value;
      if (SECRET_KEYS.test(key)) return '[REDACTED]';
      if (PII_KEYS.test(key)) return pseudonymize_(value);
      if (value instanceof Date) return value.toISOString();
      if (Array.isArray(value)) {
        return value.slice(0, 100).map(function (item) { return sanitize_(item, key); });
      }
      if (typeof value === 'object') {
        var clean = {};
        Object.keys(value).slice(0, 100).forEach(function (childKey) {
          clean[childKey] = sanitize_(value[childKey], childKey);
        });
        return clean;
      }
      if (typeof value === 'string') {
        var text = redactInline_(value);
        return text.length > config_.maxStringLength ?
          text.slice(0, config_.maxStringLength) + '[TRUNCATED]' : text;
      }
      return value;
    } catch (error) {
      Logger.log("Erro em sanitize_: " + error.message);
      throw error;
    }
  }

  function redactInline_(text) {
    try {
      return String(text)
        .replace(/(bearer\s+)[^\s,;]+/ig, '$1[REDACTED]')
        .replace(/((?:api[_-]?key|password|senha|secret|token)\s*[:=]\s*)[^\s,;]+/ig, '$1[REDACTED]')
        .replace(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/ig, '[PII_REDACTED]');
    } catch (error) {
      Logger.log("Erro em redactInline_: " + error.message);
      throw error;
    }
  }

  function pseudonymize_(value) {
    try {
      var salt = property_(config_.saltProperty);
      if (!salt) throw new Error('LOG_PSEUDONYM_SALT_NOT_CONFIGURED');
      var raw = String(salt) + ':' + String(value);
      var bytes = utilities_().computeDigest(
        utilities_().DigestAlgorithm.SHA_256,
        raw,
        utilities_().Charset.UTF_8
      );
      return bytes.map(function (byte) {
        var normalized = byte < 0 ? byte + 256 : byte;
        return ('0' + normalized.toString(16)).slice(-2);
      }).join('').slice(0, 24);
    } catch (error) {
      Logger.log("Erro em pseudonymize_: " + error.message);
      throw error;
    }
  }

  function property_(name) {
    try {
      if (adapters_.getProperty) return adapters_.getProperty(name);
      return PropertiesService.getScriptProperties().getProperty(name);
    } catch (error) {
      Logger.log("Erro em property_: " + error.message);
      throw error;
    }
  }

  function drive_() {
    return adapters_.drive || DriveApp;
  }

  function utilities_() {
    return adapters_.utilities || Utilities;
  }

  function now_() {
    return adapters_.now ? adapters_.now() : new Date();
  }

  function normalizeLevel_(level) {
    level = String(level || 'INFO').toUpperCase();
    return ['DEBUG', 'INFO', 'WARN', 'ERROR', 'AUDIT'].indexOf(level) >= 0 ?
      level : 'INFO';
  }

  var api = {
    configure: configure,
    write: write,
    info: info,
    warn: warn,
    error: error,
    auditAi: auditAi,
    pseudonymize: pseudonymize_
  };
  return api;
}());
