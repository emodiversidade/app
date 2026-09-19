/**
 * Logger global em JSONL no Drive.
 * Requer FOLDER_ID nas Script Properties.
 */
var LoggerService = (function () {
  'use strict';

  var adapters_ = {};
  var FORBIDDEN_FIELDS = {
    password: true,
    senha: true,
    token: true,
    key: true
  };

  function configure(options) {
    adapters_ = options && options.adapters || adapters_;
    return api;
  }

  function log(payload) {
    try {
      var clean = sanitize_(payload && typeof payload === 'object' ? payload : {
        message: String(payload)
      });
      clean.timestamp = now_().toISOString();
      clean.userEmail = activeUserEmail_();
      clean.scriptId = scriptId_();

      var folderId = property_('FOLDER_ID');
      if (!folderId) throw new Error('FOLDER_ID_NOT_CONFIGURED');
      var folder = drive_().getFolderById(folderId);
      var fileName = clean.timestamp.slice(0, 10) + '.log';
      var line = JSON.stringify(clean);

      withLock_(function () {
        var files = folder.getFilesByName(fileName);
        if (files.hasNext()) {
          var file = files.next();
          var current = file.getBlob().getDataAsString('UTF-8');
          file.setContent(current + (current ? '\n' : '') + line);
        } else {
          folder.createFile(fileName, line, 'text/plain');
        }
      });
      return clean;
    } catch (error) {
      Logger.log("Erro em log: " + error.message);
      throw error;
    }
  }

  function sanitize_(value) {
    try {
      if (value == null) return value;
      if (Array.isArray(value)) return value.map(sanitize_);
      if (value instanceof Date) return value.toISOString();
      if (typeof value !== 'object') return value;

      var clean = {};
      Object.keys(value).forEach(function (field) {
        if (FORBIDDEN_FIELDS[String(field).toLowerCase()]) return;
        clean[field] = sanitize_(value[field]);
      });
      return clean;
    } catch (error) {
      Logger.log("Erro em sanitize_: " + error.message);
      throw error;
    }
  }

  function activeUserEmail_() {
    try {
      try {
        return adapters_.getUserEmail ?
          String(adapters_.getUserEmail() || '') :
          String(Session.getActiveUser().getEmail() || '');
      } catch (ignored) {
        return '';
      }
    } catch (error) {
      Logger.log("Erro em activeUserEmail_: " + error.message);
      throw error;
    }
  }

  function scriptId_() {
    try {
      return adapters_.getScriptId ?
        String(adapters_.getScriptId() || '') :
        String(ScriptApp.getScriptId() || '');
    } catch (ignored) {
      return '';
    }
  }

  function property_(name) {
    try {
      return adapters_.getProperty ?
        adapters_.getProperty(name) :
        PropertiesService.getScriptProperties().getProperty(name);
    } catch (error) {
      Logger.log("Erro em property_: " + error.message);
      throw error;
    }
  }

  function drive_() {
    return adapters_.drive || DriveApp;
  }

  function now_() {
    return adapters_.now ? adapters_.now() : new Date();
  }

  function withLock_(callback) {
    try {
      var lock = adapters_.lock ||
        (typeof LockService !== 'undefined' ? LockService.getScriptLock() : null);
      if (!lock) return callback();
      if (!lock.tryLock(5000)) throw new Error('LOGGER_FILE_BUSY');
      try {
        return callback();
      } finally {
        lock.releaseLock();
      }
    } catch (error) {
      Logger.log("Erro em withLock_: " + error.message);
      throw error;
    }
  }

  // Atalhos de nível — aceitam (message, context?) ou um objeto payload.
  // Também expostos com os aliases log* para compatibilidade com projetos legados.
  function levelLog_(level, msg, ctx) {
    var payload = (msg && typeof msg === 'object') ? msg : { message: String(msg || '') };
    payload.level = payload.level || level;
    if (ctx !== undefined) payload.context = ctx;
    return log(payload);
  }

  var api = {
    configure:   configure,
    log:         log,
    // Aliases de nível (convenção moderna)
    info:        function (msg, ctx) { return levelLog_('INFO',    msg, ctx); },
    warn:        function (msg, ctx) { return levelLog_('WARN',    msg, ctx); },
    error:       function (msg, ctx) { return levelLog_('ERROR',   msg, ctx); },
    debug:       function (msg, ctx) { return levelLog_('DEBUG',   msg, ctx); },
    // Aliases com prefixo log* (compatibilidade com projetos legados)
    logInfo:     function (msg, ctx) { return levelLog_('INFO',    msg, ctx); },
    logWarning:  function (msg, ctx) { return levelLog_('WARN',    msg, ctx); },
    logError:    function (msg, ctx) { return levelLog_('ERROR',   msg, ctx); },
    logDebug:    function (msg, ctx) { return levelLog_('DEBUG',   msg, ctx); }
  };
  return api;
}());

