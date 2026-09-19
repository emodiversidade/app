/**
 * PromptContextBuilder.gs - privacy-by-construction context builder.
 * Project: Emodiversa
 */
var PromptContextBuilder = (function() {
  var ALLOWED_FIELDS = {
    'gemini.generate': ['topic','theme','query','audience','metrics','summary','items','headlines','context','directive','authors','works'],
    'themeDigest': ['theme','query','audience','headlines'],
    'report.analytics': ['projectName','focus','scope','period','metrics','analyses','summary'],
    'recommendation': ['interests','categories','works','authors','subjects','summary','directive'],
    'content.generate': ['topic','answers','guideline','constraints','audience','summary']
  };

  var BLOCKED_KEYS = [
    'email','e_mail','nome','name','cpf','ra','matricula','registration',
    'studentid','student_id','userid','user_id','id_externo','external_id',
    'telefone','phone','endereco','address','responsavel','guardian',
    'responsavel_nome','turma_nominal','nota_livre','observacao_pessoal',
    'student_name','teacher_name','password','senha','token','secret'
  ];

  var PII_PATTERNS = [
    /\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/g,
    /\b\d{3}\.?\d{3}\.?\d{3}\-?\d{2}\b/g,
    /\b(?:matr[ií]cula|ra)\s*[:#-]?\s*[A-Z0-9.\/-]{5,20}\b/gi,
    /\b(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?9?\d{4}[-\s]?\d{4}\b/g,
    /\b(aluno|aluna|estudante|professor|professora|respons[aá]vel)\s+[A-ZÀ-Ú][A-Za-zÀ-ú]+(?:\s+[A-ZÀ-Ú][A-Za-zÀ-ú]+){0,3}/g
  ];

  function isBlockedKey_(key) {
    try {
      var normalized = String(key || '').toLowerCase().replace(/[^a-z0-9_]/g, '');
      return BLOCKED_KEYS.indexOf(normalized) !== -1;
    } catch (error) {
      Logger.log("Erro em isBlockedKey_: " + error.message);
      throw error;
    }
  }

  function stripPii(text) {
    try {
      if (typeof text !== 'string') return text;
      var clean = text;
      PII_PATTERNS.forEach(function(pattern) {
        pattern.lastIndex = 0;
        clean = clean.replace(pattern, '[OMITIDO]');
      });
      return clean;
    } catch (error) {
      Logger.log("Erro em stripPii: " + error.message);
      throw error;
    }
  }

  function containsPii(text) {
    try {
      if (typeof text !== 'string') return false;
      return PII_PATTERNS.some(function(pattern) {
        pattern.lastIndex = 0;
        return pattern.test(text);
      });
    } catch (error) {
      Logger.log("Erro em containsPii: " + error.message);
      throw error;
    }
  }

  function cleanValue_(value, droppedKeys, pathPrefix) {
    try {
      if (typeof value === 'string') return stripPii(value).slice(0, 4000);
      if (typeof value === 'number' || typeof value === 'boolean' || value === null) return value;
      if (Array.isArray(value)) {
        return value.slice(0, 100).map(function(item, index) {
          return cleanValue_(item, droppedKeys, pathPrefix + '[' + index + ']');
        });
      }
      if (value && typeof value === 'object') {
        var out = {};
        Object.keys(value).forEach(function(key) {
          var full = pathPrefix ? pathPrefix + '.' + key : key;
          if (isBlockedKey_(key)) {
            droppedKeys.push(full);
            return;
          }
          out[key] = cleanValue_(value[key], droppedKeys, full);
        });
        return out;
      }
      return null;
    } catch (error) {
      Logger.log("Erro em cleanValue_: " + error.message);
      throw error;
    }
  }

  function build(useCase, rawData) {
    try {
      var allowed = ALLOWED_FIELDS[useCase];
      if (!allowed) throw new Error('PromptContextBuilder: caso de uso desconhecido "' + useCase + '".');
      rawData = rawData || {};
      var context = {};
      var droppedKeys = [];
      Object.keys(rawData).forEach(function(key) {
        if (isBlockedKey_(key) || allowed.indexOf(key) === -1) {
          droppedKeys.push(key);
          return;
        }
        context[key] = cleanValue_(rawData[key], droppedKeys, key);
      });
      return { context: context, droppedKeys: droppedKeys };
    } catch (error) {
      Logger.log("Erro em build: " + error.message);
      throw error;
    }
  }

  function logAudit(useCase, droppedKeys) {
    try {
      droppedKeys = droppedKeys || [];
      Logger.log('PromptContextBuilder [' + useCase + ']: dropped=' +
        droppedKeys.length + '; keys=' + droppedKeys.join(',') +
        '; values=NOT_LOGGED');
    } catch (error) {
      Logger.log("Erro em logAudit: " + error.message);
      throw error;
    }
  }

  return {
    ALLOWED_FIELDS: ALLOWED_FIELDS,
    BLOCKED_KEYS: BLOCKED_KEYS,
    PII_PATTERNS: PII_PATTERNS,
    build: build,
    stripPii: stripPii,
    containsPii: containsPii,
    logAudit: logAudit
  };
})();
