/**
 * GeminiResponseNormalizer.gs — Normalização defensiva de respostas JSON do Gemini
 *
 * FROTA-XX: Garante que dados vindos da IA sejam sempre seguros e validados:
 *   - Conversão segura de tipos (nunca acessa campos sem verificação prévia)
 *   - Limpeza de espaços em branco (trim) em strings
 *   - Validação de estrutura esperada
 *   - Limite de tamanho (slice para arrays, substring para strings)
 *   - Valor padrão quando campo está ausente ou malformado
 *
 * USO:
 *   var normalized = GeminiResponseNormalizer.normalize(rawData, {
 *     feedback: { type: 'string', maxLength: 500, default: 'Sem feedback disponível' },
 *     suggestions: { type: 'array', maxLength: 3, itemType: 'string', default: [] }
 *   });
 */

var GeminiResponseNormalizer = (function() {
  /**
   * Normaliza um objeto de dados em relação a um esquema esperado.
   * @param {?Object} data Dados brutos da IA (pode ser null, undefined, inválido).
   * @param {Object} schema Definição dos campos esperados.
   * @return {Object} Objeto normalizado com tipos validados e limites aplicados.
   *
   * @example
   *   normalize({ feedback: "  Bem! ", suggestions: ["A", "B", "C", "D"] },
   *             { feedback: { type: 'string', maxLength: 100 },
   *               suggestions: { type: 'array', maxLength: 3, itemType: 'string' } })
   *   // => { feedback: "Bem!", suggestions: ["A", "B", "C"] }
   */
  function normalize(data, schema) {
    data = (data && typeof data === 'object') ? data : {};
    var result = {};

    for (var field in schema) {
      if (!Object.prototype.hasOwnProperty.call(schema, field)) continue;
      var fieldDef = schema[field];
      var rawValue = data[field];
      result[field] = normalizeField_(rawValue, fieldDef);
    }

    return result;
  }

  /**
   * Normaliza um campo individual de acordo com sua definição.
   * @private
   */
  function normalizeField_(rawValue, fieldDef) {
    try {
      var type = fieldDef.type;
      var defaultValue = (fieldDef.default !== undefined) ? fieldDef.default : null;

      if (type === 'string') {
        var str = String(rawValue || '').trim();
        if (fieldDef.maxLength && str.length > fieldDef.maxLength) {
          str = str.substring(0, fieldDef.maxLength);
        }
        return str || defaultValue;
      }

      if (type === 'array') {
        if (!Array.isArray(rawValue)) {
          return defaultValue || [];
        }
        var arr = rawValue.slice(0, fieldDef.maxLength || rawValue.length);
        if (fieldDef.itemType === 'string') {
          arr = arr.map(function(item) {
            var s = String(item || '').trim();
            if (fieldDef.itemMaxLength && s.length > fieldDef.itemMaxLength) {
              s = s.substring(0, fieldDef.itemMaxLength);
            }
            return s;
          }).filter(Boolean);
        }
        return arr || defaultValue;
      }

      if (type === 'object') {
        return (rawValue && typeof rawValue === 'object') ? rawValue : defaultValue;
      }

      if (type === 'number') {
        var num = Number(rawValue);
        return isFinite(num) ? num : (defaultValue || 0);
      }

      if (type === 'boolean') {
        return Boolean(rawValue);
      }

      return rawValue || defaultValue;
    } catch (error) {
      Logger.log("Erro em normalizeField_: " + error.message);
      throw error;
    }
  }

  /**
   * Extrai texto da resposta do Gemini com segurança.
   * @param {?Object} response Resposta raw do UrlFetchApp.
   * @return {?string} Texto extraído ou null.
   */
  function extractText(response) {
    try {
      try {
        if (!response) return null;
        var data = response && response.candidates && response.candidates[0] &&
          response.candidates[0].content && response.candidates[0].content.parts &&
          response.candidates[0].content.parts[0];
        return (data && data.text) ? String(data.text).trim() : null;
      } catch (e) {
        return null;
      }
    } catch (error) {
      Logger.log("Erro em extractText: " + error.message);
      throw error;
    }
  }

  /**
   * Faz parse defensivo de JSON, removendo markdown fences se presentes.
   * @param {string} text Texto potencialmente JSON.
   * @return {?Object} Objeto parseado ou null se inválido.
   */
  function parseJson(text) {
    try {
      try {
        var cleaned = String(text || '').trim()
          .replace(/^```(?:json)?\s*/i, '')
          .replace(/\s*```$/i, '');
        return JSON.parse(cleaned);
      } catch (e) {
        return null;
      }
    } catch (error) {
      Logger.log("Erro em parseJson: " + error.message);
      throw error;
    }
  }

  return {
    normalize: normalize,
    extractText: extractText,
    parseJson: parseJson
  };
})();
