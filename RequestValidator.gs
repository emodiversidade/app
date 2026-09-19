/**
 * Validador de entrada inspirado em JSON Schema, sem dependencias externas.
 * Suporta: type, required, properties, items, enum, pattern, format,
 * minLength, maxLength, minimum, maximum e additionalProperties.
 */
var RequestValidator = (function () {
  'use strict';

  function validate(value, schema) {
    var errors = [];
    validateNode_(value, schema || {}, '$', errors);
    return { valid: errors.length === 0, errors: errors };
  }

  function validateOrError(value, schema, requestId) {
    var result = validate(value, schema);
    return result.valid ? null : ApiError.validation(result.errors, null, requestId);
  }

  function validateNode_(value, schema, path, errors) {
    try {
      if (schema.required === true && (value === undefined || value === null || value === '')) {
        add_(errors, path, 'required', 'Campo obrigatorio.');
        return;
      }
      if (value === undefined || value === null) return;

      if (schema.type && !isType_(value, schema.type)) {
        add_(errors, path, 'type', 'Tipo esperado: ' + schema.type + '.');
        return;
      }
      if (schema.enum && schema.enum.indexOf(value) === -1) {
        add_(errors, path, 'enum', 'Valor nao permitido.');
      }
      if (typeof value === 'string') validateString_(value, schema, path, errors);
      if (typeof value === 'number') validateNumber_(value, schema, path, errors);
      if (Array.isArray(value) && schema.items) {
        value.forEach(function (item, index) {
          validateNode_(item, schema.items, path + '[' + index + ']', errors);
        });
      }
      if (isPlainObject_(value)) validateObject_(value, schema, path, errors);
    } catch (error) {
      Logger.log("Erro em validateNode_: " + error.message);
      throw error;
    }
  }

  function validateObject_(value, schema, path, errors) {
    try {
      var properties = schema.properties || {};
      (Array.isArray(schema.required) ? schema.required : []).forEach(function (key) {
        if (!Object.prototype.hasOwnProperty.call(value, key) ||
            value[key] === undefined || value[key] === null || value[key] === '') {
          add_(errors, path + '.' + key, 'required', 'Campo obrigatorio.');
        }
      });
      Object.keys(properties).forEach(function (key) {
        validateNode_(value[key], properties[key], path + '.' + key, errors);
      });
      if (schema.additionalProperties === false) {
        Object.keys(value).forEach(function (key) {
          if (!Object.prototype.hasOwnProperty.call(properties, key)) {
            add_(errors, path + '.' + key, 'additionalProperties', 'Campo nao reconhecido.');
          }
        });
      }
    } catch (error) {
      Logger.log("Erro em validateObject_: " + error.message);
      throw error;
    }
  }

  function validateString_(value, schema, path, errors) {
    if (schema.minLength != null && value.length < schema.minLength) {
      add_(errors, path, 'minLength', 'Tamanho minimo: ' + schema.minLength + '.');
    }
    if (schema.maxLength != null && value.length > schema.maxLength) {
      add_(errors, path, 'maxLength', 'Tamanho maximo: ' + schema.maxLength + '.');
    }
    if (schema.pattern && !(new RegExp(schema.pattern)).test(value)) {
      add_(errors, path, 'pattern', 'Formato invalido.');
    }
    if (schema.format === 'email' &&
        !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
      add_(errors, path, 'format', 'Email invalido.');
    }
  }

  function validateNumber_(value, schema, path, errors) {
    if (schema.minimum != null && value < schema.minimum) {
      add_(errors, path, 'minimum', 'Valor minimo: ' + schema.minimum + '.');
    }
    if (schema.maximum != null && value > schema.maximum) {
      add_(errors, path, 'maximum', 'Valor maximo: ' + schema.maximum + '.');
    }
  }

  function isType_(value, type) {
    if (type === 'array') return Array.isArray(value);
    if (type === 'object') return isPlainObject_(value);
    if (type === 'integer') return typeof value === 'number' && isFinite(value) && Math.floor(value) === value;
    if (type === 'number') return typeof value === 'number' && isFinite(value);
    return typeof value === type;
  }

  function isPlainObject_(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  function add_(errors, field, rule, message) {
    try {
      errors.push({ field: field, rule: rule, message: message });
    } catch (error) {
      Logger.log("Erro em add_: " + error.message);
      throw error;
    }
  }

  return {
    validate: validate,
    validateOrError: validateOrError
  };
}());

