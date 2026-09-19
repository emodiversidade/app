/**
 * @file JsonUtils.gs
 * @description Utilidades para manipulação segura de JSON sem causar exceções indesejadas.
 *
 * English Description:
 * This utility module provides safe wrappers for parsing and stringifying JSON data, preventing runtime
 * exceptions from halting execution. It intercepts syntax errors, logs detailed warnings, and returns
 * specified fallback values when encountering invalid input. This utility ensures that parsing corrupted
 * data stored in spreadsheet columns or configuration cells does not crash the application during critical processes.
 */

var JsonUtils = (function() {
  function safeParse(text, fallback) {
    if (typeof text !== 'string') return text;
    try {
      return JSON.parse(text);
    } catch (error) {
      console.warn("Falha ao realizar safeParse de JSON:", error.message, text);
      return fallback !== undefined ? fallback : null;
    }
  }

  function safeStringify(value, fallback) {
    try {
      return JSON.stringify(value);
    } catch (error) {
      console.error("Falha ao realizar safeStringify de JSON:", error.message, value);
      return fallback !== undefined ? fallback : "";
    }
  }

  return {
    safeParse: safeParse,
    safeStringify: safeStringify
  };
})();
