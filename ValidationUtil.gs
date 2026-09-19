/**
 * @file ValidationUtil.gs
 * @description Validações e sanitizações utilitárias para entradas do sistema.
 *
 * English Description:
 * This utility module provides validation and sanitization functions for system inputs to ensure security
 * and data integrity. It includes functions to check for non-empty values, validate email formats using
 * regular expressions, and sanitize strings against cross-site scripting vulnerabilities by replacing
 * special characters with HTML entities. This helps protect the database from malicious input.
 */

var ValidationUtil = (function() {
  function isNotEmpty(value) {
    try {
      if (value === null || value === undefined) return false;
      if (typeof value === 'string') return value.trim().length > 0;
      if (Array.isArray(value)) return value.length > 0;
      return true;
    } catch (error) {
      Logger.log("Erro em isNotEmpty: " + error.message);
      throw error;
    }
  }

  function isValidEmail(email) {
    try {
      if (typeof email !== 'string') return false;
      const trimmed = email.trim();
      if (trimmed.length === 0) return false;
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      return emailRegex.test(trimmed);
    } catch (error) {
      Logger.log("Erro em isValidEmail: " + error.message);
      throw error;
    }
  }

  function sanitizeString(value) {
    try {
      if (typeof value !== 'string') return String(value || "");
      return value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    } catch (error) {
      Logger.log("Erro em sanitizeString: " + error.message);
      throw error;
    }
  }

  return {
    isNotEmpty: isNotEmpty,
    isValidEmail: isValidEmail,
    sanitizeString: sanitizeString
  };
})();
