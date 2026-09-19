/**
 * @file CacheProvider.gs
 * @description Centraliza chaves, serializacao e TTLs do cache de script.
 *
 * English Description:
 * This service provides centralized caching capabilities using Google Apps Script's CacheService to
 * optimize application performance. It defines specific time-to-live constants, manages structured
 * cache keys, and abstracts JSON serialization and deserialization processes. By caching spreadsheet
 * rows, configurations, and session data, it minimizes slow database reads and prevents hitting
 * Google Sheets API rate limits during high concurrency.
 */

var CacheProvider = (function() {
  var TTL = Object.freeze({
    SHEET_DATA_SECONDS: 300,
    LOGIN_USER_SECONDS: 300,
    LOGIN_GLOBAL_SECONDS: 60
  });

  function cache_() {
    try {
      return CacheService.getScriptCache();
    } catch (error) {
      Logger.log("Erro em cache_: " + error.message);
      throw error;
    }
  }

  function sheetDataKey_(sheetName) {
    try {
      return "sheet_data_" + String(sheetName);
    } catch (error) {
      Logger.log("Erro em sheetDataKey_: " + error.message);
      throw error;
    }
  }

  function loginUserKey_(username) {
    try {
      return "login:" + String(username || "").trim().toLowerCase();
    } catch (error) {
      Logger.log("Erro em loginUserKey_: " + error.message);
      throw error;
    }
  }

  function getJson_(key) {
    try {
      var cached = cache_().get(key);
      if (!cached) return null;
      try {
        return JSON.parse(cached);
      } catch (error) {
        cache_().remove(key);
        return null;
      }
    } catch (error) {
      Logger.log("Erro em getJson_: " + error.message);
      throw error;
    }
  }

  function putJson_(key, value, ttlSeconds) {
    try {
      cache_().put(key, JSON.stringify(value), ttlSeconds);
      return true;
    } catch (error) {
      return false;
    }
  }

  function getNumber_(key) {
    var value = Number(cache_().get(key) || 0);
    return Number.isFinite(value) && value >= 0 ? value : 0;
  }

  function incrementNumber_(key, ttlSeconds) {
    var next = getNumber_(key) + 1;
    cache_().put(key, String(next), ttlSeconds);
    return next;
  }

  function getSheetData(sheetName) {
    return getJson_(sheetDataKey_(sheetName));
  }

  function putSheetData(sheetName, rows) {
    return putJson_(sheetDataKey_(sheetName), rows, TTL.SHEET_DATA_SECONDS);
  }

  function invalidateSheetData(sheetName) {
    try {
      cache_().remove(sheetDataKey_(sheetName));
    } catch (error) {
      Logger.log("Erro em invalidateSheetData: " + error.message);
      throw error;
    }
  }

  function getLoginAttempts(username) {
    return getNumber_(loginUserKey_(username));
  }

  function incrementLoginAttempts(username) {
    return incrementNumber_(
      loginUserKey_(username),
      TTL.LOGIN_USER_SECONDS
    );
  }

  function getGlobalLoginAttempts() {
    return getNumber_("login:global_attempts");
  }

  function incrementGlobalLoginAttempts() {
    return incrementNumber_(
      "login:global_attempts",
      TTL.LOGIN_GLOBAL_SECONDS
    );
  }

  return Object.freeze({
    TTL: TTL,
    getSheetData: getSheetData,
    putSheetData: putSheetData,
    invalidateSheetData: invalidateSheetData,
    getLoginAttempts: getLoginAttempts,
    incrementLoginAttempts: incrementLoginAttempts,
    getGlobalLoginAttempts: getGlobalLoginAttempts,
    incrementGlobalLoginAttempts: incrementGlobalLoginAttempts
  });
})();
