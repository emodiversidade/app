/**
 * @file PropertiesManager.gs
 * @description Gerencia as propriedades do Apps Script (ScriptProperties e UserProperties) de forma segura e tipada.
 *
 * English Description:
 * This service acts as a typed wrapper around Google Apps Script PropertiesService to manage script
 * and user configurations. It handles data type casting for boolean, numeric, and string values, prevents
 * runtime exceptions by capturing service errors, and handles deletion of properties. It ensures
 * consistent and reliable access to persistent environment variables across different user sessions.
 */

var PropertiesManager = (function() {
  function getScriptProperty(key, type, defaultValue) {
    try {
      const value = PropertiesService.getScriptProperties().getProperty(key);
      if (value === null || value === undefined) {
        return defaultValue !== undefined ? defaultValue : null;
      }
      return castValue_(value, type || 'string');
    } catch (error) {
      console.error(`Erro ao obter ScriptProperty '${key}':`, error);
      return defaultValue !== undefined ? defaultValue : null;
    }
  }

  function setScriptProperty(key, value) {
    try {
      try {
        let stringValue;
        if (value === null || value === undefined) {
          PropertiesService.getScriptProperties().deleteProperty(key);
          return;
        } else if (typeof value === 'object') {
          stringValue = JSON.stringify(value);
        } else {
          stringValue = String(value);
        }
        PropertiesService.getScriptProperties().setProperty(key, stringValue);
      } catch (error) {
        console.error(`Erro ao salvar ScriptProperty '${key}':`, error);
        throw new Error(`Falha ao salvar propriedade no servidor: ${error.message}`);
      }
    } catch (error) {
      Logger.log("Erro em setScriptProperty: " + error.message);
      throw error; // Re-lança para tratamento superior
    }
  }

  function getUserProperty(key, type, defaultValue) {
    try {
      try {
        const value = PropertiesService.getUserProperties().getProperty(key);
        if (value === null || value === undefined) {
          return defaultValue !== undefined ? defaultValue : null;
        }
        return castValue_(value, type || 'string');
      } catch (error) {
        console.error(`Erro ao obter UserProperty '${key}':`, error);
        return defaultValue !== undefined ? defaultValue : null;
      }
    } catch (error) {
      Logger.log("Erro em getUserProperty: " + error.message);
      throw error;
    }
  }

  function setUserProperty(key, value) {
    try {
      try {
        try {
          let stringValue;
          if (value === null || value === undefined) {
            PropertiesService.getUserProperties().deleteProperty(key);
            return;
          } else if (typeof value === 'object') {
            stringValue = JSON.stringify(value);
          } else {
            stringValue = String(value);
          }
          PropertiesService.getUserProperties().setProperty(key, stringValue);
        } catch (error) {
          console.error(`Erro ao salvar UserProperty '${key}':`, error);
          throw new Error(`Falha ao salvar propriedade de usuário no servidor: ${error.message}`);
        }
      } catch (error) {
        Logger.log("Erro em setUserProperty: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em setUserProperty: " + error.message);
      throw error;
    }
  }

  function castValue_(value, type) {
    try {
      try {
        if (type === 'number') {
          const num = Number(value);
          return isNaN(num) ? null : num;
        }
        if (type === 'json') {
          try {
            return JSON.parse(value);
          } catch (e) {
            console.error("Erro no parse JSON do valor:", value);
            return null;
          }
        }
        return value;
      } catch (error) {
        Logger.log("Erro em castValue_: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em castValue_: " + error.message);
      throw error;
    }
  }

  return {
    getScriptProperty: getScriptProperty,
    setScriptProperty: setScriptProperty,
    getUserProperty: getUserProperty,
    setUserProperty: setUserProperty
  };
})();
