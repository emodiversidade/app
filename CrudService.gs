/**
 * @file CrudService.gs
 * @description Operacoes internas e validadas sobre as abas da aplicacao.
 *
 * English Description:
 * This service implements basic create, read, update, and delete operations on allowed Google Sheets.
 * It validates sheet name arguments against a strict allowlist to prevent unauthorized access, enforces
 * script locks to avoid concurrent write conflicts, and handles raw row arrays. It provides low-level
 * database operations that serve as the foundation for the higher-level service components.
 */

function getAllowedSheetNames_() {
  try {
    return Object.keys(getGameConfig_().SHEET_NAMES).map(key => getGameConfig_().SHEET_NAMES[key]);
  } catch (error) {
    Logger.log("Erro em getAllowedSheetNames_: " + error.message);
    throw error;
  }
}

function getSheet_(sheetName) {
  try {
    if (getAllowedSheetNames_().indexOf(sheetName) === -1) {
      throw new Error("Aba nao permitida.");
    }
    const sheet = getSpreadsheet_().getSheetByName(sheetName);
    if (!sheet) {
      throw new Error(`A aba '${sheetName}' nao foi encontrada.`);
    }
    return sheet;
  } catch (error) {
    Logger.log("Erro em getSheet_: " + error.message);
    throw error;
  }
}

/**
 * Anexa uma linha em uma aba permitida, sob lock de script.
 * @param {string} sheetName Nome presente em GAME_CONFIG.SHEET_NAMES.
 * @param {Array} dataArray Valores na ordem das colunas da aba.
 * @throws {Error} Aba nao permitida/encontrada ou dados vazios.
 */
function createRecord_(sheetName, dataArray) {
  try {
    if (!Array.isArray(dataArray) || !dataArray.length) {
      throw new Error("Os dados do registro sao obrigatorios.");
    }
    const lock = LockService.getScriptLock();
    lock.waitLock(5000);
    try {
      getSheet_(sheetName).appendRow(dataArray);
    } finally {
      lock.releaseLock();
    }
  } catch (error) {
    Logger.log("Erro em createRecord_: " + error.message);
    throw error;
  }
}

/**
 * Reexecuta uma operacao idempotente com backoff exponencial limitado.
 * @param {function(): *} operation Operacao de leitura segura para repetir.
 * @param {number} [attempts=3] Tentativas maximas.
 * @param {number} [baseDelayMs=200] Espera base, dobrada a cada tentativa.
 * @returns {*} Retorno da operacao na primeira tentativa bem-sucedida.
 * @throws {Error} Ultimo erro apos esgotar as tentativas.
 */
function withRetry_(operation, attempts = 3, baseDelayMs = 200) {
  let lastError;
  for (let i = 0; i < attempts; i++) {
    try {
      return operation();
    } catch (error) {
      lastError = error;
      Utilities.sleep(baseDelayMs * Math.pow(2, i));
    }
  }
  throw lastError;
}

/**
 * Le todas as linhas de uma aba permitida (cabecalho incluso na linha 0).
 * A aba Situacoes e cacheada por 5 minutos; as demais sao lidas a cada chamada.
 * @param {string} sheetName Nome presente em GAME_CONFIG.SHEET_NAMES.
 * @returns {Array<Array>} Matriz de valores da aba.
 */
function readRecords_(sheetName) {
  try {
    const isCachable = (sheetName === getGameConfig_().SHEET_NAMES.SITUATIONS);

    if (isCachable) {
      const cached = CacheProvider.getSheetData(sheetName);
      if (cached) return cached;
    }

    const data = withRetry_(() => getSheet_(sheetName).getDataRange().getValues());

    if (isCachable && data && data.length > 0) {
      CacheProvider.putSheetData(sheetName, data);
    }
  
    return data;
  } catch (error) {
    Logger.log("Erro em readRecords_: " + error.message);
    throw error;
  }
}

/**
 * Atualiza por ID as colunas cujo nome de cabecalho conste em dataObject, sob lock.
 * @param {string} sheetName Nome presente em GAME_CONFIG.SHEET_NAMES.
 * @param {string} recordId Valor da coluna ID (primeira coluna).
 * @param {Object} dataObject Pares cabecalho->valor; chaves desconhecidas sao ignoradas.
 * @throws {Error} "Registro nao encontrado." quando o ID nao existe.
 */
function updateRecord_(sheetName, recordId, dataObject) {
  try {
    try {
      if (!recordId || !dataObject || typeof dataObject !== "object") {
        throw new Error("ID e dados de atualizacao sao obrigatorios.");
      }
      const lock = LockService.getScriptLock();
      lock.waitLock(5000);
      try {
        const sheet = getSheet_(sheetName);
        const data = sheet.getDataRange().getValues();
        const headers = data[0] || [];
        for (let index = 1; index < data.length; index++) {
          if (String(data[index][0]) !== String(recordId)) {
            continue;
          }
          Object.keys(dataObject).forEach(key => {
            const column = headers.indexOf(key);
            if (column >= 0) {
              sheet.getRange(index + 1, column + 1).setValue(dataObject[key]);
            }
          });
          return;
        }
        throw new Error("Registro nao encontrado.");
      } finally {
        lock.releaseLock();
      }
    } catch (error) {
      Logger.log("Erro em updateRecord_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em updateRecord_: " + error.message);
    throw error;
  }
}

function deleteRecord_(sheetName, recordId) {
  try {
    try {
      try {
        const lock = LockService.getScriptLock();
        lock.waitLock(5000);
        try {
          const sheet = getSheet_(sheetName);
          const data = sheet.getDataRange().getValues();
          for (let index = 1; index < data.length; index++) {
            if (String(data[index][0]) === String(recordId)) {
              sheet.deleteRow(index + 1);
              return;
            }
          }
          throw new Error("Registro nao encontrado.");
        } finally {
          lock.releaseLock();
        }
      } catch (error) {
        Logger.log("Erro em deleteRecord_: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em deleteRecord_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em deleteRecord_: " + error.message);
    throw error;
  }
}
