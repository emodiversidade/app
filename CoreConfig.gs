/**
 * @file CoreConfig.gs
 * @description Configuração centralizada e resolução segura do ID da Planilha.
 *
 * English Description:
 * This configuration module manages the central database identifiers and provides automatic
 * provisioning for Google Sheets. It safely resolves the spreadsheet ID by querying the script
 * properties, falling back to local constants, or automatically creating a new spreadsheet if
 * no valid identifier is found. This utility guarantees that the application has a reliable
 * database backend configured upon initialization.
 */

var CoreConfig = (function() {
  const CONFIG = {
    SPREADSHEETS_ID: "PLACEHOLDER_SPREADSHEETS_ID",
    DB_NAME: "Emodiversa DB"
  };

  function getSpreadsheetId() {
    try {
      let spreadsheetId = PropertiesManager.getScriptProperty("SPREADSHEETS_ID");
      if (spreadsheetId && spreadsheetId.trim() !== "" && spreadsheetId !== "PLACEHOLDER_SPREADSHEETS_ID") {
        return spreadsheetId;
      }

      if (CONFIG.SPREADSHEETS_ID && CONFIG.SPREADSHEETS_ID.trim() !== "" && CONFIG.SPREADSHEETS_ID !== "PLACEHOLDER_SPREADSHEETS_ID") {
        PropertiesManager.setScriptProperty("SPREADSHEETS_ID", CONFIG.SPREADSHEETS_ID);
        return CONFIG.SPREADSHEETS_ID;
      }

      try {
        console.warn("SPREADSHEETS_ID não configurado. Iniciando auto-provisão de banco de dados...");
        const newSS = SpreadsheetApp.create(CONFIG.DB_NAME);
        spreadsheetId = newSS.getId();
        PropertiesManager.setScriptProperty("SPREADSHEETS_ID", spreadsheetId);
        // Auto-provisão concluída (log removido para produção)
        return spreadsheetId;
      } catch (error) {
        console.error("Erro crítico na auto-provisão do banco de dados:", error);
        throw new Error("Não foi possível resolver ou criar a planilha de banco de dados. " + error.message);
      }
    } catch (error) {
      Logger.log("Erro em getSpreadsheetId: " + error.message);
      throw error;
    }
  }

  return {
    getSpreadsheetId: getSpreadsheetId
  };
})();
