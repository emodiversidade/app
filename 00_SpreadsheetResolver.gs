/**
 * 00_SpreadsheetResolver.gs
 *
 * Resolvedor robusto de planilha para contexto WEBAPP/standalone.
 *
 * Em deployment de web app (nao container-bound), SpreadsheetApp
 * .getActiveSpreadsheet() retorna null, e qualquer null.getSheetByName(...)
 * derruba o fluxo — inclusive o login ("Cannot read properties of null
 * (reading 'getSheetByName')").
 *
 * Ordem de resolucao:
 *   1. getSpreadsheetId() — funcao de Config local, se existir
 *   2. getProjectSpreadsheet() — alias de projeto, se existir
 *   3. CONFIG.SPREADSHEET_ID / Config.SPREADSHEET_ID — objeto global
 *   4. Script Properties: SPREADSHEETS_ID ou SPREADSHEET_ID
 *   5. SpreadsheetApp.getActiveSpreadsheet() — container-bound fallback
 *   6. Lanca erro explicativo: "defina SPREADSHEET_ID nas Script Properties"
 *
 * O prefixo "00_" garante avaliacao antecipada no concatenador do Apps Script.
 */
function getBoundSpreadsheet_() {
  try {
    try {
      // 1. Funcao getSpreadsheetId() definida em Config/ConfigService local
      if (typeof getSpreadsheetId === 'function') {
        var idFromFunction = getSpreadsheetId();
        if (idFromFunction) return SpreadsheetApp.openById(idFromFunction);
      }

      // 2. Alias getProjectSpreadsheet() definido em Config local
      if (typeof getProjectSpreadsheet === 'function') {
        var ssFromProject = getProjectSpreadsheet();
        if (ssFromProject) return ssFromProject;
      }

      // 3. Objeto CONFIG ou Config com SPREADSHEET_ID
      if (typeof CONFIG !== 'undefined' && CONFIG && CONFIG.SPREADSHEET_ID) {
        return SpreadsheetApp.openById(CONFIG.SPREADSHEET_ID);
      }
      if (typeof Config !== 'undefined' && Config && Config.SPREADSHEET_ID) {
        return SpreadsheetApp.openById(Config.SPREADSHEET_ID);
      }

      // 4. Script Properties (suporta ambos os aliases da frota)
      var props = PropertiesService.getScriptProperties();
      var id = props.getProperty('SPREADSHEETS_ID') || props.getProperty('SPREADSHEET_ID');
      if (id) return SpreadsheetApp.openById(id);

      // 5. Container-bound fallback (funciona no editor / container)
      var active = SpreadsheetApp.getActiveSpreadsheet();
      if (active) return active;

      // 6. Nenhuma fonte disponivel — erro explicativo
      throw new Error(
        'Planilha indisponivel: defina a Script Property SPREADSHEET_ID ' +
        '(ou SPREADSHEETS_ID) no editor do Apps Script > Projeto > Propriedades.'
      );
    } catch (error) {
      Logger.log("Erro em getBoundSpreadsheet_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em getBoundSpreadsheet_: " + error.message);
    throw error;
  }
}