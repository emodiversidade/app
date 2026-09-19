/**
 * Dados sintéticos — Emodiversa
 * Gerado em 2026-06-21 01:10:43 por generate_synthetic_data_all_projects.py
 *
 * Execute populateSyntheticData() PELO EDITOR do Apps Script para popular
 * as abas de domínio com ~30 registros cada (valida os gráficos do notebook).
 * Idempotente: limpa as linhas de dados antes de reinserir.
 *
 * NÃO define onOpen() — para não colidir com o menu real do projeto.
 */

function populateSyntheticData() {
  try {
    try {
      try {
        var ss = SpreadsheetApp.getActiveSpreadsheet();
        var results = [];

        // Situacoes
        try {
          var sheet_SITUACOES = ss.getSheetByName('Situacoes') || ss.insertSheet('Situacoes');
          if (sheet_SITUACOES.getLastRow() > 1) {
            sheet_SITUACOES.deleteRows(2, sheet_SITUACOES.getLastRow() - 1);
          }
          var h_sheet_SITUACOES = ["ID", "Titulo", "Descricao", "Opcao1", "Opcao2", "Resultado1", "Resultado2", "NivelPrincesamento", "ImpactoOpcao1", "ImpactoOpcao2"];
          sheet_SITUACOES.getRange(1, 1, 1, h_sheet_SITUACOES.length).setValues([h_sheet_SITUACOES]);
          var d_sheet_SITUACOES = [
            ["SIT-0001", "Prática", "Registro de sessão experimental", "C", "C", "D", "A", "medio", "alta", "baixa"],
            ["SIT-0002", "Revisão", "Observação inicial do processo", "D", "B", "A", "D", "alto", "média", "média"],
            ["SIT-0003", "Introdução", "Dados coletados durante atividade", "B", "B", "C", "D", "alto", "média", "alta"],
            ["SIT-0004", "Prática", "Dados coletados durante atividade", "A", "D", "C", "B", "medio", "baixa", "alta"],
            ["SIT-0005", "Introdução", "Acompanhamento de evolução", "C", "B", "B", "B", "medio", "alta", "alta"],
            ["SIT-0006", "Revisão", "Acompanhamento de evolução", "D", "D", "D", "D", "medio", "alta", "alta"],
            ["SIT-0007", "Revisão", "Dados coletados durante atividade", "C", "A", "C", "B", "baixo", "baixa", "alta"],
            ["SIT-0008", "Conceitos", "Dados coletados durante atividade", "C", "C", "B", "A", "medio", "baixa", "baixa"],
            ["SIT-0009", "Prática", "Dados coletados durante atividade", "A", "A", "C", "C", "medio", "baixa", "baixa"],
            ["SIT-0010", "Revisão", "Registro de sessão experimental", "D", "B", "C", "D", "medio", "alta", "alta"],
            ["SIT-0011", "Avaliação", "Observação inicial do processo", "B", "B", "B", "B", "alto", "alta", "baixa"],
            ["SIT-0012", "Revisão", "Dados coletados durante atividade", "A", "C", "A", "D", "alto", "média", "baixa"],
            ["SIT-0013", "Revisão", "Observação inicial do processo", "A", "A", "A", "A", "alto", "alta", "média"],
            ["SIT-0014", "Avaliação", "Acompanhamento de evolução", "A", "C", "D", "B", "baixo", "baixa", "baixa"],
            ["SIT-0015", "Introdução", "Registro de sessão experimental", "D", "A", "C", "B", "medio", "alta", "baixa"],
            ["SIT-0016", "Revisão", "Observação inicial do processo", "C", "A", "A", "D", "baixo", "baixa", "alta"],
            ["SIT-0017", "Prática", "Dados coletados durante atividade", "C", "D", "C", "B", "baixo", "baixa", "média"],
            ["SIT-0018", "Conceitos", "Acompanhamento de evolução", "C", "B", "A", "A", "alto", "alta", "alta"],
            ["SIT-0019", "Revisão", "Acompanhamento de evolução", "B", "A", "A", "A", "medio", "baixa", "alta"],
            ["SIT-0020", "Avaliação", "Registro de sessão experimental", "C", "D", "A", "A", "baixo", "alta", "média"],
            ["SIT-0021", "Prática", "Registro de sessão experimental", "D", "A", "A", "A", "baixo", "alta", "baixa"],
            ["SIT-0022", "Conceitos", "Registro de sessão experimental", "C", "D", "C", "A", "medio", "média", "alta"],
            ["SIT-0023", "Introdução", "Acompanhamento de evolução", "D", "D", "C", "D", "alto", "média", "média"],
            ["SIT-0024", "Conceitos", "Observação inicial do processo", "B", "A", "B", "A", "medio", "baixa", "média"],
            ["SIT-0025", "Revisão", "Acompanhamento de evolução", "B", "D", "C", "A", "alto", "baixa", "alta"],
            ["SIT-0026", "Avaliação", "Acompanhamento de evolução", "A", "D", "D", "A", "baixo", "média", "baixa"],
            ["SIT-0027", "Prática", "Observação inicial do processo", "B", "C", "A", "D", "alto", "baixa", "alta"],
            ["SIT-0028", "Avaliação", "Registro de sessão experimental", "D", "B", "A", "A", "alto", "alta", "baixa"],
            ["SIT-0029", "Avaliação", "Acompanhamento de evolução", "B", "A", "D", "C", "baixo", "média", "média"],
            ["SIT-0030", "Introdução", "Dados coletados durante atividade", "D", "D", "B", "A", "medio", "média", "baixa"]
          ];
          sheet_SITUACOES.getRange(2, 1, d_sheet_SITUACOES.length, h_sheet_SITUACOES.length).setValues(d_sheet_SITUACOES);
          results.push('OK Situacoes: ' + d_sheet_SITUACOES.length + ' registros');
        } catch (e) {
          results.push('ERRO Situacoes: ' + e.message);
        }

        // Resultados
        try {
          var sheet_RESULTADOS = ss.getSheetByName('Resultados') || ss.insertSheet('Resultados');
          if (sheet_RESULTADOS.getLastRow() > 1) {
            sheet_RESULTADOS.deleteRows(2, sheet_RESULTADOS.getLastRow() - 1);
          }
          var h_sheet_RESULTADOS = ["ID", "ID_Usuario", "ID_Situacao", "ResultadoFinal", "ImpactoPontuacao", "DataResultado", "Dimensoes"];
          sheet_RESULTADOS.getRange(1, 1, 1, h_sheet_RESULTADOS.length).setValues([h_sheet_RESULTADOS]);
          var d_sheet_RESULTADOS = [
            ["RES-0001", "USR-558", "ativo", "D", 7.2, "2026-06-14 01:10:43", "D"],
            ["RES-0002", "USR-863", "ativo", "A", 7.0, "2026-06-19 01:10:43", "A"],
            ["RES-0003", "USR-945", "ativo", "C", 9.5, "2026-05-17 01:10:43", "B"],
            ["RES-0004", "USR-770", "inativo", "C", 6.8, "2026-06-18 01:10:43", "B"],
            ["RES-0005", "USR-964", "inativo", "C", 7.8, "2026-05-13 01:10:43", "D"],
            ["RES-0006", "USR-833", "ativo", "D", 5.6, "2026-05-04 01:10:43", "B"],
            ["RES-0007", "USR-102", "ativo", "C", 9.1, "2026-05-01 01:10:43", "C"],
            ["RES-0008", "USR-660", "ativo", "A", 5.2, "2026-06-11 01:10:43", "D"],
            ["RES-0009", "USR-243", "inativo", "A", 6.3, "2026-06-09 01:10:43", "B"],
            ["RES-0010", "USR-223", "inativo", "B", 7.3, "2026-05-02 01:10:43", "B"],
            ["RES-0011", "USR-833", "ativo", "A", 8.8, "2026-06-19 01:10:43", "A"],
            ["RES-0012", "USR-890", "inativo", "D", 8.3, "2026-05-20 01:10:43", "B"],
            ["RES-0013", "USR-610", "ativo", "A", 7.8, "2026-05-25 01:10:43", "D"],
            ["RES-0014", "USR-620", "ativo", "B", 8.7, "2026-06-13 01:10:43", "A"],
            ["RES-0015", "USR-496", "ativo", "D", 7.1, "2026-05-14 01:10:43", "B"],
            ["RES-0016", "USR-380", "inativo", "C", 9.9, "2026-05-24 01:10:43", "B"],
            ["RES-0017", "USR-241", "inativo", "C", 9.2, "2026-05-17 01:10:43", "C"],
            ["RES-0018", "USR-828", "ativo", "B", 6.0, "2026-05-07 01:10:43", "C"],
            ["RES-0019", "USR-803", "ativo", "B", 9.9, "2026-05-11 01:10:43", "B"],
            ["RES-0020", "USR-615", "ativo", "A", 8.8, "2026-05-17 01:10:43", "D"],
            ["RES-0021", "USR-696", "inativo", "C", 9.4, "2026-05-25 01:10:43", "A"],
            ["RES-0022", "USR-635", "ativo", "A", 6.9, "2026-06-18 01:10:43", "A"],
            ["RES-0023", "USR-304", "ativo", "B", 9.9, "2026-05-09 01:10:43", "C"],
            ["RES-0024", "USR-249", "ativo", "C", 6.0, "2026-05-18 01:10:43", "A"],
            ["RES-0025", "USR-467", "inativo", "A", 8.1, "2026-05-09 01:10:43", "B"],
            ["RES-0026", "USR-860", "inativo", "B", 7.6, "2026-05-02 01:10:43", "C"],
            ["RES-0027", "USR-809", "ativo", "D", 6.9, "2026-05-02 01:10:43", "C"],
            ["RES-0028", "USR-665", "ativo", "A", 8.7, "2026-06-09 01:10:43", "D"],
            ["RES-0029", "USR-949", "ativo", "C", 8.8, "2026-05-18 01:10:43", "B"],
            ["RES-0030", "USR-894", "ativo", "C", 7.2, "2026-05-09 01:10:43", "A"]
          ];
          sheet_RESULTADOS.getRange(2, 1, d_sheet_RESULTADOS.length, h_sheet_RESULTADOS.length).setValues(d_sheet_RESULTADOS);
          results.push('OK Resultados: ' + d_sheet_RESULTADOS.length + ' registros');
        } catch (e) {
          results.push('ERRO Resultados: ' + e.message);
        }

        // ProgressoJogo
        try {
          var sheet_PROGRESS = ss.getSheetByName('ProgressoJogo') || ss.insertSheet('ProgressoJogo');
          if (sheet_PROGRESS.getLastRow() > 1) {
            sheet_PROGRESS.deleteRows(2, sheet_PROGRESS.getLastRow() - 1);
          }
          var h_sheet_PROGRESS = ["ID", "ID_Usuario", "NivelAtual", "Pontuacao", "UltimaAtualizacao"];
          sheet_PROGRESS.getRange(1, 1, 1, h_sheet_PROGRESS.length).setValues([h_sheet_PROGRESS]);
          var d_sheet_PROGRESS = [
            ["PRO-0001", "USR-291", "baixo", 8.7, "criar"],
            ["PRO-0002", "USR-486", "baixo", 7.1, "visualizar"],
            ["PRO-0003", "USR-233", "baixo", 7.9, "visualizar"],
            ["PRO-0004", "USR-105", "medio", 7.3, "editar"],
            ["PRO-0005", "USR-140", "baixo", 9.6, "criar"],
            ["PRO-0006", "USR-892", "alto", 8.1, "visualizar"],
            ["PRO-0007", "USR-602", "alto", 7.8, "criar"],
            ["PRO-0008", "USR-902", "alto", 6.6, "editar"],
            ["PRO-0009", "USR-288", "alto", 5.3, "editar"],
            ["PRO-0010", "USR-525", "alto", 7.7, "criar"],
            ["PRO-0011", "USR-457", "baixo", 9.4, "visualizar"],
            ["PRO-0012", "USR-439", "medio", 6.5, "remover"],
            ["PRO-0013", "USR-674", "alto", 5.0, "visualizar"],
            ["PRO-0014", "USR-556", "baixo", 9.5, "editar"],
            ["PRO-0015", "USR-140", "alto", 9.4, "remover"],
            ["PRO-0016", "USR-915", "baixo", 6.9, "visualizar"],
            ["PRO-0017", "USR-125", "alto", 9.1, "visualizar"],
            ["PRO-0018", "USR-626", "alto", 7.2, "criar"],
            ["PRO-0019", "USR-403", "alto", 6.0, "criar"],
            ["PRO-0020", "USR-198", "baixo", 9.9, "criar"],
            ["PRO-0021", "USR-741", "baixo", 6.1, "editar"],
            ["PRO-0022", "USR-985", "alto", 7.1, "criar"],
            ["PRO-0023", "USR-310", "alto", 9.1, "remover"],
            ["PRO-0024", "USR-762", "baixo", 8.7, "editar"],
            ["PRO-0025", "USR-124", "medio", 8.6, "criar"],
            ["PRO-0026", "USR-189", "baixo", 5.3, "visualizar"],
            ["PRO-0027", "USR-671", "baixo", 7.9, "editar"],
            ["PRO-0028", "USR-364", "medio", 6.9, "remover"],
            ["PRO-0029", "USR-543", "medio", 5.1, "criar"],
            ["PRO-0030", "USR-647", "baixo", 8.2, "criar"]
          ];
          sheet_PROGRESS.getRange(2, 1, d_sheet_PROGRESS.length, h_sheet_PROGRESS.length).setValues(d_sheet_PROGRESS);
          results.push('OK ProgressoJogo: ' + d_sheet_PROGRESS.length + ' registros');
        } catch (e) {
          results.push('ERRO ProgressoJogo: ' + e.message);
        }

        // Escolhas
        try {
          var sheet_ESCOLHAS = ss.getSheetByName('Escolhas') || ss.insertSheet('Escolhas');
          if (sheet_ESCOLHAS.getLastRow() > 1) {
            sheet_ESCOLHAS.deleteRows(2, sheet_ESCOLHAS.getLastRow() - 1);
          }
          var h_sheet_ESCOLHAS = ["ID", "ID_Usuario", "ID_Situacao", "EscolhaFeita", "DataEscolha"];
          sheet_ESCOLHAS.getRange(1, 1, 1, h_sheet_ESCOLHAS.length).setValues([h_sheet_ESCOLHAS]);
          var d_sheet_ESCOLHAS = [
            ["ESC-0001", "USR-874", "ativo", "A", "2026-05-30 01:10:43"],
            ["ESC-0002", "USR-614", "ativo", "C", "2026-05-02 01:10:43"],
            ["ESC-0003", "USR-802", "ativo", "A", "2026-06-03 01:10:43"],
            ["ESC-0004", "USR-883", "inativo", "D", "2026-05-27 01:10:43"],
            ["ESC-0005", "USR-941", "ativo", "D", "2026-06-11 01:10:43"],
            ["ESC-0006", "USR-646", "ativo", "B", "2026-05-05 01:10:43"],
            ["ESC-0007", "USR-500", "ativo", "B", "2026-06-02 01:10:43"],
            ["ESC-0008", "USR-793", "ativo", "A", "2026-05-17 01:10:43"],
            ["ESC-0009", "USR-995", "ativo", "A", "2026-06-19 01:10:43"],
            ["ESC-0010", "USR-105", "inativo", "A", "2026-06-02 01:10:43"],
            ["ESC-0011", "USR-315", "ativo", "B", "2026-05-26 01:10:43"],
            ["ESC-0012", "USR-761", "ativo", "B", "2026-06-10 01:10:43"],
            ["ESC-0013", "USR-326", "ativo", "B", "2026-06-14 01:10:43"],
            ["ESC-0014", "USR-285", "ativo", "D", "2026-05-11 01:10:43"],
            ["ESC-0015", "USR-662", "ativo", "C", "2026-05-10 01:10:43"],
            ["ESC-0016", "USR-168", "ativo", "A", "2026-06-09 01:10:43"],
            ["ESC-0017", "USR-843", "inativo", "C", "2026-05-30 01:10:43"],
            ["ESC-0018", "USR-231", "ativo", "A", "2026-06-03 01:10:43"],
            ["ESC-0019", "USR-734", "ativo", "B", "2026-06-01 01:10:43"],
            ["ESC-0020", "USR-597", "inativo", "C", "2026-05-13 01:10:43"],
            ["ESC-0021", "USR-775", "ativo", "B", "2026-05-16 01:10:43"],
            ["ESC-0022", "USR-680", "ativo", "C", "2026-05-25 01:10:43"],
            ["ESC-0023", "USR-272", "ativo", "C", "2026-06-06 01:10:43"],
            ["ESC-0024", "USR-904", "ativo", "D", "2026-06-04 01:10:43"],
            ["ESC-0025", "USR-960", "ativo", "B", "2026-05-09 01:10:43"],
            ["ESC-0026", "USR-850", "ativo", "D", "2026-05-23 01:10:43"],
            ["ESC-0027", "USR-556", "ativo", "C", "2026-05-21 01:10:43"],
            ["ESC-0028", "USR-664", "ativo", "B", "2026-05-11 01:10:43"],
            ["ESC-0029", "USR-721", "ativo", "C", "2026-04-28 01:10:43"],
            ["ESC-0030", "USR-946", "ativo", "B", "2026-06-05 01:10:43"]
          ];
          sheet_ESCOLHAS.getRange(2, 1, d_sheet_ESCOLHAS.length, h_sheet_ESCOLHAS.length).setValues(d_sheet_ESCOLHAS);
          results.push('OK Escolhas: ' + d_sheet_ESCOLHAS.length + ' registros');
        } catch (e) {
          results.push('ERRO Escolhas: ' + e.message);
        }

        Logger.log(results.join('\n'));
        return results;
      } catch (error) {
        Logger.log("Erro em populateSyntheticData: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em populateSyntheticData: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em populateSyntheticData: " + error.message);
    throw error;
  }
}
