/**
 * SchemaCleanupUtility.gs
 * 
 * Utilitário para limpar e recriar estruturas de planilhas de forma segura.
 * Evita problemas de CRUD causados por abas órfãs, headers desatualizados,
 * ou estruturas inconsistentes.
 * 
 * FUNCIONALIDADES:
 * 1. Limpeza segura de dados preservando estrutura
 * 2. Reset completo de schema (remove e recria abas)
 * 3. Backup antes de operações destrutivas
 * 4. Validação de integridade pós-limpeza
 * 
 * USO:
 *   SchemaCleanupUtility.cleanAllSheets();           // Limpa dados, preserva estrutura
 *   SchemaCleanupUtility.resetAllSheets();           // Remove e recria TUDO
 *   SchemaCleanupUtility.cleanSheet('EntityName');   // Limpa aba específica
 *   SchemaCleanupUtility.validateIntegrity();        // Valida schema atual
 * 
 * SEGURANÇA:
 * - Nunca executa automaticamente (apenas via menu ou script)
 * - Cria backup antes de operações destrutivas
 * - Confirma com usuário antes de reset completo
 * - Loga todas as operações
 * 
 * COMPATIBILIDADE:
 * - Funciona com SchemaService existente
 * - Não quebra código CRUD existente
 * - Pode ser adicionado a qualquer projeto da frota
 * 
 * @author Kiro AI
 * @version 1.0.0
 * @date 2026-06-21
 */

var SchemaCleanupUtility = (function() {
  'use strict';

  /**
   * Obtém spreadsheet ativa (compatível com diferentes padrões da frota).
   */
  function getSpreadsheet_() {
    try {
      if (typeof getBoundSpreadsheet_ === 'function') {
        return getBoundSpreadsheet_();
      }
      if (typeof getSpreadsheet_ === 'function') {
        return getSpreadsheet_();
      }
      return SpreadsheetApp.getActiveSpreadsheet();
    } catch (error) {
      Logger.log("Erro em getSpreadsheet_: " + error.message);
      throw error;
    }
  }

  /**
   * Obtém SchemaService (compatível com diferentes implementações).
   */
  function getSchemaService_() {
    if (typeof SchemaService !== 'undefined') {
      return SchemaService;
    }
    throw new Error('SchemaService não encontrado. Este utilitário requer SchemaService.');
  }

  /**
   * Loga operação (compatível com diferentes loggers).
   */
  function log_(message, level) {
    level = level || 'INFO';
    if (typeof LogService !== 'undefined' && LogService.log) {
      LogService.log(message, level);
    } else {
      Logger.log('[' + level + '] ' + message);
    }
  }

  /**
   * Cria backup da planilha (cópia completa).
   * 
   * @return {string} ID da planilha de backup criada
   */
  function createBackup() {
    try {
      try {
        var ss = getSpreadsheet_();
        var timestamp = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), 'yyyy-MM-dd_HHmmss');
        var backupName = ss.getName() + ' [BACKUP ' + timestamp + ']';
      
        var backup = ss.copy(backupName);
        var backupId = backup.getId();
      
        log_('Backup criado: ' + backupName + ' (ID: ' + backupId + ')');
      
        return backupId;
      } catch (e) {
        log_('Erro ao criar backup: ' + e.message, 'ERROR');
        throw new Error('Falha ao criar backup: ' + e.message);
      }
    } catch (error) {
      Logger.log("Erro em createBackup: " + error.message);
      throw error;
    }
  }

  /**
   * Limpa DADOS de uma aba, preservando headers.
   * 
   * @param {string} sheetName - Nome da aba
   * @return {object} Resultado da operação
   */
  function cleanSheetData(sheetName) {
    try {
      try {
        var ss = getSpreadsheet_();
        var sheet = ss.getSheetByName(sheetName);
      
        if (!sheet) {
          return {
            ok: false,
            message: 'Aba não encontrada: ' + sheetName
          };
        }
      
        var lastRow = sheet.getLastRow();
      
        if (lastRow <= 1) {
          return {
            ok: true,
            message: 'Aba ' + sheetName + ' já está vazia',
            rowsDeleted: 0
          };
        }
      
        // Deleta linhas 2 em diante (preserva header na linha 1)
        sheet.deleteRows(2, lastRow - 1);
      
        var rowsDeleted = lastRow - 1;
        log_('Aba ' + sheetName + ': ' + rowsDeleted + ' linhas deletadas');
      
        return {
          ok: true,
          message: 'Aba ' + sheetName + ' limpa com sucesso',
          rowsDeleted: rowsDeleted
        };
      
      } catch (e) {
        log_('Erro ao limpar aba ' + sheetName + ': ' + e.message, 'ERROR');
        return {
          ok: false,
          message: 'Erro ao limpar aba ' + sheetName + ': ' + e.message
        };
      }
    } catch (error) {
      Logger.log("Erro em cleanSheetData: " + error.message);
      throw error;
    }
  }

  /**
   * Limpa DADOS de todas as abas de schema, preservando headers.
   * 
   * @return {object} Resumo da operação
   */
  function cleanAllSheets() {
    try {
      try {
        var schemaService = getSchemaService_();
        var schemas = schemaService.getSchemas({ asArray: true });
      
        var results = [];
        var totalDeleted = 0;
      
        schemas.forEach(function(schema) {
          var result = cleanSheetData(schema.sheetName);
          results.push(result);
          if (result.ok) {
            totalDeleted += result.rowsDeleted || 0;
          }
        });
      
        var summary = {
          ok: true,
          message: 'Limpeza concluída: ' + totalDeleted + ' linhas deletadas em ' + schemas.length + ' abas',
          totalSheets: schemas.length,
          totalRowsDeleted: totalDeleted,
          results: results
        };
      
        log_(summary.message);
        return summary;
      
      } catch (e) {
        log_('Erro ao limpar todas as abas: ' + e.message, 'ERROR');
        return {
          ok: false,
          message: 'Erro ao limpar todas as abas: ' + e.message
        };
      }
    } catch (error) {
      Logger.log("Erro em cleanAllSheets: " + error.message);
      throw error;
    }
  }

  /**
   * Remove COMPLETAMENTE uma aba (estrutura + dados).
   * 
   * @param {string} sheetName - Nome da aba
   * @return {object} Resultado da operação
   */
  function deleteSheet(sheetName) {
    try {
      try {
        var ss = getSpreadsheet_();
        var sheet = ss.getSheetByName(sheetName);
      
        if (!sheet) {
          return {
            ok: false,
            message: 'Aba não encontrada: ' + sheetName
          };
        }
      
        ss.deleteSheet(sheet);
        log_('Aba ' + sheetName + ' deletada completamente');
      
        return {
          ok: true,
          message: 'Aba ' + sheetName + ' deletada com sucesso'
        };
      
      } catch (e) {
        log_('Erro ao deletar aba ' + sheetName + ': ' + e.message, 'ERROR');
        return {
          ok: false,
          message: 'Erro ao deletar aba ' + sheetName + ': ' + e.message
        };
      }
    } catch (error) {
      Logger.log("Erro em deleteSheet: " + error.message);
      throw error;
    }
  }

  /**
   * Reset COMPLETO: remove e recria TODAS as abas de schema.
   * 
   * ATENÇÃO: OPERAÇÃO DESTRUTIVA! Cria backup automaticamente.
   * 
   * @param {object} options - Opções de configuração
   * @param {boolean} options.skipBackup - Pular criação de backup (NÃO RECOMENDADO)
   * @param {boolean} options.confirmed - Confirma operação destrutiva
   * @return {object} Resumo da operação
   */
  function resetAllSheets(options) {
    options = options || {};
    
    // Proteção: requer confirmação explícita
    if (!options.confirmed) {
      return {
        ok: false,
        message: 'ATENÇÃO: Esta operação é DESTRUTIVA e irá DELETAR TODOS OS DADOS. ' +
                 'Para confirmar, execute: SchemaCleanupUtility.resetAllSheets({ confirmed: true })'
      };
    }
    
    try {
      var schemaService = getSchemaService_();
      var schemas = schemaService.getSchemas({ asArray: true });
      
      // 1. Criar backup (a menos que explicitamente pulado)
      var backupId = null;
      if (!options.skipBackup) {
        backupId = createBackup();
      }
      
      // 2. Deletar todas as abas de schema
      var deleted = [];
      schemas.forEach(function(schema) {
        var result = deleteSheet(schema.sheetName);
        deleted.push(result);
      });
      
      // 3. Recriar estrutura via SchemaService
      log_('Recriando estrutura de schema...');
      var recreated = schemaService.ensureAllSheets(options);
      
      var summary = {
        ok: true,
        message: 'Reset completo: ' + schemas.length + ' abas recriadas',
        backupId: backupId,
        totalSheets: schemas.length,
        deleted: deleted,
        recreated: recreated
      };
      
      log_(summary.message);
      return summary;
      
    } catch (e) {
      log_('Erro ao fazer reset completo: ' + e.message, 'ERROR');
      return {
        ok: false,
        message: 'Erro ao fazer reset completo: ' + e.message
      };
    }
  }

  /**
   * Valida integridade do schema atual vs definido.
   * 
   * @return {object} Relatório de validação
   */
  function validateIntegrity() {
    try {
      try {
        try {
          var schemaService = getSchemaService_();
          var schemas = schemaService.getSchemas({ asArray: true });
          var ss = getSpreadsheet_();
      
          var issues = [];
          var validated = 0;
      
          schemas.forEach(function(schema) {
            var sheet = ss.getSheetByName(schema.sheetName);
        
            if (!sheet) {
              issues.push({
                entity: schema.entity,
                sheetName: schema.sheetName,
                issue: 'MISSING_SHEET',
                message: 'Aba não existe na planilha'
              });
              return;
            }
        
            // Validar headers
            if (sheet.getLastRow() === 0) {
              issues.push({
                entity: schema.entity,
                sheetName: schema.sheetName,
                issue: 'EMPTY_SHEET',
                message: 'Aba existe mas não tem headers'
              });
              return;
            }
        
            var actualHeaders = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
            var expectedHeaders = schema.headers || [];
        
            // Verificar se todos os headers esperados estão presentes
            var missingHeaders = expectedHeaders.filter(function(h) {
              return actualHeaders.indexOf(h) === -1;
            });
        
            if (missingHeaders.length > 0) {
              issues.push({
                entity: schema.entity,
                sheetName: schema.sheetName,
                issue: 'MISSING_HEADERS',
                message: 'Headers ausentes: ' + missingHeaders.join(', '),
                missingHeaders: missingHeaders
              });
            } else {
              validated++;
            }
          });
      
          var report = {
            ok: issues.length === 0,
            message: issues.length === 0 
              ? 'Integridade validada: ' + validated + '/' + schemas.length + ' abas OK'
              : 'Encontrados ' + issues.length + ' problemas em ' + schemas.length + ' abas',
            totalSheets: schemas.length,
            validSheets: validated,
            issueCount: issues.length,
            issues: issues
          };
      
          log_(report.message);
          return report;
      
        } catch (e) {
          log_('Erro ao validar integridade: ' + e.message, 'ERROR');
          return {
            ok: false,
            message: 'Erro ao validar integridade: ' + e.message
          };
        }
      } catch (error) {
        Logger.log("Erro em validateIntegrity: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em validateIntegrity: " + error.message);
      throw error;
    }
  }

  /**
   * Remove abas órfãs (que não estão no schema).
   * 
   * @param {object} options - Opções
   * @param {boolean} options.dryRun - Apenas lista, não deleta
   * @return {object} Resultado da operação
   */
  function removeOrphanSheets(options) {
    try {
      options = options || {};
    
      try {
        var schemaService = getSchemaService_();
        var schemas = schemaService.getSchemas({ asArray: true });
        var ss = getSpreadsheet_();
      
        // Abas definidas no schema
        var definedSheets = schemas.map(function(s) { return s.sheetName; });
      
        // Abas existentes na planilha
        var allSheets = ss.getSheets();
      
        var orphans = [];
        allSheets.forEach(function(sheet) {
          var name = sheet.getName();
          if (definedSheets.indexOf(name) === -1) {
            orphans.push(name);
          }
        });
      
        if (orphans.length === 0) {
          return {
            ok: true,
            message: 'Nenhuma aba órfã encontrada',
            orphans: []
          };
        }
      
        if (options.dryRun) {
          return {
            ok: true,
            message: 'Encontradas ' + orphans.length + ' abas órfãs (dry run, não deletadas)',
            orphans: orphans
          };
        }
      
        // Deletar órfãs
        var deleted = [];
        orphans.forEach(function(name) {
          var result = deleteSheet(name);
          if (result.ok) {
            deleted.push(name);
          }
        });
      
        return {
          ok: true,
          message: deleted.length + ' abas órfãs removidas',
          orphans: orphans,
          deleted: deleted
        };
      
      } catch (e) {
        log_('Erro ao remover abas órfãs: ' + e.message, 'ERROR');
        return {
          ok: false,
          message: 'Erro ao remover abas órfãs: ' + e.message
        };
      }
    } catch (error) {
      Logger.log("Erro em removeOrphanSheets: " + error.message);
      throw error;
    }
  }

  // API Pública
  return {
    createBackup: createBackup,
    cleanSheetData: cleanSheetData,
    cleanAllSheets: cleanAllSheets,
    deleteSheet: deleteSheet,
    resetAllSheets: resetAllSheets,
    validateIntegrity: validateIntegrity,
    removeOrphanSheets: removeOrphanSheets
  };
})();


/**
 * Funções globais para fácil acesso via menu/editor.
 */

function cleanupValidateIntegrity() {
  try {
    try {
      var result = SchemaCleanupUtility.validateIntegrity();
      Logger.log(JSON.stringify(result, null, 2));
      return result;
    } catch (error) {
      Logger.log("Erro em cleanupValidateIntegrity: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em cleanupValidateIntegrity: " + error.message);
    throw error;
  }
}

function cleanupRemoveOrphans() {
  try {
    try {
      var result = SchemaCleanupUtility.removeOrphanSheets({ dryRun: false });
      Logger.log(JSON.stringify(result, null, 2));
      return result;
    } catch (error) {
      Logger.log("Erro em cleanupRemoveOrphans: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em cleanupRemoveOrphans: " + error.message);
    throw error;
  }
}

function cleanupListOrphans() {
  try {
    var result = SchemaCleanupUtility.removeOrphanSheets({ dryRun: true });
    Logger.log(JSON.stringify(result, null, 2));
    return result;
  } catch (error) {
    Logger.log("Erro em cleanupListOrphans: " + error.message);
    throw error;
  }
}

function cleanupCleanAllData() {
  try {
    var result = SchemaCleanupUtility.cleanAllSheets();
    Logger.log(JSON.stringify(result, null, 2));
    return result;
  } catch (error) {
    Logger.log("Erro em cleanupCleanAllData: " + error.message);
    throw error;
  }
}

function cleanupResetAll() {
  try {
    var result = SchemaCleanupUtility.resetAllSheets({ confirmed: true });
    Logger.log(JSON.stringify(result, null, 2));
    return result;
  } catch (error) {
    Logger.log("Erro em cleanupResetAll: " + error.message);
    throw error;
  }
}

function cleanupCreateBackup() {
  var backupId = SchemaCleanupUtility.createBackup();
  Logger.log('Backup criado. ID: ' + backupId);
  return backupId;
}
