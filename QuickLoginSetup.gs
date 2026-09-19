/**
 * QuickLoginSetup.gs
 * Script de configuração rápida do sistema de login.
 * Execute a função setupLoginComplete() no editor do Apps Script.
 */

/**
 * Configuração completa do login em um único comando.
 * Cria aba, seed de usuários e testa login.
 */
function setupLoginComplete() {
  try {
    try {
      try {
        Logger.clear();
        Logger.log('=== INÍCIO DA CONFIGURAÇÃO DE LOGIN ===\n');
  
        try {
          // 1. Verificar configuração da planilha
          var spreadsheetId = PropertiesService.getScriptProperties().getProperty('SPREADSHEETS_ID');
          if (!spreadsheetId) {
            Logger.log('❌ ERRO: SPREADSHEETS_ID não configurado nas Script Properties');
            Logger.log('   Configure manualmente: Project Settings → Script Properties → Add Property');
            Logger.log('   Nome: SPREADSHEETS_ID');
            Logger.log('   Valor: [ID da sua planilha do Google Sheets]');
            return;
          }
          Logger.log('✅ SPREADSHEETS_ID configurado: ' + spreadsheetId);
    
          // 2. Abrir planilha
          var ss = SpreadsheetApp.openById(spreadsheetId);
          Logger.log('✅ Planilha aberta: ' + ss.getName());
    
          // 3. Criar/garantir aba Usuarios
          Logger.log('\n--- Criando/verificando aba Usuarios ---');
          var sheet = ss.getSheetByName('Usuarios');
    
          if (!sheet) {
            Logger.log('⚙️  Criando aba "Usuarios"...');
            sheet = ss.insertSheet('Usuarios');
      
            // Adicionar cabeçalho
            sheet.getRange(1, 1, 1, 4).setValues([['ID', 'Login', 'Senha', 'Perfil']]);
            sheet.getRange(1, 1, 1, 4).setFontWeight('bold');
            sheet.setFrozenRows(1);
      
            Logger.log('✅ Aba "Usuarios" criada com cabeçalho');
          } else {
            Logger.log('✅ Aba "Usuarios" já existe');
          }
    
          // 4. Verificar se já tem usuários
          var userCount = sheet.getLastRow() - 1;
          Logger.log('📊 Usuários existentes: ' + userCount);
    
          if (userCount === 0) {
            Logger.log('\n--- Criando usuários administradores ---');
      
            // Criar 15 usuários sintéticos
            var users = [
              ['SYN-ADMIN-01', 'admin01', 'admin123', 'admin'],
              ['SYN-ADMIN-02', 'admin02', 'admin123', 'admin'],
              ['SYN-ADMIN-03', 'admin03', 'admin123', 'admin'],
              ['SYN-ADMIN-04', 'admin04', 'admin123', 'admin'],
              ['SYN-ADMIN-05', 'admin05', 'admin123', 'admin'],
              ['SYN-ADMIN-06', 'admin06', 'admin123', 'admin'],
              ['SYN-ADMIN-07', 'admin07', 'admin123', 'admin'],
              ['SYN-ADMIN-08', 'admin08', 'admin123', 'admin'],
              ['SYN-ADMIN-09', 'admin09', 'admin123', 'admin'],
              ['SYN-ADMIN-10', 'admin10', 'admin123', 'admin'],
              ['SYN-ADMIN-11', 'admin11', 'admin123', 'admin'],
              ['SYN-ADMIN-12', 'admin12', 'admin123', 'admin'],
              ['SYN-ADMIN-13', 'admin13', 'admin123', 'admin'],
              ['SYN-ADMIN-14', 'admin14', 'admin123', 'admin'],
              ['SYN-ADMIN-15', 'admin15', 'admin123', 'admin']
            ];
      
            sheet.getRange(2, 1, users.length, 4).setValues(users);
            Logger.log('✅ 15 usuários administradores criados');
          } else {
            Logger.log('ℹ️  Usuários já existem, pulando criação');
          }
    
          // 5. Testar login
          Logger.log('\n--- Testando login ---');
          var testResult = doLogin('admin01', 'admin123');
    
          if (testResult.success) {
            Logger.log('✅ TESTE DE LOGIN: SUCESSO');
            Logger.log('   Usuário logado: ' + testResult.user.login);
            Logger.log('   Perfil: ' + testResult.user.perfil);
          } else {
            Logger.log('❌ TESTE DE LOGIN: FALHOU');
            Logger.log('   Mensagem: ' + testResult.message);
          }
    
          // 6. Resumo final
          Logger.log('\n=== CONFIGURAÇÃO CONCLUÍDA ===');
          Logger.log('\n📋 CREDENCIAIS DISPONÍVEIS:');
          Logger.log('   Username: admin01, admin02, ... admin15');
          Logger.log('[LGPD] Evento registrado; detalhes sensíveis omitidos.');
          Logger.log('   Perfil: admin');
          Logger.log('\n🌐 Próximo passo: Abra a aplicação web e faça login');
    
        } catch (error) {
          Logger.log('\n❌ ERRO: ' + error.message);
          Logger.log('\nStack trace:');
          Logger.log(error.stack);
        }
      } catch (error) {
        Logger.log("Erro em setupLoginComplete: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em setupLoginComplete: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em setupLoginComplete: " + error.message);
    throw error;
  }
}

/**
 * Apenas diagnóstico - não altera nada.
 */
function diagnosticarLogin() {
  try {
    try {
      try {
        Logger.clear();
        Logger.log('=== DIAGNÓSTICO DO SISTEMA DE LOGIN ===\n');
  
        try {
          // 1. Script Properties
          var spreadsheetId = PropertiesService.getScriptProperties().getProperty('SPREADSHEETS_ID');
          Logger.log('1. SPREADSHEET_ID: ' + (spreadsheetId || '❌ NÃO CONFIGURADO'));
    
          if (!spreadsheetId) {
            Logger.log('\n⚠️  Configure SPREADSHEETS_ID antes de continuar');
            return;
          }
    
          // 2. Planilha
          var ss = SpreadsheetApp.openById(spreadsheetId);
          Logger.log('2. Planilha: ✅ ' + ss.getName());
    
          // 3. Aba Usuarios
          var sheet = ss.getSheetByName('Usuarios');
          if (!sheet) {
            Logger.log('3. Aba "Usuarios": ❌ NÃO ENCONTRADA');
            Logger.log('   Abas disponíveis: ' + ss.getSheets().map(function(s) { return s.getName(); }).join(', '));
            Logger.log('\n💡 Solução: Execute setupLoginComplete()');
            return;
          }
          Logger.log('3. Aba "Usuarios": ✅ ENCONTRADA');
    
          // 4. Estrutura
          if (sheet.getLastRow() < 1) {
            Logger.log('4. Estrutura: ❌ ABA VAZIA');
            Logger.log('\n💡 Solução: Execute setupLoginComplete()');
            return;
          }
    
          var headers = sheet.getRange(1, 1, 1, 4).getValues()[0];
          Logger.log('4. Cabeçalhos: ' + JSON.stringify(headers));
    
          // 5. Usuários
          var userCount = sheet.getLastRow() - 1;
          Logger.log('5. Total de usuários: ' + userCount);
    
          if (userCount === 0) {
            Logger.log('   ❌ NENHUM USUÁRIO CADASTRADO');
            Logger.log('\n💡 Solução: Execute setupLoginComplete()');
            return;
          }
    
          Logger.log('\n📋 Primeiros usuários:');
          var users = sheet.getRange(2, 1, Math.min(5, userCount), 4).getValues();
          users.forEach(function(user) {
            Logger.log('   • Login: ' + user[1] + ' | Perfil: ' + user[3] + ' | ID: ' + user[0]);
          });
    
          // 6. Teste de login
          Logger.log('\n--- Testando autenticação ---');
          var firstUser = users[0];
          var testUsername = firstUser[1];
          var testPassword = firstUser[2];
    
          Logger.log('[LGPD] Evento registrado; detalhes sensíveis omitidos.');
          var result = doLogin(testUsername, testPassword);
    
          if (result.success) {
            Logger.log('✅ AUTENTICAÇÃO: FUNCIONANDO');
          } else {
            Logger.log('❌ AUTENTICAÇÃO: FALHOU');
            Logger.log('   Mensagem: ' + result.message);
          }
    
          Logger.log('\n=== FIM DO DIAGNÓSTICO ===');
    
        } catch (error) {
          Logger.log('\n❌ ERRO: ' + error.message);
        }
      } catch (error) {
        Logger.log("Erro em diagnosticarLogin: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em diagnosticarLogin: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em diagnosticarLogin: " + error.message);
    throw error;
  }
}

/**
 * Teste simples de login.
 */
function testarLogin() {
  try {
    try {
      try {
        Logger.clear();
        Logger.log('=== TESTE DE LOGIN ===\n');
  
        var username = 'admin01';
        var password = 'admin123';
  
        Logger.log('Tentando login com:');
        Logger.log('  Username: ' + username);
        Logger.log('[LGPD] Evento registrado; detalhes sensíveis omitidos.');
        Logger.log('');
  
        var result = doLogin(username, password);
  
        Logger.log('Resultado: ' + JSON.stringify(result, null, 2));
  
        if (result.success) {
          Logger.log('\n✅ LOGIN BEM-SUCEDIDO');
        } else {
          Logger.log('\n❌ LOGIN FALHOU: ' + result.message);
        }
      } catch (error) {
        Logger.log("Erro em testarLogin: " + error.message);
        throw error; // Re-lança para tratamento superior
      }
    } catch (error) {
      Logger.log("Erro em testarLogin: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em testarLogin: " + error.message);
    throw error;
  }
}

/**
 * Reseta tudo e reconfigura do zero.
 * CUIDADO: Apaga todos os usuários existentes!
 */
function resetarSistemaLogin() {
  try {
    try {
      var ui = SpreadsheetApp.getUi();
      var response = ui.alert(
        'Resetar Sistema de Login',
        'ATENÇÃO: Isso irá APAGAR TODOS os usuários existentes e recriar do zero.\n\nTem certeza?',
        ui.ButtonSet.YES_NO
      );
  
      if (response !== ui.Button.YES) {
        Logger.log('Operação cancelada pelo usuário');
        return;
      }
  
      Logger.clear();
      Logger.log('=== RESETANDO SISTEMA DE LOGIN ===\n');
  
      try {
        var spreadsheetId = PropertiesService.getScriptProperties().getProperty('SPREADSHEETS_ID');
        var ss = SpreadsheetApp.openById(spreadsheetId);
    
        // Apagar aba Usuarios se existir
        var sheet = ss.getSheetByName('Usuarios');
        if (sheet) {
          Logger.log('🗑️  Apagando aba Usuarios existente...');
          ss.deleteSheet(sheet);
        }
    
        // Recriar do zero
        Logger.log('🔄 Recriando sistema...\n');
        setupLoginComplete();
    
      } catch (error) {
        Logger.log('❌ ERRO: ' + error.message);
      }
    } catch (error) {
      Logger.log("Erro em resetarSistemaLogin: " + error.message);
      throw error; // Re-lança para tratamento superior
    }
  } catch (error) {
    Logger.log("Erro em resetarSistemaLogin: " + error.message);
    throw error;
  }
}
