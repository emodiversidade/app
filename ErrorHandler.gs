/**
 * @file ErrorHandler.gs
 * @description Centraliza o tratamento de erros do sistema, garantindo logs adequados e retornos seguros.
 *
 * English Description:
 * This service provides centralized error handling for the backend application, catching exceptions
 * and logging them systematically to Stackdriver and the audit sheets. It generates unique trace
 * identifiers to assist debugging, wraps errors into standardized fail responses, and presents
 * clean, user-friendly messages to the frontend. This component prevents internal system implementation
 * details from leaking to end users.
 */

var ErrorHandler = (function() {
  function handleError(error, contexto, mensagemUsuario) {
    try {
      const errorMsg = error && error.message ? error.message : String(error);
      const stack = error && error.stack ? error.stack : new Error().stack;
      const logMessage = `[Erro em ${contexto}]: ${errorMsg}\nStack: ${stack}`;
    
      console.error(logMessage);
      LoggerService.info(logMessage);

      const traceId = Utilities.getUuid().split('-')[0];

      try {
        if (typeof logAction_ === "function") {
          logAction_("ERROR", "SYSTEM", `Contexto: ${contexto} | Erro: ${errorMsg} | Trace: ${traceId}`);
        }
      } catch (auditError) {
        console.warn("Falha ao gravar erro em Auditoria:", auditError.message);
      }

      const finalUserMessage = mensagemUsuario || "Não foi possível concluir a operação.";
      return StandardReturn.fail(finalUserMessage, null, {
        code: error && error.name ? error.name : "ERROR",
        traceId: traceId,
        context: contexto
      });
    } catch (error) {
      Logger.log("Erro em handleError: " + error.message);
      throw error;
    }
  }

  return {
    handleError: handleError
  };
})();
