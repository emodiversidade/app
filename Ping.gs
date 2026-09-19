/**
 * codex-frontend-backend-healthcheck
 * Sonda de saude leve: confirma que o frontend alcanca o backend via
 * google.script.run e recebe o envelope padrao do projeto.
 *
 * - Sem efeitos colaterais, sem dependencia de sessao ou planilha.
 * - Arquivo isolado de proposito: nao altera nenhuma rota existente.
 * - Espelha o contrato local ({ success, data, message, code, traceId }),
 *   entao o callServer/unwrap do frontend desempacota .data normalmente.
 *
 * English Description:
 * This lightweight health check endpoint confirms that the frontend can successfully communicate with
 * the Google Apps Script backend. It handles incoming requests, bypasses sheet database dependencies,
 * and returns a standard success response containing basic server details and the current timestamp.
 * This component is isolated to prevent side effects and verify communication status without affecting
 * active user sessions.
 */
function ping() {
  try {
    return successResponse_({
      status: "ok",
      service: "backend",
      time: new Date().toISOString()
    });
  } catch (error) {
    Logger.log("Erro em ping: " + error.message);
    throw error;
  }
}
