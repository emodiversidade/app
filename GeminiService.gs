/**
 * @file GeminiService.gs
 * @description Reflexao pedagogica personalizada via Google Gemini a partir do
 *              perfil de autonomia da jogadora (5 pilares + estagio). Segue o
 *              padrao endurecido da frota: retry com backoff exponencial +
 *              jitter para 429/5xx e excecoes de rede, parse defensivo e
 *              fallback deterministico — a tela nunca quebra por
 *              indisponibilidade do LLM.
 *
 * English Description:
 * This service generates a personalized, non-judgmental pedagogical reflection using
 * the Google Gemini API, grounded on the player's aggregated autonomy profile (five
 * pillars, developmental stage and balance flag). It never sends raw answers or any
 * personally identifying data to the LLM — only aggregated dimension totals. When the
 * GEMINI_API_KEY script property is absent or the API is unavailable, it degrades to a
 * deterministic local reflection so the feature always returns usable text.
 *
 * CONFIGURACAO:
 *   - Propriedade de script GEMINI_API_KEY (script.google.com → Configuracoes
 *     do projeto → Propriedades do script). Sem ela, o servico responde com a
 *     reflexao local (source: "fallback") e o restante do app segue normal.
 */

const GEMINI_EMODIVERSA = Object.freeze({
  MODEL: "gemini-2.0-flash",
  BASE_URL: "https://generativelanguage.googleapis.com/v1beta/models/",
  MAX_ATTEMPTS: 3,
  BASE_DELAY_MS: 600
});

// FROTA-07: modelo lido da property do script, nunca hardcoded; cai no padrão local.
function emodiversaModel_() {
  try {
    return PropertiesService.getScriptProperties().getProperty('GEMINI_MODEL') || 'gemini-2.0-flash';
  } catch (e) {
    LoggerService.info('Emodiversa/FROTA-07 property indisponível: ' + e.message);
    return 'gemini-2.0-flash';
  }
}

/**
 * Fronteira publica: reflexao da IA sobre a jornada de autonomia da pessoa
 * autenticada. Consumida pelo frontend via callServer("getAiReflectionData").
 * @returns {{success: boolean, data: ?Object, message: string, code: string, traceId: string}}
 *   data = { reflection: string, source: "gemini"|"fallback", model: string, profile: Object }
 */
function getAiReflectionData(options) {
  try {
    const user = requireAuth_();
    options = options || {};
    if (options.consentAcknowledged !== true) {
      throw new Error('Confirme o uso do perfil agregado antes de pedir a reflexao da IA.');
    }
    try {
      ConsentService.check(user.id, 'generative');
    } catch (consentError) {
      if (consentError && consentError.isConsentError) {
        throw new Error('Consentimento generativo ausente, expirado ou revogado.');
      }
      throw consentError;
    }
    const profile = getPlayerAutonomyProfile_(user.id);
    const generated = generateAutonomyReflection_(profile);
    let data = {
      reflection: generated.reflection,
      source: generated.source,
      model: generated.source === "gemini" ? emodiversaModel_() : "local",
      profile: profile
    };
    if (generated.source === 'gemini') {
      data = HumanReviewService.decorateResult(
        'emodiversa.autonomy-reflection',
        data,
        generated.reflection,
        { ownerId: user.id }
      );
    }
    return successResponse_(data);
  } catch (error) {
    return publicError_(error);
  }
}

/**
 * Gera a reflexao: tenta o Gemini e degrada para o texto local sem lancar.
 * @param {Object} profile Perfil agregado de computeAutonomyProfile_ (+ stage).
 * @returns {{reflection: string, source: string}}
 */
function generateAutonomyReflection_(profile) {
  try {
    const apiKey = PropertiesService.getScriptProperties().getProperty("GEMINI_API_KEY");
    if (apiKey) {
      // FROTA-05: bloqueio de rate limit / quota antes de atingir o provedor.
      try {
        if (typeof AiRateLimitService !== "undefined") {
          const _rl = AiRateLimitService.check("emodiversaReflection", String(profile && profile.answered));
          if (_rl && _rl.dedupHit && _rl.cached) return _rl.cached;
        }
      } catch (e) {
        if (e && e.isAiLimit) return { reflection: fallbackAutonomyReflection_(profile), source: "fallback" };
        throw e;
      }

      const _t06 = Date.now();
      const text = callGeminiEmodiversa_(apiKey, buildAutonomyReflectionPrompt_(profile));

      // FROTA-06: auditoria minima da geracao (sem prompt nem resposta).
      try {
        if (typeof AiAuditLogService !== "undefined") {
          AiAuditLogService.record({
            useCase: "emodiversaReflection",
            model: emodiversaModel_(),
            durationMs: Date.now() - _t06,
            status: text ? "ok" : "fallback",
            fallback: !text,
            errorCode: text ? "" : "PROVIDER_DEGRADED"
          });
        }
      } catch (_) {}

      if (text) {
        return { reflection: text, source: "gemini" };
      }
    }
    return { reflection: fallbackAutonomyReflection_(profile), source: "fallback" };
  } catch (error) {
    Logger.log("Erro em generateAutonomyReflection_: " + error.message);
    throw error;
  }
}

/**
 * Prompt com APENAS dados agregados (totais por pilar, estagio, equilibrio).
 * Nenhuma resposta individual ou dado pessoal vai ao LLM.
 * @param {Object} profile Perfil agregado da jogadora.
 * @returns {string}
 */
function buildAutonomyReflectionPrompt_(profile) {
  try {
    const dims = (profile.dimensions || [])
      .map(d => `- ${d.label}: ${d.value} pontos (${d.short || ""})`)
      .join("\n");
    const stage = profile.stage && profile.stage.name ? profile.stage.name : "inicio da jornada";
    return [
      "Voce e uma educadora acolhedora do jogo Emodiversa, que desenvolve a",
      "autonomia de meninas e mulheres por meio de escolhas em situacoes do",
      "cotidiano. O jogo NAO tem resposta certa: cada escolha movimenta cinco",
      "pilares da autonomia, e refletir sobre os trade-offs ja e crescimento.",
      "",
      "Perfil agregado da jogadora (totais acumulados por pilar):",
      dims || "- (nenhuma situacao respondida ainda)",
      `Situacoes respondidas: ${profile.answered || 0}`,
      `Estagio de desenvolvimento: ${stage}`,
      `Equilibrio entre os pilares (selo Emodiversa): ${profile.balanced ? "sim" : "ainda nao"}`,
      "",
      "Escreva, em portugues do Brasil, uma reflexao de 3 a 4 frases dirigida",
      "diretamente a jogadora (use 'voce'). Regras: (1) tom acolhedor e NUNCA",
      "julgador — jamais diga que ela errou ou que um pilar esta 'ruim';",
      "(2) reconheca o pilar mais fortalecido; (3) convide com gentileza a",
      "explorar o pilar menos movimentado como oportunidade, nao como falha;",
      "(4) nao invente numeros alem dos fornecidos e nao use formatacao",
      "markdown, apenas texto corrido."
    ].join("\n");
  } catch (error) {
    Logger.log("Erro em buildAutonomyReflectionPrompt_: " + error.message);
    throw error;
  }
}

/**
 * Reflexao deterministica local, usada quando a IA esta ausente ou falha.
 * Reusa o vocabulario nao-julgador do modelo de autonomia.
 * @param {Object} profile Perfil agregado da jogadora.
 * @returns {string}
 */
function fallbackAutonomyReflection_(profile) {
  try {
    if (!profile || !profile.answered) {
      return "Sua jornada esta apenas comecando. Cada situacao respondida vai revelar " +
        "como voce equilibra os cinco pilares da sua autonomia — e nao existe resposta errada.";
    }
    const strongest = profile.strongest || {};
    const weakest = profile.weakest || {};
    const parts = [
      `Em ${profile.answered} situacoes, voce ja deixou um retrato da sua autonomia.`
    ];
    if (strongest.label) {
      parts.push(`${strongest.emoji || "✨"} Sua ${strongest.label.toLowerCase()} tem sido seu pilar mais presente.`);
    }
    if (weakest.label && weakest.key !== strongest.key) {
      parts.push(`Que tal observar, sem cobranca, os momentos que convidam sua ${weakest.label.toLowerCase()}? Cada escolha e um convite, nunca uma prova.`);
    }
    if (profile.balanced) {
      parts.push("Seus pilares estao em equilibrio — esse e o espirito do selo Emodiversa.");
    }
    return parts.join(" ");
  } catch (error) {
    Logger.log("Erro em fallbackAutonomyReflection_: " + error.message);
    throw error;
  }
}

/**
 * Ponto unico de HTTP com o Gemini (generateContent), resiliente no padrao da
 * frota: retry + backoff exponencial + jitter para 429/5xx e excecoes de rede;
 * demais 4xx falham imediatamente. Nao lanca: devolve null para o chamador
 * degradar ao fallback.
 * @param {string} apiKey Chave da API.
 * @param {string} prompt Prompt textual.
 * @returns {?string} Texto gerado ou null.
 */
function callGeminiEmodiversa_(apiKey, prompt) {
  try {
    const url = GEMINI_EMODIVERSA.BASE_URL +
      encodeURIComponent(emodiversaModel_()) +
      ":generateContent?key=" + encodeURIComponent(apiKey);
    const options = {
      method: "post",
      contentType: "application/json",
      payload: JSON.stringify({
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.5 }
      }),
      muteHttpExceptions: true
    };

    let lastError = "Falha desconhecida.";
    for (let attempt = 1; attempt <= GEMINI_EMODIVERSA.MAX_ATTEMPTS; attempt++) {
      try {
        const response = UrlFetchApp.fetch(url, options);
        const code = response.getResponseCode();
        if (code >= 200 && code < 300) {
          const json = JsonUtils.safeParse(response.getContentText(), null);
          const text = json && json.candidates && json.candidates[0] &&
            json.candidates[0].content && json.candidates[0].content.parts &&
            json.candidates[0].content.parts[0] && json.candidates[0].content.parts[0].text;
          if (text) {
            return String(text).trim();
          }
          lastError = "Resposta do Gemini sem texto utilizavel.";
          break; // 2xx mal-formado nao melhora com retry
        }
        lastError = "Gemini HTTP " + code;
        if (code !== 429 && code < 500) {
          break; // 4xx permanente (chave invalida, payload errado...)
        }
      } catch (error) {
        lastError = "Rede: " + (error && error.message ? error.message : error);
      }
      if (attempt < GEMINI_EMODIVERSA.MAX_ATTEMPTS) {
        const backoff = GEMINI_EMODIVERSA.BASE_DELAY_MS * Math.pow(2, attempt - 1) +
          Math.floor(Math.random() * GEMINI_EMODIVERSA.BASE_DELAY_MS);
        Utilities.sleep(backoff);
      }
    }
    console.warn("GeminiService degradou para fallback: " + lastError);
    return null;
  } catch (error) {
    Logger.log("Erro em callGeminiEmodiversa_: " + error.message);
    throw error;
  }
}
