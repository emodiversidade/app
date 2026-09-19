/**
 * ============================================================================
 * ThemeDigestService.gs — Radar de Novidades do Tema + Síntese via Gemini
 * ============================================================================
 * PROJETO: Emodiversa
 * TEMA: Educacao socioemocional e autonomia
 *
 * DESCRIÇÃO:
 *   Raspa novidades recentes da internet sobre o tema do projeto (feed RSS de
 *   busca do Google News, parse defensivo via XmlService) e pede ao Gemini uma
 *   síntese pedagógica curta conectando as novidades ao contexto dos
 *   estudantes. Padrão endurecido da frota: retry com backoff exponencial +
 *   jitter para HTTP 429/5xx e exceções de rede, 4xx permanente falha
 *   imediatamente, parse defensivo e fallback determinístico local — o
 *   recurso nunca quebra sem GEMINI_API_KEY ou com a internet indisponível.
 *
 * FRONTEIRA PÚBLICA (google.script.run):
 *   - getThemeNewsDigest(force?) → envelope { success, data, error/meta }
 *     data = { theme, query, items: [{title, source, published, link}],
 *              synthesis, source: "gemini"|"local", model, generatedAt,
 *              cached: boolean }
 *
 * CONFIGURAÇÃO:
 *   - Propriedade de script GEMINI_API_KEY (opcional: sem ela a síntese é
 *     gerada localmente a partir das manchetes raspadas).
 *   - Cache de 30 minutos (CacheService) para não martelar o RSS nem a IA;
 *     use getThemeNewsDigest(true) para forçar atualização.
 *
 * PRIVACIDADE: nenhum dado de aluno é enviado à internet ou à IA — apenas as
 * manchetes públicas raspadas vão ao Gemini.
 * ============================================================================
 */

var THEME_NEWS_CFG = {
  THEME: 'Educacao socioemocional e autonomia',
  QUERY: '"educacao socioemocional" OR "autonomia de meninas"',
  AUDIENCE: 'meninas e mulheres que desenvolvem autonomia por meio de escolhas no jogo',
  RSS_BASE: 'https://news.google.com/rss/search',
  MAX_ITEMS: 8,
  CACHE_KEY: 'theme_news_digest_v1',
  CACHE_SECONDS: 1800,
  GEMINI_MODEL: 'gemini-2.0-flash',
  GEMINI_BASE_URL: 'https://generativelanguage.googleapis.com/v1beta/models/',
  MAX_ATTEMPTS: 3,
  BASE_DELAY_MS: 600
};

/**
 * Fronteira pública: radar de novidades do tema com síntese pedagógica.
 * Nunca lança: qualquer falha vira envelope de erro.
 * @param {boolean} force Ignora o cache quando true.
 * @return {Object} Envelope padrão do projeto.
 */
function getThemeNewsDigest(force) {
  try {
    try {
      var cache = CacheService.getScriptCache();
      if (!force) {
        var cachedRaw = cache.get(THEME_NEWS_CFG.CACHE_KEY);
        if (cachedRaw) {
          var cachedData = themeNewsSafeParse_(cachedRaw);
          if (cachedData) {
            cachedData.cached = true;
            return themeNewsOk_(cachedData);
          }
        }
      }

      var items = fetchThemeNewsItems_();
      var synthesis = synthesizeThemeNews_(items);
      var data = {
        theme: THEME_NEWS_CFG.THEME,
        query: THEME_NEWS_CFG.QUERY,
        items: items,
        synthesis: synthesis.text,
        source: synthesis.source,
        model: synthesis.source === 'gemini' ? THEME_NEWS_CFG.GEMINI_MODEL : 'local',
        generatedAt: new Date().toISOString(),
        cached: false
      };

      // Só cacheia quando a raspagem trouxe conteúdo de verdade.
      if (items.length > 0) {
        try {
          cache.put(THEME_NEWS_CFG.CACHE_KEY, JSON.stringify(data), THEME_NEWS_CFG.CACHE_SECONDS);
        } catch (cacheError) {
          Logger.log('ThemeDigest: cache indisponivel: ' + cacheError.message);
        }
      }
      return themeNewsOk_(data);
    } catch (error) {
      return themeNewsFail_(error);
    }
  } catch (error) {
    Logger.log("Erro em getThemeNewsDigest: " + error.message);
    throw error;
  }
}

/**
 * Raspa o RSS de busca do Google News para a consulta do tema, com o mesmo
 * endurecimento das chamadas Gemini da frota. Devolve [] quando a internet
 * está indisponível (o chamador degrada para a síntese local explicativa).
 * @return {Array<{title: string, source: string, published: string, link: string}>}
 */
function fetchThemeNewsItems_() {
  try {
    var url = THEME_NEWS_CFG.RSS_BASE +
      '?q=' + encodeURIComponent(THEME_NEWS_CFG.QUERY) +
      '&hl=pt-BR&gl=BR&ceid=BR:pt-150';
    var options = { method: 'get', muteHttpExceptions: true };

    var lastError = 'Falha desconhecida.';
    for (var attempt = 1; attempt <= THEME_NEWS_CFG.MAX_ATTEMPTS; attempt++) {
      try {
        var response = UrlFetchApp.fetch(url, options);
        var code = response.getResponseCode();
        if (code >= 200 && code < 300) {
          return parseThemeNewsXml_(response.getContentText());
        }
        lastError = 'RSS HTTP ' + code;
        if (code !== 429 && code < 500) {
          break; // 4xx permanente não melhora com retry
        }
      } catch (error) {
        lastError = 'Rede: ' + (error && error.message ? error.message : error);
      }
      if (attempt < THEME_NEWS_CFG.MAX_ATTEMPTS) {
        Utilities.sleep(themeNewsBackoff_(attempt));
      }
    }
    Logger.log('ThemeDigest: raspagem indisponivel: ' + lastError);
    return [];
  } catch (error) {
    Logger.log("Erro em fetchThemeNewsItems_: " + error.message);
    throw error;
  }
}

/**
 * Parse defensivo do XML do RSS: qualquer item malformado é pulado e um feed
 * inválido inteiro vira lista vazia (nunca lança).
 * @param {string} xmlText Corpo XML do feed.
 * @return {Array<Object>}
 */
function parseThemeNewsXml_(xmlText) {
  try {
    var root = XmlService.parse(xmlText).getRootElement();
    var channel = root.getChild('channel');
    if (!channel) return [];
    var entries = channel.getChildren('item') || [];
    var items = [];
    for (var i = 0; i < entries.length && items.length < THEME_NEWS_CFG.MAX_ITEMS; i++) {
      try {
        var entry = entries[i];
        var title = String(entry.getChildText('title') || '').trim();
        if (!title) continue;
        items.push({
          title: title,
          source: String(entry.getChildText('source') || '').trim(),
          published: String(entry.getChildText('pubDate') || '').trim(),
          link: String(entry.getChildText('link') || '').trim()
        });
      } catch (itemError) {
        // item malformado: segue para o próximo
      }
    }
    return items;
  } catch (error) {
    Logger.log('ThemeDigest: XML do RSS invalido: ' + error.message);
    return [];
  }
}

/**
 * Síntese pedagógica: tenta o Gemini sobre as manchetes raspadas e degrada
 * para o resumo determinístico local sem lançar.
 * @param {Array<Object>} items Manchetes raspadas.
 * @return {{text: string, source: string}}
 */
function synthesizeThemeNews_(items) {
  try {
    try {
      var apiKey = PropertiesService.getScriptProperties().getProperty('GEMINI_API_KEY');
      if (apiKey && items.length > 0) {
        var text = callThemeNewsGemini_(apiKey, buildThemeNewsPrompt_(items));
        if (text) {
          return { text: text, source: 'gemini' };
        }
      }
      return { text: localThemeNewsSynthesis_(items), source: 'local' };
    } catch (error) {
      Logger.log("Erro em synthesizeThemeNews_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em synthesizeThemeNews_: " + error.message);
    throw error;
  }
}

/**
 * Prompt pedagógico: só manchetes públicas (título/fonte/data) vão à IA.
 * @param {Array<Object>} items Manchetes raspadas.
 * @return {string}
 */
function buildThemeNewsPrompt_(items) {
  try {
    var headlines = items.map(function(item, index) {
      var origin = item.source ? ' (' + item.source + ')' : '';
      return (index + 1) + '. ' + item.title + origin;
    }).join('\n');

    return [
      'Voce e uma educadora que prepara um radar de novidades sobre o tema',
      '"' + THEME_NEWS_CFG.THEME + '" para ' + THEME_NEWS_CFG.AUDIENCE + '.',
      '',
      'Manchetes recentes raspadas do noticiario (Google News, pt-BR):',
      headlines,
      '',
      'Escreva, em portugues do Brasil, uma sintese de 3 a 5 frases que:',
      '(1) resuma o panorama geral dessas novidades; (2) destaque a noticia',
      'mais relevante para o contexto educacional descrito; (3) sugira uma',
      'forma concreta de levar essa novidade para os estudantes. Regras: tom',
      'acolhedor e factual; nao invente fatos alem das manchetes fornecidas;',
      'nao use formatacao markdown, apenas texto corrido.'
    ].join('\n');
  } catch (error) {
    Logger.log("Erro em buildThemeNewsPrompt_: " + error.message);
    throw error;
  }
}

/**
 * Síntese determinística local (sem LLM) a partir das manchetes raspadas.
 * @param {Array<Object>} items Manchetes raspadas.
 * @return {string}
 */
function localThemeNewsSynthesis_(items) {
  try {
    if (!items || items.length === 0) {
      return 'Nao foi possivel buscar novidades sobre "' + THEME_NEWS_CFG.THEME +
        '" agora (internet ou feed indisponivel). Tente novamente mais tarde.';
    }
    var top = items.slice(0, 3).map(function(item) {
      return '"' + item.title + '"' + (item.source ? ' (' + item.source + ')' : '');
    });
    return 'Radar do tema "' + THEME_NEWS_CFG.THEME + '": foram encontradas ' +
      items.length + ' novidades recentes no noticiario. Destaques: ' +
      top.join('; ') + '. Vale levar esses assuntos para a proxima conversa com os estudantes.';
  } catch (error) {
    Logger.log("Erro em localThemeNewsSynthesis_: " + error.message);
    throw error;
  }
}

/**
 * Ponto único de HTTP com o Gemini (generateContent), resiliente no padrão da
 * frota. Não lança: devolve null para o chamador degradar ao fallback local.
 * @param {string} apiKey Chave da API.
 * @param {string} prompt Prompt textual.
 * @return {?string} Texto gerado ou null.
 */
function callThemeNewsGemini_(apiKey, prompt) {
  try {
    var url = THEME_NEWS_CFG.GEMINI_BASE_URL +
      encodeURIComponent(THEME_NEWS_CFG.GEMINI_MODEL) +
      ':generateContent?key=' + encodeURIComponent(apiKey);
    var options = {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.4 }
      }),
      muteHttpExceptions: true
    };

    var lastError = 'Falha desconhecida.';
    for (var attempt = 1; attempt <= THEME_NEWS_CFG.MAX_ATTEMPTS; attempt++) {
      try {
        var response = UrlFetchApp.fetch(url, options);
        var code = response.getResponseCode();
        if (code >= 200 && code < 300) {
          var json = themeNewsSafeParse_(response.getContentText());
          var text = json && json.candidates && json.candidates[0] &&
            json.candidates[0].content && json.candidates[0].content.parts &&
            json.candidates[0].content.parts[0] && json.candidates[0].content.parts[0].text;
          if (text) {
            return String(text).trim();
          }
          lastError = 'Resposta do Gemini sem texto utilizavel.';
          break; // 2xx malformado não melhora com retry
        }
        lastError = 'Gemini HTTP ' + code;
        if (code !== 429 && code < 500) {
          break; // 4xx permanente (chave inválida, payload errado...)
        }
      } catch (error) {
        lastError = 'Rede: ' + (error && error.message ? error.message : error);
      }
      if (attempt < THEME_NEWS_CFG.MAX_ATTEMPTS) {
        Utilities.sleep(themeNewsBackoff_(attempt));
      }
    }
    Logger.log('ThemeDigest: Gemini degradou para fallback local: ' + lastError);
    return null;
  } catch (error) {
    Logger.log("Erro em callThemeNewsGemini_: " + error.message);
    throw error;
  }
}

/** Backoff exponencial com jitter, compartilhado entre RSS e Gemini. */
function themeNewsBackoff_(attempt) {
  return THEME_NEWS_CFG.BASE_DELAY_MS * Math.pow(2, attempt - 1) +
    Math.floor(Math.random() * THEME_NEWS_CFG.BASE_DELAY_MS);
}

/** JSON.parse defensivo: devolve null em vez de lançar SyntaxError cru. */
function themeNewsSafeParse_(text) {
  try {
    try {
      if (typeof text !== 'string' || !text) return null;
      try {
        return JSON.parse(text);
      } catch (error) {
        return null;
      }
    } catch (error) {
      Logger.log("Erro em themeNewsSafeParse_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em themeNewsSafeParse_: " + error.message);
    throw error;
  }
}

/** Envelope de sucesso: usa StandardReturn quando o projeto o define. */
function themeNewsOk_(data) {
  if (typeof StandardReturn !== 'undefined' && StandardReturn && StandardReturn.ok) {
    return StandardReturn.ok(data);
  }
  return { success: true, data: data, error: null };
}

/** Envelope de falha: usa StandardReturn quando o projeto o define. */
function themeNewsFail_(error) {
  var message = error && error.message ? error.message : String(error || 'Erro desconhecido.');
  if (typeof StandardReturn !== 'undefined' && StandardReturn && StandardReturn.fail) {
    return StandardReturn.fail(message);
  }
  return { success: false, data: null, error: message };
}

