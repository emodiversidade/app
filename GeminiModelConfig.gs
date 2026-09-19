/**
 * GeminiModelConfig.gs — FROTA-07: Configuração central de modelo e parâmetros
 *
 * Fonte única de verdade para o modelo Gemini e limites operacionais.
 * Todos os projetos leem daqui; nenhum fixa modelo em código-fonte.
 *
 * Configuração via Script Properties (sem mudança de código):
 *   GEMINI_MODEL             — modelo a usar; padrão: gemini-2.0-flash
 *   GEMINI_MAX_OUTPUT_TOKENS — limite de tokens na resposta; padrão: 2048
 *   GEMINI_TEMPERATURE       — temperatura (0.0–1.0); padrão: 0.4
 *
 * Critérios de aceite (FROTA-07):
 *   ✓ Modelo lido de Script Properties — sem hardcode no fluxo de produção.
 *   ✓ Lista de modelos permitidos; falha clara para modelo fora da lista.
 *   ✓ getModel() registra o modelo efetivamente usado (via Logger auditável).
 *   ✓ Teste troca o modelo por propriedade sem editar código.
 */

'use strict';

var GeminiModelConfig = (function () {

  // ── Padrão e lista permitida ──────────────────────────────────────────────
  var DEFAULT_MODEL       = 'gemini-2.0-flash';
  var DEFAULT_MAX_TOKENS  = 2048;
  var DEFAULT_TEMPERATURE = 0.4;

  // Apenas modelos validados pela frota. Adicione novos aqui após homologação.
  var ALLOWED_MODELS = [
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.5-flash-preview-05-20',
    'gemini-1.5-flash',
    'gemini-1.5-flash-8b',
    'gemini-1.5-pro'
  ];

  // ── Leitura de propriedades ───────────────────────────────────────────────
  function _props() {
    try {
      try { return PropertiesService.getScriptProperties(); }
      catch (_) { return { getProperty: function() { return null; } }; }
    } catch (error) {
      Logger.log("Erro em _props: " + error.message);
      throw error;
    }
  }

  // ── API pública ───────────────────────────────────────────────────────────

  /**
   * Retorna o modelo configurado, validado contra a lista permitida.
   * Falha de forma clara se o valor não for reconhecido.
   *
   * @returns {string}  Ex.: 'gemini-2.0-flash'
   * @throws  {Error}   Se o modelo configurado não estiver na lista permitida.
   */
  function getModel() {
    var model = _props().getProperty('GEMINI_MODEL') || DEFAULT_MODEL;
    if (ALLOWED_MODELS.indexOf(model) === -1) {
      throw new Error(
        '[GeminiModelConfig] Modelo não permitido: "' + model + '". ' +
        'Valores aceitos: ' + ALLOWED_MODELS.join(', ') + '. ' +
        'Corrija a Script Property GEMINI_MODEL ou remova-a para usar o padrão.');
    }
    Logger.log('[GeminiModelConfig] modelo=' + model);
    return model;
  }

  /**
   * Limite máximo de tokens na resposta do modelo.
   * @returns {number}
   */
  function getMaxOutputTokens() {
    try {
      var v = _props().getProperty('GEMINI_MAX_OUTPUT_TOKENS');
      var n = v ? parseInt(v, 10) : DEFAULT_MAX_TOKENS;
      return (isNaN(n) || n < 1) ? DEFAULT_MAX_TOKENS : n;
    } catch (error) {
      Logger.log("Erro em getMaxOutputTokens: " + error.message);
      throw error;
    }
  }

  /**
   * Temperatura da geração (0.0 = determinística, 1.0 = criativa).
   * @returns {number}
   */
  function getTemperature() {
    try {
      var v = _props().getProperty('GEMINI_TEMPERATURE');
      var n = v ? parseFloat(v) : DEFAULT_TEMPERATURE;
      return (isNaN(n) || n < 0 || n > 1) ? DEFAULT_TEMPERATURE : n;
    } catch (error) {
      Logger.log("Erro em getTemperature: " + error.message);
      throw error;
    }
  }

  /**
   * Retorna todos os parâmetros operacionais de uma vez.
   * Use para montar o generationConfig do payload Gemini.
   *
   * @returns {{ model: string, maxOutputTokens: number, temperature: number }}
   */
  function getParams() {
    return {
      model:           getModel(),
      maxOutputTokens: getMaxOutputTokens(),
      temperature:     getTemperature()
    };
  }

  /**
   * Retorna o modelo padrão sem validar propriedades (útil para fallback/log).
   * @returns {string}
   */
  function getDefault() { return DEFAULT_MODEL; }

  return {
    getModel:           getModel,
    getMaxOutputTokens: getMaxOutputTokens,
    getTemperature:     getTemperature,
    getParams:          getParams,
    getDefault:         getDefault,
    ALLOWED_MODELS:     ALLOWED_MODELS,
    DEFAULT_MODEL:      DEFAULT_MODEL
  };
})();
