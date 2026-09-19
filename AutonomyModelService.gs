/**
 * @file AutonomyModelService.gs
 * @description Modelo conceitual da autonomia: vetores de impacto multidimensionais,
 *              reflexao nao-julgadora, perfil agregado da jogadora e estagios de
 *              desenvolvimento. Funcoes puras (sem planilha) para serem testaveis.
 *
 * English Description:
 * This service defines the conceptual model of the player autonomy game, implementing
 * multidimensional impact vectors, non-judgmental reflection mechanisms, and player
 * developmental stages. It provides pure, stateless functions that parse JSON representation
 * of impacts from the spreadsheet, compute cumulative profiles, identify dominant pillars,
 * and map total scores to developmental milestones. This decoupled design enables comprehensive
 * unit testing without spreadsheet database dependency.
 *
 * Decisao de design: o jogo NAO tem resposta certa. Cada escolha movimenta a
 * jogadora pelos cinco pilares (ver GAME_CONFIG.DIMENSIONS). Uma opcao pode
 * fortalecer a voz mas custar um pouco de empatia; outra pode acolher a outra
 * pessoa as custas de calar a propria vontade. O motor le esse trade-off de um
 * vetor por opcao gravado na planilha e, na ausencia dele, cai no contrato
 * legado (+10 / -5) para preservar compatibilidade.
 */

/**
 * Interpreta o vetor de impacto de uma opcao, vindo de uma celula da planilha.
 * Aceita um objeto JSON {dimensao: numero}; chaves desconhecidas e valores nao
 * numericos sao descartados. Strings vazias ou JSON invalido viram null, sinal
 * de que o chamador deve usar o fallback legado.
 * @param {*} raw Conteudo bruto da celula (string JSON, objeto ou vazio).
 * @returns {?Object<string, number>} Vetor saneado, {} se sem dimensoes, ou null.
 */
function parseImpactVector_(raw) {
  try {
    if (raw === null || raw === undefined || raw === "") {
      return null;
    }
    let parsed = raw;
    if (typeof raw === "string") {
      const trimmed = raw.trim();
      if (!trimmed) {
        return null;
      }
      try {
        parsed = JSON.parse(trimmed);
      } catch (error) {
        return null;
      }
    }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null;
    }
    const allowed = getDimensionKeySet_();
    const vector = {};
    Object.keys(parsed).forEach(key => {
      const value = Number(parsed[key]);
      if (allowed[key] && Number.isFinite(value) && value !== 0) {
        vector[key] = value;
      }
    });
    return vector;
  } catch (error) {
    Logger.log("Erro em parseImpactVector_: " + error.message);
    throw error;
  }
}

/**
 * Soma as componentes de um vetor de impacto.
 * @param {?Object<string, number>} vector Vetor saneado.
 * @returns {number} Total (positivo, negativo ou zero).
 */
function impactTotal_(vector) {
  try {
    if (!vector) {
      return 0;
    }
    return Object.keys(vector).reduce((sum, key) => sum + Number(vector[key] || 0), 0);
  } catch (error) {
    Logger.log("Erro em impactTotal_: " + error.message);
    throw error;
  }
}

/**
 * Resolve o impacto de uma opcao a partir da linha da situacao.
 * Usa o vetor da coluna ImpactoOpcao1/ImpactoOpcao2 (indices 8 e 9) quando presente;
 * caso contrario aplica o contrato legado (opcao 1 = +10, opcao 2 = -5) sem dimensoes.
 * @param {Array} situationRow Linha completa da aba Situacoes.
 * @param {number} choiceIndex 1 ou 2.
 * @returns {{vector: Object<string, number>, total: number, legacy: boolean}}
 */
function resolveOptionImpact_(situationRow, choiceIndex) {
  const column = choiceIndex === 1 ? 8 : 9;
  const vector = parseImpactVector_(situationRow ? situationRow[column] : null);
  if (vector) {
    return { vector: vector, total: impactTotal_(vector), legacy: false };
  }
  return { vector: {}, total: choiceIndex === 1 ? 10 : -5, legacy: true };
}

/**
 * Produz uma reflexao acolhedora (nunca "voce errou") a partir do vetor da
 * escolha. Destaca a dimensao mais movimentada e nomeia o trade-off quando
 * alguma dimensao recuou, convidando a pensar — nao a se sentir punida.
 * @param {?Object<string, number>} vector Vetor da opcao escolhida.
 * @param {number} total Total do vetor (ou impacto legado).
 * @returns {{dominant: ?string, dominantLabel: ?string, emoji: string, title: string, message: string, gained: Array, lost: Array}}
 */
function summarizeReflection_(vector, total) {
  try {
    const dimensions = getAutonomyDimensions_();
    const labelOf = key => {
      const found = dimensions.filter(d => d.key === key)[0];
      return found || { key: key, label: key, emoji: "✨" };
    };
    const entries = vector ? Object.keys(vector).map(key => ({ key: key, value: Number(vector[key]) })) : [];
    const gained = entries.filter(e => e.value > 0).sort((a, b) => b.value - a.value);
    const lost = entries.filter(e => e.value < 0).sort((a, b) => a.value - b.value);

    let dominant = null;
    let dominantLabel = null;
    let emoji = "💫";
    if (gained.length > 0) {
      dominant = gained[0].key;
      const meta = labelOf(dominant);
      dominantLabel = meta.label;
      emoji = meta.emoji;
    }

    let title;
    let message;
    if (gained.length > 0 && lost.length > 0) {
      const grew = labelOf(gained[0].key);
      const cost = labelOf(lost[0].key);
      title = "Toda escolha tem um peso";
      message = `Essa decisao fortaleceu sua ${grew.label.toLowerCase()}, mas pediu um pouco da sua ` +
        `${cost.label.toLowerCase()}. Nenhum caminho e perfeito — o que importa e voce perceber a troca.`;
    } else if (gained.length > 0) {
      const grew = labelOf(gained[0].key);
      title = `Voce exercitou: ${grew.label}`;
      message = `${grew.emoji} ${grew.short} Continue notando quando voce age assim na vida real.`;
    } else if (lost.length > 0) {
      const cost = labelOf(lost[0].key);
      title = "Um passo para dentro";
      message = `Dessa vez voce recuou um pouco na sua ${cost.label.toLowerCase()}. Sentir isso ja e ` +
        `autoconhecimento — e a proxima escolha pode ser diferente.`;
    } else if (total > 0) {
      title = "Voce avancou";
      message = "Cada situacao respondida e um passo na sua jornada de autonomia.";
    } else {
      title = "Voce refletiu";
      message = "Parar para pensar sobre uma situacao dificil ja e cuidar de si.";
    }

    return {
      dominant: dominant,
      dominantLabel: dominantLabel,
      emoji: emoji,
      title: title,
      message: message,
      gained: gained.map(e => ({ key: e.key, label: labelOf(e.key).label, emoji: labelOf(e.key).emoji, value: e.value })),
      lost: lost.map(e => ({ key: e.key, label: labelOf(e.key).label, emoji: labelOf(e.key).emoji, value: e.value }))
    };
  } catch (error) {
    Logger.log("Erro em summarizeReflection_: " + error.message);
    throw error;
  }
}

/**
 * Agrega os vetores persistidos no historico de resultados em um perfil de
 * autonomia: total por dimensao, dimensao mais forte, dimensao a desenvolver e
 * flag de equilibrio (todas as dimensoes acima do piso => selo "Emodiversa").
 * @param {Array<Array>} resultRows Linhas (sem cabecalho) da aba Resultados de uma jogadora.
 *   Espera o vetor JSON na coluna Dimensoes (indice 6).
 * @returns {{dimensions: Array, total: number, strongest: ?Object, weakest: ?Object, balanced: boolean, answered: number}}
 */
function computeAutonomyProfile_(resultRows) {
  try {
    const dimensions = getAutonomyDimensions_();
    const totals = {};
    dimensions.forEach(dimension => { totals[dimension.key] = 0; });

    const rows = resultRows || [];
    rows.forEach(row => {
      const vector = parseImpactVector_(row[6]);
      if (vector) {
        Object.keys(vector).forEach(key => {
          if (totals[key] !== undefined) {
            totals[key] += Number(vector[key] || 0);
          }
        });
      }
    });

    const profile = dimensions.map(dimension => ({
      key: dimension.key,
      label: dimension.label,
      emoji: dimension.emoji,
      short: dimension.short,
      value: totals[dimension.key]
    }));

    const sorted = profile.slice().sort((a, b) => b.value - a.value);
    const total = profile.reduce((sum, d) => sum + d.value, 0);
    // Equilibrio so faz sentido com pelo menos uma rodada por dimensao registrada.
    const BALANCE_FLOOR = 10;
    const balanced = rows.length >= dimensions.length &&
      profile.every(d => d.value >= BALANCE_FLOOR);

    return {
      dimensions: profile,
      total: total,
      strongest: sorted.length ? sorted[0] : null,
      weakest: sorted.length ? sorted[sorted.length - 1] : null,
      balanced: balanced,
      answered: rows.length
    };
  } catch (error) {
    Logger.log("Erro em computeAutonomyProfile_: " + error.message);
    throw error;
  }
}

/**
 * Nomeia o estagio de desenvolvimento correspondente a um nivel numerico.
 * @param {number} level Nivel atual (>= 1).
 * @returns {{key: string, name: string, description: string, level: number, isMax: boolean, next: ?Object}}
 */
function resolveLevelStage_(level) {
  try {
    const stages = getGameConfig_().LEVEL_STAGES;
    const numericLevel = Math.max(1, Number(level) || 1);
    let current = stages[0];
    let nextStage = null;
    for (let i = 0; i < stages.length; i++) {
      if (stages[i].minLevel <= numericLevel) {
        current = stages[i];
        nextStage = stages[i + 1] || null;
      }
    }
    return {
      key: current.key,
      name: current.name,
      description: current.description,
      level: numericLevel,
      isMax: nextStage === null,
      next: nextStage ? { key: nextStage.key, name: nextStage.name, minLevel: nextStage.minLevel } : null
    };
  } catch (error) {
    Logger.log("Erro em resolveLevelStage_: " + error.message);
    throw error;
  }
}
