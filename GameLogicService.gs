/**
 * @file GameLogicService.gs
 * @description Fluxo de situacoes, escolhas e impacts.
 *
 * English Description:
 * This service orchestrates the core game flow, managing situational transitions, player decisions, and
 * choice impacts. It queries the spreadsheet to retrieve situations, filters out already answered challenges
 * based on the user's progress level, and records choice details. It also computes real-time feedback and
 * updates the player's overall score and developmental levels based on the specific vectors of each choice.
 */

/**
 * Seleciona a proxima situacao nao respondida cujo nivel cabe no progresso atual;
 * se nenhuma couber, devolve a primeira nao respondida de qualquer nivel.
 * @param {string} userId ID resolvido no servidor por requireAuth_().
 * @returns {?Object} Situacao com id, titulo, descricao e opcoes, ou null ao fim.
 */
function getNextSituation_(userId) {
  try {
    const normalizedUserId = normalizeText_(userId, "Usuario", 1);
    const situations = readRecords_(getGameConfig_().SHEET_NAMES.SITUATIONS);
    const choices = readRecords_(getGameConfig_().SHEET_NAMES.CHOICES);
    const answered = new Set(
      choices.slice(1)
        .filter(row => String(row[1]) === normalizedUserId)
        .map(row => String(row[2]))
    );
    const progress = ensurePlayerProgress_(normalizedUserId);
    const level = Number(progress.NivelAtual || 1);
    const available = situations.slice(1).filter(row =>
      !answered.has(String(row[0])) && Number(row[7] || 1) <= level
    );
    const fallback = situations.slice(1).filter(row => !answered.has(String(row[0])));
    const row = available[0] || fallback[0];
    if (!row) {
      return null;
    }
    return {
      id: String(row[0]),
      titulo: String(row[1]),
      descricao: String(row[2]),
      opcao1: String(row[3]),
      opcao2: String(row[4]),
      nivelPrincesamento: Number(row[7] || 1),
      // Previa transparente do que cada caminho desenvolve: a jogadora escolhe
      // sabendo a troca, em vez de adivinhar a "resposta certa".
      opcao1Dimensoes: previewOptionDimensions_(row, 1),
      opcao2Dimensoes: previewOptionDimensions_(row, 2)
    };
  } catch (error) {
    Logger.log("Erro em getNextSituation_: " + error.message);
    throw error;
  }
}

/**
 * Resume as dimensoes positivas que uma opcao desenvolve, para exibir como
 * etiquetas na tela de situacao. Vazio quando a situacao usa o contrato legado.
 * @param {Array} situationRow Linha da aba Situacoes.
 * @param {number} choiceIndex 1 ou 2.
 * @returns {Array<{key: string, label: string, emoji: string}>}
 */
function previewOptionDimensions_(situationRow, choiceIndex) {
  try {
    const impact = resolveOptionImpact_(situationRow, choiceIndex);
    if (impact.legacy) {
      return [];
    }
    const dimensions = getAutonomyDimensions_();
    return Object.keys(impact.vector)
      .filter(key => impact.vector[key] > 0)
      .map(key => {
        const meta = dimensions.filter(d => d.key === key)[0] || { label: key, emoji: "✨" };
        return { key: key, label: meta.label, emoji: meta.emoji };
      });
  } catch (error) {
    Logger.log("Erro em previewOptionDimensions_: " + error.message);
    throw error;
  }
}

/**
 * Persiste a escolha, calcula o impacto e registra auditoria.
 * @param {string} userId ID resolvido no servidor.
 * @param {string} situationId Situacao alvo; cada pessoa responde uma unica vez.
 * @param {number} choiceIndex 1 ou 2.
 * @param {number|string} originalChoice Valor original enviado pelo cliente (indice ou texto).
 * @returns {Object} Resultado com id, description, impact, score e level.
 * @throws {Error} Escolha invalida ou situacao ja respondida.
 */
function makeChoice_(userId, situationId, choiceIndex, originalChoice) {
  const workflowLock = LockService.getUserLock();
  if (!workflowLock.tryLock(5000)) {
    throw new Error('Outra escolha esta sendo registrada. Aguarde e tente novamente.');
  }
  try {
    const normalizedUserId = normalizeText_(userId, "Usuario", 1);
    const normalizedSituationId = normalizeText_(situationId, "Situacao", 1);
    const selectedChoice = Number(choiceIndex);
    if (selectedChoice !== 1 && selectedChoice !== 2) {
      throw new Error("Escolha invalida.");
    }

    const situations = readRecords_(getGameConfig_().SHEET_NAMES.SITUATIONS);
    const situation = situations.slice(1).find(row => String(row[0]) === normalizedSituationId);
    if (!situation) {
      throw new Error('Situacao nao encontrada.');
    }

    const existingChoices = readRecords_(getGameConfig_().SHEET_NAMES.CHOICES);
    const duplicate = existingChoices.slice(1).some(row =>
      String(row[1]) === normalizedUserId && String(row[2]) === normalizedSituationId
    );
    if (duplicate) {
      throw new Error("Essa situacao ja foi respondida.");
    }

    // Resolve texto canonico: se cliente enviou texto, usa ele; senao resolve do indice
    const choiceText = (typeof originalChoice === 'string' && originalChoice !== '1' && originalChoice !== '2')
      ? String(originalChoice).trim()
      : String(selectedChoice === 1 ? situation[3] : situation[4]).trim();

    createRecord_(getGameConfig_().SHEET_NAMES.CHOICES, [
      Utilities.getUuid(),
      normalizedUserId,
      normalizedSituationId,
      selectedChoice,
      new Date().toISOString(),
      choiceText  // Nova coluna: texto canonico da opcao escolhida
    ]);
    const result = calculateImpact_(normalizedUserId, normalizedSituationId, selectedChoice);
    logAction_("CHOICE", normalizedUserId, JSON.stringify({
      situationId: normalizedSituationId,
      choice: selectedChoice,
      choiceText: choiceText,
      impact: result.impact
    }));
    return result;
  } catch (error) {
    Logger.log("Erro em makeChoice_: " + error.message);
    throw error;
  } finally {
    workflowLock.releaseLock();
  }
}

/**
 * Aplica o impacto multidimensional da escolha, atualiza o progresso e grava o
 * resultado (incluindo o vetor de dimensoes) na aba Resultados.
 *
 * O impacto vem do vetor da opcao (coluna ImpactoOpcao1/2); na ausencia dele,
 * cai no contrato legado (opcao 1 = +10, opcao 2 = -5). O total ainda alimenta
 * a pontuacao/nivel, mas o retorno traz tambem a decomposicao por dimensao, uma
 * reflexao nao-julgadora e o estagio de desenvolvimento atingido.
 * @param {string} userId ID resolvido no servidor.
 * @param {string} situationId Situacao respondida.
 * @param {number} choiceIndex 1 ou 2.
 * @returns {Object} id, description, impact, score, level, dimensions, reflection, stage.
 * @throws {Error} "Situacao nao encontrada." quando o ID nao existe.
 */
function calculateImpact_(userId, situationId, choiceIndex) {
  try {
    try {
      const rows = readRecords_(getGameConfig_().SHEET_NAMES.SITUATIONS);
      const situation = rows.slice(1).find(row => String(row[0]) === String(situationId));
      if (!situation) {
        throw new Error("Situacao nao encontrada.");
      }
      const resultText = String(choiceIndex === 1 ? situation[5] : situation[6]);
      const impact = resolveOptionImpact_(situation, choiceIndex);
      const impactScore = impact.total;
      const progress = updatePlayerScore_(userId, impactScore);
      const resultId = Utilities.getUuid();
      createRecord_(getGameConfig_().SHEET_NAMES.RESULTS, [
        resultId,
        userId,
        situationId,
        resultText,
        impactScore,
        new Date().toISOString(),
        JSON.stringify(impact.vector || {})
      ]);
      return {
        id: resultId,
        description: resultText,
        impact: impactScore,
        score: Number(progress.Pontuacao || 0),
        level: Number(progress.NivelAtual || 1),
        dimensions: impact.vector,
        reflection: summarizeReflection_(impact.vector, impactScore),
        stage: resolveLevelStage_(Number(progress.NivelAtual || 1))
      };
    } catch (error) {
      Logger.log("Erro em calculateImpact_: " + error.message);
      throw error;
    }
  } catch (error) {
    Logger.log("Erro em calculateImpact_: " + error.message);
    throw error;
  }
}

function generateUniqueId_() {
  return Utilities.getUuid();
}
