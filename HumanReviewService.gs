/**
 * FROTA-03: revisão humana obrigatória para saídas geradas por IA.
 * Conteúdo permanece em cache como rascunho; o log persistente guarda apenas
 * metadados de estado, usuário, data e versão, nunca prompt ou raciocínio.
 */
var HumanReviewService = (function () {
  var PREFIX = 'AI_REVIEW_';
  var TTL_SECONDS = 21600;

  function now_() { return new Date().toISOString(); }
  function id_() { return Utilities.getUuid(); }
  function actor_(actor) {
    try {
      var value = String(actor || '').trim();
      if (!value) throw new Error('Revisor humano obrigatório.');
      return value.slice(0, 120);
    } catch (error) {
      Logger.log("Erro em actor_: " + error.message);
      throw error;
    }
  }
  function version_(content) {
    try {
      var bytes = Utilities.computeDigest(
        Utilities.DigestAlgorithm.SHA_256,
        String(content == null ? '' : content),
        Utilities.Charset.UTF_8
      );
      return Utilities.base64EncodeWebSafe(bytes).replace(/=+$/, '').slice(0, 24);
    } catch (error) {
      Logger.log("Erro em version_: " + error.message);
      throw error;
    }
  }
  function key_(id) { return PREFIX + id; }
  function cache_() { return CacheService.getScriptCache(); }
  function props_() { return PropertiesService.getScriptProperties(); }
  function read_(id) {
    try {
      var raw = cache_().get(key_(id));
      if (!raw) throw new Error('Rascunho inexistente ou expirado.');
      return JSON.parse(raw);
    } catch (error) {
      Logger.log("Erro em read_: " + error.message);
      throw error;
    }
  }
  function write_(record) {
    try {
      cache_().put(key_(record.id), JSON.stringify(record), TTL_SECONDS);
      return record;
    } catch (error) {
      Logger.log("Erro em write_: " + error.message);
      throw error;
    }
  }
  function metadata_(record) {
    return {
      id: record.id,
      useCase: record.useCase,
      state: record.state,
      version: record.version,
      createdAt: record.createdAt,
      updatedAt: record.updatedAt,
      reviewedBy: record.reviewedBy || '',
      reviewedAt: record.reviewedAt || '',
      label: record.label
    };
  }
  function audit_(record, action) {
    try {
      var event = metadata_(record);
      event.action = action;
      props_().setProperty(key_(record.id) + '_AUDIT', JSON.stringify(event));
    } catch (error) {
      Logger.log("Erro em audit_: " + error.message);
      throw error;
    }
  }
  function createDraft(useCase, content, options) {
    try {
      options = options || {};
      var text = String(content == null ? '' : content);
      var record = {
        id: id_(),
        useCase: String(useCase || 'ai.output').slice(0, 100),
        label: 'Rascunho gerado por IA',
        state: 'draft',
        content: text,
        version: version_(text),
        createdAt: now_(),
        updatedAt: now_(),
        reviewedBy: '',
        reviewedAt: '',
        rejectionReason: '',
        externalEffectAllowed: false
      };
      record.ownerId = String(options.ownerId || '').slice(0, 120);
      write_(record);
      audit_(record, 'generated');
      return record;
    } catch (error) {
      Logger.log("Erro em createDraft: " + error.message);
      throw error;
    }
  }
  function editDraft(id, content, actor) {
    try {
      var record = read_(id);
      if (record.state === 'approved') throw new Error('Versão aprovada é imutável; crie novo rascunho.');
      record.content = String(content == null ? '' : content);
      record.version = version_(record.content);
      record.state = 'draft';
      record.updatedAt = now_();
      record.reviewedBy = actor_(actor);
      record.reviewedAt = '';
      record.externalEffectAllowed = false;
      write_(record);
      audit_(record, 'edited');
      return record;
    } catch (error) {
      Logger.log("Erro em editDraft: " + error.message);
      throw error;
    }
  }
  function reject(id, reason, actor) {
    try {
      var record = read_(id);
      record.state = 'rejected';
      record.rejectionReason = String(reason || '').slice(0, 500);
      record.reviewedBy = actor_(actor);
      record.reviewedAt = now_();
      record.updatedAt = record.reviewedAt;
      record.externalEffectAllowed = false;
      write_(record);
      audit_(record, 'rejected');
      return record;
    } catch (error) {
      Logger.log("Erro em reject: " + error.message);
      throw error;
    }
  }
  function approve(id, expectedVersion, actor) {
    try {
      var record = read_(id);
      if (record.state !== 'draft') throw new Error('Somente rascunho pode ser aprovado.');
      if (record.version !== String(expectedVersion || '')) {
        throw new Error('Versão alterada; revise novamente antes de aprovar.');
      }
      record.state = 'approved';
      record.reviewedBy = actor_(actor);
      record.reviewedAt = now_();
      record.updatedAt = record.reviewedAt;
      record.externalEffectAllowed = true;
      write_(record);
      audit_(record, 'approved');
      return record;
    } catch (error) {
      Logger.log("Erro em approve: " + error.message);
      throw error;
    }
  }
  function assertApproved(id, expectedVersion) {
    try {
      var record = read_(id);
      if (record.state !== 'approved' || !record.externalEffectAllowed) {
        throw new Error('Efeito externo bloqueado: aprovação humana obrigatória.');
      }
      if (expectedVersion && record.version !== String(expectedVersion)) {
        throw new Error('Efeito externo bloqueado: versão não aprovada.');
      }
      return record;
    } catch (error) {
      Logger.log("Erro em assertApproved: " + error.message);
      throw error;
    }
  }
  function get(id) { return read_(id); }
  function decorateResult(useCase, result, content, options) {
    var draft = createDraft(useCase, content, options || {});
    result = result && typeof result === 'object' ? result : { value: result };
    result.review = metadata_(draft);
    result.review.content = draft.content;
    result.review.externalEffectAllowed = false;
    result.review.label = 'Rascunho gerado por IA';
    return result;
  }

  return {
    createDraft: createDraft,
    editDraft: editDraft,
    reject: reject,
    approve: approve,
    assertApproved: assertApproved,
    get: get,
    decorateResult: decorateResult,
    metadata: metadata_
  };
})();

function createAiReviewDraft(useCase, content, options) {
  var user = requireAuth_();
  options = options || {};
  options.ownerId = user.id;
  return HumanReviewService.createDraft(useCase, content, options);
}
function editAiReviewDraft(id, content, actor) {
  var user = requireAuth_();
  assertAiDraftOwner_(id, user.id);
  return HumanReviewService.editDraft(id, content, user.id);
}
function rejectAiReviewDraft(id, reason, actor) {
  var user = requireAuth_();
  assertAiDraftOwner_(id, user.id);
  return HumanReviewService.reject(id, reason, user.id);
}
function approveAiReviewDraft(id, version, actor) {
  var user = requireAuth_();
  assertAiDraftOwner_(id, user.id);
  return HumanReviewService.approve(id, version, user.id);
}
function getAiReviewDraft(id) {
  try {
    var user = requireAuth_();
    assertAiDraftOwner_(id, user.id);
    return HumanReviewService.get(id);
  } catch (error) {
    Logger.log("Erro em getAiReviewDraft: " + error.message);
    throw error;
  }
}
function assertAiReviewApproved(id, version) {
  var user = requireAuth_();
  assertAiDraftOwner_(id, user.id);
  return HumanReviewService.assertApproved(id, version);
}

function assertAiDraftOwner_(id, userId) {
  var draft = HumanReviewService.get(id);
  if (draft.ownerId && String(draft.ownerId) !== String(userId)) {
    throw new Error('Rascunho pertence a outra jornada.');
  }
  return draft;
}
