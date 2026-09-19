/**
 * FROTA-15: guardas eticas modulares para IA e dados sensiveis.
 */
var EthicsGuardService = (function () {
  'use strict';
  var policies_ = {};
  var defaults_ = {
    blocked: [
      { id: 'clinical-diagnosis', pattern: '\\bdiagnostico definitivo\\b', flags: 'i' },
      { id: 'genetic-determinism', pattern: '\\bgeneticamente (?:inferior|superior)\\b', flags: 'i' }
    ],
    replacements: [
      { id: 'ableist-incapable', pattern: '\\bincapaz\\b', flags: 'gi', replacement: 'pessoa que necessita de apoio' },
      { id: 'normal-abnormal', pattern: '\\banormal\\b', flags: 'gi', replacement: 'atipico' }
    ],
    notices: {
      synthetic: 'Aviso pedagogico: os dados apresentados sao sinteticos e nao identificam pessoas reais.',
      clinical: 'Aviso: este conteudo nao possui validade clinica e nao substitui avaliacao profissional.',
      biometric: 'Aviso de privacidade: dados biometricos devem ser minimizados, pseudonimizados e acessados apenas por pessoas autorizadas.'
    }
  };

  function definePolicy(name, policy) {
    policies_[name] = mergePolicy_(policy || {});
    return api;
  }

  function inspect(content, options) {
    try {
      options = options || {};
      var policy = policies_[options.policy] || mergePolicy_({});
      var text = String(content == null ? '' : content);
      var blocked = [];
      policy.blocked.forEach(function (rule) {
        if ((new RegExp(rule.pattern, rule.flags || 'i')).test(text)) blocked.push(rule.id);
      });
      if (blocked.length) {
        audit_(options, false, blocked, []);
        return {
          ok: false,
          status: 422,
          error: {
            code: 'ETHICS_POLICY_BLOCKED',
            message: 'Conteudo bloqueado pelas politicas eticas.',
            details: blocked.map(function (id) { return { rule: id }; })
          }
        };
      }

      var replacements = [];
      policy.replacements.forEach(function (rule) {
        var regex = new RegExp(rule.pattern, rule.flags || 'gi');
        if (regex.test(text)) {
          replacements.push(rule.id);
          text = text.replace(new RegExp(rule.pattern, rule.flags || 'gi'), rule.replacement);
        }
      });
      var notices = requiredNotices_(policy, options);
      if (notices.length) text += '\n\n' + notices.join('\n');
      audit_(options, true, [], replacements);
      return {
        ok: true,
        status: 200,
        content: text,
        ethics: {
          policy: options.policy || 'default',
          substitutions: replacements,
          notices: notices,
          syntheticData: options.syntheticData === true
        }
      };
    } catch (error) {
      Logger.log("Erro em inspect: " + error.message);
      throw error;
    }
  }

  function protectData(data, options) {
    try {
      options = options || {};
      var allowed = options.allowedFields || [];
      var pseudonymFields = options.pseudonymFields || [];
      var output = {};
      Object.keys(data || {}).forEach(function (key) {
        if (allowed.length && allowed.indexOf(key) === -1) return;
        output[key] = pseudonymFields.indexOf(key) >= 0 ?
          pseudonymize_(data[key]) : data[key];
      });
      return output;
    } catch (error) {
      Logger.log("Erro em protectData: " + error.message);
      throw error;
    }
  }

  function pipeline(input, generate, options) {
    options = options || {};
    var pre = inspect(input, copy_(options, { stage: 'input' }));
    if (!pre.ok) return pre;
    var generated = generate(pre.content);
    var raw = generated && generated.data && generated.data.text != null ?
      generated.data.text : generated;
    var post = inspect(raw, copy_(options, { stage: 'output' }));
    if (!post.ok) return post;
    post.source = generated && generated.source || 'local';
    return post;
  }

  function requiredNotices_(policy, options) {
    try {
      var notices = [];
      if (options.syntheticData) notices.push(policy.notices.synthetic);
      if (options.clinicalContext) notices.push(policy.notices.clinical);
      if (options.biometricData) notices.push(policy.notices.biometric);
      return notices.filter(function (item) { return !!item; });
    } catch (error) {
      Logger.log("Erro em requiredNotices_: " + error.message);
      throw error;
    }
  }

  function audit_(options, allowed, blocked, replacements) {
    if (typeof StructuredLogService === 'undefined') return;
    StructuredLogService.auditAi('ethics.guard', {
      sessionId: options.sessionId,
      ageRange: options.ageRange,
      ethicalSubstitution: replacements.length > 0,
      errorCode: allowed ? null : 'ETHICS_POLICY_BLOCKED',
      outcome: allowed ? 'allowed' : 'blocked',
      details: {
        policy: options.policy || 'default',
        stage: options.stage || null,
        blockedRules: blocked,
        replacementRules: replacements
      }
    });
  }

  function pseudonymize_(value) {
    if (typeof StructuredLogService !== 'undefined' &&
        typeof StructuredLogService.pseudonymize === 'function') {
      return StructuredLogService.pseudonymize(value);
    }
    throw new Error('PSEUDONYMIZATION_SERVICE_REQUIRED');
  }

  function mergePolicy_(policy) {
    try {
      return {
        blocked: (policy.blocked || defaults_.blocked).slice(),
        replacements: (policy.replacements || defaults_.replacements).slice(),
        notices: copy_(defaults_.notices, policy.notices || {})
      };
    } catch (error) {
      Logger.log("Erro em mergePolicy_: " + error.message);
      throw error;
    }
  }

  function copy_(base, extra) {
    try {
      var result = {};
      Object.keys(base || {}).forEach(function (key) { result[key] = base[key]; });
      Object.keys(extra || {}).forEach(function (key) { result[key] = extra[key]; });
      return result;
    } catch (error) {
      Logger.log("Erro em copy_: " + error.message);
      throw error;
    }
  }

  var api = {
    definePolicy: definePolicy,
    inspect: inspect,
    protectData: protectData,
    pipeline: pipeline
  };
  return api;
}());

