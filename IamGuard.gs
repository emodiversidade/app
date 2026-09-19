/**
 * FROTA-11: guarda central de autenticacao e autorizacao.
 *
 * O adaptador de sessao deve retornar null para uma sessao invalida ou:
 * { userId: string, roles: string[], permissions: string[] }
 */
var IamGuard = (function () {
  'use strict';

  var policies_ = {};
  var sessionResolver_ = null;

  function configure(options) {
    try {
      options = options || {};
      if (typeof options.resolveSession === 'function') {
        sessionResolver_ = options.resolveSession;
      }
      if (options.policies) {
        Object.keys(options.policies).forEach(function (action) {
          definePolicy(action, options.policies[action]);
        });
      }
      return api;
    } catch (error) {
      Logger.log("Erro em configure: " + error.message);
      throw error;
    }
  }

  function definePolicy(action, policy) {
    if (!action || typeof action !== 'string') {
      throw new Error('IAM_ACTION_REQUIRED');
    }
    policies_[action] = normalizePolicy_(policy);
    return api;
  }

  function authorize(action, credential, context) {
    var policy = policies_[action];
    if (!policy) {
      return denied_(403, 'IAM_POLICY_NOT_FOUND', 'Acao sem politica de acesso.');
    }

    var session;
    try {
      session = resolveSession_(credential, context || {});
    } catch (error) {
      return denied_(401, 'UNAUTHORIZED', 'Sessao invalida ou expirada.');
    }

    if (!session || !session.userId) {
      return denied_(401, 'UNAUTHORIZED', 'Sessao ausente, invalida ou expirada.');
    }

    if (!matchesPolicy_(session, policy)) {
      return denied_(403, 'FORBIDDEN', 'Permissao insuficiente para esta acao.');
    }

    return {
      ok: true,
      status: 200,
      action: action,
      principal: session
    };
  }

  function guard(action, credential, handler, context) {
    var decision = authorize(action, credential, context);
    if (!decision.ok) return decision;
    if (typeof handler !== 'function') return decision;
    return handler(decision.principal, context || {});
  }

  function resolveSession_(credential, context) {
    if (sessionResolver_) return sessionResolver_(credential, context);
    return credential && typeof credential === 'object' ? credential : null;
  }

  function normalizePolicy_(policy) {
    policy = policy || {};
    return {
      public: policy.public === true,
      rolesAny: array_(policy.rolesAny || policy.roles),
      permissionsAny: array_(policy.permissionsAny || policy.permissions),
      permissionsAll: array_(policy.permissionsAll)
    };
  }

  function matchesPolicy_(session, policy) {
    try {
      if (policy.public) return true;
      var roles = array_(session.roles || session.role);
      var permissions = array_(session.permissions);
      if (permissions.indexOf('*') !== -1) return true;

      var hasConstraint = policy.rolesAny.length ||
        policy.permissionsAny.length ||
        policy.permissionsAll.length;
      if (!hasConstraint) return true;

      if (policy.rolesAny.length && intersects_(roles, policy.rolesAny)) return true;
      if (policy.permissionsAny.length &&
          intersects_(permissions, policy.permissionsAny)) return true;
      if (policy.permissionsAll.length &&
          policy.permissionsAll.every(function (permission) {
            return permissions.indexOf(permission) !== -1;
          })) return true;
      return false;
    } catch (error) {
      Logger.log("Erro em matchesPolicy_: " + error.message);
      throw error;
    }
  }

  function denied_(status, code, message) {
    if (typeof ApiError !== 'undefined' && ApiError.create) {
      return ApiError.create(status, message, code);
    }
    return {
      ok: false,
      status: status,
      statusCode: status,
      error: { code: code, message: message }
    };
  }

  function array_(value) {
    if (value == null) return [];
    return Array.isArray(value) ? value : [value];
  }

  function intersects_(left, right) {
    try {
      return left.some(function (value) { return right.indexOf(value) !== -1; });
    } catch (error) {
      Logger.log("Erro em intersects_: " + error.message);
      throw error;
    }
  }

  var api = {
    configure: configure,
    definePolicy: definePolicy,
    authorize: authorize,
    guard: guard
  };
  return api;
}());
