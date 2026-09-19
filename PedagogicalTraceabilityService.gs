      /**
       * PedagogicalTraceabilityService.gs
       *
       * Pequeno modulo de rastreabilidade pedagogica inspirado no Tool Debate.
       * Ele nao altera fluxo, autenticacao, planilhas ou UI: apenas monta um
       * envelope revisavel para ligar escolhas, evidencias, produto e revisao.
       */
      var PedagogicalTraceabilityService = (function () {
        var CFG = {
          projectName: "Emodiversa",
          shortName: "Emodiversa",
          theme: "autonomia socioemocional nao diagnostica",
          sourceStrategy: "A atividade deve ligar escolhas agregadas, cinco pilares de autonomia e reflexao acolhedora, sem diagnostico nem comparacao entre alunos.",
          learnerAgency: "O estudante reconhece padroes e escolhe proximas observacoes; a IA devolve linguagem de cuidado, nunca rotulo.",
          evidenceLabel: "vetores agregados por pilar e sinais de equilibrio",
          productLabel: "reflexao pedagogica nao diagnostica",
          reviewFocus: [
  "A reflexao evita diagnostico, ranking e promessa terapeutica?",
  "Os cinco pilares aparecem como possibilidades, nao como notas?",
  "Ha convite concreto para observar uma situacao futura?",
  "O texto usa dados agregados em vez de respostas literais sensiveis?"
]
        };

        var BLOCKED_KEYS = [
          'nome', 'name', 'student_name', 'teacher_name', 'email', 'cpf',
          'telefone', 'phone', 'responsavel', 'endereco', 'address',
          'matricula', 'ra', 'id_externo', 'observacao_pessoal'
        ];

        var PII_PATTERNS = [
          /\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/g,
          /\b\d{3}\.?\d{3}\.?\d{3}\-?\d{2}\b/g,
          /\b\d{10,11}\b/g
        ];

        function stripPii(text) {
          try {
            if (typeof text !== 'string') return text;
            var out = text;
            PII_PATTERNS.forEach(function (pattern) {
              out = out.replace(pattern, '[OMITIDO]');
            });
            return out;
          } catch (error) {
            Logger.log("Erro em stripPii: " + error.message);
            throw error;
          }
        }

        function sanitizeScalar(value, maxLen) {
          try {
            maxLen = maxLen || 500;
            if (value === null || value === undefined) return null;
            if (typeof value === 'number' || typeof value === 'boolean') return value;
            return stripPii(String(value)).slice(0, maxLen);
          } catch (error) {
            Logger.log("Erro em sanitizeScalar: " + error.message);
            throw error;
          }
        }

        function sanitizeObject(input, allowedKeys, maxLen) {
          try {
            var out = {};
            var dropped = [];
            if (!input || typeof input !== 'object') {
              return { value: out, droppedKeys: dropped };
            }
            Object.keys(input).forEach(function (key) {
              var lower = String(key).toLowerCase();
              if (BLOCKED_KEYS.indexOf(lower) !== -1) {
                dropped.push(key);
                return;
              }
              if (allowedKeys && allowedKeys.indexOf(key) === -1) {
                dropped.push(key);
                return;
              }
              var sanitized = sanitizeScalar(input[key], maxLen);
              if (sanitized !== null && sanitized !== '') out[key] = sanitized;
            });
            return { value: out, droppedKeys: dropped };
          } catch (error) {
            Logger.log("Erro em sanitizeObject: " + error.message);
            throw error;
          }
        }

        function compactList(values, maxItems, maxLen) {
          try {
            if (!Array.isArray(values)) return [];
            return values.slice(0, maxItems || 8).map(function (item) {
              if (item && typeof item === 'object') {
                return sanitizeObject(item, null, maxLen || 240).value;
              }
              return sanitizeScalar(item, maxLen || 240);
            }).filter(function (item) {
              return item !== null && item !== '';
            });
          } catch (error) {
            Logger.log("Erro em compactList: " + error.message);
            throw error;
          }
        }

        /**
         * Monta uma trilha curta e revisavel.
         *
         * Campos esperados em record:
         * - choices: escolhas ou parametros do estudante/grupo
         * - evidence: evidencias deterministicas ou agregadas
         * - product: produto gerado ou recomendado
         * - teacherNotes: observacoes docentes nao nominais
         */
        function buildTrace(record) {
          try {
            record = record || {};
            var choices = sanitizeObject(record.choices, null, 240);
            var evidence = sanitizeObject(record.evidence, null, 300);
            var product = sanitizeObject(record.product, null, 400);
            var teacherNotes = compactList(record.teacherNotes, 5, 280);
            var dropped = []
              .concat(choices.droppedKeys)
              .concat(evidence.droppedKeys)
              .concat(product.droppedKeys);

            var trace = {
              projectName: CFG.projectName,
              theme: CFG.theme,
              sourceStrategy: CFG.sourceStrategy,
              learnerAgency: CFG.learnerAgency,
              evidenceLabel: CFG.evidenceLabel,
              productLabel: CFG.productLabel,
              choices: choices.value,
              evidence: evidence.value,
              product: product.value,
              teacherNotes: teacherNotes,
              reviewQuestions: CFG.reviewFocus.slice(),
              droppedKeys: dropped,
              riskFlags: inferRiskFlags_(choices.value, evidence.value, product.value),
              createdAt: new Date().toISOString()
            };
            trace.completeness = scoreCompleteness(trace);
            return trace;
          } catch (error) {
            Logger.log("Erro em buildTrace: " + error.message);
            throw error;
          }
        }

        function inferRiskFlags_(choices, evidence, product) {
          try {
            try {
              var flags = [];
              if (Object.keys(choices).length === 0) flags.push('sem_escolhas_registradas');
              if (Object.keys(evidence).length === 0) flags.push('sem_evidencias_deterministicas');
              if (Object.keys(product).length === 0) flags.push('sem_produto_revisavel');
              var text = JSON.stringify({ choices: choices, evidence: evidence, product: product });
              if (/diagnost|ranking|nota final|incapaz|fracasso/i.test(text)) {
                flags.push('linguagem_avaliativa_revisar');
              }
              return flags;
            } catch (error) {
              Logger.log("Erro em inferRiskFlags_: " + error.message);
              throw error;
            }
          } catch (error) {
            Logger.log("Erro em inferRiskFlags_: " + error.message);
            throw error;
          }
        }

        function scoreCompleteness(trace) {
          var score = 0;
          if (Object.keys(trace.choices || {}).length > 0) score++;
          if (Object.keys(trace.evidence || {}).length > 0) score++;
          if (Object.keys(trace.product || {}).length > 0) score++;
          if ((trace.teacherNotes || []).length > 0) score++;
          if ((trace.reviewQuestions || []).length > 0) score++;
          return { score: score, max: 5, label: score + '/5' };
        }

        function buildTeacherChecklist(record) {
          try {
            var trace = buildTrace(record);
            return {
              projectName: CFG.projectName,
              checklist: [
                'Conferir se as escolhas registradas sao anonimas e suficientes.',
                'Conferir se as evidencias sustentam o produto sem expor estudante.',
                'Conferir se o produto preserva agencia do estudante.',
              ].concat(CFG.reviewFocus),
              riskFlags: trace.riskFlags,
              completeness: trace.completeness
            };
          } catch (error) {
            Logger.log("Erro em buildTeacherChecklist: " + error.message);
            throw error;
          }
        }

        function wrap(record) {
          var trace = buildTrace(record);
          if (typeof StandardReturn !== 'undefined' && StandardReturn && StandardReturn.ok) {
            return StandardReturn.ok(trace);
          }
          return { success: true, data: trace, error: null };
        }

        return {
          buildTrace: buildTrace,
          buildTeacherChecklist: buildTeacherChecklist,
          scoreCompleteness: scoreCompleteness,
          stripPii: stripPii,
          wrap: wrap
        };
      })();
