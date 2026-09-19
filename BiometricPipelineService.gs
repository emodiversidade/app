/**
 * FROTA-16: pipeline biometrico orientado a lotes para Apps Script.
 *
 * Amostras brutas ficam em chunks JSONL no Drive. Sheets recebem apenas
 * indices e metricas agregadas. Nao use este servico para diagnostico clinico.
 */
var BiometricPipelineService = (function () {
  'use strict';
  var adapters_ = {};
  var processors_ = {};
  var config_ = {
    folderProperty: 'BIOMETRIC_QUEUE_FOLDER_ID',
    maxBatchSize: 500,
    maxChunkBytes: 500000,
    schemaVersion: '1.0',
    retentionDays: 30
  };
  var SENSOR_FIELDS = {
    EEG: ['af7', 'af8', 'tp9', 'tp10', 'theta', 'alpha', 'beta'],
    ECG: ['heartRate', 'rrInterval'],
    EDA: ['conductance', 'phasic', 'tonic'],
    POG: ['x', 'y', 'pupil', 'fixation', 'saccade']
  };

  function configure(options) {
    try {
      options = options || {};
      adapters_ = options.adapters || adapters_;
      Object.keys(config_).forEach(function (key) {
        if (options[key] != null) config_[key] = options[key];
      });
      return api;
    } catch (error) {
      Logger.log("Erro em configure: " + error.message);
      throw error;
    }
  }

  function registerProcessor(sensorType, processor) {
    if (!SENSOR_FIELDS[sensorType] || typeof processor !== 'function') {
      throw new Error('INVALID_BIOMETRIC_PROCESSOR');
    }
    processors_[sensorType] = processor;
    return api;
  }

  function ingest(batch, context) {
    try {
      try {
        context = context || {};
        var auth = authorize_(context);
        if (auth && !auth.ok) return auth;
        if (!Array.isArray(batch) || !batch.length || batch.length > config_.maxBatchSize) {
          return ApiError.validation([{
            field: '$.samples',
            rule: 'batchSize',
            message: 'O lote deve conter entre 1 e ' + config_.maxBatchSize + ' amostras.'
          }]);
        }
        checkConsent_(context);
        var normalized = batch.map(function (sample, index) {
          return normalizeSample_(sample, context, index);
        });
        var chunk = {
          schemaVersion: config_.schemaVersion,
          sessionId: pseudonymize_(context.sessionId),
          subjectId: pseudonymize_(context.subjectId),
          receivedAt: now_().toISOString(),
          sampleCount: normalized.length,
          samples: normalized
        };
        var serialized = normalized.map(function (sample) {
          return JSON.stringify(sample);
        }).join('\n');
        if (serialized.length > config_.maxChunkBytes) {
          return ApiError.create(413, 'Lote biometrico excede o tamanho permitido.',
            'BIOMETRIC_BATCH_TOO_LARGE');
        }
        var location = persistChunk_(chunk, serialized);
        audit_('biometric.ingest', context, {
          sensorTypes: unique_(normalized.map(function (sample) { return sample.sensorType; })),
          sampleCount: normalized.length,
          chunkId: location.chunkId
        });
        return {
          ok: true,
          status: 202,
          data: {
            chunkId: location.chunkId,
            sampleCount: normalized.length,
            receivedAt: chunk.receivedAt,
            state: 'QUEUED'
          }
        };
      } catch (error) {
        Logger.log("Erro em ingest: " + error.message);
        throw error;
      }
    } catch (error) {
      Logger.log("Erro em ingest: " + error.message);
      throw error;
    }
  }

  function process(batch, context) {
    try {
      context = context || {};
      var groups = {};
      batch.forEach(function (sample) {
        var normalized = normalizeSample_(sample, context, 0);
        (groups[normalized.sensorType] = groups[normalized.sensorType] || []).push(normalized);
      });
      var metrics = {};
      Object.keys(groups).forEach(function (sensorType) {
        metrics[sensorType] = (processors_[sensorType] || defaultProcessor_)(groups[sensorType]);
      });
      return {
        schemaVersion: config_.schemaVersion,
        sessionId: pseudonymize_(context.sessionId),
        window: window_(batch),
        metrics: metrics,
        disclaimer: 'Metricas educacionais experimentais, sem validade clinica.'
      };
    } catch (error) {
      Logger.log("Erro em process: " + error.message);
      throw error;
    }
  }

  function normalizeSample_(sample, context, index) {
    try {
      sample = sample || {};
      var sensorType = String(sample.sensorType || '').toUpperCase();
      if (!SENSOR_FIELDS[sensorType]) {
        throw new Error('UNSUPPORTED_SENSOR_TYPE_AT_' + index);
      }
      var values = {};
      SENSOR_FIELDS[sensorType].forEach(function (field) {
        if (sample.values && sample.values[field] != null) {
          var value = Number(sample.values[field]);
          if (!isFinite(value)) throw new Error('INVALID_SENSOR_VALUE_' + field);
          values[field] = value;
        }
      });
      if (!Object.keys(values).length) throw new Error('EMPTY_SENSOR_VALUES_AT_' + index);
      return {
        schemaVersion: config_.schemaVersion,
        sensorType: sensorType,
        deviceId: sample.deviceId ? pseudonymize_(sample.deviceId) : null,
        capturedAt: new Date(sample.capturedAt || now_()).toISOString(),
        sequence: Number(sample.sequence == null ? index : sample.sequence),
        values: values,
        quality: sample.quality == null ? null : Number(sample.quality)
      };
    } catch (error) {
      Logger.log("Erro em normalizeSample_: " + error.message);
      throw error;
    }
  }

  function persistChunk_(chunk, serialized) {
    try {
      var folderId = property_(config_.folderProperty);
      if (!folderId) throw new Error('BIOMETRIC_QUEUE_FOLDER_NOT_CONFIGURED');
      var folder = drive_().getFolderById(folderId);
      var chunkId = chunk.sessionId + '_' + now_().getTime();
      var fileName = 'bio_' + chunk.receivedAt.slice(0, 10) + '_' + chunkId + '.jsonl';
      withLock_(function () {
        folder.createFile(fileName, serialized, 'application/json');
      });
      return { chunkId: chunkId, fileName: fileName };
    } catch (error) {
      Logger.log("Erro em persistChunk_: " + error.message);
      throw error;
    }
  }

  function defaultProcessor_(samples) {
    try {
      var sums = {}, counts = {};
      samples.forEach(function (sample) {
        Object.keys(sample.values).forEach(function (key) {
          sums[key] = (sums[key] || 0) + sample.values[key];
          counts[key] = (counts[key] || 0) + 1;
        });
      });
      var averages = {};
      Object.keys(sums).forEach(function (key) { averages[key] = sums[key] / counts[key]; });
      return { sampleCount: samples.length, averages: averages };
    } catch (error) {
      Logger.log("Erro em defaultProcessor_: " + error.message);
      throw error;
    }
  }

  function authorize_(context) {
    if (typeof IamGuard === 'undefined') return null;
    return IamGuard.authorize('biometric.ingest', context.credential, context);
  }
  function checkConsent_(context) {
    if (typeof ConsentService !== 'undefined') {
      ConsentService.check(pseudonymize_(context.subjectId), 'collect');
    }
  }
  function pseudonymize_(value) {
    if (value == null || value === '') throw new Error('BIOMETRIC_IDENTIFIER_REQUIRED');
    return StructuredLogService.pseudonymize(value);
  }
  function audit_(operation, context, details) {
    if (typeof StructuredLogService !== 'undefined') {
      StructuredLogService.auditAi(operation, {
        sessionId: context.sessionId,
        actorId: context.actorId,
        outcome: 'accepted',
        details: details
      });
    }
  }
  function withLock_(callback) {
    try {
      var lock = adapters_.lock || LockService.getScriptLock();
      if (!lock.tryLock(5000)) throw new Error('BIOMETRIC_QUEUE_BUSY');
      try { return callback(); } finally { lock.releaseLock(); }
    } catch (error) {
      Logger.log("Erro em withLock_: " + error.message);
      throw error;
    }
  }
  function window_(samples) {
    try {
      var times = samples.map(function (sample) {
        return new Date(sample.capturedAt).getTime();
      }).sort();
      return {
        startedAt: new Date(times[0]).toISOString(),
        endedAt: new Date(times[times.length - 1]).toISOString()
      };
    } catch (error) {
      Logger.log("Erro em window_: " + error.message);
      throw error;
    }
  }
  function unique_(values) {
    return values.filter(function (value, index) { return values.indexOf(value) === index; });
  }
  function property_(name) {
    try {
      return adapters_.getProperty ? adapters_.getProperty(name) :
        PropertiesService.getScriptProperties().getProperty(name);
    } catch (error) {
      Logger.log("Erro em property_: " + error.message);
      throw error;
    }
  }
  function drive_() { return adapters_.drive || DriveApp; }
  function now_() { return adapters_.now ? adapters_.now() : new Date(); }

  var api = {
    configure: configure,
    registerProcessor: registerProcessor,
    ingest: ingest,
    process: process
  };
  return api;
}());

