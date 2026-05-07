import { useCallback, useEffect, useMemo, useState } from 'react';
import { useAppStore } from '../../store/index';
import {
  scanModels, updateSession, deleteModel,
  preflightModel, patchGgufArch, getPatchStatus, type PatchProgress,
} from '../../api/index';
import { useVllmServer } from '../../hooks/useVllmServer';
import type {
  ModelInfo, InferenceParameters, ChatSession, PreflightResponse,
} from '../../store/types';

const STAGE_LABELS: Record<string, string> = {
  spawning: 'Spawning vLLM…',
  downloading: 'Downloading…',
  loading_weights: 'Loading weights…',
  compiling: 'Compiling CUDA graphs…',
  allocating_kv: 'Allocating KV cache…',
  starting_api: 'Starting API…',
  ready: 'Ready',
};

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${(bytes / 1e3).toFixed(1)} KB`;
}

function formatMb(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${Math.round(mb)} MB`;
}

function formatCtx(n: number): string {
  if (n >= 1024) return `${(n / 1024).toFixed(n % 1024 === 0 ? 0 : 1)}k`;
  return String(n);
}

function logCtx(value: number, max: number): number {
  // Map a 0..1000 slider value to ctx on a log curve from 512..max
  const minLog = Math.log(512);
  const maxLog = Math.log(max);
  const scale = (maxLog - minLog) / 1000;
  return Math.round(Math.exp(minLog + scale * value));
}

function invLogCtx(ctx: number, max: number): number {
  // Inverse: ctx -> slider value 0..1000
  const minLog = Math.log(512);
  const maxLog = Math.log(max);
  const scale = (maxLog - minLog) / 1000;
  return Math.round((Math.log(Math.max(512, ctx)) - minLog) / scale);
}

export default function ModelSelector() {
  const {
    models, selectedModelId, setSelectedModelId, scanningModels, setScanningModels,
    setModels, currentSessionId, sessions, updateSession: updateStoreSession,
    serverStatus, serverStage, serverError, settings,
  } = useAppStore();
  const vllmServer = useVllmServer();
  const [paramsOpen, setParamsOpen] = useState(true);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [deletingModel, setDeletingModel] = useState<string | null>(null);
  const [preflight, setPreflight] = useState<PreflightResponse | null>(null);
  const [preflightError, setPreflightError] = useState<string | null>(null);
  const [contextLength, setContextLength] = useState<number>(0);
  const [tensorParallelSize, setTensorParallelSize] = useState(1);
  const [maxNumSeqs, setMaxNumSeqs] = useState(1);
  const [gpuMemUtilOverride, setGpuMemUtilOverride] = useState<number | null>(null);
  const [additionalArgs, setAdditionalArgs] = useState<string>('');
  const [cpuOffloadGb, setCpuOffloadGb] = useState(0);
  const [kvCacheDtype, setKvCacheDtype] = useState<'auto' | 'fp8'>('auto');
  const [enforceEager, setEnforceEager] = useState(false);
  const [maxNumBatchedTokens, setMaxNumBatchedTokens] = useState(8192);
  const [baseModelRepo, setBaseModelRepo] = useState<string>('');
  const [languageModelOnly, setLanguageModelOnly] = useState(false);
  const [patchProgress, setPatchProgress] = useState<PatchProgress | null>(null);

  const currentSession = useMemo(
    () => sessions.find((s) => s.id === currentSessionId),
    [sessions, currentSessionId]
  );

  const selectedModel = useMemo(
    () => models.find((m) => m.id === (selectedModelId || currentSession?.modelId)),
    [models, selectedModelId, currentSession]
  );

  // Run preflight every time the user picks a different model.
  useEffect(() => {
    let cancelled = false;
    if (!selectedModel) {
      setPreflight(null);
      setPreflightError(null);
      return;
    }
    setPreflightError(null);
    preflightModel({
      modelPath: selectedModel.path,
      tensorParallelSize,
      maxNumSeqs,
      cpuOffloadGb,
      kvCacheDtype,
      maxNumBatchedTokens,
    }).then((res) => {
      if (cancelled) return;
      setPreflight(res);
      const recommended = res.recommendedContextLength;
      setContextLength(recommended);
      setGpuMemUtilOverride(null);
      // Auto-apply recommended CPU offload on first preflight if needed.
      if (res.recommendedCpuOffloadGb > 0 && cpuOffloadGb === 0) {
        setCpuOffloadGb(res.recommendedCpuOffloadGb);
      }
      // Auto-fill base-model repo (passed as --hf-config-path + --tokenizer)
      // for GGUFs whose arch isn't in the upstream transformers map.
      if (res.suggestedBaseRepo && !baseModelRepo) {
        setBaseModelRepo(res.suggestedBaseRepo);
      }
      if (res.prefs?.tensorParallelSize) setTensorParallelSize(res.prefs.tensorParallelSize);
      if (res.prefs?.maxNumSeqs) setMaxNumSeqs(res.prefs.maxNumSeqs);
      if (res.prefs?.maxModelLen) setContextLength(res.prefs.maxModelLen);
      if (res.prefs?.additionalArgs?.length) setAdditionalArgs(res.prefs.additionalArgs.join(' '));
    }).catch((e) => {
      if (cancelled) return;
      setPreflightError(e instanceof Error ? e.message : 'Preflight failed');
    });
    return () => { cancelled = true; };
  }, [selectedModel?.path, tensorParallelSize, maxNumSeqs]);

  // Re-run preflight (cheap) when ctx/offload/etc change so we get a fresh estimate.
  useEffect(() => {
    if (!selectedModel || !contextLength) return;
    let cancelled = false;
    preflightModel({
      modelPath: selectedModel.path,
      contextLength,
      tensorParallelSize,
      maxNumSeqs,
      cpuOffloadGb,
      kvCacheDtype,
      maxNumBatchedTokens,
    }).then((res) => { if (!cancelled) setPreflight(res); }).catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, [contextLength, tensorParallelSize, maxNumSeqs, cpuOffloadGb, kvCacheDtype, maxNumBatchedTokens, selectedModel?.path]);

  const params: InferenceParameters = currentSession?.parameters ?? {
    temperature: 0.7, topP: 0.9, topK: -1, maxTokens: 4096,
    presencePenalty: 0, frequencyPenalty: 0, repetitionPenalty: 1.0,
    seed: -1, stop: [],
  };

  const handleScan = useCallback(async () => {
    setScanningModels(true);
    try {
      const result = await scanModels();
      setModels(result.models);
    } catch { /* ignore */ }
    setScanningModels(false);
  }, [setScanningModels, setModels]);

  const handleSelectModel = useCallback(async (modelId: string) => {
    setSelectedModelId(modelId);
    if (currentSessionId) {
      try {
        await updateSession(currentSessionId, { modelId });
        updateStoreSession(currentSessionId, { modelId } as Partial<ChatSession>);
      } catch { /* ignore */ }
    }
  }, [currentSessionId, setSelectedModelId, updateStoreSession]);

  const updateParam = useCallback(async (key: string, value: number | string | string[]) => {
    if (!currentSessionId) return;
    const newParams = { ...params, [key]: value };
    try {
      await updateSession(currentSessionId, { parameters: newParams });
      updateStoreSession(currentSessionId, { parameters: newParams } as Partial<ChatSession>);
    } catch { /* ignore */ }
  }, [currentSessionId, params, updateStoreSession]);

  const handlePatch = useCallback(async () => {
    if (!selectedModel) return;
    try {
      await patchGgufArch(selectedModel.path);
      // Start polling.
      const tick = async () => {
        try {
          const s = await getPatchStatus();
          setPatchProgress(s);
          if (s.active) setTimeout(tick, 500);
          else if (s.stage === 'complete') {
            // Re-scan so the new patched model appears, then leave it for the user to pick.
            const result = await scanModels();
            setModels(result.models);
          }
        } catch { /* ignore */ }
      };
      tick();
    } catch (err) {
      setPatchProgress({
        active: false, bytesCopied: 0, totalBytes: 0,
        stage: 'error',
        error: err instanceof Error ? err.message : 'Patch failed',
      });
    }
  }, [selectedModel, setModels]);

  const handleLoad = useCallback(async () => {
    if (!selectedModel || !preflight) return;
    const memUtil = gpuMemUtilOverride ?? preflight.estimate.recommendedGpuMemUtil;
    try {
      const trimmedBase = baseModelRepo.trim();
      await vllmServer.start({
        modelPath: selectedModel.path,
        port: settings?.vllmPort || 8000,
        gpuMemoryUtilization: memUtil,
        maxModelLen: contextLength,
        quantization: null,
        tensorParallelSize,
        maxNumSeqs,
        dtype: 'auto',
        additionalArgs: additionalArgs.split(/\s+/).filter(Boolean),
        cpuOffloadGb,
        kvCacheDtype,
        maxNumBatchedTokens,
        enforceEager,
        hfConfigPath: trimmedBase || null,
        tokenizer: trimmedBase || null,
        languageModelOnly,
      });
    } catch { /* surfaced via store */ }
  }, [selectedModel, preflight, contextLength, gpuMemUtilOverride, tensorParallelSize, maxNumSeqs, additionalArgs, settings, vllmServer, cpuOffloadGb, kvCacheDtype, maxNumBatchedTokens, enforceEager, baseModelRepo, languageModelOnly]);

  const isLoading = serverStatus === 'starting';
  const isRunning = serverStatus === 'running';
  const verdictColors: Record<string, { bg: string; fg: string }> = {
    'full-gpu': { bg: 'var(--success-muted)', fg: 'var(--success)' },
    'partial-gpu': { bg: 'var(--warning-muted)', fg: 'var(--warning)' },
    'cpu-offload': { bg: 'var(--accent-muted)', fg: 'var(--accent)' },
    'too-large': { bg: 'var(--error-muted)', fg: 'var(--error)' },
  };

  const ctxMax = selectedModel?.contextLength ?? 0;
  const ctxNative = selectedModel?.nativeContextLength ?? 0;

  return (
    <div style={{ borderBottom: '1px solid var(--border)' }}>
      {/* Model picker */}
      <div style={{ padding: '12px' }}>
        <div style={{ fontFamily: 'var(--font-heading)', fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
          Model
        </div>

        <select
          className="input"
          value={selectedModel?.id || ''}
          onChange={(e) => handleSelectModel(e.target.value)}
          style={{ fontSize: 12, marginBottom: 8, fontFamily: 'var(--font-mono)' }}
        >
          <option value="">-- Select a model --</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name} ({formatBytes(m.sizeBytes)}){m.architecture ? ` — ${m.architecture}` : ''}{m.quantization ? ` [${m.quantization}]` : ''}
            </option>
          ))}
        </select>

        <button
          className="btn btn-secondary btn-sm"
          onClick={handleScan}
          disabled={scanningModels}
          style={{ width: '100%' }}
        >
          {scanningModels ? (<><div className="spinner" style={{ width: 12, height: 12 }} />Scanning...</>) : 'Scan for Models'}
        </button>

        {models.length === 0 && !scanningModels && (
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8, textAlign: 'center' }}>
            No models found. Add model directories in Settings.
          </p>
        )}

        {selectedModel && (
          <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {selectedModel.support.level === 'supported' && (
              <span className="badge" style={{ background: 'var(--success-muted)', color: 'var(--success)' }}>
                ✓ {selectedModel.support.family ?? 'vLLM-supported'}
              </span>
            )}
            {selectedModel.support.level === 'experimental' && (
              <span className="badge" style={{ background: 'var(--warning-muted)', color: 'var(--warning)' }} title={selectedModel.support.reason}>
                ⚠ {selectedModel.support.family ?? 'Experimental'} (GGUF)
              </span>
            )}
            {selectedModel.support.level === 'unsupported' && (
              <span className="badge" style={{ background: 'var(--error-muted)', color: 'var(--error)' }}>
                ✗ Not supported by vLLM
              </span>
            )}
            <span className="badge badge-accent">{selectedModel.architecture}</span>
            {selectedModel.quantization && (
              <span className="badge badge-warning">{selectedModel.quantization}</span>
            )}
            {selectedModel.isVision && <span className="badge badge-success">Vision</span>}
            <span className="badge" style={{ background: 'var(--surface-active)', color: 'var(--text-muted)' }}>
              {formatBytes(selectedModel.sizeBytes)}
            </span>
            {selectedModel.arch.numLayers !== null && (
              <span className="badge" style={{ background: 'var(--surface-active)', color: 'var(--text-muted)' }}>
                {selectedModel.arch.numLayers}L
              </span>
            )}
            {selectedModel.arch.hiddenSize !== null && (
              <span className="badge" style={{ background: 'var(--surface-active)', color: 'var(--text-muted)' }}>
                d={selectedModel.arch.hiddenSize}
              </span>
            )}
            <span className="badge" style={{ background: 'var(--surface-active)', color: 'var(--text-muted)' }}>
              ctx: {selectedModel.contextLength.toLocaleString()}
              {selectedModel.contextLength !== selectedModel.nativeContextLength
                ? ` (native ${selectedModel.nativeContextLength.toLocaleString()})`
                : ''}
            </span>
          </div>
        )}

        {/* Smart load section */}
        {selectedModel && preflight && (
          <div style={{
            marginTop: 12,
            padding: 10,
            background: 'var(--surface-active)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
          }}>
            {/* Base-model override for GGUF (passed as --hf-config-path + --tokenizer) */}
            {selectedModel.format === 'gguf' && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>
                  <span>
                    Base model (HF id) {preflight.ggufNeedsBaseConfig && (
                      <span style={{ color: 'var(--warning)', fontWeight: 600 }}> · required</span>
                    )}
                  </span>
                  {preflight.suggestedBaseRepo && baseModelRepo !== preflight.suggestedBaseRepo && (
                    <button
                      className="btn btn-ghost btn-sm"
                      style={{ fontSize: 10, padding: '0 4px' }}
                      onClick={() => setBaseModelRepo(preflight.suggestedBaseRepo!)}
                    >
                      Use suggested
                    </button>
                  )}
                </div>
                <input
                  className="input"
                  value={baseModelRepo}
                  onChange={(e) => setBaseModelRepo(e.target.value)}
                  placeholder={preflight.suggestedBaseRepo ?? 'e.g. Qwen/Qwen3-4B'}
                  style={{ padding: '4px 6px', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                  disabled={isLoading || isRunning}
                />
                <div style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
                  Passed to vLLM as <code>--hf-config-path</code> and <code>--tokenizer</code>.
                  {' '}Recommended for all GGUF loads (avoids slow / buggy tokenizer conversion).
                  {preflight.ggufNeedsBaseConfig && (
                    <span style={{ color: 'var(--warning)' }}>
                      {' '}This GGUF's architecture isn't in transformers' map, so this field is required.
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Context slider + input */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                <span>Context window</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <input
                    className="input"
                    type="number"
                    value={contextLength}
                    onChange={(e) => {
                      const v = parseInt(e.target.value, 10);
                      if (!isNaN(v)) setContextLength(v);
                    }}
                    style={{
                      width: 80, padding: '2px 6px', fontSize: 11,
                      fontFamily: 'var(--font-mono)', textAlign: 'left',
                    }}
                    disabled={isLoading || isRunning}
                  />
                  <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', minWidth: 50, textAlign: 'left' }}>
                    tok
                  </strong>
                </div>
              </div>
              <input
                type="range"
                min={0}
                max={1000}
                step={1}
                value={invLogCtx(contextLength, ctxMax)}
                onChange={(e) => setContextLength(logCtx(parseInt(e.target.value, 10), ctxMax))}
                style={{ width: '100%' }}
                disabled={isLoading || isRunning}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)' }}>
                <span>512</span>
                <span style={{ opacity: 0.7 }}>native: {ctxNative.toLocaleString()}</span>
                <span>{ctxMax.toLocaleString()}</span>
              </div>
              {(contextLength < 512 || contextLength > ctxMax) && !isLoading && !isRunning && (
                <div style={{ fontSize: 10, color: 'var(--error)', marginTop: 3 }}>
                  Value must be between 512 and {ctxMax.toLocaleString()}. Use the slider to pick a valid range.
                </div>
              )}
              {preflight.recommendedContextLength !== contextLength && contextLength >= 512 && contextLength <= ctxMax && (
                <button
                  className="btn btn-ghost btn-sm"
                  style={{ marginTop: 4, fontSize: 10, padding: '2px 6px' }}
                  onClick={() => setContextLength(preflight.recommendedContextLength)}
                >
                  Snap to recommended ({preflight.recommendedContextLength.toLocaleString()})
                </button>
              )}
            </div>

            {/* VRAM gauge */}
            <VramGauge
              estimate={preflight.estimate}
              totalGpuMb={preflight.totalGpuMb}
              freeGpuMb={preflight.freeGpuMb}
              tp={tensorParallelSize}
            />

            {/* Verdict */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: 11,
              padding: '4px 8px',
              borderRadius: 'var(--radius-sm)',
              background: verdictColors[preflight.estimate.fitVerdict].bg,
              color: verdictColors[preflight.estimate.fitVerdict].fg,
              fontWeight: 600,
            }}>
              <span>
                {preflight.estimate.fitVerdict === 'full-gpu' && 'Fits in GPU'}
                {preflight.estimate.fitVerdict === 'partial-gpu' && 'Partial GPU (other GPU mem in use)'}
                {preflight.estimate.fitVerdict === 'cpu-offload' && 'Will need CPU offload'}
                {preflight.estimate.fitVerdict === 'too-large' && 'Too large — reduce ctx, enable quant, or add GPU'}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>
                {formatMb(preflight.estimate.totalMb)}
              </span>
            </div>

            {/* GPU mem util — auto with override */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)' }}>
                <span>GPU memory pool</span>
                <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                  {((gpuMemUtilOverride ?? preflight.estimate.recommendedGpuMemUtil) * 100).toFixed(0)}%
                  {' '}≈ {formatMb((gpuMemUtilOverride ?? preflight.estimate.recommendedGpuMemUtil) * (preflight.totalGpuMb / Math.max(1, tensorParallelSize)))}
                  {gpuMemUtilOverride === null && ' (auto)'}
                </strong>
              </div>
              <div style={{ fontSize: 9, color: 'var(--text-muted)', marginBottom: 3 }}>
                vLLM reserves this entire pool on startup. Lower it to leave VRAM for other apps; raise it to allow more concurrent requests.
              </div>
              <input
                type="range"
                min={0.3}
                max={0.95}
                step={0.01}
                value={gpuMemUtilOverride ?? preflight.estimate.recommendedGpuMemUtil}
                onChange={(e) => setGpuMemUtilOverride(parseFloat(e.target.value))}
                style={{ width: '100%' }}
                disabled={isLoading || isRunning}
              />
              {gpuMemUtilOverride !== null && (
                <button
                  className="btn btn-ghost btn-sm"
                  style={{ fontSize: 10, padding: '2px 6px' }}
                  onClick={() => setGpuMemUtilOverride(null)}
                >
                  Reset to auto
                </button>
              )}
            </div>

            {/* Advanced */}
            <div>
              <button
                className="btn btn-ghost btn-sm"
                style={{ fontSize: 10, padding: '2px 6px' }}
                onClick={() => setAdvancedOpen(!advancedOpen)}
              >
                {advancedOpen ? '▼' : '▶'} Advanced
              </button>
              {advancedOpen && (
                <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <div style={{ flex: 1 }}>
                      <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>TP size</label>
                      <input
                        className="input"
                        type="number" min={1} max={8}
                        value={tensorParallelSize}
                        onChange={(e) => setTensorParallelSize(Math.max(1, parseInt(e.target.value) || 1))}
                        style={{ padding: '4px 6px', fontSize: 11 }}
                        disabled={isLoading || isRunning}
                      />
                    </div>
                    <div style={{ flex: 1 }}>
                      <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Max seqs</label>
                      <input
                        className="input"
                        type="number" min={1} max={1024}
                        value={maxNumSeqs}
                        onChange={(e) => setMaxNumSeqs(Math.max(1, parseInt(e.target.value) || 1))}
                        style={{ padding: '4px 6px', fontSize: 11 }}
                        disabled={isLoading || isRunning}
                      />
                    </div>
                  </div>
                  <div>
                    <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      CPU offload (GB) — spill weights to RAM if VRAM is tight
                    </label>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <input
                        type="range"
                        min={0}
                        max={Math.min(64, Math.floor(((preflight.systemRamMb || 0) / 1024) * 0.5))}
                        step={1}
                        value={cpuOffloadGb}
                        onChange={(e) => setCpuOffloadGb(parseInt(e.target.value, 10))}
                        style={{ flex: 1 }}
                        disabled={isLoading || isRunning}
                      />
                      <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', minWidth: 36, textAlign: 'right' }}>
                        {cpuOffloadGb} GB
                      </span>
                    </div>
                    {preflight.recommendedCpuOffloadGb > 0 && cpuOffloadGb !== preflight.recommendedCpuOffloadGb && (
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ fontSize: 10, padding: '2px 6px', marginTop: 2 }}
                        onClick={() => setCpuOffloadGb(preflight.recommendedCpuOffloadGb)}
                      >
                        Snap to recommended ({preflight.recommendedCpuOffloadGb} GB)
                      </button>
                    )}
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <div style={{ flex: 1 }}>
                      <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>KV cache dtype</label>
                      <select
                        className="input"
                        value={kvCacheDtype}
                        onChange={(e) => setKvCacheDtype(e.target.value as 'auto' | 'fp8')}
                        style={{ padding: '4px 6px', fontSize: 11 }}
                        disabled={isLoading || isRunning}
                      >
                        <option value="auto">auto (fp16)</option>
                        <option value="fp8">fp8 (½ KV memory)</option>
                      </select>
                    </div>
                    <div style={{ flex: 1 }}>
                      <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Prefill chunk</label>
                      <input
                        className="input"
                        type="number" min={512} max={32768} step={512}
                        value={maxNumBatchedTokens}
                        onChange={(e) => setMaxNumBatchedTokens(Math.max(512, parseInt(e.target.value) || 8192))}
                        style={{ padding: '4px 6px', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                        disabled={isLoading || isRunning}
                      />
                    </div>
                  </div>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                    <input
                      type="checkbox"
                      checked={enforceEager}
                      onChange={(e) => setEnforceEager(e.target.checked)}
                      disabled={isLoading || isRunning}
                    />
                    --enforce-eager (skip CUDA graph capture; saves ~500 MB, lower throughput)
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                    <input
                      type="checkbox"
                      checked={languageModelOnly}
                      onChange={(e) => setLanguageModelOnly(e.target.checked)}
                      disabled={isLoading || isRunning}
                    />
                    --language-model-only (hybrid multimodal models — Llama-4, Mistral-3, Qwen3.5, Step3 — skip vision/audio modules to free VRAM for KV cache)
                  </label>
                  <div>
                    <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Additional vLLM args</label>
                    <input
                      className="input"
                      value={additionalArgs}
                      onChange={(e) => setAdditionalArgs(e.target.value)}
                      placeholder="--disable-log-requests"
                      style={{ padding: '4px 6px', fontSize: 11, fontFamily: 'var(--font-mono)' }}
                      disabled={isLoading || isRunning}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Last load error */}
            {serverStatus === 'error' && serverError && (
              <div style={{
                fontSize: 11,
                background: 'var(--error-muted)',
                color: 'var(--error)',
                padding: '6px 8px',
                borderRadius: 'var(--radius-sm)',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}>
                {serverError}
              </div>
            )}

            {/* Stage indicator */}
            {isLoading && (
              <div style={{ fontSize: 11, color: 'var(--warning)', display: 'flex', alignItems: 'center', gap: 6 }}>
                <div className="spinner" style={{ width: 12, height: 12 }} />
                {STAGE_LABELS[serverStage ?? 'spawning'] ?? 'Loading…'}
              </div>
            )}

            {/* Architecture warnings */}
            {selectedModel.support.level === 'unsupported' && (
              <div style={{
                fontSize: 11,
                background: 'var(--error-muted)',
                color: 'var(--error)',
                padding: '6px 8px',
                borderRadius: 'var(--radius-sm)',
                whiteSpace: 'pre-wrap',
              }}>
                {selectedModel.support.reason ?? 'This architecture is not supported by vLLM.'}
              </div>
            )}
            {selectedModel.support.level === 'experimental' && (
              <div style={{
                fontSize: 11,
                background: 'var(--warning-muted)',
                color: 'var(--warning)',
                padding: '6px 8px',
                borderRadius: 'var(--radius-sm)',
                whiteSpace: 'pre-wrap',
              }}>
                {selectedModel.support.reason ?? 'GGUF loading for this family is experimental — load may fail.'}
                {' '}Loading is allowed; if it fails, try the safetensors release of the same model.
              </div>
            )}

            {/* Action button */}
            <div style={{ display: 'flex', gap: 6 }}>
              {!isRunning && (() => {
                const blocked = selectedModel.support.level === 'unsupported';
                const warn = selectedModel.support.level === 'experimental';
                const needsBase = preflight.ggufNeedsBaseConfig && !baseModelRepo.trim();
                const ctxOutOfRange = contextLength < 512 || contextLength > ctxMax;
                const bg = blocked || needsBase || ctxOutOfRange ? 'var(--surface-active)' : warn ? 'var(--warning-muted)' : 'var(--success-muted)';
                const fg = blocked || needsBase || ctxOutOfRange ? 'var(--text-muted)' : warn ? 'var(--warning)' : 'var(--success)';
                const border = blocked || needsBase || ctxOutOfRange ? 'var(--border)' : warn ? 'var(--warning)' : 'var(--success)';
                const label = isLoading
                  ? 'Loading…'
                  : blocked
                    ? 'Unsupported architecture'
                    : needsBase
                      ? 'Provide a base model first'
                      : ctxOutOfRange
                        ? 'Context out of range'
                        : warn
                          ? 'Load anyway (experimental)'
                          : 'Load Model';
                return (
                  <button
                    className="btn btn-sm"
                    style={{
                      flex: 1, background: bg, color: fg, border: `1px solid ${border}`,
                      fontSize: 11, cursor: blocked || needsBase || ctxOutOfRange ? 'not-allowed' : 'pointer',
                    }}
                    onClick={handleLoad}
                    disabled={isLoading || preflight.estimate.fitVerdict === 'too-large' || blocked || needsBase || ctxOutOfRange}
                    title={blocked ? selectedModel.support.reason : needsBase ? 'Set the Base model field' : ctxOutOfRange ? `Context must be between 512 and ${ctxMax.toLocaleString()}` : ''}
                  >
                    {label}
                  </button>
                );
              })()}
              {isRunning && (
                <button
                  className="btn btn-sm"
                  style={{ flex: 1, background: 'var(--warning-muted)', color: 'var(--warning)', border: '1px solid var(--warning)', fontSize: 11 }}
                  onClick={() => vllmServer.stop()}
                >
                  Unload
                </button>
              )}
              <button
                className="btn btn-sm"
                style={{ background: 'var(--error-muted)', color: 'var(--error)', border: '1px solid var(--error)', fontSize: 11 }}
                onClick={() => setDeletingModel(selectedModel.path)}
                disabled={isLoading}
              >
                Delete
              </button>
            </div>
          </div>
        )}

        {selectedModel && preflightError && (
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--error)' }}>
            {preflightError}
          </div>
        )}
      </div>

      {/* Delete confirmation */}
      {deletingModel && (
        <div className="confirm-overlay" onClick={() => setDeletingModel(null)}>
          <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
            <p>Permanently delete this model from disk? This cannot be undone.</p>
            <div className="confirm-actions">
              <button className="btn btn-secondary btn-sm" onClick={() => setDeletingModel(null)}>Cancel</button>
              <button className="btn btn-danger btn-sm" onClick={async () => {
                try {
                  const res = await deleteModel(deletingModel);
                  setModels(res.models);
                } catch { /* ignore */ }
                setDeletingModel(null);
              }}>Delete Forever</button>
            </div>
          </div>
        </div>
      )}

      {/* Inference Parameters */}
      <div style={{ padding: '0 12px 12px' }}>
        <div
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            cursor: 'pointer', padding: '8px 0', userSelect: 'none',
          }}
          onClick={() => setParamsOpen(!paramsOpen)}
        >
          <span style={{ fontFamily: 'var(--font-heading)', fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Sampling
          </span>
          <span style={{ fontSize: 12, color: 'var(--text-muted)', transform: paramsOpen ? 'rotate(90deg)' : 'none', transition: 'transform var(--transition)' }}>›</span>
        </div>

        {paramsOpen && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <ParamSlider label="Temperature" value={params.temperature} min={0} max={2} step={0.05} onChange={(v) => updateParam('temperature', v)} />
            <ParamSlider label="Top-P" value={params.topP} min={0} max={1} step={0.05} onChange={(v) => updateParam('topP', v)} />
            <ParamSlider label="Top-K" value={params.topK} min={-1} max={200} step={1} onChange={(v) => updateParam('topK', v)} format={(v) => v === -1 ? 'Off' : String(v)} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 11, color: 'var(--text-muted)', minWidth: 80 }}>Max Tokens</span>
              <input
                className="input" type="number" value={params.maxTokens}
                onChange={(e) => updateParam('maxTokens', parseInt(e.target.value) || 4096)}
                style={{ width: 70, padding: '4px 6px', fontSize: 11, fontFamily: 'var(--font-mono)' }}
              />
            </div>
            <ParamSlider label="Presence" value={params.presencePenalty} min={-2} max={2} step={0.1} onChange={(v) => updateParam('presencePenalty', v)} />
            <ParamSlider label="Frequency" value={params.frequencyPenalty} min={-2} max={2} step={0.1} onChange={(v) => updateParam('frequencyPenalty', v)} />
            <ParamSlider label="Repetition" value={params.repetitionPenalty} min={1} max={2} step={0.05} onChange={(v) => updateParam('repetitionPenalty', v)} />
          </div>
        )}
      </div>
    </div>
  );
}

function VramGauge({ estimate, totalGpuMb, freeGpuMb, tp }: {
  estimate: { weightsMb: number; kvCacheMb: number; activationsMb: number; overheadMb: number; totalMb: number; perGpuMb: number };
  totalGpuMb: number;
  freeGpuMb: number;
  tp: number;
}) {
  const perGpuTotalMb = tp > 0 ? totalGpuMb / tp : totalGpuMb;
  const perGpuFreeMb = tp > 0 ? freeGpuMb / tp : freeGpuMb;
  const usedPct = Math.min(100, (estimate.perGpuMb / perGpuTotalMb) * 100);
  const freePct = Math.min(100, ((perGpuTotalMb - perGpuFreeMb) / perGpuTotalMb) * 100);

  const segs = [
    { mb: estimate.weightsMb / tp, color: 'var(--accent)', label: 'weights' },
    { mb: estimate.kvCacheMb / tp, color: 'var(--warning)', label: 'KV cache' },
    { mb: estimate.activationsMb / tp, color: 'var(--success)', label: 'activations' },
    { mb: estimate.overheadMb, color: 'var(--text-muted)', label: 'overhead' },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>
        <span>Per-GPU VRAM estimate</span>
        <span style={{ fontFamily: 'var(--font-mono)' }}>
          {formatMb(estimate.perGpuMb)} / {formatMb(perGpuTotalMb)}
        </span>
      </div>
      <div style={{ position: 'relative', height: 10, background: 'var(--bg)', borderRadius: 4, overflow: 'hidden', display: 'flex' }}>
        {segs.map((s, i) => (
          <div key={i} style={{
            width: `${Math.min(100, (s.mb / perGpuTotalMb) * 100)}%`,
            background: s.color,
            opacity: 0.85,
          }} title={`${s.label}: ${formatMb(s.mb)}`} />
        ))}
        {/* Free-line marker */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          left: `${100 - (perGpuFreeMb / perGpuTotalMb * 100)}%`,
          width: 1, background: 'var(--text-primary)', opacity: 0.4,
        }} title={`${formatMb(perGpuTotalMb - perGpuFreeMb)} currently in use`} />
      </div>
      <div style={{ display: 'flex', gap: 8, fontSize: 9, color: 'var(--text-muted)', marginTop: 3, flexWrap: 'wrap' }}>
        {segs.map(s => (
          <span key={s.label} style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
            <span style={{ width: 6, height: 6, background: s.color, borderRadius: 1 }} />
            {s.label} {formatMb(s.mb)}
          </span>
        ))}
      </div>
      {usedPct > 95 && (
        <div style={{ fontSize: 10, color: 'var(--error)', marginTop: 3 }}>
          Estimated VRAM exceeds device capacity at this context size.
        </div>
      )}
      {freePct > 30 && (
        <div style={{ fontSize: 10, color: 'var(--warning)', marginTop: 3 }}>
          {freePct.toFixed(0)}% of GPU is already in use by other processes.
        </div>
      )}
    </div>
  );
}

function ParamSlider({
  label, value, min, max, step, onChange, format,
}: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; format?: (v: number) => string;
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <span style={{ fontSize: 11, color: 'var(--text-muted)', minWidth: 80 }}>{label}</span>
      <div className="slider-group" style={{ flex: 1 }}>
        <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} />
        <span className="slider-value" style={{ minWidth: 30 }}>
          {format ? format(value) : value % 1 === 0 ? String(value) : value.toFixed(2)}
        </span>
      </div>
    </div>
  );
}
