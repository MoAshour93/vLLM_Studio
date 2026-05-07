import { useState, useCallback, useRef, useEffect } from 'react';
import { useAppStore } from '../../store/index';
import { useVllmServer } from '../../hooks/useVllmServer';
import type { VllmServerConfig, GpuStats } from '../../store/types';

export default function ServerPanel() {
  const {
    serverStatus, serverLogs, gpuStats, settings,
    models, vllmVersion,
  } = useAppStore();
  const vllmServer = useVllmServer();
  const [config, setConfig] = useState<VllmServerConfig>({
    modelPath: '',
    port: settings?.vllmPort || 8000,
    gpuMemoryUtilization: settings?.gpuMemoryUtilization || 0.9,
    maxModelLen: null,
    quantization: null,
    tensorParallelSize: 1,
    maxNumSeqs: 256,
    dtype: 'auto',
    additionalArgs: [],
  });
  const [logsOpen, setLogsOpen] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (logEndRef.current && logsOpen) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [serverLogs, logsOpen]);

  const handleStart = useCallback(async () => {
    if (!config.modelPath) return;
    try {
      await vllmServer.start(config);
    } catch { /* error handled by store */ }
  }, [config, vllmServer]);

  const handleStop = useCallback(async () => {
    await vllmServer.stop();
  }, [vllmServer]);

  const copyApiUrl = useCallback(() => {
    const port = config.port || 8000;
    navigator.clipboard.writeText(`http://localhost:${port}/v1`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [config.port]);

  const statusLabel = serverStatus.toUpperCase();
  const statusClass = `status-dot ${serverStatus}`;

  return (
    <div style={{ borderTop: '1px solid var(--border)', display: 'flex', flexDirection: 'column' }}>
      {/* Server Header */}
      <div style={{ padding: '12px' }}>
        <div style={{ fontFamily: 'var(--font-heading)', fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
          vLLM Server
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 10px',
          background: 'var(--surface-active)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border)',
          marginBottom: 8,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className={statusClass} />
            <span style={{ fontSize: 12, fontWeight: 600, fontFamily: 'var(--font-mono)', letterSpacing: '0.04em', color: 'var(--text-secondary)' }}>
              {statusLabel}
            </span>
          </div>
          {vllmVersion && (
            <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              v{vllmVersion}
            </span>
          )}
        </div>

        {/* Progress bar for starting state */}
        {serverStatus === 'starting' && (
          <div style={{
            height: 3,
            background: 'var(--border)',
            borderRadius: 2,
            marginBottom: 8,
            overflow: 'hidden',
          }}>
            <div style={{
              height: '100%',
              width: '30%',
              background: 'var(--warning)',
              borderRadius: 2,
              animation: 'loading-bar 2s ease-in-out infinite',
            }} />
          </div>
        )}

        {/* Error message */}
        {serverStatus === 'error' && (
          <div style={{
            fontSize: 11,
            color: 'var(--error)',
            background: 'var(--error-muted)',
            padding: '6px 8px',
            borderRadius: 'var(--radius-sm)',
            marginBottom: 8,
            wordBreak: 'break-word',
          }}>
            {useAppStore.getState().serverError || 'Unknown error'}
          </div>
        )}

        {/* Config inputs (only when stopped) */}
        {serverStatus === 'stopped' || serverStatus === 'error' ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <select
              className="input"
              value={config.modelPath}
              onChange={(e) => setConfig((c) => ({ ...c, modelPath: e.target.value }))}
              style={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}
            >
              <option value="">-- Select model to load --</option>
              {models.map((m) => (
                <option key={m.id} value={m.path}>
                  {m.name} ({m.architecture})
                </option>
              ))}
            </select>

            <div style={{ display: 'flex', gap: 8 }}>
              <div style={{ flex: 1 }}>
                <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 2 }}>Port</label>
                <input
                  className="input"
                  type="number"
                  value={config.port}
                  onChange={(e) => setConfig((c) => ({ ...c, port: parseInt(e.target.value) || 8000 }))}
                  style={{ padding: '4px 6px', fontSize: 11 }}
                />
              </div>
              <div style={{ flex: 1 }}>
                <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 2 }}>TP Size</label>
                <input
                  className="input"
                  type="number"
                  min={1}
                  max={8}
                  value={config.tensorParallelSize}
                  onChange={(e) => setConfig((c) => ({ ...c, tensorParallelSize: parseInt(e.target.value) || 1 }))}
                  style={{ padding: '4px 6px', fontSize: 11 }}
                />
              </div>
            </div>

            <div>
              <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 2 }}>
                GPU Memory Util: {config.gpuMemoryUtilization.toFixed(1)}
              </label>
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.05}
                value={config.gpuMemoryUtilization}
                onChange={(e) => setConfig((c) => ({ ...c, gpuMemoryUtilization: parseFloat(e.target.value) }))}
                style={{ width: '100%' }}
              />
            </div>

            <div>
              <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 2 }}>Quantization</label>
              <select
                className="input"
                value={config.quantization || 'none'}
                onChange={(e) => setConfig((c) => ({ ...c, quantization: e.target.value === 'none' ? null : e.target.value }))}
                style={{ fontSize: 11, padding: '4px 6px' }}
              >
                <option value="none">None</option>
                <option value="awq">AWQ</option>
                <option value="gptq">GPTQ</option>
                <option value="fp8">FP8</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 2 }}>Max Model Length (optional)</label>
              <input
                className="input"
                type="number"
                value={config.maxModelLen || ''}
                onChange={(e) => setConfig((c) => ({ ...c, maxModelLen: e.target.value ? parseInt(e.target.value) : null }))}
                placeholder="Auto"
                style={{ padding: '4px 6px', fontSize: 11 }}
              />
            </div>

            <div>
              <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginBottom: 2 }}>Additional Arguments</label>
              <input
                className="input"
                type="text"
                placeholder="e.g. --enforce-eager --disable-log-requests"
                onChange={(e) => setConfig((c) => ({ ...c, additionalArgs: e.target.value.split(/\s+/).filter(Boolean) }))}
                style={{ padding: '4px 6px', fontSize: 11, fontFamily: 'var(--font-mono)' }}
              />
            </div>

            <button
              className="btn btn-primary btn-sm"
              onClick={handleStart}
              disabled={!config.modelPath}
              style={{ width: '100%', marginTop: 4 }}
            >
              Start Server
            </button>
          </div>
        ) : (
          <button
            className="btn btn-danger btn-sm"
            onClick={handleStop}
            style={{ width: '100%' }}
          >
            Stop Server
          </button>
        )}
      </div>

      {/* API Info */}
      {serverStatus === 'running' && (
        <div style={{ padding: '0 12px 8px' }}>
          <div style={{
            fontSize: 10,
            color: 'var(--text-muted)',
            marginBottom: 4,
            fontFamily: 'var(--font-mono)',
          }}>
            OpenAI-compatible API
          </div>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '6px 8px',
            background: 'var(--bg)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--success)',
          }}>
            <code style={{ flex: 1, fontSize: 11 }}>http://localhost:{config.port}/v1</code>
            <button className="btn-icon" onClick={copyApiUrl} title="Copy URL">
              {copied ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                </svg>
              )}
            </button>
          </div>
        </div>
      )}

      {/* GPU Stats */}
      {gpuStats.length > 0 && (
        <div style={{ padding: '0 12px 8px' }}>
          <div style={{ fontFamily: 'var(--font-heading)', fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>
            GPU Monitor
          </div>
          {gpuStats.map((gpu) => (
            <GpuStatBar key={gpu.index} gpu={gpu} />
          ))}
        </div>
      )}

      {/* Logs */}
      <div style={{ display: 'flex', flexDirection: 'column', borderTop: '1px solid var(--border)' }}>
        <div
          style={{
            padding: '8px 12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
            userSelect: 'none',
            flexShrink: 0,
          }}
          onClick={() => setLogsOpen(!logsOpen)}
        >
          <span style={{ fontFamily: 'var(--font-heading)', fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Server Logs ({serverLogs.length})
          </span>
          <span style={{ fontSize: 12, color: 'var(--text-muted)', transform: logsOpen ? 'rotate(90deg)' : 'none', transition: 'transform var(--transition)' }}>
            ›
          </span>
        </div>

        {logsOpen && (
          <div style={{
            maxHeight: 300,
            overflow: 'auto',
            padding: '8px 12px',
            background: 'var(--bg)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            lineHeight: 1.6,
          }}
            ref={logEndRef}
          >
            {serverLogs.length === 0 ? (
              <span style={{ color: 'var(--text-muted)' }}>No logs yet.</span>
            ) : (
              serverLogs.map((line, i) => {
                let color = 'var(--text-muted)';
                if (line.includes('[stderr]')) color = 'var(--error)';
                else if (line.includes('ERROR') || line.includes('Error')) color = 'var(--error)';
                else if (line.includes('WARNING') || line.includes('Warning')) color = 'var(--warning)';
                else if (line.includes('Uvicorn running') || line.includes('startup complete')) color = 'var(--success)';

                return (
                  <div key={i} style={{ color, wordBreak: 'break-all' }}>
                    {line}
                  </div>
                );
              })
            )}
            <div ref={logEndRef} />
          </div>
        )}
      </div>

      {/* Loading bar animation keyframe */}
      <style>{`
        @keyframes loading-bar {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(400%); }
        }
      `}</style>
    </div>
  );
}

function GpuStatBar({ gpu }: { gpu: GpuStats }) {
  const usedPercent = gpu.totalMemoryMb > 0 ? (gpu.usedMemoryMb / gpu.totalMemoryMb) * 100 : 0;
  const usedGb = (gpu.usedMemoryMb / 1024).toFixed(1);
  const totalGb = (gpu.totalMemoryMb / 1024).toFixed(1);

  let barColor = 'var(--success)';
  if (usedPercent > 80) barColor = 'var(--error)';
  else if (usedPercent > 60) barColor = 'var(--warning)';

  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>
        <span>GPU {gpu.index}: {gpu.name}</span>
        <span>{gpu.utilizationPercent}%</span>
      </div>
      <div style={{ height: 4, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${usedPercent}%`,
          background: barColor,
          borderRadius: 2,
          transition: 'width 0.5s ease',
        }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
        <span>{usedGb} / {totalGb} GB VRAM</span>
        {gpu.temperatureC !== null && <span>{gpu.temperatureC}°C</span>}
      </div>
    </div>
  );
}
