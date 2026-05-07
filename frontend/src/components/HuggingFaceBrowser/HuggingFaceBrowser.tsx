import { useState, useCallback, useEffect, useRef } from 'react';
import { useAppStore } from '../../store/index';
import { uuid } from '../../utils/uuid';

interface HfModel {
  id: string;
  author: string;
  modelId: string;
  sha: string;
  lastModified: string;
  tags: string[];
  pipelineTag: string;
  downloads: number;
  likes: number;
  libraryName: string;
  createdAt: string;
  sizeBytes: number;
  quantizations: string[];
  hasConfig: boolean;
  hasTokenizer: boolean;
  modelFilesCount: number;
  isGguf: boolean;
  ggufQuants: Array<{ name: string; size: number }>;
  support: { level: 'supported' | 'experimental' | 'unsupported'; family: string | null; ggufSupported: boolean; reason?: string };
}

interface DownloadState {
  active: boolean;
  modelId: string;
  progress: number;
  stage: string;
  targetDir: string;
  error: string | null;
}

export default function HuggingFaceBrowser() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<HfModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState('downloads');
  const [vllmOnly, setVllmOnly] = useState(true);
  const [selectedModel, setSelectedModel] = useState<HfModel | null>(null);
  const [downloadState, setDownloadState] = useState<DownloadState | null>(null);
  const [downloading, setDownloading] = useState(false);
  const searchTimeout = useRef<ReturnType<typeof setTimeout>>();

  const { models, serverStatus } = useAppStore();

  const search = useCallback(async (q: string, sort: string, vllmOnlyArg: boolean) => {
    if (!q.trim()) {
      setResults([]);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`/api/huggingface/search?query=${encodeURIComponent(q)}&limit=30&sort=${sort}&vllmOnly=${vllmOnlyArg ? 'true' : 'false'}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      const results: HfModel[] = data.results || [];

      // For GGUF models, fetch detailed model info to get accurate quant sizes
      const enriched = await Promise.all(
        results.map(async (model) => {
          if (!model.isGguf) return model;

          try {
            const detailRes = await fetch(`/api/huggingface/model/${encodeURIComponent(model.modelId)}`);
            const detail = await detailRes.json();
            if (detail.ggufQuants?.length > 0) {
              return { ...model, ggufQuants: detail.ggufQuants };
            }
          } catch { /* keep original quants */ }
          return model;
        })
      );

      setResults(enriched);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleQueryChange = useCallback((value: string) => {
    setQuery(value);
    if (searchTimeout.current) clearTimeout(searchTimeout.current);
    searchTimeout.current = setTimeout(() => {
      search(value, sortBy, vllmOnly);
    }, 400);
  }, [search, sortBy, vllmOnly]);

  const handleSortChange = useCallback((sort: string) => {
    setSortBy(sort);
    if (query.trim()) search(query, sort, vllmOnly);
  }, [query, search, vllmOnly]);

  const handleToggleVllmOnly = useCallback(() => {
    setVllmOnly((v) => {
      const next = !v;
      if (query.trim()) search(query, sortBy, next);
      return next;
    });
  }, [query, search, sortBy]);

  const handleDownload = useCallback(async (model: HfModel, quantization?: string) => {
    setDownloading(true);
    setSelectedModel(model);
    setDownloadState({
      active: true,
      modelId: quantization ? `${model.modelId} [${quantization}]` : model.modelId,
      progress: 0,
      stage: 'starting',
      targetDir: '',
      error: null,
    });

    try {
      const body: Record<string, string> = { modelId: model.modelId };
      if (quantization) body.quantization = quantization;

      const res = await fetch('/api/huggingface/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Download failed');
    } catch (err) {
      setDownloadState((prev) => prev ? {
        ...prev,
        active: false,
        stage: 'error',
        error: err instanceof Error ? err.message : 'Download failed',
      } : null);
    } finally {
      setDownloading(false);
    }
  }, []);

  const handleCancelDownload = useCallback(async () => {
    try {
      await fetch('/api/huggingface/download/cancel', { method: 'POST' });
    } catch { /* ignore */ }
  }, []);

  // Listen for download progress via a quick poll since WS is for vllm
  useEffect(() => {
    if (!downloading && !downloadState?.active) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch('/api/huggingface/download/status');
        const state = await res.json() as DownloadState;
        setDownloadState(state);
        if (!state.active) clearInterval(interval);
      } catch { /* ignore */ }
    }, 1000);

    return () => clearInterval(interval);
  }, [downloading]);

  const formatSize = (bytes: number) => {
    if (!bytes || bytes < 0) return '0 B';
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
    if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
    return `${bytes} B`;
  };

  const estimateFp16Size = (modelId: string): number | null => {
    // Parse parameter count from model name (e.g. "Qwen3.5-2B", "Llama-3-70B")
    const m = modelId.match(/(\d+\.?\d*)\s*[Bb]\b/);
    if (!m) return null;
    const b = parseFloat(m[1]);
    if (isNaN(b) || b <= 0) return null;
    return b * 1e9 * 2; // params × 2 bytes per FP16 weight
  };

  const formatEta = (sec: number) => {
    if (!sec || sec <= 0) return '';
    if (sec < 60) return `${sec}s left`;
    if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s left`;
    return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m left`;
  };

  const formatNumber = (n: number) => {
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return String(n);
  };

  const formatDate = (ts: string) => {
    const d = new Date(ts);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    const days = Math.floor(diff / 86400000);
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 30) return `${days}d ago`;
    if (days < 365) return `${Math.floor(days / 30)}mo ago`;
    return `${Math.floor(days / 365)}y ago`;
  };

  return (
    <div style={{ borderBottom: '1px solid var(--border)', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '12px' }}>
        <div style={{
          fontFamily: 'var(--font-heading)',
          fontSize: 11,
          fontWeight: 600,
          color: 'var(--text-muted)',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          marginBottom: 8,
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <path d="M8 14s1.5 2 4 2 4-2 4-2" />
            <line x1="9" y1="9" x2="9.01" y2="9" />
            <line x1="15" y1="9" x2="15.01" y2="9" />
          </svg>
          HuggingFace Models
        </div>

        <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
          <input
            className="input"
            type="text"
            placeholder="Search models on HuggingFace..."
            value={query}
            onChange={(e) => handleQueryChange(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && query.trim() && search(query, sortBy, vllmOnly)}
            style={{ flex: 1, fontSize: 11, padding: '6px 10px' }}
          />
          <button
            className="btn btn-primary btn-sm"
            onClick={() => query.trim() && search(query, sortBy, vllmOnly)}
            disabled={loading || !query.trim()}
          >
            {loading ? (
              <div className="spinner" style={{ width: 12, height: 12, borderWidth: 1.5 }} />
            ) : 'Search'}
          </button>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6, gap: 6 }}>
          <button
            className="btn btn-sm"
            style={{
              fontSize: 10,
              padding: '3px 8px',
              background: vllmOnly ? 'var(--success-muted)' : 'var(--surface-hover)',
              color: vllmOnly ? 'var(--success)' : 'var(--text-muted)',
              border: `1px solid ${vllmOnly ? 'var(--success)' : 'transparent'}`,
            }}
            title="Only show models from architectures vLLM can actually load"
            onClick={handleToggleVllmOnly}
          >
            {vllmOnly ? '✓ vLLM-supported only' : 'All models'}
          </button>
        </div>

        <div style={{ display: 'flex', gap: 4, marginBottom: 6 }}>
          {['downloads', 'likes', 'lastModified'].map((s) => (
            <button
              key={s}
              className="btn btn-sm"
              style={{
                fontSize: 10,
                padding: '2px 8px',
                background: sortBy === s ? 'var(--accent-muted)' : 'var(--surface-hover)',
                color: sortBy === s ? 'var(--accent)' : 'var(--text-muted)',
                border: sortBy === s ? '1px solid var(--accent)' : '1px solid transparent',
              }}
              onClick={() => handleSortChange(s)}
            >
              {s === 'lastModified' ? 'Recent' : s.charAt(0).toUpperCase() + s.slice(1)}
            </button>
          ))}
        </div>

        {/* Download Progress Bar */}
        {downloadState && (downloadState.active || downloadState.stage === 'complete' || downloadState.stage === 'error') && (
          <div style={{
            padding: '8px 10px',
            background: 'var(--surface-active)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            marginBottom: 8,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                {downloadState.modelId}
              </span>
              <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginLeft: 8 }}>
                {downloadState.progress}%
              </span>
            </div>
            <div style={{ height: 4, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${downloadState.progress}%`,
                background: downloadState.stage === 'error' ? 'var(--error)'
                  : downloadState.stage === 'complete' ? 'var(--success)'
                  : 'var(--accent)',
                borderRadius: 2,
                transition: 'width 0.3s ease',
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3, fontSize: 10, color: 'var(--text-muted)' }}>
              <span style={{
                color: downloadState.stage === 'error' ? 'var(--error)'
                  : downloadState.stage === 'complete' ? 'var(--success)'
                  : 'var(--accent)',
              }}>
                {downloadState.stage === 'starting' ? 'Starting...'
                  : downloadState.stage === 'downloading' ? 'Downloading...'
                  : downloadState.stage === 'complete' ? 'Complete!'
                  : downloadState.stage === 'error' ? downloadState.error || 'Error'
                  : downloadState.stage === 'cancelled' ? 'Cancelled'
                  : downloadState.stage}
              </span>
              {downloadState.stage === 'downloading' && (downloadState as any).speedBps > 0 && (
                <span style={{ fontFamily: 'var(--font-mono)' }}>
                  {formatSize((downloadState as any).speedBps)}/s
                  {(downloadState as any).etaSec > 0 && ` · ${formatEta((downloadState as any).etaSec)}`}
                </span>
              )}
              {downloadState.stage === 'complete' && downloadState.targetDir && (
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9 }}>
                  {downloadState.targetDir}
                </span>
              )}
              {downloadState.active && (
                <button
                  className="btn btn-sm"
                  style={{ fontSize: 10, padding: '1px 6px', color: 'var(--error)' }}
                  onClick={handleCancelDownload}
                >
                  Cancel
                </button>
              )}
            </div>
          </div>
        )}

        {error && (
          <div style={{ fontSize: 11, color: 'var(--error)', padding: '4px 0', marginBottom: 4 }}>
            {error}
          </div>
        )}
      </div>

      {/* Results */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '0 12px 8px' }}>
        {results.length === 0 && !loading && query && (
          <div style={{ textAlign: 'center', padding: '20px 0', color: 'var(--text-muted)', fontSize: 12 }}>
            {loading ? 'Searching...' : 'No models found.'}
          </div>
        )}

        {results.map((model) => {
          const isDownloadingThis = downloadState?.active && downloadState.modelId.includes(model.modelId);

          return (
            <div
              key={model.modelId}
              style={{
                padding: '8px 10px',
                marginBottom: 6,
                background: 'var(--surface-hover)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)',
                transition: 'all var(--transition)',
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.borderColor = 'var(--accent)';
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.borderColor = 'var(--border)';
              }}
            >
              {/* Model header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: 'var(--text-primary)',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}>
                    {model.modelId}
                  </div>

                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
                    {model.support.level === 'supported' && (
                      <span className="badge" style={{ fontSize: 9, background: 'var(--success-muted)', color: 'var(--success)' }}>
                        ✓ {model.support.family ?? 'vLLM'}
                      </span>
                    )}
                    {model.support.level === 'experimental' && (
                      <span
                        className="badge"
                        style={{ fontSize: 9, background: 'var(--warning-muted)', color: 'var(--warning)' }}
                        title={model.support.reason ?? 'GGUF support experimental for this family'}
                      >
                        ⚠ {model.support.family ?? 'Experimental'}
                      </span>
                    )}
                    {model.support.level === 'unsupported' && (
                      <span
                        className="badge"
                        style={{ fontSize: 9, background: 'var(--error-muted)', color: 'var(--error)' }}
                        title={model.support.reason ?? 'Not supported by vLLM'}
                      >
                        ✗ Unsupported
                      </span>
                    )}
                    {model.isGguf ? (
                      <span className="badge badge-warning" style={{ fontSize: 9, padding: '1px 5px' }}>
                        GGUF ({model.ggufQuants.length} quants)
                      </span>
                    ) : model.quantizations.length > 0 ? model.quantizations.map((q) => (
                      <span key={q} className="badge badge-accent" style={{ fontSize: 9, padding: '1px 5px' }}>
                        {q}
                      </span>
                    )) : (
                      <span className="badge" style={{ fontSize: 9, background: 'var(--surface-active)', color: 'var(--text-muted)' }}>
                        {model.pipelineTag}
                      </span>
                    )}
                    {model.hasTokenizer && (
                      <span className="badge" style={{ fontSize: 9, background: 'var(--surface-active)', color: 'var(--text-muted)' }}>
                        Tokenizer
                      </span>
                    )}
                  </div>

                  <div style={{ display: 'flex', gap: 10, marginTop: 4, fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    <span title="Downloads">⬇ {formatNumber(model.downloads)}</span>
                    <span title="Likes">♥ {formatNumber(model.likes)}</span>
                    {model.sizeBytes > 0 && <span title="Repo size">{formatSize(model.sizeBytes)}</span>}
                    {(() => {
                      const fp16 = estimateFp16Size(model.modelId);
                      if (fp16) return <span title="Estimated 16-bit model size (weights only)">~{formatSize(fp16)} fp16</span>;
                      return null;
                    })()}
                    <span>Modified {formatDate(model.lastModified)}</span>
                  </div>
                </div>

                {/* Full download button for non-GGUF models */}
                {!model.isGguf && (
                  <button
                    className="btn btn-sm"
                    style={{
                      fontSize: 10,
                      padding: '3px 10px',
                      background: isDownloadingThis ? 'var(--warning-muted)' : 'var(--accent-muted)',
                      color: isDownloadingThis ? 'var(--warning)' : 'var(--accent)',
                      border: '1px solid',
                      borderColor: isDownloadingThis ? 'var(--warning)' : 'var(--accent)',
                      flexShrink: 0,
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDownload(model);
                    }}
                    disabled={downloadState?.active}
                  >
                    {isDownloadingThis ? 'Downloading...' : 'Download'}
                  </button>
                )}
              </div>

              {/* GGUF quant list */}
              {model.isGguf && model.ggufQuants.length > 0 && (
                <div style={{
                  marginTop: 8,
                  paddingTop: 8,
                  borderTop: '1px solid var(--border)',
                }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 6, fontWeight: 600 }}>
                    Available Quantizations — click to download:
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                    {model.ggufQuants.map((quant) => {
                      const isThisQuant = downloadState?.active &&
                        downloadState.modelId.includes(model.modelId) &&
                        downloadState.modelId.includes(`[${quant.name}]`);

                      return (
                        <div
                          key={quant.name}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '4px 8px',
                            borderRadius: 'var(--radius-sm)',
                            background: 'var(--bg)',
                            gap: 8,
                          }}
                        >
                          <span style={{
                            fontSize: 11,
                            fontWeight: 600,
                            fontFamily: 'var(--font-mono)',
                            color: 'var(--accent)',
                            minWidth: 70,
                          }}>
                            {quant.name}
                          </span>
                          <span style={{
                            fontSize: 10,
                            color: 'var(--text-muted)',
                            fontFamily: 'var(--font-mono)',
                            flex: 1,
                          }}>
                            {quant.size > 0 ? formatSize(quant.size) : '—'}
                          </span>
                          <button
                            className="btn btn-sm"
                            style={{
                              fontSize: 10,
                              padding: '2px 10px',
                              background: isThisQuant ? 'var(--warning-muted)' : 'var(--accent-muted)',
                              color: isThisQuant ? 'var(--warning)' : 'var(--accent)',
                              border: '1px solid',
                              borderColor: isThisQuant ? 'var(--warning)' : 'var(--accent)',
                              flexShrink: 0,
                            }}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDownload(model, quant.name);
                            }}
                            disabled={downloadState?.active}
                          >
                            {isThisQuant ? '...' : 'Download'}
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
