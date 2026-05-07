import { useState, useCallback } from 'react';
import { useAppStore } from '../../store/index';
import { updateSettings } from '../../api/index';
import { useTheme } from '../../hooks/useTheme';

interface SettingsModalProps {
  onClose: () => void;
}

export default function SettingsModal({ onClose }: SettingsModalProps) {
  const { settings, setSettings } = useAppStore();
  const { theme, setTheme } = useTheme();
  const [localSettings, setLocalSettings] = useState(settings ? { ...settings } : null);
  const [newDir, setNewDir] = useState('');
  const [saving, setSaving] = useState(false);

  const handleSave = useCallback(async () => {
    if (!localSettings) return;
    setSaving(true);
    try {
      const updated = await updateSettings(localSettings);
      setSettings(updated);
      if (updated.theme) setTheme(updated.theme);
    } catch { /* ignore */ }
    setSaving(false);
  }, [localSettings, setSettings, setTheme]);

  if (!localSettings) return null;

  const update = (key: string, value: unknown) => {
    setLocalSettings((prev) => prev ? { ...prev, [key]: value } : prev);
  };

  const addDir = () => {
    const dir = newDir.trim();
    if (!dir || localSettings.modelScanDirs.includes(dir)) return;
    setLocalSettings((prev) => prev ? {
      ...prev,
      modelScanDirs: [...prev.modelScanDirs, dir],
    } : prev);
    setNewDir('');
  };

  const removeDir = (dir: string) => {
    setLocalSettings((prev) => prev ? {
      ...prev,
      modelScanDirs: prev.modelScanDirs.filter((d) => d !== dir),
    } : prev);
  };

  const handleExport = async () => {
    try {
      const { getSessions } = await import('../../api/index');
      const res = await getSessions();
      const allSessions = [];
      for (const s of res.sessions) {
        const { getSession } = await import('../../api/index');
        const full = await getSession(s.id);
        allSessions.push(full);
      }
      const blob = new Blob([JSON.stringify(allSessions, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `vllm-studio-export-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch { /* ignore */ }
  };

  const handleImport = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        if (Array.isArray(data)) {
          for (const session of data) {
            await import('../../api/index').then(({ createSession }) =>
              createSession({ title: session.title, systemPrompt: session.systemPrompt })
            );
          }
        }
      } catch { /* ignore */ }
    };
    input.click();
  };

  const handleClearHistory = () => {
    if (confirm('Are you sure you want to delete all chat history? This cannot be undone.')) {
      // Clear through the store
      useAppStore.getState().setSessions([]);
      useAppStore.getState().setCurrentSessionId(null);
      useAppStore.getState().setMessages([]);
    }
  };

  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" style={{ width: 680 }}>
        <div className="modal-header">
          <h2>Settings</h2>
          <button className="btn-icon" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <div className="modal-body">
          {/* Model Scan Directories */}
          <div className="modal-section">
            <h3>Model Scan Directories</h3>
            <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
              <input
                className="input"
                value={newDir}
                onChange={(e) => setNewDir(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addDir()}
                placeholder="/path/to/models..."
              />
              <button className="btn btn-primary btn-sm" onClick={addDir}>Add</button>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {localSettings.modelScanDirs.map((dir) => (
                <div key={dir} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', background: 'var(--surface-hover)', borderRadius: 'var(--radius-sm)', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                  <span style={{ flex: 1, wordBreak: 'break-all', color: 'var(--text-secondary)' }}>{dir}</span>
                  <button className="btn-icon" onClick={() => removeDir(dir)} style={{ fontSize: 14, color: 'var(--text-muted)' }}>
                    ×
                  </button>
                </div>
              ))}
              {localSettings.modelScanDirs.length === 0 && (
                <p style={{ fontSize: 12, color: 'var(--text-muted)', padding: '8px 0' }}>No directories added. Add paths to scan for HuggingFace models.</p>
              )}
            </div>
          </div>

          {/* Default Inference Params */}
          <div className="modal-section">
            <h3>Default System Prompt</h3>
            <textarea
              className="textarea"
              value={localSettings.defaultSystemPrompt}
              onChange={(e) => update('defaultSystemPrompt', e.target.value)}
              placeholder="Enter a default system prompt for new chats..."
              rows={3}
            />
          </div>

          <div className="modal-section">
            <h3>Default Inference Parameters</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 16px' }}>
              <SliderParam label="Temperature" value={localSettings.defaultTemperature} min={0} max={2} step={0.05} onChange={(v) => update('defaultTemperature', v)} />
              <SliderParam label="Top-P" value={localSettings.defaultTopP} min={0} max={1} step={0.05} onChange={(v) => update('defaultTopP', v)} />
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 12, color: 'var(--text-secondary)', minWidth: 100 }}>Max Tokens</span>
                <input
                  className="input"
                  type="number"
                  value={localSettings.defaultMaxTokens}
                  onChange={(e) => update('defaultMaxTokens', parseInt(e.target.value) || 4096)}
                  style={{ width: 80 }}
                />
              </div>
            </div>
          </div>

          {/* UI Settings */}
          <div className="modal-section">
            <h3>Interface</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Theme</span>
                <select
                  className="input"
                  value={localSettings.theme}
                  onChange={(e) => update('theme', e.target.value)}
                  style={{ width: 140 }}
                >
                  <option value="dark">Dark</option>
                  <option value="light">Light</option>
                  <option value="system">System</option>
                </select>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Backend Port</span>
                <input
                  className="input"
                  type="number"
                  value={localSettings.backendPort}
                  onChange={(e) => update('backendPort', parseInt(e.target.value) || 3333)}
                  style={{ width: 100 }}
                />
              </div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>vLLM Default Port</span>
                <input
                  className="input"
                  type="number"
                  value={localSettings.vllmPort}
                  onChange={(e) => update('vllmPort', parseInt(e.target.value) || 8000)}
                  style={{ width: 100 }}
                />
              </div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Auto-start vLLM on launch</span>
                <label className="toggle">
                  <input type="checkbox" checked={localSettings.autoStartVllm} onChange={(e) => update('autoStartVllm', e.target.checked)} />
                  <span className="toggle-slider" />
                </label>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Send on Enter (Shift+Enter for newline)</span>
                <label className="toggle">
                  <input type="checkbox" checked={localSettings.sendOnEnter} onChange={(e) => update('sendOnEnter', e.target.checked)} />
                  <span className="toggle-slider" />
                </label>
              </div>
            </div>
          </div>

          {/* Keyboard Shortcuts */}
          <div className="modal-section">
            <h3>Keyboard Shortcuts</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '6px 24px', fontSize: 12 }}>
              {[
                ['New Chat', 'Ctrl+N'],
                ['Focus Search', 'Ctrl+/'],
                ['Toggle Config Panel', 'Ctrl+\''],
                ['Open Settings', 'Ctrl+,'],
                ['Send Message', 'Ctrl+Enter'],
                ['Close Modal', 'Escape'],
              ].map(([label, key]) => (
                <>
                  <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                  <code style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>{key}</code>
                </>
              ))}
            </div>
          </div>

          {/* Data Management */}
          <div className="modal-section">
            <h3>Data Management</h3>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-secondary btn-sm" onClick={handleExport}>Export All Chats (JSON)</button>
              <button className="btn btn-secondary btn-sm" onClick={handleImport}>Import Chats</button>
              <button className="btn btn-danger btn-sm" onClick={handleClearHistory}>Clear All History</button>
            </div>
          </div>

          {/* About */}
          <div className="modal-section">
            <h3>About</h3>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6 }}>
              <p><strong style={{ color: 'var(--text-secondary)' }}>VLLM Studio v1.0.0</strong></p>
              <p>Production-grade local AI chat and inference management for vLLM on Linux with NVIDIA CUDA GPUs.</p>
              <p style={{ marginTop: 4 }}>
                Built with React, TypeScript, Express, SQLite, and vLLM.
              </p>
            </div>
          </div>

          {/* Save */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
            <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
              {saving ? 'Saving...' : 'Save Settings'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function SliderParam({ label, value, min, max, step, onChange }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ fontSize: 12, color: 'var(--text-secondary)', minWidth: 90 }}>{label}</span>
      <div className="slider-group" style={{ flex: 1 }}>
        <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} />
        <span className="slider-value">{value.toFixed(2)}</span>
      </div>
    </div>
  );
}
