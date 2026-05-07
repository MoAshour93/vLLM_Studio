import { useEffect, useCallback } from 'react';
import { useAppStore } from './store/index';
import type { SystemResources } from './store/types';
import { useTheme } from './hooks/useTheme';
import { useVllmServer } from './hooks/useVllmServer';
import {
  getModels, getSessions, getFolders, getSettings,
  getSystemResources,
} from './api/index';
import Sidebar from './components/Sidebar/Sidebar';
import ChatWindow from './components/ChatWindow/ChatWindow';
import ConfigPanel from './components/ConfigPanel/ConfigPanel';
import SettingsModal from './components/SettingsModal/SettingsModal';
import ReleaseNotesModal from './components/ReleaseNotesModal/ReleaseNotesModal';
import AboutModal from './components/AboutModal/AboutModal';
import ThemeToggle from './components/ThemeToggle/ThemeToggle';

export default function App() {
  const theme = useTheme();
  const {
    setSessions, setModels, setFolders, setSettings,
    configPanelOpen, setConfigPanelOpen,
    settingsOpen, setSettingsOpen,
    releaseNotesOpen, setReleaseNotesOpen,
    aboutOpen, setAboutOpen,
    serverStatus, settings,
    currentSessionId, setCurrentSessionId, setMessages,
    sysResources, setSysResources,
    serverInfo,
  } = useAppStore();
  const vllmServer = useVllmServer();

  useEffect(() => {
    loadInitialData();
    vllmServer.fetchStatus();
  }, []);

  useEffect(() => {
    const pollSys = async () => {
      try {
        const res = await getSystemResources();
        useAppStore.getState().setSysResources(res);
      } catch { /* ignore */ }
    };
    pollSys();
    const iv = setInterval(pollSys, 8000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;

      if (mod && e.key === 'n') {
        e.preventDefault();
        handleNewChat();
      } else if (mod && e.key === '/') {
        e.preventDefault();
        document.querySelector<HTMLInputElement>('[data-search]')?.focus();
      } else if (mod && e.key === ',') {
        e.preventDefault();
        setSettingsOpen(true);
      } else if (mod && e.shiftKey && e.key === 'S') {
        e.preventDefault();
      } else if (e.key === 'Escape') {
        setSettingsOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const loadInitialData = async () => {
    try {
      const [modelsRes, sessionsRes, foldersRes, settingsRes] = await Promise.all([
        getModels().catch(() => ({ models: [], vllmVersion: '' })),
        getSessions().catch(() => ({ sessions: [] })),
        getFolders().catch(() => ({ folders: [] })),
        getSettings().catch(() => null),
      ]);

      setModels(modelsRes.models);
      if (modelsRes.vllmVersion) {
        useAppStore.getState().setVllmVersion(modelsRes.vllmVersion);
      }
      setSessions(sessionsRes.sessions);
      setFolders(foldersRes.folders);
      if (settingsRes) {
        setSettings(settingsRes);
        if (settingsRes.theme) {
          theme.setTheme(settingsRes.theme);
        }
      }

      if (sessionsRes.sessions.length > 0 && !currentSessionId) {
        const firstSession = sessionsRes.sessions[0];
        setCurrentSessionId(firstSession.id);
      }
    } catch {
      // Ignore initial load errors — UI will show empty states
    }
  };

  const handleNewChat = useCallback(async () => {
    try {
      const { createSession } = await import('./api/index');
      const state = useAppStore.getState();
      const model = state.models.find((m) => m.id === state.selectedModelId);
      const title = model ? `${model.name.split('/').pop()?.replace(/-GGUF$/i, '')} Chat` : 'New Chat';
      const session = await createSession({ title });
      state.setSessions([session, ...state.sessions]);
      state.setCurrentSessionId(session.id);
      state.setMessages([]);
    } catch {
      // ignore
    }
  }, []);

  const handleSelectSession = useCallback(async (id: string) => {
    setCurrentSessionId(id);
    try {
      const { getSession } = await import('./api/index');
      const session = await getSession(id);
      if (session.messages) {
        setMessages(session.messages);
      }
    } catch {
      setMessages([]);
    }
  }, [setCurrentSessionId, setMessages]);

  return (
    <div
      style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--bg)', color: 'var(--text-primary)' }}
      data-theme={theme.theme}
    >
      <div className="topbar">
        <div className="topbar-logo">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
          </svg>
          <span>VLLM Studio</span>
        </div>
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          {serverInfo?.isBehind && serverInfo?.upgradeCommand && (
            <div
              title={`Click to copy: ${serverInfo.upgradeCommand}`}
              onClick={() => navigator.clipboard.writeText(serverInfo.upgradeCommand!).catch(() => {})}
              style={{
                display: 'flex', alignItems: 'center', gap: 4,
                padding: '3px 8px', borderRadius: 'var(--radius-sm)',
                background: 'var(--warning-muted)', color: 'var(--warning)',
                border: '1px solid var(--warning)',
                fontSize: 10, fontFamily: 'var(--font-mono)',
                cursor: 'pointer', userSelect: 'none',
              }}
            >
              vLLM {serverInfo.vllmVersion} → {serverInfo.latestStable} (click to copy upgrade)
            </div>
          )}
          <ServerStatusBadge status={serverStatus} />
          <ThemeToggle />
          <button className="btn btn-ghost btn-sm" onClick={() => setReleaseNotesOpen(true)} title="Release Notes">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
              <line x1="8" y1="7" x2="16" y2="7" />
              <line x1="8" y1="11" x2="14" y2="11" />
            </svg>
          </button>
          <button className="btn btn-ghost btn-sm" onClick={() => setAboutOpen(true)} title="About">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
          </button>
          <button className="btn btn-ghost btn-sm" onClick={() => setSettingsOpen(true)} title="Settings">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33-1.82 10 10 0 0 0-1.46-2.54 1.65 1.65 0 0 0-1.82-.33" />
              <path d="M4.6 9a1.65 1.65 0 0 0-.33 1.82 10 10 0 0 0 1.46 2.54 1.65 1.65 0 0 0 1.82.33" />
            </svg>
          </button>
        </div>
      </div>

      <div className="app-layout">
        <Sidebar
          onNewChat={handleNewChat}
          onSelectSession={handleSelectSession}
        />

        <ChatWindow onNewChat={handleNewChat} />

        {configPanelOpen && <ConfigPanel />}

        {!configPanelOpen && (
          <button
            className="btn-icon"
            onClick={() => setConfigPanelOpen(true)}
            style={{
              position: 'absolute',
              top: 8,
              right: 8,
              zIndex: 2,
              background: 'var(--bg)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              padding: 6,
            }}
            title="Open config panel"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="4" y1="6" x2="20" y2="6" />
              <line x1="8" y1="12" x2="20" y2="12" />
              <line x1="4" y1="18" x2="20" y2="18" />
            </svg>
          </button>
        )}
      </div>

      {settingsOpen && <SettingsModal onClose={() => setSettingsOpen(false)} />}
      {releaseNotesOpen && <ReleaseNotesModal onClose={() => setReleaseNotesOpen(false)} />}
      {aboutOpen && <AboutModal onClose={() => setAboutOpen(false)} />}
      <SysBar resources={sysResources} />
    </div>
  );
}

function ServerStatusBadge({ status }: { status: string }) {
  const label = status.toUpperCase();
  const dotClass = `status-dot ${status}`;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '4px 10px', borderRadius: 'var(--radius-sm)', background: 'var(--surface-active)', border: '1px solid var(--border)' }}>
      <span className={dotClass} />
      <span style={{ fontSize: 11, fontWeight: 600, fontFamily: 'var(--font-mono)', letterSpacing: '0.05em', color: 'var(--text-secondary)' }}>{label}</span>
    </div>
  );
}

function SysBar({ resources }: { resources: SystemResources | null }) {
  if (!resources) return null;

  const gpu = resources.gpus[0];
  const gpuPct = gpu ? gpu.utilizationPercent : 0;
  const ramUsedGb = (resources.ramUsedBytes / (1024 ** 3)).toFixed(1);
  const ramTotalGb = (resources.ramTotalBytes / (1024 ** 3)).toFixed(1);
  const ramPct = Math.round((resources.ramUsedBytes / resources.ramTotalBytes) * 100);

  return (
    <div style={{
      position: 'fixed',
      bottom: 0,
      left: 0,
      padding: '3px 10px',
      background: 'var(--surface)',
      borderTop: '1px solid var(--border)',
      borderRight: '1px solid var(--border)',
      borderTopRightRadius: 'var(--radius-sm)',
      display: 'flex',
      gap: '12px',
      fontSize: 10,
      fontFamily: 'var(--font-mono)',
      color: 'var(--text-muted)',
      zIndex: 5,
    }}>
      <span>CPU: <strong style={{ color: 'var(--text-secondary)' }}>{resources.cpuPercent}%</strong></span>
      <span>RAM: <strong style={{ color: 'var(--text-secondary)' }}>{ramUsedGb}/{ramTotalGb}GB</strong> ({ramPct}%)</span>
      {gpu && (
        <>
          <span>GPU: <strong style={{ color: 'var(--text-secondary)' }}>{gpuPct}%</strong></span>
          <span>VRAM: <strong style={{ color: 'var(--text-secondary)' }}>{(gpu.usedMemoryMb/1024).toFixed(1)}/{gpu.name.endsWith('GB') ? gpu.name : `${(gpu.totalMemoryMb/1024).toFixed(0)}GB`}</strong></span>
        </>
      )}
    </div>
  );
}
