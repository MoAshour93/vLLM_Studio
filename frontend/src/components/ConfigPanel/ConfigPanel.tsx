import { useState, useCallback, useRef, useEffect } from 'react';
import { useAppStore } from '../../store/index';
import ModelSelector from '../ModelSelector/ModelSelector';
import ServerPanel from '../ServerPanel/ServerPanel';
import HuggingFaceBrowser from '../HuggingFaceBrowser/HuggingFaceBrowser';

type Tab = 'inference' | 'server' | 'download';

const TABS: { id: Tab; label: string }[] = [
  { id: 'inference', label: 'Inference' },
  { id: 'server', label: 'Server' },
  { id: 'download', label: 'Download' },
];

const MIN_WIDTH = 280;
const MAX_WIDTH = 600;
const DEFAULT_WIDTH = 320;

function getStoredWidth(): number {
  try {
    const stored = localStorage.getItem('vllm-studio-config-width');
    if (stored) {
      const w = parseInt(stored, 10);
      if (w >= MIN_WIDTH && w <= MAX_WIDTH) return w;
    }
  } catch { /* ignore */ }
  return DEFAULT_WIDTH;
}

function storeWidth(w: number): void {
  try {
    localStorage.setItem('vllm-studio-config-width', String(w));
  } catch { /* ignore */ }
}

export default function ConfigPanel() {
  const [activeTab, setActiveTab] = useState<Tab>('inference');
  const [panelWidth, setPanelWidth] = useState(getStoredWidth);
  const { setConfigPanelOpen } = useAppStore();
  const isDragging = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(0);
  const panelRef = useRef<HTMLDivElement>(null);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    startX.current = e.clientX;
    startWidth.current = panelWidth;
    document.body.style.cursor = 'ew-resize';
    document.body.style.userSelect = 'none';
  }, [panelWidth]);

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      const dx = startX.current - e.clientX;
      const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startWidth.current + dx));
      setPanelWidth(newWidth);
    };

    const onMouseUp = () => {
      if (isDragging.current) {
        isDragging.current = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        const currentWidth = panelRef.current ? panelRef.current.offsetWidth : panelWidth;
        storeWidth(currentWidth);
      }
    };

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    return () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
  }, [panelWidth]);

  return (
    <div
      ref={panelRef}
      style={{
        width: panelWidth,
        flexShrink: 0,
        borderLeft: '1px solid var(--border)',
        background: 'var(--surface)',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      {/* Resize handle */}
      <div
        onMouseDown={onMouseDown}
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          bottom: 0,
          width: 5,
          cursor: 'ew-resize',
          zIndex: 20,
          background: 'transparent',
          transition: 'background 0.15s',
        }}
        onMouseEnter={(e) => {
          (e.target as HTMLElement).style.background = 'var(--accent)';
          (e.target as HTMLElement).style.opacity = '0.4';
        }}
        onMouseLeave={(e) => {
          (e.target as HTMLElement).style.background = 'transparent';
          (e.target as HTMLElement).style.opacity = '1';
        }}
      />

      {/* Header */}
      <div style={{
        padding: '8px 12px 8px 16px',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <span style={{
          fontFamily: 'var(--font-heading)',
          fontSize: 12,
          fontWeight: 600,
          color: 'var(--text-secondary)',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        }}>
          Configuration
        </span>
        <button
          className="btn-icon"
          onClick={() => setConfigPanelOpen(false)}
          title="Close panel"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* Tab bar */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
      }}>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              flex: 1,
              padding: '8px 0',
              fontSize: 11,
              fontWeight: 600,
              fontFamily: 'var(--font-body)',
              color: activeTab === tab.id ? 'var(--accent)' : 'var(--text-muted)',
              background: 'transparent',
              border: 'none',
              borderBottom: activeTab === tab.id ? '2px solid var(--accent)' : '2px solid transparent',
              cursor: 'pointer',
              transition: 'all var(--transition)',
              letterSpacing: '0.03em',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        overflowX: 'hidden',
        minHeight: 0,
      }}>
        {activeTab === 'inference' && <ModelSelector />}
        {activeTab === 'server' && <ServerPanel />}
        {activeTab === 'download' && <HuggingFaceBrowser />}
      </div>
    </div>
  );
}
