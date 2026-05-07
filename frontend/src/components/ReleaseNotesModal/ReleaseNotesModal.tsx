import { useState, useMemo } from 'react';

interface ReleaseNotesModalProps {
  onClose: () => void;
}

interface ReleaseSection {
  key: string;
  icon: string;
  label: string;
  summary: string;
  items: string[];
}

interface Release {
  version: string;
  date: string;
  type: 'Major' | 'Minor' | 'Patch' | 'Beta' | 'Hotfix';
  codename: string;
  sections: ReleaseSection[];
}

const SECTION_KEYS = ['whatsNew', 'improvements', 'bugFixes', 'security', 'breakingChanges', 'knownIssues', 'deprecated', 'dependencyUpdates'] as const;

const RELEASES: Release[] = [
  {
    version: 'v1.0.0',
    date: '7 May 2026',
    type: 'Major',
    codename: 'PagedAttention',
    sections: [
      {
        key: 'whatsNew',
        icon: '\u{1F680}',
        label: "What's New",
        summary: 'Initial release of VLLM Studio — a purpose-built local AI chat and inference management interface for vLLM on Linux.',
        items: [
          'vLLM subprocess management with one-click start, stop, and restart',
          'OpenAI-compatible API server with copyable base URL',
          'Local model scanner for HuggingFace (safetensors) and GGUF formats',
          'Streaming chat interface with markdown rendering and syntax highlighting',
          'Vision model support with image and PDF attachment uploads',
          'Chat history with folder organization, renaming, pinning, and fuzzy search',
          'Configurable inference parameters per session (temperature, top-p, max tokens, penalty)',
          'Live GPU memory monitoring via nvidia-smi with utilization and temperature',
          'Real-time vLLM process log streaming',
          'HuggingFace Hub browser with quantized GGUF download support',
          'VRAM estimation engine for pre-flight resource planning',
          'GGUF binary metadata parser and architecture patching for community models',
          'Dark and light theme with system preference detection',
          'Apache 2.0 open-source license',
        ],
      },
      {
        key: 'improvements',
        icon: '\u26A1',
        label: 'Improvements',
        summary: 'Performance and usability refinements since pre-release.',
        items: [
          'Virtualized chat message rendering with react-virtuoso for smooth scrolling with large histories',
          'SQLite WAL mode for concurrent read/write performance',
          'Auto-detection of model architecture, quantization, and context length including RoPE scaling',
          'Heuristic base model guessing for GGUF files to auto-configure tokenizer',
          'Intelligent vLLM startup stage detection (spawning → loading weights → compiling → allocating KV cache → ready)',
          'Actionable error messages for CUDA OOM, unsupported architecture, and shape mismatch errors',
        ],
      },
      {
        key: 'bugFixes',
        icon: '\u{1F41B}',
        label: 'Bug Fixes',
        summary: 'Resolved issues identified during development and testing.',
        items: [
          'Fixed vLLM process orphan when backend terminates unexpectedly',
          'Fixed GGUF metadata parsing for files with non-standard metadata ordering',
          'Fixed chat input clearing on session switch while streaming',
          'Fixed scroll position reset when new messages arrive while scrolled up',
          'Fixed settings persistence race condition on rapid save',
        ],
      },
      {
        key: 'security',
        icon: '\u{1F512}',
        label: 'Security',
        summary: 'Security hardening measures in this release.',
        items: [
          'All API inputs validated with Zod schemas on the backend',
          'CORS restricted to localhost connections by default',
          'No telemetry or external data collection of any kind',
        ],
      },
      {
        key: 'breakingChanges',
        icon: '\u{1F4A5}',
        label: 'Breaking Changes',
        summary: 'None — this is the initial public release.',
        items: [],
      },
      {
        key: 'knownIssues',
        icon: '\u26A0\uFE0F',
        label: 'Known Issues',
        summary: 'Acknowledged limitations in this release.',
        items: [
          'Linux + NVIDIA CUDA only — Windows and macOS are not supported (planned for v2+)',
          'Multi-GPU tensor parallelism requires manual configuration',
          'Some GGUF community models require architecture patching before they can be loaded',
          'Large model downloads (>20GB) may timeout on slow connections',
          'Vision model support requires vLLM >= 0.6.0',
        ],
      },
      {
        key: 'deprecated',
        icon: '\u{1F5D1}\uFE0F',
        label: 'Deprecated',
        summary: 'No deprecated features in this initial release.',
        items: [],
      },
      {
        key: 'dependencyUpdates',
        icon: '\u{1F4E6}',
        label: 'Dependency Updates',
        summary: 'Core dependencies pinned in this release.',
        items: [
          'vLLM 0.7.3+ (required)',
          'React 18.3.1 with TypeScript 5.6',
          'Express 4.21 with better-sqlite3 11.6',
          'Zustand 5.0 for state management',
          'react-markdown 9.0 with highlight.js 11.10 for code rendering',
          'Python 3.10+ with huggingface_hub for model downloads',
        ],
      },
    ],
  },
];

const TYPE_COLORS: Record<Release['type'], { bg: string; text: string }> = {
  Major: { bg: 'var(--accent-muted)', text: 'var(--accent)' },
  Minor: { bg: 'var(--success-muted)', text: 'var(--success)' },
  Patch: { bg: 'var(--success-muted)', text: 'var(--success)' },
  Beta: { bg: 'var(--warning-muted)', text: 'var(--warning)' },
  Hotfix: { bg: 'var(--error-muted)', text: 'var(--error)' },
};

const scrollToSection = (key: string) => {
    const el = document.getElementById(`section-${key}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

export default function ReleaseNotesModal({ onClose }: ReleaseNotesModalProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedReleases, setExpandedReleases] = useState<Set<string>>(() => new Set(RELEASES.map((r) => r.version)));

  const toggleRelease = (version: string) => {
    setExpandedReleases((prev) => {
      const next = new Set(prev);
      if (next.has(version)) next.delete(version); else next.add(version);
      return next;
    });
  };

  const filteredReleases = useMemo(() => {
    if (!searchQuery.trim()) return RELEASES;
    const q = searchQuery.toLowerCase();
    return RELEASES.filter((r) => {
      if (r.version.toLowerCase().includes(q)) return true;
      if (r.codename.toLowerCase().includes(q)) return true;
      return r.sections.some((s) => {
        if (s.summary.toLowerCase().includes(q)) return true;
        return s.items.some((item) => item.toLowerCase().includes(q));
      });
    });
  }, [searchQuery]);

  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" style={{ width: 780, maxHeight: '90vh', display: 'flex', flexDirection: 'column' }}>
        <div className="modal-header">
          <h2>Release Notes</h2>
          <button className="btn-icon" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Header */}
        <div style={{ padding: '20px 24px 0', flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
            <span style={{ fontFamily: 'var(--font-heading)', fontSize: 20, fontWeight: 700 }}>VLLM Studio</span>
            <span className="badge badge-accent" style={{ fontSize: 13, padding: '4px 10px' }}>v1.0.0</span>
          </div>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
            Track what's new, what's fixed, and what's coming.{' '}
            <a href="https://github.com/Moashour93/vllm-studio/releases" target="_blank" rel="noopener noreferrer" style={{ fontWeight: 500 }}>
              View on GitHub →
            </a>
          </p>

          {/* Search & Section Toggles */}
          <div style={{ marginBottom: 12 }}>
            <input
              className="input"
              placeholder="Search release notes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{ maxWidth: 280, fontSize: 12, marginBottom: 10 }}
            />
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={{
              fontSize: 11, fontWeight: 600, fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)', textTransform: 'uppercase',
              letterSpacing: '0.05em', marginBottom: 6,
            }}>
              Table of Contents
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {SECTION_KEYS.map((key) => {
                const section = RELEASES[0].sections.find((s) => s.key === key);
                return (
                  <button
                    key={key}
                    onClick={() => scrollToSection(key)}
                    style={{
                      display: 'inline-flex', alignItems: 'center', gap: 5,
                      padding: '4px 10px',
                      borderRadius: 100,
                      fontSize: 11, fontWeight: 500,
                      border: '1px solid var(--border)',
                      background: 'transparent',
                      color: 'var(--text-secondary)',
                      cursor: 'pointer',
                      transition: 'all var(--transition)',
                    }}
                    onMouseEnter={(e) => {
                      (e.target as HTMLElement).style.background = 'var(--accent-muted)';
                      (e.target as HTMLElement).style.borderColor = 'var(--accent)';
                      (e.target as HTMLElement).style.color = 'var(--accent)';
                    }}
                    onMouseLeave={(e) => {
                      (e.target as HTMLElement).style.background = 'transparent';
                      (e.target as HTMLElement).style.borderColor = 'var(--border)';
                      (e.target as HTMLElement).style.color = 'var(--text-secondary)';
                    }}
                  >
                    <span style={{ fontSize: 12 }}>{section?.icon || ''}</span>
                    {section?.label || key}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Release list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '0 24px 24px' }}>
          {/* Roadmap */}
          <div style={{
            marginBottom: 24, padding: '16px 20px',
            background: 'var(--surface-active)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
          }}>
            <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: 14, fontWeight: 600, marginBottom: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
              </svg>
              Roadmap — Coming Next
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <RoadmapItem version="v1.1.0" status="In Progress">
                Multi-GPU inference dashboard, chat message editing, improved download queue management
              </RoadmapItem>
              <RoadmapItem version="v1.2.0" status="Planned">
                Embeddings API endpoint, RAG pipeline integration, model benchmarking tools
              </RoadmapItem>
              <RoadmapItem version="v2.0.0" status="Under Consideration">
                Windows/macOS support, Docker container deployment, remote API server mode, plugin system
              </RoadmapItem>
            </div>
          </div>

          {filteredReleases.map((release) => (
            <ReleaseCard
              key={release.version}
              release={release}
              expanded={expandedReleases.has(release.version)}
              onToggle={() => toggleRelease(release.version)}
              searchQuery={searchQuery}
            />
          ))}

          {filteredReleases.length === 0 && (
            <div className="empty-state" style={{ padding: 40 }}>
              <p>No releases match your search.</p>
            </div>
          )}
        </div>

        {/* vLLM note */}
        <div style={{
          margin: '0 24px', padding: '12px 16px',
          background: 'var(--accent-muted)', border: '1px solid var(--accent)',
          borderRadius: 'var(--radius-sm)', opacity: 0.7, flexShrink: 0,
        }}>
          <p style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            <strong style={{ color: 'var(--accent)' }}>Note:</strong> LLM model support is bound by what the vLLM engine supports.
            Check the full documentation at{' '}
            <a href="https://docs.vllm.ai" target="_blank" rel="noopener noreferrer">docs.vllm.ai</a>{' '}
            for the latest list of supported model architectures and configuration options.
          </p>
        </div>

        {/* Footer */}
        <div style={{
          padding: '12px 24px', borderTop: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          fontSize: 11, color: 'var(--text-muted)', flexShrink: 0,
        }}>
          <span>
            <a href="https://github.com/Moashour93/vllm-studio/commits/main" target="_blank" rel="noopener noreferrer">Full commit history</a>
            {' · '}
            <a href="https://github.com/Moashour93/vllm-studio/issues/new" target="_blank" rel="noopener noreferrer">Report a bug</a>
          </span>
          <span>Last updated: 7 May 2026</span>
        </div>
      </div>
    </div>
  );
}

function RoadmapItem({ version, status, children }: { version: string; status: 'In Progress' | 'Planned' | 'Under Consideration'; children: string }) {
  const colorMap = {
    'In Progress': { bg: 'var(--warning-muted)', text: 'var(--warning)' },
    'Planned': { bg: 'var(--accent-muted)', text: 'var(--accent)' },
    'Under Consideration': { bg: 'var(--surface-hover)', text: 'var(--text-muted)' },
  };
  const colors = colorMap[status];
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', minWidth: 64 }}>{version}</span>
      <span style={{
        padding: '1px 7px', borderRadius: 100, fontSize: 10, fontWeight: 500,
        background: colors.bg, color: colors.text, fontFamily: 'var(--font-mono)',
        whiteSpace: 'nowrap',
      }}>
        {status}
      </span>
      <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{children}</span>
    </div>
  );
}

function ReleaseCard({ release, expanded, onToggle, searchQuery }: {
  release: Release;
  expanded: boolean;
  onToggle: () => void;
  searchQuery: string;
}) {
  const typeColors = TYPE_COLORS[release.type];

  return (
    <div id={`release-${release.version}`} style={{
      marginBottom: 24, border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)', overflow: 'hidden',
      background: 'var(--surface)',
    }}>
      <button
        onClick={onToggle}
        style={{
          display: 'flex', alignItems: 'center', gap: 12,
          width: '100%', padding: '16px 20px',
          background: 'var(--surface-hover)', border: 'none', cursor: 'pointer',
          textAlign: 'left', font: 'inherit', color: 'inherit',
        }}
      >
        <svg
          width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
          style={{
            transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)',
            transition: 'transform var(--transition)',
            flexShrink: 0, color: 'var(--text-muted)',
          }}
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>
          {release.version}
        </span>
        <span style={{
          padding: '2px 10px', borderRadius: 100, fontSize: 10, fontWeight: 600,
          background: typeColors.bg, color: typeColors.text,
          fontFamily: 'var(--font-mono)', letterSpacing: '0.03em',
        }}>
          {release.type}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>
          "{release.codename}"
        </span>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          {release.date}
        </span>
      </button>

      {expanded && (
        <div style={{ padding: '20px' }}>
          {release.sections.map((section) => (
            <ReleaseSection
              key={section.key}
              section={section}
              searchQuery={searchQuery}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ReleaseSection({ section, searchQuery }: {
  section: ReleaseSection;
  searchQuery: string;
}) {
  if (section.items.length === 0 && !section.summary) return null;

  return (
    <div
      id={`section-${section.key}`}
      style={{ marginBottom: 16 }}
    >
      <h4 style={{
        fontFamily: 'var(--font-heading)', fontSize: 13, fontWeight: 600,
        color: 'var(--text-primary)', marginBottom: 6,
        display: 'flex', alignItems: 'center', gap: 6,
      }}>
        <span>{section.icon}</span> {section.label}
      </h4>
      {section.summary && (
        <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: section.items.length > 0 ? 8 : 0, lineHeight: 1.6 }}>
          <HighlightText text={section.summary} query={searchQuery} />
        </p>
      )}
      {section.items.length > 0 && (
        <ul style={{ paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 4 }}>
          {section.items.map((item, i) => (
            <li key={i} style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              <HighlightText text={item} query={searchQuery} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function HighlightText({ text, query }: { text: string; query: string }) {
  if (!query.trim()) return <>{text}</>;
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const parts = text.split(new RegExp(`(${escaped})`, 'gi'));
  return (
    <>
      {parts.map((part, i) =>
        part.toLowerCase() === query.toLowerCase() ? (
          <mark key={i} style={{
            background: 'var(--accent-muted)',
            color: 'var(--accent)',
            borderRadius: 2,
            padding: '0 2px',
          }}>
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </>
  );
}
