interface AboutModalProps {
  onClose: () => void;
}

const TECH_STACK = {
  Frontend: ['React 18', 'TypeScript 5.6', 'Vite 5', 'Zustand 5', 'react-virtuoso', 'react-markdown', 'highlight.js', 'remark-gfm'],
  Backend: ['Node.js 18+', 'Express 4', 'TypeScript 5.6', 'better-sqlite3', 'WebSocket (ws)', 'Zod', 'tsx'],
  'ML / Inference': ['vLLM', 'Python 3.10+', 'huggingface_hub', 'FlashInfer'],
  'Build & Quality': ['Vite', 'ESLint', 'tsc (strict)', 'concurrently'],
};

export default function AboutModal({ onClose }: AboutModalProps) {
  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" style={{ width: 680, maxHeight: '90vh' }}>
        <div className="modal-header">
          <h2>About VLLM Studio</h2>
          <button className="btn-icon" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <div className="modal-body" style={{ padding: '24px' }}>

          {/* Section 1 — App Credits */}
          <div style={{ marginBottom: 28, textAlign: 'center' }}>
            <div style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              width: 56, height: 56, borderRadius: 'var(--radius-lg)',
              background: 'var(--accent-muted)', marginBottom: 12,
            }}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
              </svg>
            </div>
            <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
              VLLM Studio
            </h2>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 8 }}>
              A fast, Linux-native AI inference interface built on vLLM
            </p>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 12 }}>
              <span className="badge badge-accent">v1.0.0</span>
              <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                Apache 2.0
              </span>
            </div>
            <a
              href="https://github.com/Moashour93/vllm-studio"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-secondary"
              style={{ textDecoration: 'none' }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style={{ flexShrink: 0 }}>
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
              </svg>
              github.com/Moashour93/vllm-studio
            </a>
          </div>

          {/* Section 2 — About the Author */}
          <div style={{ marginBottom: 28 }}>
            <h3 className="modal-section-label" style={{
              fontFamily: 'var(--font-heading)', fontSize: 12, fontWeight: 600,
              color: 'var(--text-muted)', textTransform: 'uppercase',
              letterSpacing: '0.08em', marginBottom: 12,
            }}>
              About the Author
            </h3>
            <div style={{
              display: 'flex', gap: 0,
              background: 'var(--surface-active)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)', overflow: 'hidden',
            }}>
              {/* Left accent bar */}
              <div style={{
                width: 4, flexShrink: 0,
                background: 'linear-gradient(180deg, var(--accent) 0%, var(--accent-hover) 100%)',
              }} />
              <div style={{ padding: '20px 24px', flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 14 }}>
                  {/* Avatar placeholder */}
                  <div style={{
                    width: 52, height: 52, borderRadius: '50%',
                    background: 'linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                  }}>
                    <span style={{
                      fontFamily: 'var(--font-heading)', fontSize: 18, fontWeight: 700,
                      color: '#fff', letterSpacing: '0.02em',
                    }}>
                      MA
                    </span>
                  </div>
                  <div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-heading)' }}>
                      Mohamed Ashour
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.6, marginTop: 2 }}>
                      MRICS (Chartered Quantity Surveyor) · MBCS (Chartered Data Analyst) · Civil Engineer
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--accent)', fontWeight: 500, marginTop: 2 }}>
                      Digital Transformation Manager
                    </div>
                  </div>
                </div>
                <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: 14 }}>
                  Mohamed is a chartered surveyor and data analyst with a decade of combined experience spanning
                  construction engineering, digital transformation, and AI/LLM development. He has worked on major UK
                  and international infrastructure programmes and specialises in building AI-powered tools for the
                  construction and cost intelligence sector. VLLM Studio was built out of a genuine frustration with
                  existing local inference tools on Linux — and a desire for something faster, cleaner, and purpose-built
                  for vLLM.
                </p>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <a
                    href="https://www.github.com/Moashour93"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-secondary btn-sm"
                    style={{ textDecoration: 'none' }}
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                    </svg>
                    GitHub
                  </a>
                  <a
                    href="https://www.linkedin.com/in/mohamed-ashour-0727/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-secondary btn-sm"
                    style={{ textDecoration: 'none' }}
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                    </svg>
                    LinkedIn
                  </a>
                  <a
                    href="mailto:mo_ashour1@outlook.com"
                    className="btn btn-secondary btn-sm"
                    style={{ textDecoration: 'none' }}
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="2" y="4" width="20" height="16" rx="2" /><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7" />
                    </svg>
                    Email
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Section 3 — AI Collaborator Credit */}
          <div style={{ marginBottom: 28 }}>
            <h3 style={{
              fontFamily: 'var(--font-heading)', fontSize: 12, fontWeight: 600,
              color: 'var(--text-muted)', textTransform: 'uppercase',
              letterSpacing: '0.08em', marginBottom: 12,
            }}>
              AI Collaborator Credit
            </h3>
            <div style={{
              padding: '16px 20px', background: 'var(--surface-active)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                <div style={{
                  width: 32, height: 32, borderRadius: 'var(--radius-sm)',
                  background: 'var(--accent-muted)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                  </svg>
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                    DeepSeek V4 (Pro)
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    Primary AI pair programmer
                  </div>
                </div>
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                Used throughout the architecture, backend, and frontend development of VLLM Studio.
              </p>
              <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6, marginTop: 6, fontStyle: 'italic' }}>
                This application was built with the assistance of large language models. All design decisions,
                architecture choices, and final implementation were directed and reviewed by the author.
              </p>
            </div>
          </div>

          {/* Section 4 — Tech Stack */}
          <div style={{ marginBottom: 28 }}>
            <h3 style={{
              fontFamily: 'var(--font-heading)', fontSize: 12, fontWeight: 600,
              color: 'var(--text-muted)', textTransform: 'uppercase',
              letterSpacing: '0.08em', marginBottom: 12,
            }}>
              Tech Stack
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {(Object.entries(TECH_STACK) as [string, string[]][]).map(([category, items]) => (
                <div key={category} style={{
                  padding: '14px 16px', background: 'var(--surface-active)',
                  border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
                }}>
                  <div style={{
                    fontSize: 11, fontWeight: 600, fontFamily: 'var(--font-mono)',
                    color: 'var(--accent)', letterSpacing: '0.03em',
                    textTransform: 'uppercase', marginBottom: 8,
                  }}>
                    {category}
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {items.map((item) => (
                      <span key={item} style={{
                        padding: '2px 8px', fontSize: 11, borderRadius: 'var(--radius-sm)',
                        background: 'var(--surface-hover)', color: 'var(--text-secondary)',
                        fontFamily: 'var(--font-mono)',
                      }}>
                        {item}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* vLLM disclaimer */}
          <div style={{
            marginBottom: 28, padding: '14px 18px',
            background: 'var(--accent-muted)', border: '1px solid var(--accent)',
            borderRadius: 'var(--radius-sm)', opacity: 0.8,
          }}>
            <p style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              <strong style={{ color: 'var(--accent)' }}>Important:</strong> LLM model support in VLLM Studio is
              bound by what the vLLM inference engine supports. For the latest list of supported model
              architectures, quantization methods, and configuration options, refer to the official vLLM
              documentation at{' '}
              <a href="https://docs.vllm.ai" target="_blank" rel="noopener noreferrer">docs.vllm.ai</a>.
            </p>
          </div>

          {/* Section 5 — Contact & Feedback */}
          <div style={{
            padding: '20px 24px', background: 'var(--surface-active)',
            border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
            textAlign: 'center',
          }}>
            <h3 style={{
              fontFamily: 'var(--font-heading)', fontSize: 12, fontWeight: 600,
              color: 'var(--text-muted)', textTransform: 'uppercase',
              letterSpacing: '0.08em', marginBottom: 8,
            }}>
              Contact &amp; Feedback
            </h3>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 14 }}>
              Found a bug? Have a feature request? Reach out via GitHub Issues or email directly.
            </p>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
              <a
                href="https://github.com/Moashour93/vllm-studio/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-primary btn-sm"
                style={{ textDecoration: 'none' }}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                </svg>
                Open GitHub Issues
              </a>
              <a
                href="mailto:mo_ashour1@outlook.com"
                className="btn btn-secondary btn-sm"
                style={{ textDecoration: 'none' }}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="4" width="20" height="16" rx="2" /><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7" />
                </svg>
                Send Email
              </a>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
