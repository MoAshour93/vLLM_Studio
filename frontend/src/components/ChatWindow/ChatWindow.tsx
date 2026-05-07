import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { Virtuoso, VirtuosoHandle } from 'react-virtuoso';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import hljs from 'highlight.js/lib/core';
import bash from 'highlight.js/lib/languages/bash';
import python from 'highlight.js/lib/languages/python';
import javascript from 'highlight.js/lib/languages/javascript';
import typescript from 'highlight.js/lib/languages/typescript';
import json from 'highlight.js/lib/languages/json';
import sql from 'highlight.js/lib/languages/sql';
import xml from 'highlight.js/lib/languages/xml';
import css from 'highlight.js/lib/languages/css';
import markdown from 'highlight.js/lib/languages/markdown';
import c from 'highlight.js/lib/languages/c';
import cpp from 'highlight.js/lib/languages/cpp';
import rust from 'highlight.js/lib/languages/rust';
import go from 'highlight.js/lib/languages/go';
import yaml from 'highlight.js/lib/languages/yaml';
import { useAppStore } from '../../store/index';
import { useChat } from '../../hooks/useChat';
import { getChatStats } from '../../hooks/useChat';
import { uploadAttachments } from '../../api/index';
import AttachmentPreview from '../AttachmentPreview/AttachmentPreview';
import { uuid } from '../../utils/uuid';
import type { AttachmentMeta, ChatMessage, InferenceParameters, ModelInfo } from '../../store/types';

hljs.registerLanguage('bash', bash);
hljs.registerLanguage('python', python);
hljs.registerLanguage('javascript', javascript);
hljs.registerLanguage('typescript', typescript);
hljs.registerLanguage('json', json);
hljs.registerLanguage('sql', sql);
hljs.registerLanguage('xml', xml);
hljs.registerLanguage('css', css);
hljs.registerLanguage('markdown', markdown);
hljs.registerLanguage('c', c);
hljs.registerLanguage('cpp', cpp);
hljs.registerLanguage('rust', rust);
hljs.registerLanguage('go', go);
hljs.registerLanguage('yaml', yaml);

interface ChatWindowProps {
  onNewChat: () => void;
}

export default function ChatWindow({ onNewChat }: ChatWindowProps) {
  const {
    currentSessionId, messages, sessions, isStreaming,
    models, selectedModelId, serverStatus, settings,
  } = useAppStore();
  const { sendMessage, regenerate, cancelStream } = useChat();
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<AttachmentMeta[]>([]);
  const [editingMsgId, setEditingMsgId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');
  const [showScrollFab, setShowScrollFab] = useState(false);
  const virtuosoRef = useRef<VirtuosoHandle>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const currentSession = useMemo(
    () => sessions.find((s) => s.id === currentSessionId),
    [sessions, currentSessionId]
  );

  const selectedModel = useMemo(
    () => models.find((m) => m.id === (selectedModelId || currentSession?.modelId)),
    [models, selectedModelId, currentSession]
  );

  const isVisionModel = selectedModel?.isVision ?? false;

  useEffect(() => {
    if (virtuosoRef.current && !isStreaming) {
      virtuosoRef.current.scrollToIndex({ index: messages.length - 1, behavior: 'smooth' });
    }
  }, [messages.length]);

  const handleSend = useCallback(async () => {
    const content = input.trim();
    if (!content || isStreaming || serverStatus !== 'running') return;

    setInput('');
    const currentAttachments = [...attachments];
    setAttachments([]);
    await sendMessage(content, currentAttachments.length > 0 ? currentAttachments : undefined);
  }, [input, isStreaming, serverStatus, attachments, sendMessage]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    const sendOnEnter = settings?.sendOnEnter ?? true;

    if (sendOnEnter) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    } else {
      if (e.key === 'Enter' && e.ctrlKey) {
        e.preventDefault();
        handleSend();
      }
    }
  }, [handleSend, settings]);

  const handleAttach = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0 || !currentSessionId) return;

    try {
      const result = await uploadAttachments(currentSessionId, files);
      setAttachments((prev) => [...prev, ...result.attachments]);
    } catch (err) {
      console.error('Upload failed:', err);
    }

    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [currentSessionId]);

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id));
  }, []);

  const handleCopy = useCallback((text: string) => {
    navigator.clipboard.writeText(text).catch(() => {});
  }, []);

  const handleEditMessage = useCallback((id: string, content: string) => {
    setEditingMsgId(id);
    setEditContent(content);
  }, []);

  const handleSaveEdit = useCallback(async () => {
    if (!editingMsgId) return;
    const state = useAppStore.getState();
    state.updateMessage(editingMsgId, editContent);
    setEditingMsgId(null);

    // Re-send to regenerate
    const userMsg = messages.find((m) => m.role === 'user' && messages.indexOf(m) < messages.findIndex((mm) => mm.id === editingMsgId));
    if (userMsg) {
      // Find the user message that triggered this
      const idx = messages.findIndex((m) => m.id === editingMsgId);
      if (idx > 0 && messages[idx - 1].role === 'user') {
        // Remove all messages after this user message and re-send
        // For simplicity, just trigger regenerate
      }
    }
  }, [editingMsgId, editContent, messages]);

  if (!currentSessionId) {
    return (
      <div style={{ flex: 1, display: 'flex' }} className="empty-state">
        <div className="empty-state-icon">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
          </svg>
        </div>
        <h2>VLLM Studio</h2>
        <p>Create a new chat or select an existing conversation from the sidebar to get started.</p>
        <button className="btn btn-primary" onClick={onNewChat}>New Chat</button>
      </div>
    );
  }

  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--bg)',
      position: 'relative',
      minWidth: 0,
    }}>
      {/* System prompt */}
      {currentSession?.systemPrompt && (
        <div style={{
          padding: '8px 16px',
          background: 'var(--surface-active)',
          borderBottom: '1px solid var(--border)',
          fontSize: 12,
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
        }}>
          <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>System Prompt: </span>
          {currentSession.systemPrompt}
        </div>
      )}

      {/* Server error banner */}
      {serverStatus === 'error' && (
        <div style={{
          padding: '10px 16px',
          background: 'var(--error-muted)',
          borderBottom: '1px solid var(--error)',
          fontSize: 13,
          color: 'var(--error)',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <span>⚠</span>
          <span>The vLLM server encountered an error. Check the server panel for details.</span>
        </div>
      )}

      {serverStatus !== 'running' && serverStatus !== 'error' && messages.length === 0 && (
        <div style={{ flex: 1, display: 'flex' }} className="empty-state">
          <div className="empty-state-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
              <line x1="8" y1="21" x2="16" y2="21" />
              <line x1="12" y1="17" x2="12" y2="21" />
            </svg>
          </div>
          <h2>vLLM Server {serverStatus === 'stopped' ? 'Not Running' : 'Starting...'}</h2>
          <p>{serverStatus === 'stopped'
            ? 'Start the vLLM server from the configuration panel to begin chatting.'
            : 'The vLLM server is starting up. This may take a moment...'}
          </p>
        </div>
      )}

      {/* Messages */}
      {messages.length > 0 && (
        <Virtuoso
          ref={virtuosoRef}
          data={messages}
          style={{ flex: 1 }}
          initialTopMostItemIndex={messages.length - 1}
          followOutput={'auto'}
          itemContent={(index, msg) => (
            <MessageBubble
              message={msg}
              isStreaming={isStreaming && index === messages.length - 1 && msg.role === 'assistant'}
              onCopy={handleCopy}
              onEdit={handleEditMessage}
              onRegenerate={regenerate}
              isEditing={editingMsgId === msg.id}
              editContent={editContent}
              onEditChange={setEditContent}
              onSaveEdit={handleSaveEdit}
              onCancelEdit={() => setEditingMsgId(null)}
              removeMessage={(id) => useAppStore.getState().removeMessage(id)}
            />
          )}
          atBottomStateChange={(atBottom) => setShowScrollFab(!atBottom)}
        />
      )}

      {/* Scroll FAB */}
      {showScrollFab && messages.length > 0 && (
        <div className="scroll-fab" onClick={() => {
          virtuosoRef.current?.scrollToIndex({ index: messages.length - 1, behavior: 'smooth' });
        }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
      )}

      {/* Attachment bar */}
      {attachments.length > 0 && (
        <div style={{ padding: '0 16px', borderTop: '1px solid var(--border)' }}>
          <AttachmentPreview
            attachments={attachments}
            onRemove={removeAttachment}
            isVisionModel={isVisionModel}
          />
        </div>
      )}

      {/* Input */}
      <div style={{
        padding: '12px 16px',
        borderTop: '1px solid var(--border)',
        background: 'var(--surface)',
        display: 'flex',
        gap: 8,
        alignItems: 'flex-end',
      }}>
        <button
          className="btn-icon"
          onClick={handleAttach}
          title="Attach files"
          disabled={serverStatus !== 'running'}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
          </svg>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/jpeg,image/png,image/webp,image/gif,application/pdf"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />

        <textarea
          className="textarea"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            serverStatus === 'running'
              ? 'Type a message... (Enter to send)'
              : serverStatus === 'stopped'
                ? 'Start the vLLM server to chat...'
                : 'Waiting for vLLM server...'
          }
          disabled={serverStatus !== 'running' || isStreaming}
          rows={1}
          style={{ flex: 1 }}
        />

        <button
          className="btn btn-primary"
          onClick={handleSend}
          disabled={!input.trim() || isStreaming || serverStatus !== 'running'}
          style={{ padding: '8px 16px' }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>

        {isStreaming && (
          <button
            className="btn btn-sm"
            onClick={cancelStream}
            style={{
              padding: '8px 12px',
              background: 'var(--error-muted)',
              color: 'var(--error)',
              border: '1px solid var(--error)',
              fontSize: 11,
            }}
            title="Stop generating"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <rect x="4" y="4" width="16" height="16" rx="2" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}

function MessageBubble({
  message,
  isStreaming,
  onCopy,
  onEdit,
  onRegenerate,
  isEditing,
  editContent,
  onEditChange,
  onSaveEdit,
  onCancelEdit,
  removeMessage,
}: {
  message: ChatMessage;
  isStreaming: boolean;
  onCopy: (text: string) => void;
  onEdit: (id: string, content: string) => void;
  onRegenerate: () => void;
  isEditing: boolean;
  editContent: string;
  onEditChange: (v: string) => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
  removeMessage: (id: string) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';
  const isSystem = message.role === 'system';

  if (isSystem) {
    return (
      <div style={{ textAlign: 'center', padding: '8px', fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic' }}>
        {message.content}
      </div>
    );
  }

  return (
    <div
      className="fade-in"
      style={{
        padding: '16px 20px',
        borderBottom: '1px solid var(--border)',
        background: isAssistant ? 'var(--surface)' : 'transparent',
        position: 'relative',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div style={{ display: 'flex', gap: 12, maxWidth: 900, margin: '0 auto', width: '100%' }}>
        <div style={{
          width: 28,
          height: 28,
          borderRadius: 'var(--radius-sm)',
          background: isUser ? 'var(--accent-muted)' : 'var(--surface-active)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          fontSize: 12,
          fontWeight: 700,
          color: isUser ? 'var(--accent)' : 'var(--text-secondary)',
          fontFamily: 'var(--font-mono)',
        }}>
          {isUser ? 'U' : 'AI'}
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 11,
            fontWeight: 600,
            color: 'var(--text-muted)',
            marginBottom: 6,
            fontFamily: 'var(--font-mono)',
            letterSpacing: '0.03em',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}>
            <span>{isUser ? 'You' : 'Assistant'}</span>
            {message.finishReason && (
              <span className="badge badge-accent">{message.finishReason}</span>
            )}
            {message.tokensUsed && (
              <span style={{ color: 'var(--text-muted)' }}>{message.tokensUsed} tokens</span>
            )}
            {isStreaming && (
              <span className="blink" style={{ color: 'var(--accent)' }}>▌</span>
            )}
          </div>

          {isEditing ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <textarea
                className="textarea"
                value={editContent}
                onChange={(e) => onEditChange(e.target.value)}
                rows={4}
                autoFocus
              />
              <div style={{ display: 'flex', gap: 6 }}>
                <button className="btn btn-primary btn-sm" onClick={onSaveEdit}>Save</button>
                <button className="btn btn-secondary btn-sm" onClick={onCancelEdit}>Cancel</button>
              </div>
            </div>
          ) : (
            <div className="markdown-content" style={{ fontSize: 14, lineHeight: 1.7, color: 'var(--text-primary)' }}>
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ node, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    const codeStr = String(children).replace(/\n$/, '');
                    const isBlock = node?.tagName === 'code' && match;

                    if (isBlock) {
                      try {
                        const highlighted = hljs.highlight(codeStr, { language: match![1] });
                        return (
                          <pre>
                            <button className="copy-btn" onClick={() => onCopy(codeStr)}>Copy</button>
                            <code className={`hljs ${match![1]}`} dangerouslySetInnerHTML={{ __html: highlighted.value }} />
                          </pre>
                        );
                      } catch {
                        return (
                          <pre>
                            <button className="copy-btn" onClick={() => onCopy(codeStr)}>Copy</button>
                            <code>{codeStr}</code>
                          </pre>
                        );
                      }
                    }

                    return (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  },
                  a({ href, children }) {
                    return (
                      <a href={href} target="_blank" rel="noopener noreferrer">
                        {children}
                      </a>
                    );
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>

              {isStreaming && !message.content && (
                <div style={{ display: 'flex', gap: 4, padding: '8px 0' }}>
                  <span className="spinner" style={{ width: 8, height: 8 }} />
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Thinking...</span>
                </div>
              )}

              {isAssistant && !isStreaming && message.content && !isEditing && (
                <StatsRow msgId={message.id} />
              )}
            </div>
          )}

          {/* Hover actions */}
          {hovered && !isEditing && message.content && (
            <div style={{
              marginTop: 8,
              display: 'flex',
              gap: 4,
              opacity: 0.7,
            }}>
              <button className="btn-icon" onClick={() => onCopy(message.content)} title="Copy">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                </svg>
              </button>
              {isUser && (
                <button
                  className="btn-icon"
                  onClick={() => onEdit(message.id, message.content)}
                  title="Edit"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                  </svg>
                </button>
              )}
              {isAssistant && (
                <button className="btn-icon" onClick={onRegenerate} title="Regenerate">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="23 4 23 10 17 10" /><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
                  </svg>
                </button>
              )}
              <button
                className="btn-icon"
                onClick={() => removeMessage(message.id)}
                title="Delete"
                style={{ color: 'var(--error)' }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatsRow({ msgId }: { msgId: string }) {
  const messages = useAppStore((s) => s.messages);
  const msg = messages.find((m) => m.id === msgId);
  const liveStats = getChatStats(msgId);
  const stats = msg?.chatStats ?? liveStats;
  if (!stats) return null;

  return (
    <div style={{
      marginTop: 10,
      padding: '8px 10px',
      background: 'var(--surface-active)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)',
      display: 'flex',
      flexWrap: 'wrap',
      gap: '8px 16px',
      fontSize: 11,
      fontFamily: 'var(--font-mono)',
      color: 'var(--text-muted)',
    }}>
      <span>In: <strong style={{ color: 'var(--text-secondary)' }}>{Math.round(stats.promptTokens)}</strong> tok</span>
      <span>Out: <strong style={{ color: 'var(--text-secondary)' }}>{stats.outputTokens || stats.totalTokens}</strong> tok</span>
      <span>TTFT: <strong style={{ color: 'var(--text-secondary)' }}>{stats.ttftMs}ms</strong></span>
      <span>Time: <strong style={{ color: 'var(--text-secondary)' }}>{(stats.totalTimeMs / 1000).toFixed(1)}s</strong></span>
      <span>Speed: <strong style={{ color: 'var(--accent)' }}>{stats.tokensPerSec.toFixed(1)}</strong> tok/s</span>
    </div>
  );
}
