import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { useAppStore } from '../../store/index';
import { createSession, deleteSession, updateSession, getSessions, createFolder, deleteFolder, updateFolder, getFolders } from '../../api/index';
import type { ChatSession, ChatFolder } from '../../store/types';

interface SidebarProps {
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
}

const DEFAULT_FOLDER_ID = 'default';

export default function Sidebar({ onNewChat, onSelectSession }: SidebarProps) {
  const {
    sessions, currentSessionId, folders, searchQuery, setSearchQuery,
    setSessions, removeSession, updateSession,
  } = useAppStore();
  const [collapsedFolders, setCollapsedFolders] = useState<Set<string>>(new Set());
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);
  const [newFolderName, setNewFolderName] = useState('');
  const [editingFolderId, setEditingFolderId] = useState<string | null>(null);
  const [editingFolderName, setEditingFolderName] = useState('');
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [deletingFolderId, setDeletingFolderId] = useState<string | null>(null);
  const [deleteFolderChoice, setDeleteFolderChoice] = useState<'keep' | 'delete' | null>(null);

  // New chat goes to "Defaults" folder automatically
  useEffect(() => {
    if (newFolderName === '_' && folderInputRef.current) {
      folderInputRef.current.focus();
    }
  }, [newFolderName]);

  const pinnedFirst = useMemo(() => {
    return (list: ChatSession[]) => {
      const pinned = list.filter((s) => s.pinned > 0);
      const unpinned = list.filter((s) => s.pinned === 0);
      return [...pinned, ...unpinned];
    };
  }, []);

  // Group sessions by folder
  const sessionsByFolder = useMemo(() => {
    const map = new Map<string, ChatSession[]>();
    let filtered = sessions;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter((s) => s.title.toLowerCase().includes(q));
    }
    for (const s of filtered) {
      const fid = s.folder || DEFAULT_FOLDER_ID;
      if (!map.has(fid)) map.set(fid, []);
      map.get(fid)!.push(s);
    }
    // Sort folders: pinned folders first, then alphabetically
    const sorted = new Map([...map.entries()].sort(([a], [b]) => {
      const fa = folders.find((f) => f.id === a);
      const fb = folders.find((f) => f.id === b);
      return (fa?.name ?? a).localeCompare(fb?.name ?? b);
    }));
    for (const [fid, list] of sorted) {
      sorted.set(fid, pinnedFirst(list));
    }
    return sorted;
  }, [sessions, searchQuery, folders]);

  const toggleFolder = useCallback((folderId: string) => {
    setCollapsedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(folderId)) next.delete(folderId);
      else next.add(folderId);
      return next;
    });
  }, []);

  const handleSelect = useCallback((id: string) => {
    onSelectSession(id);
  }, [onSelectSession]);

  const handleDelete = useCallback(async (id: string) => {
    try {
      await deleteSession(id);
      removeSession(id);
    } catch { /* ignore */ }
    setDeletingSessionId(null);
  }, [removeSession]);

  const handleRename = useCallback(async (id: string) => {
    if (!editTitle.trim()) { setEditingSessionId(null); return; }
    try {
      await updateSession(id, { title: editTitle.trim() });
      updateSession(id, { title: editTitle.trim() } as Parameters<typeof updateSession>[1]);
    } catch { /* ignore */ }
    setEditingSessionId(null);
  }, [editTitle, updateSession]);

  const handlePin = useCallback(async (id: string, currentPinned: number) => {
    try {
      const newPinned = currentPinned > 0 ? 0 : 1;
      await updateSession(id, { pinned: newPinned });
      updateSession(id, { pinned: newPinned } as Parameters<typeof updateSession>[1]);
    } catch { /* ignore */ }
  }, [updateSession]);

  const formatDate = (ts: number) => {
    const d = new Date(ts);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    const days = Math.floor(diff / 86400000);
    if (days === 0) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days}d ago`;
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  };

  const handleAddFolder = useCallback(async () => {
    const name = folderInputRef.current?.value.trim();
    if (!name) return;
    try {
      await createFolder({ name });
      const updated = await getFolders();
      useAppStore.getState().setFolders(updated.folders);
    } catch { /* ignore */ }
    setNewFolderName('');
  }, []);

  const handleRenameFolder = useCallback(async (id: string) => {
    if (!editingFolderName.trim()) { setEditingFolderId(null); return; }
    try {
      await updateFolder(id, { name: editingFolderName.trim() });
      const updated = await getFolders();
      useAppStore.getState().setFolders(updated.folders);
    } catch { /* ignore */ }
    setEditingFolderId(null);
  }, [editingFolderName]);

  const handleDeleteFolderConfirm = useCallback(async (id: string, choice: 'keep' | 'delete') => {
    try {
      if (choice === 'keep') {
        // Move all sessions in this folder to defaults, then delete folder
        const inFolder = sessions.filter((s) => s.folder === id);
        for (const s of inFolder) {
          await updateSession(s.id, { folder: DEFAULT_FOLDER_ID });
          updateSession(s.id, { folder: DEFAULT_FOLDER_ID } as Parameters<typeof updateSession>[1]);
        }
      } else {
        // Delete all sessions in this folder
        const inFolder = sessions.filter((s) => s.folder === id);
        for (const s of inFolder) {
          await deleteSession(s.id);
          removeSession(s.id);
        }
      }
      await deleteFolder(id);
      const updated = await getFolders();
      useAppStore.getState().setFolders(updated.folders);
    } catch { /* ignore */ }
    setDeletingFolderId(null);
    setDeleteFolderChoice(null);
  }, [sessions, updateSession, removeSession]);

  const handleMoveToFolder = useCallback(async (sessionId: string, folderId: string) => {
    try {
      await updateSession(sessionId, { folder: folderId });
      updateSession(sessionId, { folder: folderId } as Parameters<typeof updateSession>[1]);
    } catch { /* ignore */ }
  }, [updateSession]);

  const folderName = (fid: string) => folders.find((f) => f.id === fid)?.name ?? fid;

  return (
    <div style={{
      width: 'var(--sidebar-width)',
      flexShrink: 0,
      borderRight: '1px solid var(--border)',
      background: 'var(--surface)',
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
    }}>
      {/* New Chat + Search */}
      <div style={{ padding: '10px 12px', borderBottom: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 8 }}>
        <button
          className="btn btn-primary"
          onClick={onNewChat}
          style={{ width: '100%', justifyContent: 'flex-start', padding: '8px 14px' }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New Chat
        </button>
        <input
          className="input"
          type="text"
          placeholder="Search chats..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{ fontSize: 12, padding: '6px 10px' }}
        />
      </div>

      {/* Scrollable area: folder tree + sessions */}
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
        {sessionsByFolder.size === 0 ? (
          <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
            {searchQuery ? 'No matching chats' : 'No chats yet'}
          </div>
        ) : (
          [...sessionsByFolder.entries()].map(([fid, folderSessions]) => {
            const collapsed = collapsedFolders.has(fid);
            const isDefault = fid === DEFAULT_FOLDER_ID;
            return (
              <div key={fid}>
                {/* Folder header */}
                <div
                  style={{
                    display: 'flex', alignItems: 'center', gap: 4,
                    padding: '6px 12px 4px', cursor: 'pointer',
                    fontSize: 11, color: 'var(--text-muted)',
                    fontFamily: 'var(--font-heading)', fontWeight: 600,
                    textTransform: 'uppercase', letterSpacing: '0.05em',
                    userSelect: 'none',
                  }}
                  onClick={() => toggleFolder(fid)}
                >
                  <span style={{ fontSize: 10, width: 12, textAlign: 'center' }}>
                    {collapsed ? '▸' : '▾'}
                  </span>
                  <span style={{
                    width: 7, height: 7, borderRadius: '50%',
                    background: folders.find((f) => f.id === fid)?.color || 'var(--text-muted)',
                    flexShrink: 0,
                  }} />
                  <span>{folderName(fid)}</span>
                  <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 400 }}>
                    {folderSessions.length}
                  </span>
                  {!isDefault && (
                    <button
                      className="btn-icon"
                      title={`Delete ${folderName(fid)} folder`}
                      onClick={(e) => { e.stopPropagation(); setDeletingFolderId(fid); }}
                      style={{ padding: 0, fontSize: 12, color: 'var(--error)', opacity: 0.5 }}
                    >
                      ×
                    </button>
                  )}
                </div>

                {/* Sessions in this folder */}
                {!collapsed && folderSessions.map((session) => (
                  <div
                    key={session.id}
                    style={{
                      padding: '8px 12px 8px 28px',
                      margin: '0 6px 1px',
                      borderRadius: 'var(--radius-sm)',
                      cursor: 'pointer',
                      background: session.id === currentSessionId ? 'var(--surface-active)' : 'transparent',
                      border: session.id === currentSessionId ? '1px solid var(--border-light)' : '1px solid transparent',
                      transition: 'all var(--transition)',
                    }}
                    onClick={() => handleSelect(session.id)}
                    onMouseEnter={(e) => {
                      const el = e.currentTarget;
                      if (session.id !== currentSessionId) el.style.background = 'var(--surface-hover)';
                    }}
                    onMouseLeave={(e) => {
                      const el = e.currentTarget;
                      if (session.id !== currentSessionId) el.style.background = 'transparent';
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      {session.pinned > 0 && (
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="var(--accent)" style={{ flexShrink: 0 }}>
                          <path d="M16 12V4h1V2H7v2h1v8l-2 2v2h5.2v5h1.6v-5H18v-2l-2-2z" />
                        </svg>
                      )}
                      {editingSessionId === session.id ? (
                        <input
                          className="input"
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          onBlur={() => handleRename(session.id)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleRename(session.id);
                            if (e.key === 'Escape') setEditingSessionId(null);
                          }}
                          autoFocus
                          style={{ flex: 1, fontSize: 12, padding: '3px 6px' }}
                          onClick={(e) => e.stopPropagation()}
                        />
                      ) : (
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{
                            fontSize: 13, fontWeight: 500, color: 'var(--text-primary)',
                            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                          }}>
                            {session.title}
                          </div>
                          <div style={{
                            fontSize: 10, color: 'var(--text-muted)',
                            fontFamily: 'var(--font-mono)', marginTop: 1,
                          }}>
                            {formatDate(session.updatedAt)}
                          </div>
                        </div>
                      )}

                      {/* Hover actions */}
                      <div
                        style={{ display: 'flex', gap: 1, opacity: 0.4 }}
                        onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; }}
                        onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.4'; }}
                      >
                        <button
                          className="btn-icon" title="Pin"
                          onClick={(e) => { e.stopPropagation(); handlePin(session.id, session.pinned); }}
                          style={{ padding: 1, fontSize: 11 }}
                        >
                          <svg width="10" height="10" viewBox="0 0 24 24" fill={session.pinned > 0 ? 'var(--accent)' : 'none'} stroke="currentColor" strokeWidth="2">
                            <path d="M16 12V4h1V2H7v2h1v8l-2 2v2h5.2v5h1.6v-5H18v-2l-2-2z" />
                          </svg>
                        </button>
                        <button
                          className="btn-icon" title="Rename"
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditingSessionId(session.id);
                            setEditTitle(session.title);
                          }}
                          style={{ padding: 1, fontSize: 11 }}
                        >
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                          </svg>
                        </button>
                        <button
                          className="btn-icon" title="Delete"
                          onClick={(e) => {
                            e.stopPropagation();
                            setDeletingSessionId(session.id);
                          }}
                          style={{ padding: 1, fontSize: 11, color: 'var(--error)' }}
                        >
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="3 6 5 6 21 6" />
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                          </svg>
                        </button>
                        <select
                          className="input"
                          value={session.folder || DEFAULT_FOLDER_ID}
                          onChange={(e) => { e.stopPropagation(); handleMoveToFolder(session.id, e.target.value); }}
                          onClick={(e) => e.stopPropagation()}
                          style={{ fontSize: 8, padding: '1px 1px', width: 44, fontFamily: 'var(--font-mono)' }}
                          title="Move to folder"
                        >
                          {folders.map((f) => (
                            <option key={f.id} value={f.id}>{f.name}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            );
          })
        )}

        {/* Manage folders */}
        <div style={{ padding: '8px 12px', borderTop: '1px solid var(--border)', fontSize: 11, color: 'var(--text-muted)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
            <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Manage folders
            </div>
          </div>
          {newFolderName === '_' ? (
            <div style={{ marginBottom: 4 }}>
              <input
                className="input"
                ref={folderInputRef}
                placeholder="New folder name"
                onBlur={() => setNewFolderName('')}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleAddFolder();
                  if (e.key === 'Escape') setNewFolderName('');
                }}
                style={{ padding: '2px 6px', fontSize: 10, fontFamily: 'var(--font-mono)', width: '100%' }}
              />
            </div>
          ) : (
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => setNewFolderName('_')}
              style={{ fontSize: 10, width: '100%', justifyContent: 'center', padding: '3px' }}
            >
              + New Folder
            </button>
          )}
          {folders.filter((f) => f.id !== DEFAULT_FOLDER_ID).map((f) => {
            const count = sessions.filter((s) => s.folder === f.id).length;
            return (
              <div key={f.id} style={{
                display: 'flex', alignItems: 'center', gap: 4,
                padding: '2px 0', fontSize: 10,
              }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: f.color || 'var(--text-muted)', flexShrink: 0 }} />
                {editingFolderId === f.id ? (
                  <input
                    className="input" autoFocus
                    value={editingFolderName}
                    onChange={(e) => setEditingFolderName(e.target.value)}
                    onBlur={() => handleRenameFolder(f.id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleRenameFolder(f.id);
                      if (e.key === 'Escape') setEditingFolderId(null);
                    }}
                    style={{ padding: '1px 4px', fontSize: 9, flex: 1 }}
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <span
                    onDoubleClick={() => { setEditingFolderId(f.id); setEditingFolderName(f.name); }}
                    style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', cursor: 'default' }}
                  >
                    {f.name}
                  </span>
                )}
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{count}</span>
                <button
                  className="btn-icon"
                  title={`Delete ${f.name}`}
                  onClick={(e) => { e.stopPropagation(); setDeletingFolderId(f.id); }}
                  style={{ padding: 0, fontSize: 10, color: 'var(--error)', opacity: 0.5 }}
                >
                  ×
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Delete session confirmation */}
      {deletingSessionId && (
        <div className="confirm-overlay" onClick={() => setDeletingSessionId(null)}>
          <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
            <p>Delete this chat? This action cannot be undone.</p>
            <div className="confirm-actions">
              <button className="btn btn-secondary btn-sm" onClick={() => setDeletingSessionId(null)}>Cancel</button>
              <button className="btn btn-danger btn-sm" onClick={() => handleDelete(deletingSessionId)}>Delete</button>
            </div>
          </div>
        </div>
      )}

      {/* Delete folder confirmation */}
      {deletingFolderId && (
        <div className="confirm-overlay" onClick={() => { setDeletingFolderId(null); setDeleteFolderChoice(null); }}>
          <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
            <p style={{ margin: '0 0 10px' }}>
              Delete <strong>{folderName(deletingFolderId)}</strong>?
            </p>
            {(sessions.filter((s) => s.folder === deletingFolderId).length > 0) ? (
              <>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '0 0 12px' }}>
                  This folder contains {sessions.filter((s) => s.folder === deletingFolderId).length} chat(s).
                </p>
                <div className="confirm-actions" style={{ flexDirection: 'column', gap: 6 }}>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => handleDeleteFolderConfirm(deletingFolderId, 'keep')}
                    style={{ width: '100%' }}
                  >
                    Move chats to Defaults, delete folder
                  </button>
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => handleDeleteFolderConfirm(deletingFolderId, 'delete')}
                    style={{ width: '100%' }}
                  >
                    Delete folder and all chats
                  </button>
                  <button className="btn btn-ghost btn-sm" onClick={() => { setDeletingFolderId(null); setDeleteFolderChoice(null); }} style={{ width: '100%' }}>
                    Cancel
                  </button>
                </div>
              </>
            ) : (
              <div className="confirm-actions">
                <button className="btn btn-secondary btn-sm" onClick={() => { setDeletingFolderId(null); setDeleteFolderChoice(null); }}>Cancel</button>
                <button className="btn btn-danger btn-sm" onClick={() => handleDeleteFolderConfirm(deletingFolderId, 'delete')}>Delete</button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
