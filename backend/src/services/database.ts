import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';
import { v4 as uuid } from 'uuid';
import {
  ChatSession,
  ChatMessage,
  ChatFolder,
  AppSettings,
  DEFAULT_SETTINGS,
  InferenceParameters,
  DEFAULT_INFERENCE_PARAMS,
  AttachmentMeta,
} from '../types/index.js';

const SCHEMA_VERSION = 1;

let db: Database.Database;

export function getDbPath(): string {
  const dataDir = process.env.DATA_DIR || './data';
  return path.resolve(dataDir, 'vllm-studio.db');
}

export function initDatabase(): void {
  const dbPath = getDbPath();
  const dataDir = path.dirname(dbPath);

  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }

  db = new Database(dbPath);
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');

  runMigrations();

  ensureDefaultFolder();
  ensureDefaultSettings();
}

function runMigrations(): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS schema_version (
      version INTEGER PRIMARY KEY
    );
  `);

  const row = db.prepare('SELECT MAX(version) as version FROM schema_version').get() as { version: number | null };

  const currentVersion = row?.version ?? 0;

  if (currentVersion < 1) {
    db.exec(`
      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        model_id TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        folder TEXT DEFAULT 'default',
        pinned INTEGER DEFAULT 0,
        system_prompt TEXT,
        parameters TEXT
      );

      CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK(role IN ('user','assistant','system','tool')),
        content TEXT NOT NULL,
        attachments TEXT,
        created_at INTEGER NOT NULL,
        tokens_used INTEGER,
        model_id TEXT,
        finish_reason TEXT
      );

      CREATE TABLE IF NOT EXISTS folders (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        colour TEXT,
        created_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS model_cache (
        path TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        size_bytes INTEGER,
        architecture TEXT,
        quantization TEXT,
        context_length INTEGER,
        is_vision INTEGER DEFAULT 0,
        last_checked INTEGER NOT NULL,
        directory_mtime INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
      CREATE INDEX IF NOT EXISTS idx_sessions_folder ON sessions(folder);
      CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
    `);

    db.prepare('INSERT INTO schema_version (version) VALUES (?)').run(1);
  }

  if (currentVersion < 2) {
    db.exec(`
      CREATE TABLE IF NOT EXISTS model_prefs (
        path TEXT PRIMARY KEY,
        max_model_len INTEGER,
        gpu_memory_utilization REAL,
        quantization TEXT,
        dtype TEXT,
        tensor_parallel_size INTEGER,
        max_num_seqs INTEGER,
        additional_args TEXT,
        updated_at INTEGER NOT NULL
      );
    `);
    const addCol = (name: string, decl: string) => {
      try { db.exec(`ALTER TABLE model_cache ADD COLUMN ${name} ${decl}`); } catch { /* exists */ }
    };
    addCol('native_context_length', 'INTEGER');
    addCol('model_file', 'TEXT');
    addCol('format', 'TEXT');
    addCol('spec_json', 'TEXT');
    db.prepare('INSERT INTO schema_version (version) VALUES (?)').run(2);
  }

  if (currentVersion < 3) {
    const addCol = (name: string, decl: string) => {
      try { db.exec(`ALTER TABLE model_cache ADD COLUMN ${name} ${decl}`); } catch { /* exists */ }
    };
    addCol('gguf_architecture', 'TEXT');
    db.prepare('INSERT INTO schema_version (version) VALUES (?)').run(3);
  }
}

function ensureDefaultFolder(): void {
  const existing = db.prepare('SELECT id FROM folders WHERE id = ?').get('default');
  if (!existing) {
    db.prepare(`
      INSERT INTO folders (id, name, colour, created_at)
      VALUES (?, ?, ?, ?)
    `).run('default', 'Defaults', '#5B8CFF', Date.now());
  }
}

function ensureDefaultSettings(): void {
  const existing = db.prepare('SELECT key FROM settings WHERE key = ?').get('app_settings');
  if (!existing) {
    db.prepare('INSERT INTO settings (key, value) VALUES (?, ?)').run(
      'app_settings',
      JSON.stringify(DEFAULT_SETTINGS)
    );
  }
}

// ---- Sessions ----
export function getSessions(folder?: string, search?: string): ChatSession[] {
  let query = 'SELECT * FROM sessions';
  const conditions: string[] = [];
  const params: unknown[] = [];

  if (folder) {
    conditions.push('folder = ?');
    params.push(folder);
  }

  if (search) {
    conditions.push("(title LIKE ? OR id IN (SELECT DISTINCT session_id FROM messages WHERE content LIKE ?))");
    const term = `%${search}%`;
    params.push(term, term);
  }

  if (conditions.length > 0) {
    query += ' WHERE ' + conditions.join(' AND ');
  }

  query += ' ORDER BY pinned DESC, updated_at DESC';

  const rows = db.prepare(query).all(...params) as Array<Record<string, unknown>>;
  return rows.map(mapSession);
}

export function getSession(id: string): ChatSession | null {
  const row = db.prepare('SELECT * FROM sessions WHERE id = ?').get(id) as Record<string, unknown> | undefined;
  return row ? mapSession(row) : null;
}

export function createSession(data: {
  title?: string;
  modelId?: string | null;
  folder?: string;
  systemPrompt?: string | null;
  parameters?: InferenceParameters;
}): ChatSession {
  const id = uuid();
  const now = Date.now();
  const params = data.parameters ?? DEFAULT_INFERENCE_PARAMS;

  db.prepare(`
    INSERT INTO sessions (id, title, model_id, created_at, updated_at, folder, pinned, system_prompt, parameters)
    VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
  `).run(
    id,
    data.title ?? 'New Chat',
    data.modelId ?? null,
    now,
    now,
    data.folder ?? 'default',
    data.systemPrompt ?? null,
    JSON.stringify(params),
  );

  return getSession(id)!;
}

export function updateSession(id: string, updates: {
  title?: string;
  folder?: string;
  pinned?: number;
  modelId?: string | null;
  systemPrompt?: string | null;
  parameters?: InferenceParameters;
}): ChatSession | null {
  const sets: string[] = [];
  const params: unknown[] = [];

  if (updates.title !== undefined) {
    sets.push('title = ?');
    params.push(updates.title);
  }
  if (updates.folder !== undefined) {
    sets.push('folder = ?');
    params.push(updates.folder);
  }
  if (updates.pinned !== undefined) {
    sets.push('pinned = ?');
    params.push(updates.pinned);
  }
  if (updates.modelId !== undefined) {
    sets.push('model_id = ?');
    params.push(updates.modelId);
  }
  if (updates.systemPrompt !== undefined) {
    sets.push('system_prompt = ?');
    params.push(updates.systemPrompt);
  }
  if (updates.parameters !== undefined) {
    sets.push('parameters = ?');
    params.push(JSON.stringify(updates.parameters));
  }

  if (sets.length === 0) return getSession(id);

  sets.push('updated_at = ?');
  params.push(Date.now());
  params.push(id);

  db.prepare(`UPDATE sessions SET ${sets.join(', ')} WHERE id = ?`).run(...params);
  return getSession(id);
}

export function deleteSession(id: string): boolean {
  const result = db.prepare('DELETE FROM sessions WHERE id = ?').run(id);
  return result.changes > 0;
}

// ---- Messages ----
export function getMessages(sessionId: string, limit?: number, offset?: number): ChatMessage[] {
  const query = limit !== undefined
    ? 'SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC LIMIT ? OFFSET ?'
    : 'SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC';

  const params: unknown[] = [sessionId];
  if (limit !== undefined) {
    params.push(limit, offset ?? 0);
  }

  const rows = db.prepare(query).all(...params) as Array<Record<string, unknown>>;
  return rows.map(mapMessage);
}

export function addMessage(data: {
  sessionId: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  attachments?: AttachmentMeta[] | null;
  modelId?: string | null;
  tokensUsed?: number | null;
  finishReason?: string | null;
}): ChatMessage {
  const id = uuid();
  const now = Date.now();

  db.prepare(`
    INSERT INTO messages (id, session_id, role, content, attachments, created_at, tokens_used, model_id, finish_reason)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    id,
    data.sessionId,
    data.role,
    data.content,
    data.attachments ? JSON.stringify(data.attachments) : null,
    now,
    data.tokensUsed ?? null,
    data.modelId ?? null,
    data.finishReason ?? null,
  );

  db.prepare('UPDATE sessions SET updated_at = ? WHERE id = ?').run(now, data.sessionId);

  return getMessage(id)!;
}

export function getMessage(id: string): ChatMessage | null {
  const row = db.prepare('SELECT * FROM messages WHERE id = ?').get(id) as Record<string, unknown> | undefined;
  return row ? mapMessage(row) : null;
}

export function updateMessage(id: string, content: string): ChatMessage | null {
  db.prepare('UPDATE messages SET content = ? WHERE id = ?').run(content, id);
  return getMessage(id);
}

export function deleteMessage(id: string): boolean {
  const result = db.prepare('DELETE FROM messages WHERE id = ?').run(id);
  return result.changes > 0;
}

export function getLastAssistantMessage(sessionId: string): ChatMessage | null {
  const row = db.prepare(
    "SELECT * FROM messages WHERE session_id = ? AND role = 'assistant' ORDER BY created_at DESC LIMIT 1"
  ).get(sessionId) as Record<string, unknown> | undefined;
  return row ? mapMessage(row) : null;
}

export function getMessageCount(sessionId: string): number {
  const row = db.prepare('SELECT COUNT(*) as count FROM messages WHERE session_id = ?').get(sessionId) as { count: number };
  return row.count;
}

// ---- Folders ----
export function getFolders(): ChatFolder[] {
  const rows = db.prepare('SELECT * FROM folders ORDER BY created_at ASC').all() as Array<Record<string, unknown>>;
  return rows.map(mapFolder);
}

export function createFolder(data: { name: string; color?: string | null }): ChatFolder {
  const id = uuid();
  const now = Date.now();

  db.prepare('INSERT INTO folders (id, name, colour, created_at) VALUES (?, ?, ?, ?)').run(
    id, data.name, data.color ?? null, now
  );

  return {
    id,
    name: data.name,
    color: data.color ?? null,
    createdAt: now,
  };
}

export function updateFolder(id: string, updates: { name?: string; color?: string | null }): ChatFolder | null {
  const sets: string[] = [];
  const params: unknown[] = [];

  if (updates.name !== undefined) {
    sets.push('name = ?');
    params.push(updates.name);
  }
  if (updates.color !== undefined) {
    sets.push('colour = ?');
    params.push(updates.color);
  }

  if (sets.length === 0) {
    const row = db.prepare('SELECT * FROM folders WHERE id = ?').get(id) as Record<string, unknown> | undefined;
    return row ? mapFolder(row) : null;
  }

  params.push(id);
  db.prepare(`UPDATE folders SET ${sets.join(', ')} WHERE id = ?`).run(...params);

  const row = db.prepare('SELECT * FROM folders WHERE id = ?').get(id) as Record<string, unknown> | undefined;
  return row ? mapFolder(row) : null;
}

export function deleteFolder(id: string): boolean {
  if (id === 'default') return false;

  db.prepare("UPDATE sessions SET folder = 'default' WHERE folder = ?").run(id);
  const result = db.prepare('DELETE FROM folders WHERE id = ?').run(id);
  return result.changes > 0;
}

// ---- Settings ----
export function getSettings(): AppSettings {
  const row = db.prepare("SELECT value FROM settings WHERE key = 'app_settings'").get() as { value: string } | undefined;
  if (!row) return DEFAULT_SETTINGS;

  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(row.value) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function updateSettings(updates: Partial<AppSettings>): AppSettings {
  const current = getSettings();
  const merged = { ...current, ...updates };

  db.prepare("INSERT OR REPLACE INTO settings (key, value) VALUES ('app_settings', ?)").run(JSON.stringify(merged));

  return merged;
}

// ---- Model Cache ----
export function getCachedModels(): Array<Record<string, unknown>> {
  return db.prepare('SELECT * FROM model_cache ORDER BY name ASC').all() as Array<Record<string, unknown>>;
}

export function upsertModelCache(model: {
  path: string;
  name: string;
  sizeBytes: number;
  architecture: string;
  ggufArchitecture: string | null;
  quantization: string | null;
  contextLength: number;
  nativeContextLength: number;
  isVision: boolean;
  directoryMtime: number;
  modelFile: string | null;
  format: 'hf' | 'gguf';
  specJson: string;
}): void {
  db.prepare(`
    INSERT OR REPLACE INTO model_cache
      (path, name, size_bytes, architecture, gguf_architecture, quantization, context_length, native_context_length, is_vision, last_checked, directory_mtime, model_file, format, spec_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    model.path,
    model.name,
    model.sizeBytes,
    model.architecture,
    model.ggufArchitecture,
    model.quantization,
    model.contextLength,
    model.nativeContextLength,
    model.isVision ? 1 : 0,
    Date.now(),
    model.directoryMtime,
    model.modelFile,
    model.format,
    model.specJson,
  );
}

// ---- Model preferences (per-model load defaults) ----
export interface ModelPrefs {
  path: string;
  maxModelLen: number | null;
  gpuMemoryUtilization: number | null;
  quantization: string | null;
  dtype: string | null;
  tensorParallelSize: number | null;
  maxNumSeqs: number | null;
  additionalArgs: string[];
}

export function getModelPrefs(modelPath: string): ModelPrefs | null {
  const row = db.prepare('SELECT * FROM model_prefs WHERE path = ?').get(modelPath) as Record<string, unknown> | undefined;
  if (!row) return null;
  let additionalArgs: string[] = [];
  try {
    if (row.additional_args) additionalArgs = JSON.parse(row.additional_args as string);
  } catch { /* ignore */ }
  return {
    path: row.path as string,
    maxModelLen: (row.max_model_len as number) ?? null,
    gpuMemoryUtilization: (row.gpu_memory_utilization as number) ?? null,
    quantization: (row.quantization as string) ?? null,
    dtype: (row.dtype as string) ?? null,
    tensorParallelSize: (row.tensor_parallel_size as number) ?? null,
    maxNumSeqs: (row.max_num_seqs as number) ?? null,
    additionalArgs,
  };
}

export function setModelPrefs(prefs: ModelPrefs): void {
  db.prepare(`
    INSERT OR REPLACE INTO model_prefs
      (path, max_model_len, gpu_memory_utilization, quantization, dtype, tensor_parallel_size, max_num_seqs, additional_args, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    prefs.path,
    prefs.maxModelLen,
    prefs.gpuMemoryUtilization,
    prefs.quantization,
    prefs.dtype,
    prefs.tensorParallelSize,
    prefs.maxNumSeqs,
    JSON.stringify(prefs.additionalArgs ?? []),
    Date.now(),
  );
}

export function clearModelCache(): void {
  db.prepare('DELETE FROM model_cache').run();
}

export function getDb(): Database.Database {
  return db;
}

// ---- Mappers ----
function mapSession(row: Record<string, unknown>): ChatSession {
  let params: InferenceParameters = DEFAULT_INFERENCE_PARAMS;
  try {
    if (row.parameters) {
      params = { ...DEFAULT_INFERENCE_PARAMS, ...JSON.parse(row.parameters as string) };
    }
  } catch { /* keep defaults */ }

  return {
    id: row.id as string,
    title: row.title as string,
    modelId: (row.model_id as string) ?? null,
    createdAt: row.created_at as number,
    updatedAt: row.updated_at as number,
    folder: (row.folder as string) ?? 'default',
    pinned: (row.pinned as number) ?? 0,
    systemPrompt: (row.system_prompt as string) ?? null,
    parameters: params,
  };
}

function mapMessage(row: Record<string, unknown>): ChatMessage {
  let attachments: AttachmentMeta[] | null = null;
  try {
    if (row.attachments) {
      attachments = JSON.parse(row.attachments as string);
    }
  } catch { /* keep null */ }

  return {
    id: row.id as string,
    sessionId: row.session_id as string,
    role: row.role as ChatMessage['role'],
    content: row.content as string,
    attachments,
    createdAt: row.created_at as number,
    tokensUsed: (row.tokens_used as number) ?? null,
    modelId: (row.model_id as string) ?? null,
    finishReason: (row.finish_reason as string) ?? null,
  };
}

function mapFolder(row: Record<string, unknown>): ChatFolder {
  return {
    id: row.id as string,
    name: row.name as string,
    color: (row.colour as string) ?? null,
    createdAt: row.created_at as number,
  };
}
