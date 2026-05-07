import type { ModelInfo, ChatSession, ChatMessage, ChatFolder, AttachmentMeta, AppSettings, GpuStats, SystemResources, PreflightResponse, ModelPrefs } from '../store/types';

const BASE_URL = '';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${path}`;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);

  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!res.ok) {
      const text = await res.text();
      let message: string;
      try {
        const parsed = JSON.parse(text);
        message = parsed.error || parsed.details || text;
      } catch {
        message = text || `HTTP ${res.status}`;
      }
      throw new Error(message);
    }

    return res.json();
  } finally {
    clearTimeout(timeout);
  }
}

export async function getModels(): Promise<{ models: ModelInfo[]; vllmVersion: string }> {
  return request('/api/models');
}

export async function scanModels(): Promise<{ models: ModelInfo[] }> {
  return request('/api/models/scan', { method: 'POST' });
}

export interface ServerStatusResponse {
  status: string;
  stage: string | null;
  error: string | null;
  port: number;
  vllmVersion: string;
  vllmArchsCount: number;
  transformersGgufArchs: string[];
  latestStable: string | null;
  latestNightly: string | null;
  isBehind: boolean;
  upgradeCommand: string | null;
}

export async function getServerStatus(): Promise<ServerStatusResponse> {
  return request('/api/server/status');
}

export async function startServer(config: Record<string, unknown>): Promise<{ status: string }> {
  return request('/api/server/start', { method: 'POST', body: JSON.stringify(config) });
}

export async function stopServer(): Promise<{ status: string }> {
  return request('/api/server/stop', { method: 'POST' });
}

export async function getServerLogs(lines?: number): Promise<{ logs: string[] }> {
  const q = lines ? `?lines=${lines}` : '';
  return request(`/api/server/logs${q}`);
}

export async function sendChatCompletion(data: Record<string, unknown>): Promise<Record<string, unknown>> {
  return request('/api/chat/completions', { method: 'POST', body: JSON.stringify(data) });
}

export interface ChatStats {
  promptTokens: number;
  outputTokens: number;
  ttftMs: number;
  totalTimeMs: number;
  tokensPerSec: number;
  totalTokens: number;
}

export type StreamChunk = { content: string } | { stats: ChatStats };

export async function* streamChatCompletion(
  data: Record<string, unknown>,
  controller: AbortController,
): AsyncGenerator<StreamChunk, void, void> {
  const url = `${BASE_URL}/api/chat/completions`;

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...data, stream: true }),
    signal: controller.signal,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') return;

        try {
          const parsed = JSON.parse(data);
          if (parsed.error) throw new Error(parsed.error);
          if (parsed.stats) yield { stats: parsed.stats as ChatStats };
          else if (parsed.content) yield { content: parsed.content as string };
        } catch (e) {
          if (e instanceof SyntaxError) continue;
          throw e;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export async function getSessions(folder?: string, search?: string): Promise<{ sessions: ChatSession[] }> {
  const params = new URLSearchParams();
  if (folder) params.set('folder', folder);
  if (search) params.set('search', search);
  const qs = params.toString();
  return request(`/api/sessions${qs ? `?${qs}` : ''}`);
}

export async function createSession(data: Record<string, unknown>): Promise<ChatSession> {
  return request('/api/sessions', { method: 'POST', body: JSON.stringify(data) });
}

export async function getSession(id: string): Promise<ChatSession> {
  return request(`/api/sessions/${id}`);
}

export async function updateSession(id: string, data: Record<string, unknown>): Promise<ChatSession> {
  return request(`/api/sessions/${id}`, { method: 'PATCH', body: JSON.stringify(data) });
}

export async function deleteSession(id: string): Promise<{ success: boolean }> {
  return request(`/api/sessions/${id}`, { method: 'DELETE' });
}

export async function uploadAttachments(sessionId: string, files: FileList): Promise<{ attachments: AttachmentMeta[] }> {
  const formData = new FormData();
  formData.append('sessionId', sessionId);
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);

  try {
    const res = await fetch(`${BASE_URL}/api/attachments/upload`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(text);
    }

    return res.json();
  } finally {
    clearTimeout(timeout);
  }
}

export async function getFolders(): Promise<{ folders: ChatFolder[] }> {
  return request('/api/folders');
}

export async function createFolder(data: { name: string; color?: string | null }): Promise<ChatFolder> {
  return request('/api/folders', { method: 'POST', body: JSON.stringify(data) });
}

export async function updateFolder(id: string, data: Record<string, unknown>): Promise<ChatFolder> {
  return request(`/api/folders/${id}`, { method: 'PATCH', body: JSON.stringify(data) });
}

export async function deleteFolder(id: string): Promise<{ success: boolean }> {
  return request(`/api/folders/${id}`, { method: 'DELETE' });
}

export async function getSettings(): Promise<AppSettings> {
  return request('/api/settings');
}

export async function updateSettings(data: Record<string, unknown>): Promise<AppSettings> {
  return request('/api/settings', { method: 'PATCH', body: JSON.stringify(data) });
}

export async function getGpuStats(): Promise<GpuStats[]> {
  return request('/api/system/gpu');
}

export async function getSystemResources(): Promise<SystemResources> {
  return request('/api/system/resources');
}

export async function deleteModel(modelPath: string): Promise<{ success: boolean; models: ModelInfo[] }> {
  return request('/api/models/delete', { method: 'POST', body: JSON.stringify({ modelPath }) });
}

export async function preflightModel(args: {
  modelPath: string;
  contextLength?: number;
  tensorParallelSize?: number;
  maxNumSeqs?: number;
  cpuOffloadGb?: number;
  kvCacheDtype?: 'auto' | 'fp16' | 'fp8' | 'fp8_e4m3' | 'fp8_e5m2';
  maxNumBatchedTokens?: number;
}): Promise<PreflightResponse> {
  return request('/api/models/preflight', { method: 'POST', body: JSON.stringify(args) });
}

export async function getModelPrefs(modelPath: string): Promise<{ prefs: ModelPrefs | null }> {
  const qs = new URLSearchParams({ modelPath }).toString();
  return request(`/api/models/prefs?${qs}`);
}

export async function setModelPrefsApi(prefs: Partial<ModelPrefs> & { modelPath: string }): Promise<{ success: boolean }> {
  return request('/api/models/prefs', { method: 'POST', body: JSON.stringify(prefs) });
}

export interface PatchProgress {
  active: boolean;
  modelPath?: string;
  bytesCopied: number;
  totalBytes: number;
  stage: 'starting' | 'copying' | 'complete' | 'error';
  error?: string;
  outputFile?: string;
  newArch?: string;
}

export async function patchGgufArch(modelPath: string, newArchitecture?: string): Promise<{ started: boolean; targetArch: string }> {
  return request('/api/models/patch-gguf-arch', {
    method: 'POST',
    body: JSON.stringify({ modelPath, newArchitecture }),
  });
}

export async function getPatchStatus(): Promise<PatchProgress> {
  return request('/api/models/patch-gguf-arch/status');
}
