import { create } from 'zustand';
import type { ChatSession, ChatMessage, ChatFolder, AppSettings, ModelInfo, ServerStatus, GpuStats, SystemResources, ChatMessageStats } from './types';
import type { ServerStatusResponse } from '../api/index';

interface AppState {
  // Sessions
  sessions: ChatSession[];
  currentSessionId: string | null;
  messages: ChatMessage[];
  folders: ChatFolder[];

  // Models
  models: ModelInfo[];
  selectedModelId: string | null;
  scanningModels: boolean;

  // Server
  serverStatus: ServerStatus;
  serverError: string | null;
  serverStage: string | null;
  serverLogs: string[];
  gpuStats: GpuStats[];
  sysResources: SystemResources | null;
  vllmVersion: string;
  serverInfo: ServerStatusResponse | null;

  // UI
  settings: AppSettings | null;
  settingsOpen: boolean;
  configPanelOpen: boolean;
  releaseNotesOpen: boolean;
  aboutOpen: boolean;
  theme: 'dark' | 'light' | 'system';
  searchQuery: string;
  isStreaming: boolean;

  // Actions
  setSessions: (sessions: ChatSession[]) => void;
  setCurrentSessionId: (id: string | null) => void;
  setMessages: (messages: ChatMessage[]) => void;
  addMessage: (msg: ChatMessage) => void;
  updateMessage: (id: string, content: string) => void;
  updateMessageStats: (id: string, stats: ChatMessageStats) => void;
  removeMessage: (id: string) => void;
  setFolders: (folders: ChatFolder[]) => void;
  setModels: (models: ModelInfo[]) => void;
  setSelectedModelId: (id: string | null) => void;
  setScanningModels: (v: boolean) => void;
  setServerStatus: (status: ServerStatus, error?: string | null, stage?: string | null) => void;
  appendServerLog: (log: string) => void;
  setServerLogs: (logs: string[]) => void;
  setGpuStats: (stats: GpuStats[]) => void;
  setSysResources: (r: SystemResources) => void;
  setVllmVersion: (v: string) => void;
  setServerInfo: (info: ServerStatusResponse) => void;
  setSettings: (s: AppSettings) => void;
  setSettingsOpen: (v: boolean) => void;
  setConfigPanelOpen: (v: boolean) => void;
  setReleaseNotesOpen: (v: boolean) => void;
  setAboutOpen: (v: boolean) => void;
  setTheme: (t: 'dark' | 'light' | 'system') => void;
  setSearchQuery: (q: string) => void;
  setIsStreaming: (v: boolean) => void;
  updateSession: (id: string, updates: Partial<ChatSession>) => void;
  removeSession: (id: string) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  sessions: [],
  currentSessionId: null,
  messages: [],
  folders: [],
  models: [],
  selectedModelId: null,
  scanningModels: false,
  serverStatus: 'stopped',
  serverError: null,
  serverStage: null,
  serverLogs: [],
  gpuStats: [],
  sysResources: null,
  vllmVersion: '',
  serverInfo: null,
  settings: null,
  settingsOpen: false,
  configPanelOpen: true,
  releaseNotesOpen: false,
  aboutOpen: false,
  theme: 'dark',
  searchQuery: '',
  isStreaming: false,

  setSessions: (sessions) => set({ sessions }),
  setCurrentSessionId: (id) => set({ currentSessionId: id, messages: [] }),
  setMessages: (messages) => set({ messages }),
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  updateMessage: (id, content) => set((s) => ({
    messages: s.messages.map((m) => (m.id === id ? { ...m, content } : m)),
  })),
  updateMessageStats: (id, stats) => set((s) => ({
    messages: s.messages.map((m) => (m.id === id ? { ...m, chatStats: stats } : m)),
  })),
  removeMessage: (id) => set((s) => ({
    messages: s.messages.filter((m) => m.id !== id),
  })),
  setFolders: (folders) => set({ folders }),
  setModels: (models) => set({ models }),
  setSelectedModelId: (id) => set({ selectedModelId: id }),
  setScanningModels: (v) => set({ scanningModels: v }),
  setServerStatus: (status, error, stage) => set({
    serverStatus: status,
    serverError: error ?? null,
    serverStage: stage === undefined ? get().serverStage : (stage ?? null),
  }),
  appendServerLog: (log) => set((s) => {
    const logs = [...s.serverLogs, log];
    if (logs.length > 500) return { serverLogs: logs.slice(-500) };
    return { serverLogs: logs };
  }),
  setServerLogs: (logs) => set({ serverLogs: logs }),
  setGpuStats: (stats) => set({ gpuStats: stats }),
  setSysResources: (r) => set({ sysResources: r }),
  setVllmVersion: (v) => set({ vllmVersion: v }),
  setServerInfo: (info) => set({ serverInfo: info }),
  setSettings: (s) => set({ settings: s }),
  setSettingsOpen: (v) => set({ settingsOpen: v }),
  setConfigPanelOpen: (v) => set({ configPanelOpen: v }),
  setReleaseNotesOpen: (v) => set({ releaseNotesOpen: v }),
  setAboutOpen: (v) => set({ aboutOpen: v }),
  setTheme: (t) => set({ theme: t }),
  setSearchQuery: (q) => set({ searchQuery: q }),
  setIsStreaming: (v) => set({ isStreaming: v }),
  updateSession: (id, updates) => set((s) => ({
    sessions: s.sessions.map((ss) => (ss.id === id ? { ...ss, ...updates } : ss)),
  })),
  removeSession: (id) => set((s) => ({
    sessions: s.sessions.filter((ss) => ss.id !== id),
    currentSessionId: s.currentSessionId === id ? null : s.currentSessionId,
  })),
}));
