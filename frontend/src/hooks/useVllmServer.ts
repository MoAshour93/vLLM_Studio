import { useCallback, useEffect, useRef } from 'react';
import { useAppStore } from '../store/index';
import { getServerStatus, getServerLogs, getGpuStats, startServer, stopServer } from '../api/index';
import type { VllmServerConfig } from '../store/types';

const MAX_RECONNECT_ATTEMPTS = 5;

export function useVllmServer() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>();
  const reconnectDelayRef = useRef(1000);
  const reconnectAttemptsRef = useRef(0);
  const mountedRef = useRef(true);
  const intentionalCloseRef = useRef(false);

  const connectWs = useCallback(() => {
    if (!mountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) {
          ws.close();
          return;
        }
        reconnectDelayRef.current = 1000;
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const msg = JSON.parse(event.data);
          const store = useAppStore.getState();
          switch (msg.type) {
            case 'server_status':
              store.setServerStatus(msg.data.status, msg.data.error, msg.data.stage);
              break;
            case 'log':
              store.appendServerLog(msg.data.line);
              break;
            case 'gpu_stats':
              store.setGpuStats(msg.data);
              break;
            case 'hf_download_progress':
              break;
            case 'error':
              store.appendServerLog(`[Error] ${JSON.stringify(msg.data)}`);
              break;
          }
        } catch { /* ignore */ }
      };

      ws.onclose = () => {
        wsRef.current = null;
        if (!mountedRef.current || intentionalCloseRef.current) return;

        if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttemptsRef.current++;
          const delay = reconnectDelayRef.current;
          reconnectDelayRef.current = Math.min(delay * 2, 30000);
          reconnectTimeoutRef.current = setTimeout(connectWs, delay);
        }
      };

      ws.onerror = () => {
        // Let onclose handle cleanup
      };
    } catch {
      if (!mountedRef.current) return;
      if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttemptsRef.current++;
        reconnectTimeoutRef.current = setTimeout(connectWs, 5000);
      }
    }
  }, []); // No dependencies — uses useAppStore.getState() directly

  useEffect(() => {
    mountedRef.current = true;
    intentionalCloseRef.current = false;
    connectWs();

    return () => {
      mountedRef.current = false;
      intentionalCloseRef.current = true;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = undefined;
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []); // Run once on mount

  const fetchStatus = useCallback(async () => {
    try {
      const status = await getServerStatus();
      const store = useAppStore.getState();
      store.setServerStatus(status.status as 'stopped' | 'starting' | 'running' | 'error', status.error, status.stage ?? null);
      store.setVllmVersion(status.vllmVersion);
      store.setServerInfo(status);
    } catch {
      // ignore
    }
  }, []);

  const fetchLogs = useCallback(async () => {
    try {
      const { logs } = await getServerLogs(200);
      useAppStore.getState().setServerLogs(logs);
    } catch {
      // ignore
    }
  }, []);

  const fetchGpuStats = useCallback(async () => {
    try {
      const stats = await getGpuStats();
      useAppStore.getState().setGpuStats(stats);
    } catch {
      // ignore
    }
  }, []);

  const start = useCallback(async (config: VllmServerConfig) => {
    try {
      useAppStore.getState().setServerStatus('starting', null, 'spawning');
      await startServer({ ...config });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to start server';
      useAppStore.getState().setServerStatus('error', msg, null);
      throw err;
    }
  }, []);

  const stop = useCallback(async () => {
    try {
      await stopServer();
      useAppStore.getState().setServerStatus('stopped');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to stop server';
      useAppStore.getState().setServerStatus('error', msg);
    }
  }, []);

  return { fetchStatus, fetchLogs, fetchGpuStats, start, stop };
}
