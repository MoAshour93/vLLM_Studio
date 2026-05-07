import { useCallback, useRef } from 'react';
import { useAppStore } from '../store/index';
import { streamChatCompletion } from '../api/index';
import type { ChatStats } from '../api/index';
import type { AttachmentMeta } from '../store/types';
import { uuid } from '../utils/uuid';

const statsMap = new Map<string, ChatStats>();

export function useChat() {
  const store = useAppStore();
  const abortRef = useRef<AbortController | null>(null);

  const cancelStream = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const sendMessage = useCallback(async (content: string, attachments?: AttachmentMeta[]) => {
    const { currentSessionId, messages, setIsStreaming, addMessage, settings } = useAppStore.getState();
    if (!currentSessionId || !content.trim()) return null;

    const session = useAppStore.getState().sessions.find((s) => s.id === currentSessionId);
    const params = session?.parameters ?? undefined;

    const userMsgId = uuid();
    const assistantMsgId = uuid();

    addMessage({
      id: userMsgId,
      sessionId: currentSessionId,
      role: 'user',
      content: content.trim(),
      attachments: attachments ?? null,
      createdAt: Date.now(),
      tokensUsed: null,
      modelId: null,
      finishReason: null,
    });

    const assistantMsg = {
      id: assistantMsgId,
      sessionId: currentSessionId,
      role: 'assistant' as const,
      content: '',
      attachments: null,
      createdAt: Date.now(),
      tokensUsed: null,
      modelId: null,
      finishReason: null,
    };
    addMessage(assistantMsg);

    setIsStreaming(true);

    const chatMessages = [...messages, { role: 'user', content: content.trim() }];

    let fullContent = '';

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const streamData: Record<string, unknown> = {
        sessionId: currentSessionId,
        messages: chatMessages.map((m) => ({ role: m.role, content: m.content })),
        parameters: params,
      };

      // Include attachment content for multimodal models
      if (attachments && attachments.length > 0) {
        streamData.attachments = attachments
          .filter((a) => a.content)
          .map((a) => ({
            type: a.type,
            mimeType: a.mimeType,
            content: a.content,
          }));
      }

      const generator = streamChatCompletion(streamData, controller);

      for await (const chunk of generator) {
        if ('content' in chunk) {
          fullContent += chunk.content;
          const state = useAppStore.getState();
          state.updateMessage(assistantMsgId, fullContent);
        } else if ('stats' in chunk) {
          statsMap.set(assistantMsgId, chunk.stats);
          const state = useAppStore.getState();
          state.updateMessageStats(assistantMsgId, chunk.stats);
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        // User cancelled — leave partial content
      } else {
        const errorMsg = err instanceof Error ? err.message : 'Unknown error';
        const state = useAppStore.getState();
        state.updateMessage(assistantMsgId, `Error: ${errorMsg}`);
      }
    } finally {
      abortRef.current = null;
      setIsStreaming(false);
    }

    return fullContent;
  }, []);

  const regenerate = useCallback(async () => {
    const { messages, removeMessage } = useAppStore.getState();
    if (messages.length < 2) return;

    const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant');
    const lastUser = [...messages].reverse().find((m) => m.role === 'user');

    if (lastAssistant) removeMessage(lastAssistant.id);

    if (lastUser) {
      return sendMessage(lastUser.content, lastUser.attachments ?? undefined);
    }
  }, [sendMessage]);

  return { sendMessage, regenerate, cancelStream };
}

export function getChatStats(msgId: string): ChatStats | undefined {
  return statsMap.get(msgId);
}
