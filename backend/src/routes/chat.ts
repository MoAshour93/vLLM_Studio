import { Router, Request, Response } from 'express';
import { ChatCompletionRequestSchema, InferenceParamsSchema, DEFAULT_INFERENCE_PARAMS, type InferenceParameters } from '../types/index.js';
import { addMessage, getMessages, getLastAssistantMessage } from '../services/database.js';
import { getStatus, SERVED_MODEL_NAME } from '../services/vllmManager.js';
import { getSettings } from '../services/database.js';

const router = Router();

const BASE_URL = `http://127.0.0.1:${process.env.VLLM_PORT || 8000}/v1`;

router.post('/completions', async (req: Request, res: Response) => {
  try {
    const parsed = ChatCompletionRequestSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid request', details: parsed.error.format() });
      return;
    }

    const { sessionId, messages, parameters, stream, attachments } = parsed.data;

    if (getStatus() !== 'running') {
      res.status(503).json({ error: 'vLLM server is not running. Start the server first.' });
      return;
    }

    const settings = getSettings();
    const vllmPort = settings.vllmPort;

    const inferenceParams: InferenceParameters = parameters ?? DEFAULT_INFERENCE_PARAMS;

    // Build messages array for vLLM. If the last user message has attachments,
    // construct a multi-modal content array (images as data URLs, PDFs as text prepended).
    let vllmMessages: Record<string, unknown>[];
    const attachmentData = attachments && attachments.length > 0 ? attachments : [];
    const hasAttachments = attachmentData.length > 0;

    if (hasAttachments) {
      vllmMessages = messages.map((m, idx) => {
        const isLastUser = m.role === 'user' && idx === messages.length - 1;
        if (!isLastUser) return { role: m.role, content: m.content };

        // Build multi-modal content array
        const contentParts: Record<string, unknown>[] = [];
        // User text comes first
        contentParts.push({ type: 'text', text: m.content });

        let pdfTexts: string[] = [];

        for (const att of attachmentData) {
          if (att.type === 'image' && att.content) {
            contentParts.push({
              type: 'image_url',
              image_url: { url: att.content },
            });
          } else if (att.type === 'pdf' && att.content) {
            pdfTexts.push(att.content);
          }
        }

        // Append PDF text after images
        if (pdfTexts.length > 0) {
          contentParts.push({
            type: 'text',
            text: '\n\n--- Document Content ---\n\n' + pdfTexts.join('\n\n---\n\n'),
          });
        }

        return { role: m.role, content: contentParts };
      });
    } else {
      vllmMessages = messages.map(m => ({ role: m.role, content: m.content }));
    }

    const body: Record<string, unknown> = {
      model: SERVED_MODEL_NAME,
      messages: vllmMessages,
      temperature: inferenceParams.temperature ?? 0.7,
      top_p: inferenceParams.topP ?? 0.9,
      max_tokens: inferenceParams.maxTokens ?? 1024,
      presence_penalty: inferenceParams.presencePenalty ?? 0,
      frequency_penalty: inferenceParams.frequencyPenalty ?? 0,
      stream: stream ?? false,
    };

    if (inferenceParams.topK !== undefined && inferenceParams.topK > 0) {
      body.top_k = inferenceParams.topK;
    }
    if (inferenceParams.repetitionPenalty !== undefined) {
      body.repetition_penalty = inferenceParams.repetitionPenalty;
    }
    if (inferenceParams.seed !== undefined && inferenceParams.seed >= 0) {
      body.seed = inferenceParams.seed;
    }
    if (inferenceParams.stop && inferenceParams.stop.length > 0) {
      body.stop = inferenceParams.stop;
    }

    const fetchUrl = `http://127.0.0.1:${vllmPort}/v1/chat/completions`;

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      const fetchResp = await fetch(fetchUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!fetchResp.ok) {
        const errText = await fetchResp.text();
        res.write(`data: ${JSON.stringify({ error: errText })}\n\n`);
        res.end();
        return;
      }

      saveUserMessage(sessionId, messages[messages.length - 1], inferenceParams, attachmentData);

      const reader = fetchResp.body?.getReader();
      if (!reader) {
        res.write(`data: ${JSON.stringify({ error: 'No response body' })}\n\n`);
        res.end();
        return;
      }

      const decoder = new TextDecoder();
      let fullContent = '';
      let finishReason: string | null = null;
      let tokenCount = 0;
      const startTime = Date.now();
      let firstTokenTime = 0;
      let promptTokens = 0;
      let completionTokens = 0;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6).trim();
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data);
              const delta = parsed.choices?.[0]?.delta?.content;
              if (delta) {
                if (tokenCount === 0) firstTokenTime = Date.now();
                tokenCount++;
                fullContent += delta;
                res.write(`data: ${JSON.stringify({ content: delta })}\n\n`);
              }
              if (parsed.choices?.[0]?.finish_reason) {
                finishReason = parsed.choices[0].finish_reason;
              }
              if (parsed.usage) {
                promptTokens = parsed.usage.prompt_tokens ?? 0;
                completionTokens = parsed.usage.completion_tokens ?? 0;
              }
            } catch { /* skip malformed line */ }
          }
        }
      } finally {
        reader.releaseLock();
      }

      if (fullContent) {
        const genTime = Date.now() - startTime;
        const ttft = firstTokenTime ? firstTokenTime - startTime : 0;
        const tps = genTime > 0 ? (tokenCount / (genTime / 1000)) : 0;

        const stats = {
          promptTokens: promptTokens || messages.reduce((acc, m) => acc + m.content.length / 4, 0),
          outputTokens: completionTokens || tokenCount,
          ttftMs: ttft,
          totalTimeMs: genTime,
          tokensPerSec: parseFloat(tps.toFixed(1)),
          totalTokens: tokenCount,
        };

        res.write(`data: ${JSON.stringify({ stats })}\n\n`);

        addMessage({
          sessionId,
          role: 'assistant',
          content: fullContent,
          modelId: 'current',
          finishReason,
          tokensUsed: tokenCount,
        });
      }

      res.write(`data: [DONE]\n\n`);
      res.end();

    } else {
      // Non-streaming
      saveUserMessage(sessionId, messages[messages.length - 1], inferenceParams, attachmentData);

      const fetchResp = await fetch(fetchUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!fetchResp.ok) {
        const errText = await fetchResp.text();
        res.status(502).json({ error: `vLLM error: ${errText}` });
        return;
      }

      const result = await fetchResp.json();
      const content = result.choices?.[0]?.message?.content ?? '';
      const finishReason = result.choices?.[0]?.finish_reason ?? null;
      const tokensUsed = result.usage?.completion_tokens ?? null;

      if (content) {
        addMessage({
          sessionId,
          role: 'assistant',
          content,
          modelId: 'current',
          tokensUsed,
          finishReason,
        });
      }

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Date.now(),
        model: SERVED_MODEL_NAME,
        choices: [{
          index: 0,
          message: { role: 'assistant', content },
          finish_reason: finishReason,
        }],
        usage: result.usage,
      });
    }

  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    if (!res.headersSent) {
      res.status(500).json({ error: msg });
    }
  }
});

function saveUserMessage(
  sessionId: string,
  message: { role: string; content: string },
  inferenceParams: Record<string, unknown>,
  attachments?: Array<{ type: string; mimeType: string; content?: string | null }> | null,
): void {
  const savedAttachments = attachments?.length
    ? attachments.map((a) => ({
        id: '', name: '', type: a.type as 'image' | 'pdf',
        mimeType: a.mimeType, sizeBytes: 0, path: '',
      }))
    : null;
  addMessage({
    sessionId,
    role: message.role as 'user',
    content: message.content,
    attachments: savedAttachments as any,
  });
}

export default router;
