import { Router, Request, Response } from 'express';
import { CreateSessionSchema, UpdateSessionSchema, AddMessageSchema, CreateFolderSchema, UpdateFolderSchema } from '../types/index.js';
import {
  getSessions,
  getSession,
  createSession,
  updateSession,
  deleteSession,
  getMessages,
  addMessage,
  getMessage,
  updateMessage,
  deleteMessage,
  getLastAssistantMessage,
  getMessageCount,
  getFolders,
  createFolder,
  updateFolder,
  deleteFolder,
} from '../services/database.js';

const router = Router();

function param(req: Request, name: string): string {
  const val = req.params[name];
  return Array.isArray(val) ? val[0] : val;
}

router.get('/sessions', (req: Request, res: Response) => {
  try {
    const folder = req.query.folder as string | undefined;
    const search = req.query.search as string | undefined;
    const sessions = getSessions(folder, search);
    res.json({ sessions });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.post('/sessions', (req: Request, res: Response) => {
  try {
    const parsed = CreateSessionSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid session data', details: parsed.error.format() });
      return;
    }

    const session = createSession(parsed.data);
    res.status(201).json(session);
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.get('/sessions/:id', (req: Request, res: Response) => {
  try {
    const id = param(req, 'id');
    const session = getSession(id);
    if (!session) {
      res.status(404).json({ error: 'Session not found' });
      return;
    }

    const messages = getMessages(session.id);
    const messageCount = getMessageCount(session.id);

    res.json({ ...session, messages, messageCount });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.patch('/sessions/:id', (req: Request, res: Response) => {
  try {
    const id = param(req, 'id');
    const parsed = UpdateSessionSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid update data', details: parsed.error.format() });
      return;
    }

    const session = updateSession(id, parsed.data);
    if (!session) {
      res.status(404).json({ error: 'Session not found' });
      return;
    }

    res.json(session);
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.delete('/sessions/:id', (req: Request, res: Response) => {
  try {
    const id = param(req, 'id');
    const result = deleteSession(id);
    if (!result) {
      res.status(404).json({ error: 'Session not found' });
      return;
    }
    res.json({ success: true });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.get('/sessions/:id/messages', (req: Request, res: Response) => {
  try {
    const id = param(req, 'id');
    const limit = req.query.limit ? parseInt(req.query.limit as string, 10) : undefined;
    const offset = req.query.offset ? parseInt(req.query.offset as string, 10) : undefined;
    const messages = getMessages(id, limit, offset);
    res.json({ messages });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.post('/sessions/:id/messages', (req: Request, res: Response) => {
  try {
    const id = param(req, 'id');
    const parsed = AddMessageSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid message data', details: parsed.error.format() });
      return;
    }

    const message = addMessage({
      sessionId: id,
      ...parsed.data,
    });

    res.status(201).json(message);
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.get('/folders', (_req: Request, res: Response) => {
  try {
    const folders = getFolders();
    res.json({ folders });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.post('/folders', (req: Request, res: Response) => {
  try {
    const parsed = CreateFolderSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid folder data', details: parsed.error.format() });
      return;
    }

    const folder = createFolder(parsed.data);
    res.status(201).json(folder);
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.patch('/folders/:id', (req: Request, res: Response) => {
  try {
    const id = param(req, 'id');
    const parsed = UpdateFolderSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid folder data', details: parsed.error.format() });
      return;
    }

    const folder = updateFolder(id, parsed.data);
    if (!folder) {
      res.status(404).json({ error: 'Folder not found' });
      return;
    }

    res.json(folder);
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.delete('/folders/:id', (req: Request, res: Response) => {
  try {
    const id = param(req, 'id');
    const result = deleteFolder(id);
    if (!result) {
      res.status(400).json({ error: 'Cannot delete default folder or folder not found' });
      return;
    }
    res.json({ success: true });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

export default router;
