import { Router, Request, Response } from 'express';
import { SettingsUpdateSchema } from '../types/index.js';
import { getSettings, updateSettings } from '../services/database.js';

const router = Router();

router.get('/', (_req: Request, res: Response) => {
  try {
    const settings = getSettings();
    res.json(settings);
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.patch('/', (req: Request, res: Response) => {
  try {
    const parsed = SettingsUpdateSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid settings', details: parsed.error.format() });
      return;
    }

    const settings = updateSettings(parsed.data);
    res.json(settings);
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

export default router;
