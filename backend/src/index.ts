import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cors from 'cors';
import http from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { fileURLToPath } from 'url';
import multer from 'multer';
import { v4 as uuid } from 'uuid';

import { initDatabase, getSettings } from './services/database.js';
import { setOnLog, setOnStatusChange, getGpuStats, getStatus, getLogs, attachToExisting } from './services/vllmManager.js';
import modelsRouter from './routes/models.js';
import chatRouter from './routes/chat.js';
import serverRouter from './routes/server.js';
import historyRouter from './routes/history.js';
import settingsRouter from './routes/settings.js';
import huggingfaceRouter from './routes/huggingface.js';
import { setOnDownloadProgress } from './services/hfDownloader.js';
import { ensureGgufArchPatches } from './services/vllmIntrospect.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = parseInt(process.env.PORT || '3333', 10);
const HOST = process.env.HOST || '127.0.0.1';

initDatabase();
ensureGgufArchPatches();

const app = express();

app.use(cors({
  origin: true,
  credentials: true,
}));

app.use(express.json({ limit: '1mb' }));

const dataDir = path.resolve(process.env.DATA_DIR || './data');
const attachmentsDir = path.join(dataDir, 'attachments');

if (!fs.existsSync(attachmentsDir)) {
  fs.mkdirSync(attachmentsDir, { recursive: true });
}

const upload = multer({
  dest: attachmentsDir,
  limits: {
    fileSize: 100 * 1024 * 1024,
    files: 10,
  },
  fileFilter: (_req, file, cb) => {
    const allowedMimes = [
      'image/jpeg', 'image/png', 'image/webp', 'image/gif',
      'application/pdf',
    ];
    if (allowedMimes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Unsupported file type. Only JPG, PNG, WebP, GIF, and PDF are allowed.'));
    }
  },
});

// Serve frontend static files
const frontendDist = path.resolve(__dirname, '../../frontend/dist');
if (fs.existsSync(frontendDist)) {
  app.use(express.static(frontendDist));
}

// ---- API Routes ----
app.use('/api/models', modelsRouter);
app.use('/api/chat', chatRouter);
app.use('/api/server', serverRouter);
app.use('/api', historyRouter);
app.use('/api/settings', settingsRouter);
app.use('/api/huggingface', huggingfaceRouter);

// ---- Attachment upload ----
app.post('/api/attachments/upload', upload.array('files', 10), async (req, res) => {
  try {
    const files = req.files as Express.Multer.File[];
    if (!files || files.length === 0) {
      res.status(400).json({ error: 'No files uploaded' });
      return;
    }

    const sessionId = req.body.sessionId || 'shared';

    const sessionDir = path.join(attachmentsDir, sessionId);
    if (!fs.existsSync(sessionDir)) {
      fs.mkdirSync(sessionDir, { recursive: true });
    }

    const results = [];

    for (const file of files) {
      const ext = path.extname(file.originalname);
      const destName = `${uuid()}${ext}`;
      const destPath = path.join(sessionDir, destName);

      fs.renameSync(file.path, destPath);

      const isPdf = file.mimetype === 'application/pdf';
      const isImage = file.mimetype.startsWith('image/');
      const fileType = isPdf ? 'pdf' : 'image';

      if (isImage && file.size > 20 * 1024 * 1024) {
        res.status(400).json({ error: `Image ${file.originalname} exceeds 20MB limit` });
        return;
      }
      if (isPdf && file.size > 50 * 1024 * 1024) {
        res.status(400).json({ error: `PDF ${file.originalname} exceeds 50MB limit` });
        return;
      }

      // Extract content for inline inclusion
      let textContent: string | null = null;
      try {
        if (isPdf) {
          // Extract text from PDF
          const pdfParse = (await import('pdf-parse')).default;
          const dataBuffer = fs.readFileSync(destPath);
          const pdfData = await pdfParse(dataBuffer);
          textContent = pdfData.text.slice(0, 50000); // cap at 50K chars
        } else if (isImage) {
          // Encode image as base64 data URL
          const imgBuffer = fs.readFileSync(destPath);
          textContent = `data:${file.mimetype};base64,${imgBuffer.toString('base64')}`;
        }
      } catch (e) {
        console.warn(`Failed to extract content from ${file.originalname}:`, e);
      }

      results.push({
        id: destName.replace(ext, ''),
        name: file.originalname,
        type: fileType,
        mimeType: file.mimetype,
        sizeBytes: file.size,
        path: destPath,
        content: textContent,
      });
    }

    res.json({ attachments: results });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

// ---- System GPU info ----
app.get('/api/system/gpu', (_req, res) => {
  try {
    const stats = getGpuStats();
    res.json(stats);
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

// ---- System resources ----
app.get('/api/system/resources', (_req, res) => {
  try {
    const gpus = getGpuStats();

    const totalRam = os.totalmem();
    const freeRam = os.freemem();
    const usedRam = totalRam - freeRam;
    const cpuUsage = os.loadavg()[0] * 100 / os.cpus().length;

    res.json({
      gpus,
      ramTotalBytes: totalRam,
      ramUsedBytes: usedRam,
      ramFreeBytes: freeRam,
      cpuPercent: Math.min(Math.round(cpuUsage), 100),
      timestamp: Date.now(),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

// ---- SPA fallback ----
if (fs.existsSync(frontendDist)) {
  app.get('*', (_req, res) => {
    res.sendFile(path.join(frontendDist, 'index.html'));
  });
}

// ---- HTTP & WebSocket Server ----
const server = http.createServer(app);

const wss = new WebSocketServer({ server, path: '/ws' });

wss.on('connection', (ws: WebSocket) => {
  console.log('WebSocket client connected');

  // Send current status on connect
  try {
    ws.send(JSON.stringify({
      type: 'server_status',
      data: { status: getStatus() },
      timestamp: Date.now(),
    }));
  } catch { /* ignore */ }

  ws.on('close', () => {
    console.log('WebSocket client disconnected');
  });

  ws.on('error', (err) => {
    console.error('WebSocket error:', err.message);
  });
});

// Broadcast to all connected clients
function broadcast(msgType: string, data: unknown): void {
  const message = JSON.stringify({
    type: msgType,
    data,
    timestamp: Date.now(),
  });

  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      try {
        client.send(message);
      } catch { /* ignore */ }
    }
  });
}

// ---- vLLM Manager callbacks ----
setOnLog((line: string) => {
  broadcast('log', { line });
});

setOnStatusChange((status: string, error?: string, stage?: string) => {
  broadcast('server_status', { status, error, stage });
});

// ---- HF Download Progress ----
setOnDownloadProgress((state) => {
  broadcast('hf_download_progress', state);
});

// ---- GPU Stats Polling ----
let gpuPollInterval: ReturnType<typeof setInterval> | null = null;

function startGpuBroadcast(): void {
  if (gpuPollInterval) return;
  gpuPollInterval = setInterval(() => {
    const stats = getGpuStats();
    if (stats.length > 0) {
      broadcast('gpu_stats', stats);
    }
  }, 3000);
}

startGpuBroadcast();

// ---- Attach to existing vLLM ----
attachToExisting();

// ---- Start ----
server.listen(PORT, HOST, () => {
  console.log(`VLLM Studio backend running at http://${HOST}:${PORT}`);
  console.log(`WebSocket available at ws://${HOST}:${PORT}/ws`);
  console.log(`Data directory: ${dataDir}`);
});

// ---- Graceful shutdown ----
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  server.close();
  if (gpuPollInterval) clearInterval(gpuPollInterval);
  process.exit(0);
});

process.on('SIGTERM', () => {
  server.close();
  if (gpuPollInterval) clearInterval(gpuPollInterval);
  process.exit(0);
});
