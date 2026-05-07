import { Router, Request, Response } from 'express';
import { execSync } from 'child_process';
import { startDownload, cancelDownload, getDownloadState, setOnDownloadProgress } from '../services/hfDownloader.js';
import { getSettings } from '../services/database.js';
import { detectSupport, listSupportedFamilies, SUPPORTED_ARCHS } from '../services/supportedArchitectures.js';

const router = Router();

interface HfModelResult {
  id: string;
  author: string;
  modelId: string;
  sha: string;
  lastModified: string;
  tags: string[];
  pipelineTag: string;
  downloads: number;
  likes: number;
  libraryName: string;
  createdAt: string;
  private: boolean;
  gated: boolean;
  cardData: Record<string, unknown>;
  config?: {
    architectures?: string[];
    model_type?: string;
  };
  siblings: Array<{
    rfilename: string;
    size?: number;
  }>;
}

router.get('/supported-families', (_req: Request, res: Response) => {
  res.json({ families: listSupportedFamilies(), archs: SUPPORTED_ARCHS });
});

router.get('/search', async (req: Request, res: Response) => {
  try {
    const query = (req.query.query as string) || '';
    const limit = parseInt((req.query.limit as string) || '30', 10);
    const page = parseInt((req.query.page as string) || '0', 10);
    const sort = (req.query.sort as string) || 'downloads';
    const vllmOnly = (req.query.vllmOnly as string) !== 'false'; // default ON

    const params = new URLSearchParams({
      search: query,
      sort,
      direction: '-1',
      limit: String(Math.min(limit, 100)),
      full: 'true',
    });

    const url = `https://huggingface.co/api/models?${params.toString()}`;

    const response = await fetch(url, {
      headers: {
        'Accept': 'application/json',
        'User-Agent': 'VLLM-Studio/1.0',
      },
    });

    if (!response.ok) {
      res.status(response.status).json({ error: `HuggingFace API returned ${response.status}` });
      return;
    }

    const data = await response.json() as HfModelResult[];

    // Filter for text-generation models and parse quantization info
    const results = data
      .filter((m) => {
        if (m.private || m.gated) return false;
        const pipeline = m.pipelineTag || '';
        const tags = (m.tags || []).join(' ').toLowerCase();
        const relevant = pipeline.includes('text-generation') ||
          pipeline.includes('conversational') ||
          tags.includes('llm') ||
          tags.includes('chat') ||
          tags.includes('text-generation') ||
          tags.includes('transformers');
        return relevant;
      })
      .map((m) => {
        const siblings = m.siblings || [];
        const modelFiles = siblings.filter((s) =>
          s.rfilename.endsWith('.safetensors') ||
          s.rfilename.endsWith('.bin') ||
          s.rfilename.endsWith('.pth') ||
          s.rfilename.endsWith('.pt')
        );

        const quantizations = detectAvailableQuantizations(siblings);
        const isGguf = quantizations.includes('GGUF');
        const totalSize = siblings.reduce((sum, s) => sum + (s.size || 0), 0);
        const hasConfig = siblings.some((s) => s.rfilename === 'config.json');
        const hasTokenizer = siblings.some((s) =>
          s.rfilename === 'tokenizer.json' ||
          s.rfilename === 'tokenizer_config.json'
        );

        // Detect vLLM support from HF metadata (architecture if present, else model_type, else tags, else id heuristic).
        const support = detectSupport({
          hfArchitecture: m.config?.architectures?.[0],
          hfModelType: m.config?.model_type,
          tags: m.tags,
          modelId: m.id || m.modelId,
          isGguf,
        });

        return {
          id: m.id || m.modelId,
          author: m.author || m.id?.split('/')[0] || 'unknown',
          modelId: m.id || m.modelId,
          sha: m.sha?.slice(0, 7) || '',
          lastModified: m.lastModified,
          tags: m.tags || [],
          pipelineTag: m.pipelineTag || 'text-generation',
          downloads: m.downloads || 0,
          likes: m.likes || 0,
          libraryName: m.libraryName || '',
          createdAt: m.createdAt,
          sizeBytes: totalSize,
          quantizations,
          hasConfig,
          hasTokenizer,
          modelFilesCount: modelFiles.length,
          isGguf,
          ggufQuants: detectGgufQuants(siblings),
          support,
        };
      })
      // vllmOnly hides only fully unsupported families. Experimental are kept (with warning pill).
      .filter((r) => !vllmOnly || r.support.level !== 'unsupported')
      .slice(page * limit, (page + 1) * limit);

    res.json({ results, total: data.length, page, vllmOnly });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.get('/model/*', async (req: Request, res: Response) => {
  try {
    const modelId = decodeURIComponent(req.params[0] as string);
    const url = `https://huggingface.co/api/models/${modelId}?blobs=true`;

    const response = await fetch(url, {
      headers: {
        'Accept': 'application/json',
        'User-Agent': 'VLLM-Studio/1.0',
      },
    });

    if (!response.ok) {
      res.status(response.status).json({ error: `HuggingFace API returned ${response.status}` });
      return;
    }

    const data = await response.json();
    const siblings: Array<{ rfilename: string; size?: number }> = data.siblings || [];

    const ggufQuants = detectGgufQuants(siblings);
    const quantizations = detectAvailableQuantizations(siblings);

    res.json({
      modelId: data.id || data.modelId,
      author: data.author,
      sha: data.sha?.slice(0, 7),
      tags: data.tags || [],
      pipelineTag: data.pipelineTag,
      downloads: data.downloads || 0,
      likes: data.likes || 0,
      quantizations,
      isGguf: quantizations.includes('GGUF'),
      ggufQuants,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

function detectAvailableQuantizations(siblings: Array<{ rfilename: string }>): string[] {
  const filenames = siblings.map((s) => s.rfilename.toLowerCase()).join(' ');
  const quants: string[] = [];

  if (filenames.includes('awq') || filenames.includes('awq_model')) quants.push('AWQ');
  if (filenames.includes('gptq') || filenames.includes('gptq_model')) quants.push('GPTQ');
  if (filenames.includes('fp8') || filenames.includes('fp8_scales')) quants.push('FP8');
  if (filenames.includes('.gguf')) quants.push('GGUF');
  if ((filenames.includes('.safetensors') || filenames.includes('.bin')) &&
    !quants.includes('AWQ') && !quants.includes('GPTQ') && !quants.includes('FP8')) {
    quants.push('Full (16-bit)');
  }

  return quants;
}

function detectGgufQuants(siblings: Array<{ rfilename: string; size?: number }>): Array<{ name: string; size: number }> {
  const ggufFiles = siblings.filter((s) => s.rfilename.toLowerCase().endsWith('.gguf'));
  if (ggufFiles.length === 0) return [];

  const quants: Map<string, { size: number; filename: string }> = new Map();

  for (const f of ggufFiles) {
    const match = f.rfilename.match(/[Qq](\d+[._]\d*[._]?\w*)(?=\.gguf)/i);
    if (match) {
      const qName = `Q${match[1]}`;
      if (!quants.has(qName) || (f.size || 0) > (quants.get(qName)?.size || 0)) {
        quants.set(qName, { size: f.size || 0, filename: f.rfilename });
      }
    } else {
      // Fallback: try to extract any quantization-like pattern
      const parts = f.rfilename.replace('.gguf', '').split(/[-_.]/);
      for (const part of parts) {
        if (/^[Qq]\d/i.test(part) || /^(f16|f32|bf16|iq\d|q\d)/i.test(part)) {
          if (!quants.has(part.toUpperCase()) || (f.size || 0) > (quants.get(part.toUpperCase())?.size || 0)) {
            quants.set(part.toUpperCase(), { size: f.size || 0, filename: f.rfilename });
          }
        }
      }
    }
  }

  return Array.from(quants.entries())
    .map(([name, info]) => ({ name, size: info.size }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

router.get('/download/status', (_req: Request, res: Response) => {
  const state = getDownloadState();
  res.json(state);
});

router.post('/download', async (req: Request, res: Response) => {
  try {
    const { modelId, quantization } = req.body;
    if (!modelId || typeof modelId !== 'string') {
      res.status(400).json({ error: 'modelId is required' });
      return;
    }

    const settings = getSettings();
    const scanDirs = settings.modelScanDirs || [];
    const sanitizedModel = modelId.replace('/', '__');
    const baseDir = scanDirs.length > 0
      ? `${scanDirs[0]}/${sanitizedModel}`
      : `./data/models/${sanitizedModel}`;

    const targetDir = quantization
      ? `${baseDir}/${quantization}`
      : baseDir;

    try {
      await startDownload(modelId, targetDir, quantization || undefined);
      res.json({ success: true, targetDir, modelId, quantization: quantization || null });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Download failed';
      res.status(500).json({ error: msg });
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.post('/download/cancel', (_req: Request, res: Response) => {
  try {
    cancelDownload();
    res.json({ success: true });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

export default router;
