import { Router, Request, Response } from 'express';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { scanModelDirectories, scanSingleModel } from '../services/modelScanner.js';
import { getSettings, clearModelCache, getModelPrefs, setModelPrefs } from '../services/database.js';
import { getVllmVersion, getGpuStats } from '../services/vllmManager.js';
import { estimateVram, recommendMaxContext, recommendCpuOffload } from '../services/vramEstimator.js';
import { guessBaseModelRepo } from '../services/baseModelGuess.js';
import { isGgufArchSupportedUpstream } from '../services/supportedArchitectures.js';
import { patchGgufArchitecture, suggestTargetArch } from '../services/ggufPatcher.js';
import type { ModelInfo } from '../types/index.js';

const router = Router();

function getScanDirs(): string[] {
  const settings = getSettings();
  const dirs = [...settings.modelScanDirs];
  const downloadDir = path.resolve(process.env.DATA_DIR || './data', 'models');
  if (!dirs.includes(downloadDir) && !dirs.some(d => path.resolve(d) === downloadDir)) {
    dirs.push(downloadDir);
  }
  return dirs;
}

interface FitInfo {
  fit: 'full-gpu' | 'partial-gpu' | 'cpu-offload' | 'too-large';
  label: string;
  gpuMemoryMb: number;
  systemRamMb: number;
  requiredMemoryMb: number;
}

const FIT_LABEL: Record<FitInfo['fit'], string> = {
  'full-gpu': 'Fits in GPU',
  'partial-gpu': 'Partial GPU',
  'cpu-offload': 'CPU offload',
  'too-large': 'Too large',
};

function getFitInfo(model: ModelInfo): FitInfo {
  const gpus = getGpuStats();
  const totalGpuMb = gpus.reduce((s, g) => s + g.totalMemoryMb, 0);
  const freeGpuMb = gpus.reduce((s, g) => s + g.freeMemoryMb, 0);
  const systemRamMb = Math.round(os.totalmem() / (1024 * 1024));

  const ctx = Math.min(model.contextLength || 4096, 32768);
  const est = estimateVram({
    spec: model.arch,
    contextLength: ctx,
    tensorParallelSize: 1,
    totalGpuMb,
    freeGpuMb,
    systemRamMb,
    maxNumSeqs: 1,
  });

  return {
    fit: est.fitVerdict,
    label: FIT_LABEL[est.fitVerdict],
    gpuMemoryMb: totalGpuMb,
    systemRamMb,
    requiredMemoryMb: Math.round(est.totalMb),
  };
}

function enhanceWithFit(models: ModelInfo[]): Array<ModelInfo & { fitInfo: FitInfo }> {
  return models.map(m => ({ ...m, fitInfo: getFitInfo(m) }));
}

router.get('/', (_req: Request, res: Response) => {
  try {
    const scanDirs = getScanDirs();
    const models = scanModelDirectories(scanDirs);
    const vllmVersion = getVllmVersion();
    const enhanced = enhanceWithFit(models);
    res.json({ models: enhanced, vllmVersion });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.post('/scan', (_req: Request, res: Response) => {
  try {
    const scanDirs = getScanDirs();
    const models = scanModelDirectories(scanDirs, true);
    const enhanced = enhanceWithFit(models);
    res.json({ models: enhanced });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.post('/preflight', (req: Request, res: Response) => {
  try {
    const {
      modelPath, contextLength, tensorParallelSize, maxNumSeqs,
      cpuOffloadGb, kvCacheDtype, maxNumBatchedTokens,
    } = req.body as {
      modelPath: string;
      contextLength?: number;
      tensorParallelSize?: number;
      maxNumSeqs?: number;
      cpuOffloadGb?: number;
      kvCacheDtype?: 'auto' | 'fp16' | 'fp8' | 'fp8_e4m3' | 'fp8_e5m2';
      maxNumBatchedTokens?: number;
    };
    if (!modelPath || typeof modelPath !== 'string') {
      res.status(400).json({ error: 'modelPath is required' });
      return;
    }

    const model = scanSingleModel(modelPath);
    if (!model) {
      res.status(404).json({ error: 'Model not found or unreadable at given path' });
      return;
    }

    const gpus = getGpuStats();
    const totalGpuMb = gpus.reduce((s, g) => s + g.totalMemoryMb, 0);
    const freeGpuMb = gpus.reduce((s, g) => s + g.freeMemoryMb, 0);
    const systemRamMb = Math.round(os.totalmem() / (1024 * 1024));
    const tp = Math.max(1, tensorParallelSize ?? 1);
    const batch = Math.max(1, maxNumSeqs ?? 1);

    const baseInput = {
      spec: model.arch,
      tensorParallelSize: tp,
      totalGpuMb,
      freeGpuMb,
      systemRamMb,
      maxNumSeqs: batch,
      maxNumBatchedTokens: maxNumBatchedTokens ?? 8192,
      kvCacheDtype: kvCacheDtype ?? 'auto',
      cpuOffloadGb: cpuOffloadGb ?? 0,
    };

    const recommendedContextLength = recommendMaxContext({
      ...baseInput,
      nativeContext: model.nativeContextLength,
      extendedContext: model.contextLength,
    });

    const ctx = contextLength && contextLength > 0
      ? Math.min(contextLength, model.contextLength)
      : recommendedContextLength;

    const estimate = estimateVram({ ...baseInput, contextLength: ctx });

    // Suggest a CPU offload that lets us stay below 90% per-GPU.
    const recommendedCpuOffloadGb = (cpuOffloadGb ?? 0) === 0
      ? recommendCpuOffload({ ...baseInput, contextLength: ctx, targetUtil: 0.90 })
      : 0;

    const prefs = getModelPrefs(model.path);

    const ggufNeedsBase = model.format === 'gguf'
      && model.ggufArchitecture !== null
      && !isGgufArchSupportedUpstream(model.ggufArchitecture);
    const suggestedBaseRepo = model.format === 'gguf'
      ? guessBaseModelRepo(model.path)
      : null;

    res.json({
      model,
      gpus,
      totalGpuMb,
      freeGpuMb,
      systemRamMb,
      contextLength: ctx,
      recommendedContextLength,
      recommendedCpuOffloadGb,
      suggestedBaseRepo,
      ggufNeedsBaseConfig: ggufNeedsBase,
      estimate,
      prefs,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.get('/prefs', (req: Request, res: Response) => {
  try {
    const modelPath = req.query.modelPath as string | undefined;
    if (!modelPath) {
      res.status(400).json({ error: 'modelPath query param is required' });
      return;
    }
    const prefs = getModelPrefs(modelPath);
    res.json({ prefs });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.post('/prefs', (req: Request, res: Response) => {
  try {
    const { modelPath, ...rest } = req.body as Record<string, unknown>;
    if (!modelPath || typeof modelPath !== 'string') {
      res.status(400).json({ error: 'modelPath is required' });
      return;
    }
    setModelPrefs({
      path: modelPath,
      maxModelLen: typeof rest.maxModelLen === 'number' ? rest.maxModelLen : null,
      gpuMemoryUtilization: typeof rest.gpuMemoryUtilization === 'number' ? rest.gpuMemoryUtilization : null,
      quantization: typeof rest.quantization === 'string' ? rest.quantization : null,
      dtype: typeof rest.dtype === 'string' ? rest.dtype : null,
      tensorParallelSize: typeof rest.tensorParallelSize === 'number' ? rest.tensorParallelSize : null,
      maxNumSeqs: typeof rest.maxNumSeqs === 'number' ? rest.maxNumSeqs : null,
      additionalArgs: Array.isArray(rest.additionalArgs) ? rest.additionalArgs as string[] : [],
    });
    res.json({ success: true });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

// In-memory state for the active patch. UI polls /patch-gguf-arch/status while running.
let patchState: {
  active: boolean;
  modelPath: string;
  bytesCopied: number;
  totalBytes: number;
  stage: 'starting' | 'copying' | 'complete' | 'error';
  error?: string;
  outputFile?: string;
  newArch?: string;
} | null = null;

router.get('/patch-gguf-arch/status', (_req: Request, res: Response) => {
  res.json(patchState ?? { active: false });
});

router.post('/patch-gguf-arch', async (req: Request, res: Response) => {
  try {
    if (patchState?.active) {
      res.status(409).json({ error: 'A patch operation is already running' });
      return;
    }
    const { modelPath, newArchitecture } = req.body as {
      modelPath: string;
      newArchitecture?: string;
    };
    if (!modelPath || typeof modelPath !== 'string') {
      res.status(400).json({ error: 'modelPath is required' });
      return;
    }
    const m = scanSingleModel(modelPath);
    if (!m) { res.status(404).json({ error: 'Model not found' }); return; }
    if (m.format !== 'gguf' || !m.modelFile) {
      res.status(400).json({ error: 'Model is not a single-file GGUF' });
      return;
    }
    const targetArch = (newArchitecture && newArchitecture.trim())
      || suggestTargetArch(m.ggufArchitecture ?? '')
      || null;
    if (!targetArch) {
      res.status(400).json({ error: `Could not infer a target architecture for "${m.ggufArchitecture}"; pass newArchitecture explicitly.` });
      return;
    }

    patchState = {
      active: true,
      modelPath,
      bytesCopied: 0,
      totalBytes: 0,
      stage: 'starting',
      newArch: targetArch,
    };
    res.json({ started: true, targetArch });

    // Run async — UI polls /patch-gguf-arch/status.
    patchGgufArchitecture({
      inputModelPath: modelPath,
      inputGgufFile: m.modelFile,
      newArchitecture: targetArch,
      onProgress: (bytes, total) => {
        if (patchState) {
          patchState.bytesCopied = bytes;
          patchState.totalBytes = total;
          patchState.stage = 'copying';
        }
      },
    }).then((result) => {
      patchState = {
        active: false,
        modelPath,
        bytesCopied: result.bytesWritten,
        totalBytes: result.bytesWritten,
        stage: 'complete',
        outputFile: result.outputFile,
        newArch: result.newArch,
      };
      // Re-scan so the new patched copy appears.
      try { scanModelDirectories(getScanDirs(), true); } catch { /* ignore */ }
    }).catch((err) => {
      const msg = err instanceof Error ? err.message : 'Patch failed';
      patchState = {
        active: false, modelPath, bytesCopied: 0, totalBytes: 0,
        stage: 'error', error: msg,
      };
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.post('/delete', (req: Request, res: Response) => {
  try {
    const { modelPath } = req.body;
    if (!modelPath || typeof modelPath !== 'string') {
      res.status(400).json({ error: 'modelPath is required' });
      return;
    }
    const resolved = path.resolve(modelPath);
    const scanDirs = getScanDirs();
    const downloadDir = path.resolve(process.env.DATA_DIR || './data', 'models');
    const allDirs = [...scanDirs, downloadDir];
    const allowed = allDirs.some(d => resolved.startsWith(path.resolve(d) + path.sep) || resolved === path.resolve(d));
    if (!allowed) {
      res.status(403).json({ error: 'Model path is not within allowed directories' });
      return;
    }
    if (fs.existsSync(resolved)) {
      fs.rmSync(resolved, { recursive: true, force: true });
    }
    clearModelCache();
    const newModels = scanModelDirectories(scanDirs, true);
    const enhanced = enhanceWithFit(newModels);
    res.json({ success: true, models: enhanced });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

export default router;
