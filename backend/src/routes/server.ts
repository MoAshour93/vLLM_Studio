import { Router, Request, Response } from 'express';
import { StartServerSchema } from '../types/index.js';
import {
  startVllm, stopVllm, restartVllm,
  getStatus, getError, getLogs, getGpuStats, getVllmVersion, getStage,
} from '../services/vllmManager.js';
import { scanSingleModel } from '../services/modelScanner.js';
import { setModelPrefs } from '../services/database.js';
import { getVllmCapabilities, isGgufArchSupportedByInstalledTransformers } from '../services/vllmIntrospect.js';
import { getLatestRelease, compareVersions } from '../services/vllmRelease.js';

const router = Router();

router.get('/status', async (_req: Request, res: Response) => {
  try {
    const caps = getVllmCapabilities();
    const release = await getLatestRelease().catch(() => null);
    const isBehind = release?.latestStable
      ? compareVersions(caps.vllmVersion, release.latestStable) < 0
      : false;

    res.json({
      status: getStatus(),
      stage: getStage(),
      error: getError(),
      port: Number(process.env.VLLM_PORT || 8000),
      vllmVersion: caps.vllmVersion,
      vllmArchsCount: caps.vllmArchs.length,
      transformersGgufArchs: caps.transformersGgufArchs,
      latestStable: release?.latestStable ?? null,
      latestNightly: release?.latestNightly ?? null,
      isBehind,
      upgradeCommand: isBehind
        ? 'pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly'
        : null,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.get('/capabilities', (_req: Request, res: Response) => {
  res.json(getVllmCapabilities());
});

function buildStartConfig(parsed: ReturnType<typeof StartServerSchema.parse>) {
  let ggufFilePath = parsed.ggufFilePath || null;
  let loadFormat: 'auto' | 'gguf' | 'safetensors' = parsed.loadFormat;

  // Auto-resolve GGUF: if modelPath is a directory containing a single .gguf, target that file.
  let model: ReturnType<typeof scanSingleModel> = null;
  if (!ggufFilePath) {
    model = scanSingleModel(parsed.modelPath);
    if (model && model.format === 'gguf' && model.modelFile) {
      ggufFilePath = model.modelFile;
      if (loadFormat === 'auto') loadFormat = 'gguf';
    }
  }

  // Only pass --hf-config-path when the GGUF architecture is not natively
  // supported by the installed transformers.  For native archs (llama, mistral,
  // qwen2, qwen3, etc.) the override causes conflicts with the GGUF loader's
  // internal model_type mappings.
  //
  // --tokenizer is always passed for multimodal GGUFs (Qwen3.5, Qwen2-VL, etc.)
  // because vLLM requires it, and is recommended by the vLLM docs for all GGUFs
  // to avoid slow tokenizer conversion. We only strip it for simple text models
  // when no base model is explicitly configured.
  let hfConfigPath = parsed.hfConfigPath;
  let tokenizer = parsed.tokenizer;
  if (!model) model = scanSingleModel(parsed.modelPath);
  if (model?.format === 'gguf' && model.ggufArchitecture) {
    if (isGgufArchSupportedByInstalledTransformers(model.ggufArchitecture)) {
      hfConfigPath = null;
      tokenizer = null;
    }
  }

  // Cap maxModelLen at the model's derived limit (max_position_embeddings
  // from the GGUF config.json).  vLLM rejects values above this.
  let maxModelLen = parsed.maxModelLen;
  if (model && model.nativeContextLength && model.nativeContextLength > 0) {
    const derivedMax = Math.min(model.nativeContextLength, model.contextLength || Infinity);
    if (maxModelLen && maxModelLen > derivedMax) {
      maxModelLen = derivedMax;
    }
  }

  return {
    modelPath: parsed.modelPath,
    port: parsed.port,
    gpuMemoryUtilization: parsed.gpuMemoryUtilization,
    maxModelLen,
    quantization: parsed.quantization,
    tensorParallelSize: parsed.tensorParallelSize,
    maxNumSeqs: parsed.maxNumSeqs,
    dtype: parsed.dtype,
    additionalArgs: parsed.additionalArgs,
    loadFormat,
    ggufFilePath,
    cpuOffloadGb: parsed.cpuOffloadGb,
    maxNumBatchedTokens: parsed.maxNumBatchedTokens,
    kvCacheDtype: parsed.kvCacheDtype,
    enforceEager: parsed.enforceEager,
    hfConfigPath,
    tokenizer,
    languageModelOnly: parsed.languageModelOnly,
  };
}

function persistPrefs(parsed: ReturnType<typeof StartServerSchema.parse>) {
  try {
    setModelPrefs({
      path: parsed.modelPath,
      maxModelLen: parsed.maxModelLen,
      gpuMemoryUtilization: parsed.gpuMemoryUtilization,
      quantization: parsed.quantization,
      dtype: parsed.dtype,
      tensorParallelSize: parsed.tensorParallelSize,
      maxNumSeqs: parsed.maxNumSeqs,
      additionalArgs: parsed.additionalArgs,
    });
  } catch { /* non-fatal */ }
}

// Permissive guard: only refuse what we can't even read; let vLLM be authoritative
// for everything else. Architecture compat is now surfaced by the empirical
// error mapper in vllmManager.ts when (and if) the load actually fails.
function checkSupport(modelPath: string): { ok: true } | { ok: false; reason: string } {
  const m = scanSingleModel(modelPath);
  if (!m) return { ok: false, reason: `Could not read model at ${modelPath}` };
  return { ok: true };
}

router.post('/start', async (req: Request, res: Response) => {
  try {
    const parsed = StartServerSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid config', details: parsed.error.format() });
      return;
    }
    const guard = checkSupport(parsed.data.modelPath);
    if (!guard.ok) {
      res.status(400).json({ error: guard.reason });
      return;
    }
    persistPrefs(parsed.data);
    await startVllm(buildStartConfig(parsed.data));
    res.json({ status: 'starting' });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.post('/stop', async (_req: Request, res: Response) => {
  try {
    await stopVllm();
    res.json({ status: 'stopped' });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.post('/restart', async (req: Request, res: Response) => {
  try {
    const parsed = StartServerSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid config', details: parsed.error.format() });
      return;
    }
    const guard = checkSupport(parsed.data.modelPath);
    if (!guard.ok) {
      res.status(400).json({ error: guard.reason });
      return;
    }
    persistPrefs(parsed.data);
    await restartVllm(buildStartConfig(parsed.data));
    res.json({ status: 'running' });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

router.get('/logs', (req: Request, res: Response) => {
  try {
    const lines = parseInt(req.query.lines as string, 10) || 200;
    res.json({ logs: getLogs(lines) });
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error';
    res.status(500).json({ error: msg });
  }
});

export { getGpuStats };
export default router;
