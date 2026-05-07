import { execFileSync } from 'child_process';

// Introspects the *installed* vLLM and transformers to learn what they actually
// support. Cached for the lifetime of the backend process — supported archs
// don't change without a restart.

export interface VllmCapabilities {
  vllmVersion: string;
  vllmArchs: string[];                 // ModelRegistry.get_supported_archs()
  transformersGgufArchs: string[];     // GGUF_CONFIG_MAPPING keys
  introspectionError: string | null;
}

let cached: VllmCapabilities | null = null;

function getPython(): string {
  return process.env.VLLM_PYTHON || 'python3';
}

const PROBE = `
import json
out = {"vllmVersion": "", "vllmArchs": [], "transformersGgufArchs": [], "error": None}
try:
    import vllm
    out["vllmVersion"] = vllm.__version__
    try:
        from vllm.model_executor.models.registry import ModelRegistry
        out["vllmArchs"] = sorted(ModelRegistry.get_supported_archs())
    except Exception as e:
        out["error"] = f"ModelRegistry: {e}"
    try:
        from transformers.integrations.ggml import GGUF_CONFIG_MAPPING
        out["transformersGgufArchs"] = sorted(GGUF_CONFIG_MAPPING.keys())
    except Exception as e:
        prev = out["error"] or ""
        out["error"] = (prev + " | " if prev else "") + f"GGUF map: {e}"
except Exception as e:
    out["error"] = f"vllm import failed: {e}"
print(json.dumps(out))
`;

export function getVllmCapabilities(force = false): VllmCapabilities {
  if (cached && !force) return cached;
  try {
    const raw = execFileSync(getPython(), ['-c', PROBE], {
      timeout: 30_000,
      encoding: 'utf-8',
      maxBuffer: 8 * 1024 * 1024,
    }).trim();
    const parsed = JSON.parse(raw) as {
      vllmVersion: string;
      vllmArchs: string[];
      transformersGgufArchs: string[];
      error: string | null;
    };
    cached = {
      vllmVersion: parsed.vllmVersion || 'unknown',
      vllmArchs: parsed.vllmArchs || [],
      transformersGgufArchs: parsed.transformersGgufArchs || [],
      introspectionError: parsed.error,
    };
    return cached;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    cached = {
      vllmVersion: 'unknown',
      vllmArchs: [],
      transformersGgufArchs: [],
      introspectionError: `Probe failed: ${msg}`,
    };
    return cached;
  }
}

export function isHfArchSupportedByInstalledVllm(arch: string | null | undefined): boolean {
  if (!arch) return false;
  const caps = getVllmCapabilities();
  return caps.vllmArchs.includes(arch);
}

export function isGgufArchSupportedByInstalledTransformers(ggufArch: string | null | undefined): boolean {
  if (!ggufArch) return false;
  const caps = getVllmCapabilities();
  return caps.transformersGgufArchs.includes(ggufArch.toLowerCase());
}

// Community GGUFs often use non-canonical architecture labels (e.g. "qwen35"
// for Qwen3.5).  The transformers library's GGUF_CONFIG_MAPPING only contains
// the official labels.  This patches the installed transformers so that
// recognized community labels are mapped to their official counterparts —
// avoiding the need to physically patch every GGUF file.

let patchedTransformers = false;

export function ensureGgufArchPatches(): void {
  if (patchedTransformers) return;
  try {
    const py = getPython();
    const scriptPath = new URL('./gguf_setup.py', import.meta.url).pathname;
    const decoded = decodeURIComponent(scriptPath);
    execFileSync(py, [decoded], { timeout: 15000, encoding: 'utf-8' });
    cached = null;
    getVllmCapabilities(true);
    patchedTransformers = true;
  } catch (err) {
    console.warn('GGUF arch patching skipped (venv may need setup):',
      err instanceof Error ? err.message.slice(0, 120) : String(err).slice(0, 120));
  }
}
