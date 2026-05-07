import fs from 'fs';
import path from 'path';
import { ModelInfo, ModelArchSpec, ModelSupport } from '../types/index.js';
import { getCachedModels, upsertModelCache, clearModelCache } from './database.js';
import { readGgufMetadata, ggufArchToVllm, bitsPerWeightFromFileType } from './gguf.js';
import { lookupByGgufArchitecture, lookupByHfArchitecture } from './supportedArchitectures.js';

const VISION_ARCHITECTURES = [
  'llava', 'llava_next', 'llava_onevision', 'internvl', 'internvl2',
  'phi3_v', 'phi3_vision', 'fuyu', 'paligemma', 'pixtral', 'qwen2_vl',
  'qwen_vl', 'minicpmv', 'idefics3', 'llama4', 'blip', 'mllama',
];

const VISION_MODEL_NAMES = [
  'llava', 'vision', 'vl', 'internvl', 'phi-3-vision', 'paligemma',
  'fuyu', 'pixtral', 'minicpm', 'idefics', 'mllama',
];

function isVisionModel(architecture: string, modelPath: string): boolean {
  const archLower = architecture.toLowerCase();
  const pathLower = modelPath.toLowerCase();
  for (const arch of VISION_ARCHITECTURES) {
    if (archLower.includes(arch)) return true;
  }
  for (const name of VISION_MODEL_NAMES) {
    if (pathLower.includes(name)) return true;
  }
  return false;
}

function detectQuantization(modelPath: string, config: Record<string, unknown>): string | null {
  const quantConfig = config.quantization_config as Record<string, unknown> | undefined;
  if (quantConfig) {
    const method = quantConfig.quant_method as string | undefined;
    if (method) return method;
  }
  try {
    const files = fs.readdirSync(modelPath);
    if (files.some(f => /awq/i.test(f))) return 'awq';
    if (files.some(f => /gptq/i.test(f))) return 'gptq';
    if (files.some(f => /fp8/i.test(f))) return 'fp8';
  } catch { /* ignore */ }
  return null;
}

interface ContextResult { effective: number; native: number }

function detectHfContext(config: Record<string, unknown>): ContextResult {
  const native =
    (typeof config.max_position_embeddings === 'number' && config.max_position_embeddings) ||
    (typeof config.model_max_length === 'number' && config.model_max_length) ||
    (typeof config.max_sequence_length === 'number' && config.max_sequence_length) ||
    2048;

  const rope = config.rope_scaling as Record<string, unknown> | undefined;
  if (rope) {
    const factor = typeof rope.factor === 'number' ? rope.factor : null;
    const original = typeof rope.original_max_position_embeddings === 'number'
      ? rope.original_max_position_embeddings as number : null;
    if (factor && factor > 1) {
      const base = original ?? native;
      return { effective: Math.floor(base * factor), native: original ?? native };
    }
  }
  return { effective: native, native };
}

function getDirectorySize(dirPath: string): number {
  let size = 0;
  try {
    const files = fs.readdirSync(dirPath);
    for (const file of files) {
      const filePath = path.join(dirPath, file);
      try {
        const stat = fs.statSync(filePath);
        if (stat.isDirectory()) size += getDirectorySize(filePath);
        else size += stat.size;
      } catch { /* skip */ }
    }
  } catch { /* skip */ }
  return size;
}

function buildModelName(dirPath: string): string {
  const base = path.basename(dirPath);
  const parent = path.basename(path.dirname(dirPath));
  if (parent && parent !== 'models' && parent !== 'data' && parent !== '.' && parent !== '..') {
    return `${parent}/${base}`;
  }
  return base;
}

function emptySpec(): ModelArchSpec {
  return {
    numLayers: null, hiddenSize: null, numHeads: null, numKvHeads: null,
    headDim: null, vocabSize: null, intermediateSize: null, totalParams: null,
    fileSizeBytes: null, dtype: null, kvCacheDtype: null, bitsPerWeight: null,
    quantization: null,
  };
}

function specFromHfConfig(config: Record<string, unknown>): ModelArchSpec {
  const num = (k: string): number | null => typeof config[k] === 'number' ? config[k] as number : null;
  const numLayers = num('num_hidden_layers');
  const hiddenSize = num('hidden_size');
  const numHeads = num('num_attention_heads');
  const numKvHeads = num('num_key_value_heads') ?? numHeads;
  const headDim = num('head_dim') ?? (hiddenSize && numHeads ? Math.floor(hiddenSize / numHeads) : null);
  const dtype = (config.torch_dtype as string) || null;

  const quantConfig = config.quantization_config as Record<string, unknown> | undefined;
  const bits = quantConfig && typeof quantConfig.bits === 'number' ? quantConfig.bits as number : null;

  return {
    ...emptySpec(),
    numLayers, hiddenSize, numHeads, numKvHeads, headDim,
    vocabSize: num('vocab_size'),
    intermediateSize: num('intermediate_size'),
    dtype, kvCacheDtype: dtype,
    bitsPerWeight: bits,
  };
}

interface GgufFallback {
  architecture: string;
  contextLength: number;
  numLayers: number | null;
  hiddenSize: number | null;
  numHeads: number | null;
  numKvHeads: number | null;
}

function ggufFromFilename(dirPath: string, ggufFile: string): GgufFallback {
  const combined = (path.basename(dirPath) + ' ' + ggufFile).toLowerCase();
  if (combined.includes('qwen3.5') || combined.includes('qwen35')) {
    return { architecture: 'Qwen2ForCausalLM', contextLength: 262144, numLayers: 36, hiddenSize: 4096, numHeads: 32, numKvHeads: 8 };
  }
  if (combined.includes('qwen3')) {
    return { architecture: 'Qwen2ForCausalLM', contextLength: 131072, numLayers: 28, hiddenSize: 3584, numHeads: 28, numKvHeads: 4 };
  }
  if (combined.includes('qwen2.5')) {
    return { architecture: 'Qwen2ForCausalLM', contextLength: 131072, numLayers: 28, hiddenSize: 3584, numHeads: 28, numKvHeads: 4 };
  }
  if (combined.includes('qwen2')) {
    return { architecture: 'Qwen2ForCausalLM', contextLength: 32768, numLayers: 28, hiddenSize: 3584, numHeads: 28, numKvHeads: 4 };
  }
  if (combined.includes('llama-3.1') || combined.includes('llama-3.2') || combined.includes('llama-3')) {
    return { architecture: 'LlamaForCausalLM', contextLength: 131072, numLayers: 32, hiddenSize: 4096, numHeads: 32, numKvHeads: 8 };
  }
  if (combined.includes('llama-2')) {
    return { architecture: 'LlamaForCausalLM', contextLength: 4096, numLayers: 32, hiddenSize: 4096, numHeads: 32, numKvHeads: 32 };
  }
  if (combined.includes('mistral')) {
    return { architecture: 'MistralForCausalLM', contextLength: 32768, numLayers: 32, hiddenSize: 4096, numHeads: 32, numKvHeads: 8 };
  }
  if (combined.includes('mixtral')) {
    return { architecture: 'MixtralForCausalLM', contextLength: 32768, numLayers: 32, hiddenSize: 4096, numHeads: 32, numKvHeads: 8 };
  }
  if (combined.includes('gemma-2')) {
    return { architecture: 'Gemma2ForCausalLM', contextLength: 8192, numLayers: 26, hiddenSize: 2304, numHeads: 8, numKvHeads: 4 };
  }
  if (combined.includes('phi-3')) {
    return { architecture: 'Phi3ForCausalLM', contextLength: 131072, numLayers: 32, hiddenSize: 3072, numHeads: 32, numKvHeads: 32 };
  }
  return { architecture: 'LlamaForCausalLM', contextLength: 4096, numLayers: null, hiddenSize: null, numHeads: null, numKvHeads: null };
}

function scanGgufModel(dirPath: string): ModelInfo | null {
  let files: string[];
  try { files = fs.readdirSync(dirPath); } catch { return null; }

  const ggufFile = files.find(f => f.toLowerCase().endsWith('.gguf'));
  if (!ggufFile) return null;

  const ggufPath = path.join(dirPath, ggufFile);
  const meta = readGgufMetadata(ggufPath);

  if (!meta) {
    // Metadata unreadable (e.g. corrupted by older version). Fall back to filename heuristics.
    const fb = ggufFromFilename(dirPath, ggufFile);
    let fileSize = 0;
    try { fileSize = fs.statSync(ggufPath).size; } catch { /* ignore */ }
    const headDim = fb.hiddenSize && fb.numHeads ? Math.floor(fb.hiddenSize / fb.numHeads) : null;
    const arch: ModelArchSpec = {
      numLayers: fb.numLayers, hiddenSize: fb.hiddenSize, numHeads: fb.numHeads,
      numKvHeads: fb.numKvHeads, headDim,
      vocabSize: null, intermediateSize: null,
      totalParams: null, fileSizeBytes: fileSize,
      dtype: 'fp16', kvCacheDtype: 'fp16',
      bitsPerWeight: 5.5, quantization: 'gguf',
    };
    return {
      id: Buffer.from(dirPath).toString('base64'),
      name: buildModelName(dirPath) + ' (metadata unreadable — using filename heuristics)',
      path: dirPath,
      modelFile: ggufPath,
      format: 'gguf',
      sizeBytes: fileSize,
      architecture: fb.architecture,
      ggufArchitecture: null,
      quantization: inferGgufQuantLabel(ggufFile),
      contextLength: fb.contextLength,
      nativeContextLength: fb.contextLength,
      isVision: isVisionModel(fb.architecture, dirPath),
      arch,
      support: lookupByHfArchitecture(fb.architecture),
    };
  }

  const fileSize = (() => {
    try { return fs.statSync(ggufPath).size; } catch { return 0; }
  })();

  const native = meta.contextLength ?? meta.ropeOriginalContextLength ?? 2048;
  const effective = meta.ropeScalingFactor && meta.ropeScalingFactor > 1
    ? Math.max(native, Math.floor((meta.ropeOriginalContextLength ?? native) * meta.ropeScalingFactor))
    : native;

  const bpw = bitsPerWeightFromFileType(meta.fileType);
  const headDim = meta.embeddingLength && meta.headCount
    ? Math.floor(meta.embeddingLength / meta.headCount)
    : null;

  const arch: ModelArchSpec = {
    numLayers: meta.blockCount,
    hiddenSize: meta.embeddingLength,
    numHeads: meta.headCount,
    numKvHeads: meta.headCountKv ?? meta.headCount,
    headDim,
    vocabSize: null,
    intermediateSize: meta.feedForwardLength,
    totalParams: null,
    fileSizeBytes: fileSize,
    dtype: 'fp16',
    kvCacheDtype: 'fp16',
    bitsPerWeight: bpw,
    quantization: 'gguf',
  };

  const quantLabel = inferGgufQuantLabel(ggufFile);

  const support = lookupByGgufArchitecture(meta.architecture);
  return {
    id: Buffer.from(dirPath).toString('base64'),
    name: buildModelName(dirPath),
    path: dirPath,
    modelFile: ggufPath,
    format: 'gguf',
    sizeBytes: fileSize,
    architecture: ggufArchToVllm(meta.architecture),
    ggufArchitecture: meta.architecture,
    quantization: quantLabel,
    contextLength: effective,
    nativeContextLength: native,
    isVision: isVisionModel(meta.architecture, dirPath),
    arch,
    support,
  };
}

function inferGgufQuantLabel(filename: string): string {
  const m = filename.toLowerCase().match(/(q\d_[01k][_a-z0-9]*|f16|f32|bf16|iq\d_[a-z]+)/);
  return m ? m[0].toUpperCase() : 'GGUF';
}

function scanHfModel(dirPath: string): ModelInfo | null {
  const configPath = path.join(dirPath, 'config.json');
  if (!fs.existsSync(configPath)) return null;

  let files: string[];
  try { files = fs.readdirSync(dirPath); } catch { return null; }

  const hasWeights = files.some(f => f.endsWith('.safetensors') || f.endsWith('.bin'));
  if (!hasWeights) return null;

  let config: Record<string, unknown>;
  try {
    config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
  } catch {
    return null;
  }

  const architecture = (config.architectures as string[] | undefined)?.[0] ?? 'unknown';
  const quant = detectQuantization(dirPath, config);
  const ctx = detectHfContext(config);
  const sizeBytes = getDirectorySize(dirPath);
  const arch = specFromHfConfig(config);
  arch.fileSizeBytes = sizeBytes;
  arch.quantization = quant;

  return {
    id: Buffer.from(dirPath).toString('base64'),
    name: buildModelName(dirPath),
    path: dirPath,
    modelFile: null,
    format: 'hf',
    sizeBytes,
    architecture,
    ggufArchitecture: null,
    quantization: quant,
    contextLength: ctx.effective,
    nativeContextLength: ctx.native,
    isVision: isVisionModel(architecture, dirPath),
    arch,
    support: lookupByHfArchitecture(architecture),
  };
}

export function scanSingleModel(dirPath: string): ModelInfo | null {
  try {
    const stat = fs.statSync(dirPath);
    if (!stat.isDirectory()) return null;
  } catch {
    return null;
  }
  try {
    return scanGgufModel(dirPath) ?? scanHfModel(dirPath);
  } catch (err) {
    console.error(`scanSingleModel(${dirPath}) failed:`, err);
    return null;
  }
}

export function getModelInfo(modelPath: string): ModelInfo | null {
  if (!fs.existsSync(modelPath)) return null;
  return scanSingleModel(modelPath);
}

export function scanModelDirectories(directories: string[], force: boolean = false): ModelInfo[] {
  if (!force) {
    const cached = getCachedModels();
    if (cached.length > 0) {
      const models: ModelInfo[] = [];
      for (const row of cached) {
        const p = row.path as string;
        const mtime = row.directory_mtime as number;
        try {
          const stat = fs.statSync(p);
          if (stat.mtimeMs <= mtime + 1000) {
            const blob = row.spec_json as string | null;
            let spec: ModelArchSpec = emptySpec();
            try { if (blob) spec = JSON.parse(blob); } catch { /* keep default */ }
            const fmt = ((row.format as string) || 'hf') as 'hf' | 'gguf';
            const ggufArch = (row.gguf_architecture as string) || null;
            const archStr = row.architecture as string;
            const support = fmt === 'gguf' && ggufArch
              ? lookupByGgufArchitecture(ggufArch)
              : lookupByHfArchitecture(archStr);
            models.push({
              id: Buffer.from(p).toString('base64'),
              name: row.name as string,
              path: p,
              modelFile: (row.model_file as string) || null,
              format: fmt,
              sizeBytes: row.size_bytes as number,
              architecture: archStr,
              ggufArchitecture: ggufArch,
              quantization: (row.quantization as string) ?? null,
              contextLength: row.context_length as number,
              nativeContextLength: (row.native_context_length as number) ?? (row.context_length as number),
              isVision: !!(row.is_vision),
              arch: spec,
              support,
            });
            continue;
          }
        } catch { /* re-scan below */ }
        const fresh = scanSingleModel(p);
        if (fresh) {
          saveToCache(fresh);
          models.push(fresh);
        }
      }
      return models;
    }
  }

  if (force) clearModelCache();

  const models: ModelInfo[] = [];
  for (const dir of directories) {
    if (!fs.existsSync(dir)) continue;
    let entries: fs.Dirent[];
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); }
    catch { continue; }

    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const fullPath = path.join(dir, entry.name);
      const model = scanSingleModel(fullPath);
      if (model) {
        saveToCache(model);
        models.push(model);
      } else {
        // One level deeper for quantization subdirs (e.g. repo/Q4_K_M/file.gguf)
        try {
          const subEntries = fs.readdirSync(fullPath, { withFileTypes: true });
          for (const sub of subEntries) {
            if (!sub.isDirectory()) continue;
            const subPath = path.join(fullPath, sub.name);
            const subModel = scanSingleModel(subPath);
            if (subModel) {
              saveToCache(subModel);
              models.push(subModel);
            }
          }
        } catch { /* ignore */ }
      }
    }
  }

  return models;
}

function saveToCache(model: ModelInfo): void {
  try {
    const stat = fs.statSync(model.path);
    upsertModelCache({
      path: model.path,
      name: model.name,
      sizeBytes: model.sizeBytes,
      architecture: model.architecture,
      ggufArchitecture: model.ggufArchitecture,
      quantization: model.quantization,
      contextLength: model.contextLength,
      nativeContextLength: model.nativeContextLength,
      isVision: model.isVision,
      directoryMtime: stat.mtimeMs,
      modelFile: model.modelFile,
      format: model.format,
      specJson: JSON.stringify(model.arch),
    });
  } catch { /* ignore */ }
}

export function validateModelPath(modelPath: string, allowedDirs: string[]): boolean {
  const resolved = path.resolve(modelPath);
  for (const dir of allowedDirs) {
    const resolvedDir = path.resolve(dir);
    if (resolved.startsWith(resolvedDir)) return true;
  }
  return false;
}
