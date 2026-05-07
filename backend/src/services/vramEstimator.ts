import { ModelArchSpec } from '../types/index.js';

export interface VramEstimate {
  weightsMb: number;
  kvCacheMb: number;
  activationsMb: number;
  overheadMb: number;
  totalMb: number;
  perGpuMb: number;
  cpuOffloadMb: number;
  recommendedGpuMemUtil: number;
  fits: boolean;
  fitVerdict: 'full-gpu' | 'partial-gpu' | 'cpu-offload' | 'too-large';
}

export interface EstimateInput {
  spec: ModelArchSpec;
  contextLength: number;
  tensorParallelSize: number;
  totalGpuMb: number;
  freeGpuMb: number;
  systemRamMb: number;
  // For batched serving — vLLM caches per-request blocks; default 1 (LM-Studio-style chat).
  maxNumSeqs?: number;
  // Concurrent prefill chunk size; vLLM default is 8192. Activation memory scales with this, not full ctx.
  maxNumBatchedTokens?: number;
  // KV cache dtype override: fp8 halves KV cache.
  kvCacheDtype?: 'auto' | 'fp16' | 'fp8' | 'fp8_e4m3' | 'fp8_e5m2';
  // CPU offload (GiB). Reduces weights resident in VRAM.
  cpuOffloadGb?: number;
}

const DTYPE_BYTES: Record<string, number> = {
  fp32: 4, float32: 4,
  fp16: 2, float16: 2, bf16: 2, bfloat16: 2,
  fp8: 1, int8: 1,
  int4: 0.5, awq: 0.5, gptq: 0.5,
};

function weightBytesPerParam(spec: ModelArchSpec): number {
  if (spec.bitsPerWeight && spec.bitsPerWeight > 0) {
    return spec.bitsPerWeight / 8;
  }
  if (spec.quantization) {
    const q = spec.quantization.toLowerCase();
    if (q.includes('awq') || q.includes('gptq') || q.includes('q4')) return 0.55;
    if (q.includes('fp8') || q.includes('q8')) return 1.05;
    if (q.includes('q5')) return 0.7;
    if (q.includes('q6')) return 0.85;
    if (q.includes('q3')) return 0.45;
    if (q.includes('q2')) return 0.3;
  }
  const dtype = (spec.dtype || 'fp16').toLowerCase();
  return DTYPE_BYTES[dtype] ?? 2;
}

function kvCacheBytes(spec: ModelArchSpec, ctx: number, batch: number, kvDtype: string | undefined): number {
  if (!spec.numLayers || !spec.headDim || !spec.numKvHeads) return 0;
  // 2 (K + V) * layers * ctx * num_kv_heads * head_dim * bytes
  const dtype = (kvDtype || spec.kvCacheDtype || 'fp16').toLowerCase();
  const dtypeBytes = dtype.includes('fp8') || dtype === 'int8' ? 1 : 2;
  return 2 * spec.numLayers * ctx * spec.numKvHeads * spec.headDim * dtypeBytes * batch;
}

function activationsBytes(spec: ModelArchSpec, prefillTokens: number): number {
  // Activations are sized by the prefill chunk (vLLM's `max_num_batched_tokens`),
  // NOT the full ctx — KV cache holds the rest. Default chunked-prefill = 8192.
  if (!spec.hiddenSize) return 64 * 1024 * 1024;
  // ~4 working buffers of (chunk × hidden) in fp16 + small temporaries.
  return 4 * prefillTokens * spec.hiddenSize * 2 + 64 * 1024 * 1024;
}

export function estimateVram(input: EstimateInput): VramEstimate {
  const { spec, contextLength, tensorParallelSize, totalGpuMb, freeGpuMb, systemRamMb } = input;
  const batch = Math.max(1, input.maxNumSeqs ?? 1);
  const prefill = Math.max(1024, Math.min(input.maxNumBatchedTokens || 8192, contextLength));
  const cpuOffloadGiB = Math.max(0, input.cpuOffloadGb ?? 0);

  let paramBytesAll = 0;
  if (spec.totalParams && spec.totalParams > 0) {
    paramBytesAll = spec.totalParams * weightBytesPerParam(spec);
  } else if (spec.fileSizeBytes && spec.fileSizeBytes > 0) {
    // For GGUF the file IS the weights; this is the most accurate estimate.
    paramBytesAll = spec.fileSizeBytes;
  }

  const cpuOffloadBytes = Math.min(cpuOffloadGiB * 1024 * 1024 * 1024, paramBytesAll);
  const paramBytesGpu = paramBytesAll - cpuOffloadBytes;

  const kvBytes = kvCacheBytes(spec, contextLength, batch, input.kvCacheDtype);
  const actBytes = activationsBytes(spec, prefill);
  // CUDA context (~300 MB) + cuBLAS / cuDNN handles (~150 MB) + vLLM block manager (~150 MB).
  const overheadBytes = 600 * 1024 * 1024;

  const weightsMb = paramBytesGpu / (1024 * 1024);
  const cpuOffloadMb = cpuOffloadBytes / (1024 * 1024);
  const kvCacheMb = kvBytes / (1024 * 1024);
  const activationsMb = actBytes / (1024 * 1024);
  const overheadMb = overheadBytes / (1024 * 1024);
  const totalMb = weightsMb + kvCacheMb + activationsMb + overheadMb;

  const tp = Math.max(1, tensorParallelSize);
  const perGpuMb = (weightsMb + kvCacheMb + activationsMb) / tp + overheadMb;

  const perGpuTotalMb = totalGpuMb / tp;

  let fitVerdict: VramEstimate['fitVerdict'] = 'too-large';
  if (totalMb <= freeGpuMb * 0.95) fitVerdict = 'full-gpu';
  else if (totalMb <= totalGpuMb * 0.95) fitVerdict = 'partial-gpu';
  else if (cpuOffloadMb > 0 && perGpuMb <= perGpuTotalMb * 0.95) fitVerdict = 'cpu-offload';
  else if (totalMb + cpuOffloadMb <= systemRamMb * 0.8 + totalGpuMb * 0.95) fitVerdict = 'cpu-offload';

  // gpu-memory-utilization: target the peak per-GPU footprint with ~8% headroom for CUDA graphs etc.
  // Cap at 0.92 so the OS / desktop session always retains some VRAM.
  const targetUtil = perGpuTotalMb > 0
    ? Math.max(0.25, Math.min(0.92, (perGpuMb * 1.08) / perGpuTotalMb))
    : 0.85;

  return {
    weightsMb,
    kvCacheMb,
    activationsMb,
    overheadMb,
    totalMb,
    perGpuMb,
    recommendedGpuMemUtil: Number(targetUtil.toFixed(2)),
    fits: fitVerdict === 'full-gpu' || fitVerdict === 'partial-gpu',
    fitVerdict,
    cpuOffloadMb,
  };
}

// Pick the largest contextLength that still fits in the available GPU pool.
export function recommendMaxContext(input: Omit<EstimateInput, 'contextLength'> & { nativeContext: number; extendedContext: number }): number {
  const upper = Math.max(input.nativeContext, input.extendedContext);
  const candidates = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, upper];
  const unique = Array.from(new Set(candidates.filter(v => v <= upper))).sort((a, b) => a - b);

  let best = unique[0] || 2048;
  for (const ctx of unique) {
    const est = estimateVram({ ...input, contextLength: ctx });
    if (est.fitVerdict === 'full-gpu' || est.fitVerdict === 'partial-gpu') {
      best = ctx;
    } else {
      break;
    }
  }
  return best;
}

// Recommend a CPU offload (GiB) so weights + KV + activations + overhead per GPU
// fit under `targetUtil` (default 0.90) of total GPU memory. Returns 0 if not needed.
export function recommendCpuOffload(input: Omit<EstimateInput, 'cpuOffloadGb'> & { targetUtil?: number }): number {
  const target = input.targetUtil ?? 0.90;
  const tp = Math.max(1, input.tensorParallelSize);
  const perGpuTotalMb = input.totalGpuMb / tp;

  // Try with no offload first.
  const est0 = estimateVram({ ...input, cpuOffloadGb: 0 });
  if (est0.perGpuMb <= perGpuTotalMb * target) return 0;

  // Need to offload (perGpuMb - target * perGpuTotalMb) MB of weights.
  const overflowMb = est0.perGpuMb - perGpuTotalMb * target;
  // perGpuMb attributes (weights/tp). Offloading X GB removes X/tp GB from per-GPU footprint.
  const offloadMb = overflowMb * tp + 256; // +256 MB safety
  return Math.max(0, Math.ceil(offloadMb / 1024));
}
