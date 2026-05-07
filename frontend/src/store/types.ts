export interface ModelArchSpec {
  numLayers: number | null;
  hiddenSize: number | null;
  numHeads: number | null;
  numKvHeads: number | null;
  headDim: number | null;
  vocabSize: number | null;
  intermediateSize: number | null;
  totalParams: number | null;
  fileSizeBytes: number | null;
  dtype: string | null;
  kvCacheDtype: string | null;
  bitsPerWeight: number | null;
  quantization: string | null;
}

export type ModelSupportLevel = 'supported' | 'experimental' | 'unsupported';

export interface ModelSupport {
  level: ModelSupportLevel;
  family: string | null;
  ggufSupported: boolean;
  reason?: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  path: string;
  modelFile: string | null;
  format: 'hf' | 'gguf';
  sizeBytes: number;
  architecture: string;
  ggufArchitecture: string | null;
  quantization: string | null;
  contextLength: number;
  nativeContextLength: number;
  isVision: boolean;
  arch: ModelArchSpec;
  support: ModelSupport;
  fitInfo?: FitInfo;
}

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

export interface PreflightResponse {
  model: ModelInfo;
  gpus: GpuStats[];
  totalGpuMb: number;
  freeGpuMb: number;
  systemRamMb: number;
  contextLength: number;
  recommendedContextLength: number;
  recommendedCpuOffloadGb: number;
  suggestedBaseRepo: string | null;
  ggufNeedsBaseConfig: boolean;
  estimate: VramEstimate;
  prefs: ModelPrefs | null;
}

export interface ModelPrefs {
  path: string;
  maxModelLen: number | null;
  gpuMemoryUtilization: number | null;
  quantization: string | null;
  dtype: string | null;
  tensorParallelSize: number | null;
  maxNumSeqs: number | null;
  additionalArgs: string[];
}

export interface FitInfo {
  fit: 'full-gpu' | 'partial-gpu' | 'cpu-offload' | 'too-large';
  label: string;
  gpuMemoryMb: number;
  systemRamMb: number;
  requiredMemoryMb: number;
}

export interface ChatSession {
  id: string;
  title: string;
  modelId: string | null;
  createdAt: number;
  updatedAt: number;
  folder: string;
  pinned: number;
  systemPrompt: string | null;
  parameters: InferenceParameters;
  messages?: ChatMessage[];
  messageCount?: number;
}

export interface ChatMessage {
  id: string;
  sessionId: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  attachments: AttachmentMeta[] | null;
  createdAt: number;
  tokensUsed: number | null;
  modelId: string | null;
  finishReason: string | null;
  chatStats?: ChatMessageStats | null;
}

export interface ChatMessageStats {
  promptTokens: number;
  outputTokens: number;
  ttftMs: number;
  totalTimeMs: number;
  tokensPerSec: number;
  totalTokens: number;
}

export interface AttachmentMeta {
  id: string;
  name: string;
  type: 'image' | 'pdf';
  mimeType: string;
  sizeBytes: number;
  path: string;
  content?: string | null;
}

export interface ChatFolder {
  id: string;
  name: string;
  color: string | null;
  createdAt: number;
}

export interface InferenceParameters {
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  presencePenalty: number;
  frequencyPenalty: number;
  repetitionPenalty: number;
  seed: number;
  stop: string[];
}

export const DEFAULT_INFERENCE_PARAMS: InferenceParameters = {
  temperature: 0.7,
  topP: 0.9,
  topK: -1,
  maxTokens: 1024,
  presencePenalty: 0,
  frequencyPenalty: 0,
  repetitionPenalty: 1.0,
  seed: -1,
  stop: [],
};

export interface AppSettings {
  modelScanDirs: string[];
  defaultSystemPrompt: string;
  theme: 'dark' | 'light' | 'system';
  defaultTemperature: number;
  defaultTopP: number;
  defaultMaxTokens: number;
  backendPort: number;
  vllmPort: number;
  autoStartVllm: boolean;
  gpuMemoryUtilization: number;
  sendOnEnter: boolean;
}

export type ServerStatus = 'stopped' | 'starting' | 'running' | 'error';

export interface GpuStats {
  index: number;
  name: string;
  totalMemoryMb: number;
  usedMemoryMb: number;
  freeMemoryMb: number;
  utilizationPercent: number;
  temperatureC: number | null;
}

export interface SystemResources {
  gpus: GpuStats[];
  ramTotalBytes: number;
  ramUsedBytes: number;
  ramFreeBytes: number;
  cpuPercent: number;
  timestamp: number;
}

export interface VllmServerConfig {
  modelPath: string;
  port: number;
  gpuMemoryUtilization: number;
  maxModelLen: number | null;
  quantization: string | null;
  tensorParallelSize: number;
  maxNumSeqs: number;
  dtype: string;
  additionalArgs: string[];
  loadFormat?: 'auto' | 'gguf' | 'safetensors';
  ggufFilePath?: string | null;
  cpuOffloadGb?: number;
  maxNumBatchedTokens?: number;
  kvCacheDtype?: 'auto' | 'fp16' | 'fp8' | 'fp8_e4m3' | 'fp8_e5m2';
  enforceEager?: boolean;
  hfConfigPath?: string | null;
  tokenizer?: string | null;
  languageModelOnly?: boolean;
}

export interface ChatCompletionRequest {
  sessionId: string;
  messages: { role: string; content: string }[];
  parameters?: Partial<InferenceParameters>;
  stream?: boolean;
}
