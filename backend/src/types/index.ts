import { z } from 'zod';

// ---- Model types ----
export interface ModelArchSpec {
  // Layer/attention config used to size KV cache and weights.
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
  modelFile: string | null;       // For GGUF: the actual .gguf file. For HF: same as path.
  format: 'hf' | 'gguf';
  sizeBytes: number;
  architecture: string;
  ggufArchitecture: string | null; // Raw GGUF arch string ("qwen2", "qwen35", etc.)
  quantization: string | null;
  contextLength: number;          // Effective (post-RoPE-scaling) context length.
  nativeContextLength: number;    // Pre-scaling base context length.
  isVision: boolean;
  arch: ModelArchSpec;
  support: ModelSupport;
}

// ---- Chat session types ----
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
}

export interface AttachmentMeta {
  id: string;
  name: string;
  type: 'image' | 'pdf';
  mimeType: string;
  sizeBytes: number;
  path: string;
}

export interface ChatFolder {
  id: string;
  name: string;
  color: string | null;
  createdAt: number;
}

// ---- Inference parameters ----
export const InferenceParamsSchema = z.object({
  temperature: z.number().min(0).max(2).default(0.7),
  topP: z.number().min(0).max(1).default(0.9),
  topK: z.number().int().min(-1).max(200).default(-1),
  maxTokens: z.number().int().min(1).default(1024),
  presencePenalty: z.number().min(-2).max(2).default(0),
  frequencyPenalty: z.number().min(-2).max(2).default(0),
  repetitionPenalty: z.number().min(1).max(2).default(1.0),
  seed: z.number().int().min(-1).default(-1),
  stop: z.array(z.string()).max(4).default([]),
});

export type InferenceParameters = z.infer<typeof InferenceParamsSchema>;

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

// ---- vLLM Server config ----
export interface VllmServerConfig {
  modelPath: string;
  port: number;
  host: string;
  gpuMemoryUtilization: number;
  maxModelLen: number | null;
  quantization: string | null;
  tensorParallelSize: number;
  maxNumSeqs: number;
  dtype: string;
  additionalArgs: string[];
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

// ---- App settings ----
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

export const DEFAULT_SETTINGS: AppSettings = {
  modelScanDirs: ['./data/models'],
  defaultSystemPrompt: '',
  theme: 'dark',
  defaultTemperature: 0.7,
  defaultTopP: 0.9,
  defaultMaxTokens: 1024,
  backendPort: 3333,
  vllmPort: 8000,
  autoStartVllm: false,
  gpuMemoryUtilization: 0.9,
  sendOnEnter: true,
};

// ---- WebSocket message types ----
export type WsMessageType = 'log' | 'gpu_stats' | 'server_status' | 'error';

export interface WsMessage {
  type: WsMessageType;
  data: unknown;
  timestamp: number;
}

// ---- API request/response schemas ----
export const StartServerSchema = z.object({
  modelPath: z.string().min(1),
  port: z.number().int().min(1024).max(65535).default(8000),
  gpuMemoryUtilization: z.number().min(0.1).max(1.0).default(0.9),
  maxModelLen: z.number().int().positive().nullable().default(null),
  quantization: z.enum(['none', 'awq', 'gptq', 'fp8']).nullable().default(null),
  tensorParallelSize: z.number().int().min(1).max(8).default(1),
  maxNumSeqs: z.number().int().min(1).default(256),
  dtype: z.string().default('auto'),
  additionalArgs: z.array(z.string()).default([]),
  loadFormat: z.enum(['auto', 'gguf', 'safetensors']).default('auto'),
  ggufFilePath: z.string().nullable().default(null),
  cpuOffloadGb: z.number().min(0).max(2048).default(0),
  maxNumBatchedTokens: z.number().int().min(0).default(0),
  kvCacheDtype: z.enum(['auto', 'fp16', 'fp8', 'fp8_e4m3', 'fp8_e5m2']).default('auto'),
  enforceEager: z.boolean().default(false),
  hfConfigPath: z.string().nullable().default(null),
  tokenizer: z.string().nullable().default(null),
  languageModelOnly: z.boolean().default(false),
});

export const CreateSessionSchema = z.object({
  title: z.string().min(1).max(500).default('New Chat'),
  modelId: z.string().nullable().default(null),
  folder: z.string().default('default'),
  systemPrompt: z.string().nullable().default(null),
  parameters: InferenceParamsSchema.optional(),
});

export const UpdateSessionSchema = z.object({
  title: z.string().min(1).max(500).optional(),
  folder: z.string().optional(),
  pinned: z.number().min(0).max(1).optional(),
  modelId: z.string().nullable().optional(),
  systemPrompt: z.string().nullable().optional(),
  parameters: InferenceParamsSchema.optional(),
});

export const CreateFolderSchema = z.object({
  name: z.string().min(1).max(100),
  color: z.string().nullable().default(null),
});

export const UpdateFolderSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  color: z.string().nullable().optional(),
});

export const AddMessageSchema = z.object({
  role: z.enum(['user', 'assistant', 'system', 'tool']),
  content: z.string(),
  attachments: z.array(z.object({
    id: z.string(),
    name: z.string(),
    type: z.enum(['image', 'pdf']),
    mimeType: z.string(),
    sizeBytes: z.number(),
    path: z.string(),
  })).nullable().default(null),
  modelId: z.string().nullable().default(null),
  tokensUsed: z.number().int().nullable().default(null),
  finishReason: z.string().nullable().default(null),
});

export const ChatCompletionRequestSchema = z.object({
  sessionId: z.string(),
  messages: z.array(z.object({
    role: z.enum(['user', 'assistant', 'system']),
    content: z.string(),
  })),
  attachments: z.array(z.object({
    type: z.enum(['image', 'pdf']),
    mimeType: z.string(),
    content: z.string(),
  })).optional(),
  parameters: InferenceParamsSchema.optional(),
  stream: z.boolean().default(false),
});

export const SettingsUpdateSchema = z.object({
  modelScanDirs: z.array(z.string()).optional(),
  defaultSystemPrompt: z.string().optional(),
  theme: z.enum(['dark', 'light', 'system']).optional(),
  defaultTemperature: z.number().min(0).max(2).optional(),
  defaultTopP: z.number().min(0).max(1).optional(),
  defaultMaxTokens: z.number().int().min(1).optional(),
  backendPort: z.number().int().min(1024).max(65535).optional(),
  vllmPort: z.number().int().min(1024).max(65535).optional(),
  autoStartVllm: z.boolean().optional(),
  gpuMemoryUtilization: z.number().min(0.1).max(1.0).optional(),
  sendOnEnter: z.boolean().optional(),
});
