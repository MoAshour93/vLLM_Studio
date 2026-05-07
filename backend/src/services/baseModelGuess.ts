// Heuristic: from a local GGUF dir/file path, guess the canonical HuggingFace
// base-model repo id whose config.json + tokenizer can stand in for the GGUF.
// Used to auto-fill --hf-config-path and --tokenizer per vLLM's GGUF docs.

import path from 'path';

interface Candidate {
  pattern: RegExp;
  // (match) => "Org/Model-Name" — first group is the size token (e.g. "9B").
  toRepo: (m: RegExpMatchArray) => string;
}

const CANDIDATES: Candidate[] = [
  // Qwen3.x — fall back to closest released Qwen3 size.
  { pattern: /qwen[ _-]*3\.6[ _-]*([\d.]+b)/i,        toRepo: (m) => `Qwen/Qwen3-${m[1].toUpperCase()}` },
  { pattern: /qwen[ _-]*3\.5[ _-]*([\d.]+b)/i,        toRepo: (m) => `Qwen/Qwen3-${m[1].toUpperCase()}` },
  { pattern: /qwen[ _-]*3[ _-]*([\d.]+b)[ _-]*moe/i,  toRepo: (m) => `Qwen/Qwen3-${m[1].toUpperCase()}-A3B` },
  { pattern: /qwen[ _-]*3[ _-]*([\d.]+b)/i,           toRepo: (m) => `Qwen/Qwen3-${m[1].toUpperCase()}` },
  { pattern: /qwen[ _-]*2\.5[ _-]*coder[ _-]*([\d.]+b)/i, toRepo: (m) => `Qwen/Qwen2.5-Coder-${m[1].toUpperCase()}-Instruct` },
  { pattern: /qwen[ _-]*2\.5[ _-]*([\d.]+b)/i,        toRepo: (m) => `Qwen/Qwen2.5-${m[1].toUpperCase()}-Instruct` },
  { pattern: /qwen[ _-]*2[ _-]*([\d.]+b)/i,           toRepo: (m) => `Qwen/Qwen2-${m[1].toUpperCase()}-Instruct` },

  // Llama 3.x
  { pattern: /llama[ _-]*3\.3[ _-]*([\d.]+b)/i,       toRepo: (m) => `meta-llama/Llama-3.3-${m[1].toUpperCase()}-Instruct` },
  { pattern: /llama[ _-]*3\.2[ _-]*([\d.]+b)/i,       toRepo: (m) => `meta-llama/Llama-3.2-${m[1].toUpperCase()}-Instruct` },
  { pattern: /llama[ _-]*3\.1[ _-]*([\d.]+b)/i,       toRepo: (m) => `meta-llama/Llama-3.1-${m[1].toUpperCase()}-Instruct` },
  { pattern: /llama[ _-]*3[ _-]*([\d.]+b)/i,          toRepo: (m) => `meta-llama/Meta-Llama-3-${m[1].toUpperCase()}-Instruct` },
  { pattern: /llama[ _-]*2[ _-]*([\d.]+b)/i,          toRepo: (m) => `meta-llama/Llama-2-${m[1].toLowerCase()}-chat-hf` },

  // Mistral / Mixtral
  { pattern: /mixtral[ _-]*([\d.]+x[\d.]+b)/i,        toRepo: (m) => `mistralai/Mixtral-${m[1].toUpperCase()}-Instruct-v0.1` },
  { pattern: /mistral[ _-]*nemo/i,                    toRepo: () => `mistralai/Mistral-Nemo-Instruct-2407` },
  { pattern: /mistral[ _-]*([\d.]+b)/i,               toRepo: (m) => `mistralai/Mistral-${m[1].toUpperCase()}-Instruct-v0.3` },

  // Gemma
  { pattern: /gemma[ _-]*3[ _-]*([\d.]+b)/i,          toRepo: (m) => `google/gemma-3-${m[1].toLowerCase()}-it` },
  { pattern: /gemma[ _-]*2[ _-]*([\d.]+b)/i,          toRepo: (m) => `google/gemma-2-${m[1].toLowerCase()}-it` },

  // Phi
  { pattern: /phi[ _-]*4[ _-]*mini/i,                 toRepo: () => `microsoft/Phi-4-mini-instruct` },
  { pattern: /phi[ _-]*4/i,                           toRepo: () => `microsoft/phi-4` },
  { pattern: /phi[ _-]*3\.5[ _-]*mini/i,              toRepo: () => `microsoft/Phi-3.5-mini-instruct` },
  { pattern: /phi[ _-]*3[ _-]*mini/i,                 toRepo: () => `microsoft/Phi-3-mini-128k-instruct` },

  // DeepSeek R1 distills (Llama or Qwen based)
  { pattern: /deepseek[ _-]*r1[ _-]*distill[ _-]*qwen[ _-]*([\d.]+b)/i, toRepo: (m) => {
    const b = parseFloat(m[1]);
    if (b <= 1.5) return `Qwen/Qwen2-${m[1].toUpperCase()}-Instruct`;
    return `Qwen/Qwen2-${Math.ceil(b)}B-Instruct`;
  }},
  { pattern: /deepseek[ _-]*r1[ _-]*distill[ _-]*llama[ _-]*([\d.]+b)/i, toRepo: (m) => `meta-llama/Llama-3.1-${m[1].toUpperCase()}-Instruct` },

  // Generic Qwen without a version prefix (e.g. "Qwen-1.5B", "Qwen-14B")
  { pattern: /qwen[ _-]*([\d.]+b)/i, toRepo: (m) => `Qwen/Qwen2-${m[1].toUpperCase()}-Instruct` },
  { pattern: /deepseek[ _-]*v3/i,                     toRepo: () => `deepseek-ai/DeepSeek-V3` },
  { pattern: /deepseek[ _-]*v2/i,                     toRepo: () => `deepseek-ai/DeepSeek-V2-Chat` },
  { pattern: /deepseek[ _-]*coder[ _-]*v2[ _-]*lite/i, toRepo: () => `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` },

  // GLM
  { pattern: /glm[ _-]*4[ _-]*([\d.]+b)/i,            toRepo: (m) => `THUDM/glm-4-${m[1].toLowerCase()}-chat` },

  // Cohere
  { pattern: /command[ _-]*r[ _-]*plus/i,             toRepo: () => `CohereLabs/c4ai-command-r-plus` },
  { pattern: /command[ _-]*r/i,                       toRepo: () => `CohereLabs/c4ai-command-r-v01` },

  // StarCoder2
  { pattern: /starcoder[ _-]*2[ _-]*([\d.]+b)/i,      toRepo: (m) => `bigcode/starcoder2-${m[1].toLowerCase()}` },
];

export function guessBaseModelRepo(modelPath: string): string | null {
  const haystack = path.basename(modelPath).replace(/__/g, '/').replace(/\.gguf$/i, '');
  const dirHaystack = modelPath; // also test the full path so e.g. ".../unsloth__Qwen3.5-9B-GGUF/Q4_K_M/..." works
  for (const c of CANDIDATES) {
    const m = haystack.match(c.pattern) || dirHaystack.match(c.pattern);
    if (m) return c.toRepo(m);
  }
  return null;
}
