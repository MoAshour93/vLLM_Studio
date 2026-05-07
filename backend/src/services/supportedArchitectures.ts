// Canonical registry of model architectures supported by vLLM, pulled from
// https://docs.vllm.ai/en/latest/models/supported_models.html
//
// GGUF support is authoritatively driven by the upstream `transformers` mapping in
// src/transformers/integrations/ggml.py (GGUF_CONFIG_MAPPING keys). When loading a GGUF,
// vLLM delegates config extraction to transformers, so anything in that list works.
//
// Three-state support:
//   - 'supported'    : official, well-tested
//   - 'experimental' : architecture is in vLLM's docs OR transformers' GGUF list,
//                      but combination is less battle-tested. Loading is still allowed.
//   - 'unsupported'  : not in vLLM's list at all — loading will fail.

import {
  isHfArchSupportedByInstalledVllm,
  isGgufArchSupportedByInstalledTransformers,
} from './vllmIntrospect.js';

export type SupportLevel = 'supported' | 'experimental' | 'unsupported';

export interface SupportedArch {
  hfArch: string;
  ggufArch: string[];     // values of `general.architecture` in GGUF metadata (lowercase)
  hfModelTypes: string[]; // values of `model_type` in HF config.json (also used as HF tag)
  family: string;
  modality: 'text' | 'vision' | 'audio' | 'embedding';
}

// Authoritative list of GGUF architectures supported by transformers/vLLM.
// Source: huggingface/transformers src/transformers/integrations/ggml.py
const GGUF_SUPPORTED: ReadonlySet<string> = new Set([
  'llama', 'mistral', 'qwen2', 'qwen2_moe', 'qwen3', 'qwen3_moe',
  'gpt_oss', 'lfm2', 'falcon', 'phi3', 'bloom', 't5', 'stablelm',
  'gpt2', 'starcoder2', 'mamba', 'nemotron', 'gemma2', 'gemma3',
  'umt5', 'deci', 'minimax_m2',
]);

export const SUPPORTED_ARCHS: SupportedArch[] = [
  // ---- Llama family ----
  { hfArch: 'LlamaForCausalLM',          ggufArch: ['llama'],         hfModelTypes: ['llama'],          family: 'Llama',         modality: 'text' },
  { hfArch: 'LLaMAForCausalLM',          ggufArch: ['llama'],         hfModelTypes: ['llama'],          family: 'Llama (alt)',   modality: 'text' },
  { hfArch: 'MllamaForConditionalGeneration', ggufArch: [],           hfModelTypes: ['mllama'],         family: 'Llama 3.2 Vision', modality: 'vision' },

  // ---- Qwen family ----
  { hfArch: 'Qwen2ForCausalLM',          ggufArch: ['qwen2'],         hfModelTypes: ['qwen2'],          family: 'Qwen2 / 2.5',   modality: 'text' },
  { hfArch: 'Qwen3ForCausalLM',          ggufArch: ['qwen3'],         hfModelTypes: ['qwen3'],          family: 'Qwen3',         modality: 'text' },
  { hfArch: 'Qwen2MoeForCausalLM',       ggufArch: ['qwen2_moe'],     hfModelTypes: ['qwen2_moe'],      family: 'Qwen2 MoE',     modality: 'text' },
  { hfArch: 'Qwen3MoeForCausalLM',       ggufArch: ['qwen3_moe'],     hfModelTypes: ['qwen3_moe'],      family: 'Qwen3 MoE',     modality: 'text' },
  { hfArch: 'Qwen2VLForConditionalGeneration', ggufArch: [],          hfModelTypes: ['qwen2_vl'],       family: 'Qwen2-VL',      modality: 'vision' },
  { hfArch: 'Qwen2_5_VLForConditionalGeneration', ggufArch: [],       hfModelTypes: ['qwen2_5_vl'],     family: 'Qwen2.5-VL',    modality: 'vision' },

  // ---- Mistral / Mixtral ----
  // Note: Mixtral GGUFs typically use 'llama' arch, not a dedicated 'mixtral' arch.
  { hfArch: 'MistralForCausalLM',        ggufArch: ['mistral', 'llama'], hfModelTypes: ['mistral'],     family: 'Mistral',       modality: 'text' },
  { hfArch: 'MixtralForCausalLM',        ggufArch: ['llama'],         hfModelTypes: ['mixtral'],        family: 'Mixtral',       modality: 'text' },

  // ---- DeepSeek ----
  { hfArch: 'DeepseekV2ForCausalLM',     ggufArch: ['deepseek2'],     hfModelTypes: ['deepseek_v2'],    family: 'DeepSeek V2',   modality: 'text' },
  { hfArch: 'DeepseekV3ForCausalLM',     ggufArch: ['deepseek3'],     hfModelTypes: ['deepseek_v3'],    family: 'DeepSeek V3',   modality: 'text' },

  // ---- Gemma ----
  { hfArch: 'GemmaForCausalLM',          ggufArch: ['gemma'],         hfModelTypes: ['gemma'],          family: 'Gemma',         modality: 'text' },
  { hfArch: 'Gemma2ForCausalLM',         ggufArch: ['gemma2'],        hfModelTypes: ['gemma2'],         family: 'Gemma 2',       modality: 'text' },
  { hfArch: 'Gemma3ForCausalLM',         ggufArch: ['gemma3'],        hfModelTypes: ['gemma3'],         family: 'Gemma 3',       modality: 'text' },
  { hfArch: 'Gemma3ForConditionalGeneration', ggufArch: [],           hfModelTypes: ['gemma3_text'],    family: 'Gemma 3 Vision',modality: 'vision' },

  // ---- Phi ----
  { hfArch: 'PhiForCausalLM',            ggufArch: ['phi2'],          hfModelTypes: ['phi'],            family: 'Phi-1 / Phi-2', modality: 'text' },
  { hfArch: 'Phi3ForCausalLM',           ggufArch: ['phi3'],          hfModelTypes: ['phi3'],           family: 'Phi-3 / Phi-4', modality: 'text' },
  { hfArch: 'Phi3SmallForCausalLM',      ggufArch: ['phi3'],          hfModelTypes: ['phi3small'],      family: 'Phi-3 Small',   modality: 'text' },
  { hfArch: 'Phi3VForCausalLM',          ggufArch: [],                hfModelTypes: ['phi3_v'],         family: 'Phi-3 Vision',  modality: 'vision' },
  { hfArch: 'PhiMoEForCausalLM',         ggufArch: [],                hfModelTypes: ['phimoe'],         family: 'Phi MoE',       modality: 'text' },

  // ---- ChatGLM / GLM ----
  { hfArch: 'ChatGLMModel',              ggufArch: ['chatglm'],       hfModelTypes: ['chatglm'],        family: 'ChatGLM',       modality: 'text' },
  { hfArch: 'ChatGLMForConditionalGeneration', ggufArch: ['chatglm'], hfModelTypes: ['chatglm'],        family: 'ChatGLM',       modality: 'text' },
  { hfArch: 'GlmForCausalLM',            ggufArch: ['glm'],           hfModelTypes: ['glm'],            family: 'GLM',           modality: 'text' },
  { hfArch: 'Glm4ForCausalLM',           ggufArch: ['glm4'],          hfModelTypes: ['glm4'],           family: 'GLM-4',         modality: 'text' },

  // ---- Cohere ----
  { hfArch: 'CohereForCausalLM',         ggufArch: ['command-r'],     hfModelTypes: ['cohere'],         family: 'Cohere Command-R', modality: 'text' },
  { hfArch: 'Cohere2ForCausalLM',        ggufArch: ['command-r'],     hfModelTypes: ['cohere2'],        family: 'Cohere v2',     modality: 'text' },

  // ---- Falcon / Starcoder / StableLM ----
  { hfArch: 'FalconForCausalLM',         ggufArch: ['falcon'],        hfModelTypes: ['falcon'],         family: 'Falcon',        modality: 'text' },
  { hfArch: 'FalconMambaForCausalLM',    ggufArch: [],                hfModelTypes: ['falcon_mamba'],   family: 'Falcon Mamba',  modality: 'text' },
  { hfArch: 'Starcoder2ForCausalLM',     ggufArch: ['starcoder2'],    hfModelTypes: ['starcoder2'],     family: 'StarCoder2',    modality: 'text' },
  { hfArch: 'StableLmForCausalLM',       ggufArch: ['stablelm'],      hfModelTypes: ['stablelm'],       family: 'StableLM',      modality: 'text' },

  // ---- Granite ----
  { hfArch: 'GraniteForCausalLM',        ggufArch: ['granite'],       hfModelTypes: ['granite'],        family: 'Granite',       modality: 'text' },
  { hfArch: 'GraniteMoeForCausalLM',     ggufArch: ['granitemoe'],    hfModelTypes: ['granitemoe'],     family: 'Granite MoE',   modality: 'text' },

  // ---- InternLM ----
  { hfArch: 'InternLMForCausalLM',       ggufArch: ['internlm'],      hfModelTypes: ['internlm'],       family: 'InternLM',      modality: 'text' },
  { hfArch: 'InternLM2ForCausalLM',      ggufArch: ['internlm2'],     hfModelTypes: ['internlm2'],      family: 'InternLM2',     modality: 'text' },
  { hfArch: 'InternLM3ForCausalLM',      ggufArch: ['internlm3'],     hfModelTypes: ['internlm3'],      family: 'InternLM3',     modality: 'text' },

  // ---- OLMo / Aquila / Baichuan / Orion ----
  { hfArch: 'OlmoForCausalLM',           ggufArch: ['olmo'],          hfModelTypes: ['olmo'],           family: 'OLMo',          modality: 'text' },
  { hfArch: 'Olmo2ForCausalLM',          ggufArch: ['olmo2'],         hfModelTypes: ['olmo2'],          family: 'OLMo 2',        modality: 'text' },
  { hfArch: 'Olmo3ForCausalLM',          ggufArch: ['olmo3'],         hfModelTypes: ['olmo3'],          family: 'OLMo 3',        modality: 'text' },
  { hfArch: 'AquilaForCausalLM',         ggufArch: ['aquila'],        hfModelTypes: ['aquila'],         family: 'Aquila',        modality: 'text' },
  { hfArch: 'BaiChuanForCausalLM',       ggufArch: ['baichuan'],      hfModelTypes: ['baichuan'],       family: 'Baichuan',      modality: 'text' },
  { hfArch: 'OrionForCausalLM',          ggufArch: ['orion'],         hfModelTypes: ['orion'],          family: 'Orion',         modality: 'text' },

  // ---- Mamba / Jamba ----
  { hfArch: 'MambaForCausalLM',          ggufArch: ['mamba'],         hfModelTypes: ['mamba'],          family: 'Mamba',         modality: 'text' },
  { hfArch: 'Mamba2ForCausalLM',         ggufArch: ['mamba2'],        hfModelTypes: ['mamba2'],         family: 'Mamba 2',       modality: 'text' },
  { hfArch: 'JambaForCausalLM',          ggufArch: ['jamba'],         hfModelTypes: ['jamba'],          family: 'Jamba',         modality: 'text' },

  // ---- NVIDIA / EXAONE / Misc ----
  { hfArch: 'NemotronForCausalLM',       ggufArch: ['nemotron'],      hfModelTypes: ['nemotron'],       family: 'Nemotron',      modality: 'text' },
  { hfArch: 'ExaoneForCausalLM',         ggufArch: ['exaone'],        hfModelTypes: ['exaone'],         family: 'EXAONE',        modality: 'text' },
  { hfArch: 'MiniCPMForCausalLM',        ggufArch: ['minicpm'],       hfModelTypes: ['minicpm'],        family: 'MiniCPM',       modality: 'text' },
  { hfArch: 'MiniMaxText01ForCausalLM',  ggufArch: ['minimax_m2'],    hfModelTypes: ['minimax_text_01'],family: 'MiniMax Text-01', modality: 'text' },

  // ---- gpt-oss / Liquid LFM2 ----
  { hfArch: 'GptOssForCausalLM',         ggufArch: ['gpt_oss'],       hfModelTypes: ['gpt_oss'],        family: 'gpt-oss',       modality: 'text' },
  { hfArch: 'Lfm2ForCausalLM',           ggufArch: ['lfm2'],          hfModelTypes: ['lfm2'],           family: 'Liquid LFM2',   modality: 'text' },
  { hfArch: 'DeciLMForCausalLM',         ggufArch: ['deci'],          hfModelTypes: ['deci'],           family: 'Deci',          modality: 'text' },

  // ---- Older / OPT / GPT-NeoX / Bloom ----
  { hfArch: 'OPTForCausalLM',            ggufArch: ['opt'],           hfModelTypes: ['opt'],            family: 'OPT',           modality: 'text' },
  { hfArch: 'GPTNeoXForCausalLM',        ggufArch: ['gptneox'],       hfModelTypes: ['gpt_neox'],       family: 'GPT-NeoX / Pythia', modality: 'text' },
  { hfArch: 'BloomForCausalLM',          ggufArch: ['bloom'],         hfModelTypes: ['bloom'],          family: 'BLOOM',         modality: 'text' },
  { hfArch: 'GPT2LMHeadModel',           ggufArch: ['gpt2'],          hfModelTypes: ['gpt2'],           family: 'GPT-2',         modality: 'text' },
  { hfArch: 'GPTBigCodeForCausalLM',     ggufArch: ['gptbigcode'],    hfModelTypes: ['gpt_bigcode'],    family: 'StarCoder',     modality: 'text' },
  { hfArch: 'GPTJForCausalLM',           ggufArch: ['gptj'],          hfModelTypes: ['gptj'],           family: 'GPT-J',         modality: 'text' },

  // ---- Vision (additional) ----
  { hfArch: 'LlavaForConditionalGeneration',     ggufArch: [], hfModelTypes: ['llava'],          family: 'LLaVA',         modality: 'vision' },
  { hfArch: 'LlavaNextForConditionalGeneration', ggufArch: [], hfModelTypes: ['llava_next'],     family: 'LLaVA-NeXT',    modality: 'vision' },
  { hfArch: 'LlavaOnevisionForConditionalGeneration', ggufArch: [], hfModelTypes: ['llava_onevision'], family: 'LLaVA-OneVision', modality: 'vision' },
  { hfArch: 'PaliGemmaForConditionalGeneration', ggufArch: [], hfModelTypes: ['paligemma'],      family: 'PaliGemma',     modality: 'vision' },
  { hfArch: 'Idefics2ForConditionalGeneration',  ggufArch: [], hfModelTypes: ['idefics2'],       family: 'Idefics2',      modality: 'vision' },
  { hfArch: 'Idefics3ForConditionalGeneration',  ggufArch: [], hfModelTypes: ['idefics3'],       family: 'Idefics3',      modality: 'vision' },
  { hfArch: 'MiniCPMV',                          ggufArch: [], hfModelTypes: ['minicpmv'],       family: 'MiniCPM-V',     modality: 'vision' },
  { hfArch: 'PixtralForConditionalGeneration',   ggufArch: [], hfModelTypes: ['pixtral'],        family: 'Pixtral',       modality: 'vision' },
];

const HF_ARCH_INDEX = new Map<string, SupportedArch>();
const GGUF_ARCH_INDEX = new Map<string, SupportedArch>();
const MODEL_TYPE_INDEX = new Map<string, SupportedArch>();
for (const a of SUPPORTED_ARCHS) {
  HF_ARCH_INDEX.set(a.hfArch.toLowerCase(), a);
  for (const g of a.ggufArch) GGUF_ARCH_INDEX.set(g.toLowerCase(), a);
  for (const t of a.hfModelTypes) MODEL_TYPE_INDEX.set(t.toLowerCase(), a);
}

export interface SupportLookup {
  level: SupportLevel;
  family: string | null;
  ggufSupported: boolean;
  reason?: string;
}

function ggufLevelFor(arch: SupportedArch | undefined, ggufArchSeen?: string | null): SupportLevel {
  // 1) If the GGUF metadata explicitly names an architecture transformers supports, it's supported.
  if (ggufArchSeen && GGUF_SUPPORTED.has(ggufArchSeen.toLowerCase())) return 'supported';
  // 2) If we mapped to a registered family AND it has at least one ggufArch in the upstream list, supported.
  if (arch) {
    for (const g of arch.ggufArch) {
      if (GGUF_SUPPORTED.has(g.toLowerCase())) return 'supported';
    }
    // 3) Family is in vLLM's docs but its GGUF path isn't in transformers' upstream list — experimental.
    return 'experimental';
  }
  return 'unsupported';
}

export function lookupByHfArchitecture(arch: string | null | undefined): SupportLookup {
  if (!arch) return { level: 'unsupported', family: null, ggufSupported: false, reason: 'no architecture metadata' };
  const installedSupports = isHfArchSupportedByInstalledVllm(arch);
  const a = HF_ARCH_INDEX.get(arch.toLowerCase());

  if (installedSupports) {
    return { level: 'supported', family: a?.family ?? arch, ggufSupported: a ? ggufLevelFor(a) === 'supported' : false };
  }
  if (a) {
    // Known family but not in the installed vLLM build.
    return {
      level: 'experimental',
      family: a.family,
      ggufSupported: false,
      reason: `${a.family} (${arch}) is documented as supported but not present in your installed vLLM. Try upgrading vLLM.`,
    };
  }
  return {
    level: 'unsupported',
    family: null,
    ggufSupported: false,
    reason: `architecture "${arch}" is not in your installed vLLM's registry. Upgrade vLLM or use the Transformers modeling backend (--model-impl transformers).`,
  };
}

// Now consults the *installed* transformers GGUF list at runtime so we don't
// lag the upstream library.
export function lookupByGgufArchitecture(arch: string | null | undefined): SupportLookup {
  if (!arch) return { level: 'unsupported', family: null, ggufSupported: false, reason: 'no GGUF architecture metadata' };
  const a = GGUF_ARCH_INDEX.get(arch.toLowerCase());
  const installedSupports = isGgufArchSupportedByInstalledTransformers(arch);

  if (installedSupports) {
    return {
      level: 'supported',
      family: a?.family ?? arch,
      ggufSupported: true,
    };
  }
  if (a) {
    return {
      level: 'experimental',
      family: a.family,
      ggufSupported: false,
      reason: `${a.family} runs in vLLM but the installed transformers does not yet support "${arch}" as a GGUF architecture. Try upgrading transformers, or load the safetensors release.`,
    };
  }
  return {
    level: 'experimental',
    family: null,
    ggufSupported: false,
    reason: `GGUF architecture "${arch}" is not in the installed transformers' GGUF map. vLLM may still load it if you provide a base HuggingFace repo whose config + tokenizer match this model.`,
  };
}

export function lookupByModelType(modelType: string | null | undefined): SupportLookup {
  if (!modelType) return { level: 'unsupported', family: null, ggufSupported: false };
  const a = MODEL_TYPE_INDEX.get(modelType.toLowerCase());
  if (a) {
    const gguf = ggufLevelFor(a) === 'supported';
    return { level: 'supported', family: a.family, ggufSupported: gguf };
  }
  return { level: 'unsupported', family: null, ggufSupported: false };
}

export function lookupByTags(tags: string[] | undefined): SupportLookup {
  if (!tags || tags.length === 0) return { level: 'unsupported', family: null, ggufSupported: false };
  for (const t of tags) {
    const hit = lookupByModelType(t);
    if (hit.level !== 'unsupported') return hit;
  }
  return { level: 'unsupported', family: null, ggufSupported: false };
}

// Best-effort detection. `isGguf` makes us also account for GGUF-loadability.
export function detectSupport(args: {
  hfArchitecture?: string | null;
  ggufArchitecture?: string | null;
  hfModelType?: string | null;
  tags?: string[];
  modelId?: string;
  isGguf?: boolean;
}): SupportLookup {
  if (args.ggufArchitecture) {
    const r = lookupByGgufArchitecture(args.ggufArchitecture);
    return r;
  }
  if (args.hfArchitecture) {
    const r = lookupByHfArchitecture(args.hfArchitecture);
    if (r.level !== 'unsupported' || r.reason?.includes('not in vLLM')) {
      return finaliseGgufFlag(r, args.isGguf);
    }
  }
  if (args.hfModelType) {
    const r = lookupByModelType(args.hfModelType);
    if (r.level !== 'unsupported') return finaliseGgufFlag(r, args.isGguf);
  }
  if (args.tags) {
    const r = lookupByTags(args.tags);
    if (r.level !== 'unsupported') return finaliseGgufFlag(r, args.isGguf);
  }
  if (args.modelId) {
    const id = args.modelId.toLowerCase();
    for (const a of SUPPORTED_ARCHS) {
      for (const t of a.hfModelTypes) {
        if (id.includes(t.toLowerCase())) {
          const gguf = ggufLevelFor(a) === 'supported';
          return finaliseGgufFlag({ level: 'supported', family: a.family, ggufSupported: gguf }, args.isGguf);
        }
      }
    }
  }
  return { level: 'unsupported', family: null, ggufSupported: false, reason: 'could not determine architecture' };
}

function finaliseGgufFlag(r: SupportLookup, isGguf?: boolean): SupportLookup {
  if (!isGguf) return r;
  if (r.level === 'unsupported') return r;
  if (r.ggufSupported) return r;
  // Family is supported but its GGUF path is not in transformers' list — flag as experimental.
  return {
    level: 'experimental',
    family: r.family,
    ggufSupported: false,
    reason: `${r.family ?? 'This family'} runs in vLLM, but its GGUF path is not in transformers' supported list — loading may fail.`,
  };
}

export function listSupportedFamilies(): Array<{ family: string; modality: string; ggufSupported: boolean }> {
  return SUPPORTED_ARCHS.map(a => ({
    family: a.family,
    modality: a.modality,
    ggufSupported: a.ggufArch.some(g => GGUF_SUPPORTED.has(g.toLowerCase())),
  }));
}

export function isGgufArchSupportedUpstream(arch: string | null | undefined): boolean {
  if (!arch) return false;
  return GGUF_SUPPORTED.has(arch.toLowerCase());
}
