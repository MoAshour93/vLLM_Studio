import fs from 'fs';

export interface GgufMetadata {
  architecture: string;
  contextLength: number | null;
  embeddingLength: number | null;
  blockCount: number | null;
  headCount: number | null;
  headCountKv: number | null;
  feedForwardLength: number | null;
  ropeScalingType: string | null;
  ropeScalingFactor: number | null;
  ropeOriginalContextLength: number | null;
  ggufVersion: number;
  tensorCount: number;
  fileType: number | null;
  raw: Record<string, unknown>;
}

const MAGIC = 0x46554747; // 'GGUF' little-endian

const enum ValueType {
  UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3,
  UINT32 = 4, INT32 = 5, FLOAT32 = 6, BOOL = 7,
  STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
}

class Cursor {
  offset = 0;
  constructor(private fd: number, private buf: Buffer, private fileSize: number) {}

  private ensure(n: number): void {
    if (this.offset + n > this.buf.length) {
      // grow buffer by reading more from disk
      const nextSize = Math.min(this.fileSize, Math.max(this.buf.length * 2, this.offset + n));
      if (nextSize <= this.buf.length) {
        throw new Error(`GGUF read past EOF at offset ${this.offset}`);
      }
      const grown = Buffer.alloc(nextSize);
      this.buf.copy(grown);
      const need = nextSize - this.buf.length;
      const tmp = Buffer.alloc(need);
      const read = fs.readSync(this.fd, tmp, 0, need, this.buf.length);
      tmp.copy(grown, this.buf.length, 0, read);
      this.buf = grown;
    }
  }

  u8(): number { this.ensure(1); return this.buf.readUInt8(this.offset++); }
  i8(): number { this.ensure(1); const v = this.buf.readInt8(this.offset); this.offset += 1; return v; }
  u16(): number { this.ensure(2); const v = this.buf.readUInt16LE(this.offset); this.offset += 2; return v; }
  i16(): number { this.ensure(2); const v = this.buf.readInt16LE(this.offset); this.offset += 2; return v; }
  u32(): number { this.ensure(4); const v = this.buf.readUInt32LE(this.offset); this.offset += 4; return v; }
  i32(): number { this.ensure(4); const v = this.buf.readInt32LE(this.offset); this.offset += 4; return v; }
  f32(): number { this.ensure(4); const v = this.buf.readFloatLE(this.offset); this.offset += 4; return v; }
  u64(): number { this.ensure(8); const v = this.buf.readBigUInt64LE(this.offset); this.offset += 8; return Number(v); }
  i64(): number { this.ensure(8); const v = this.buf.readBigInt64LE(this.offset); this.offset += 8; return Number(v); }
  f64(): number { this.ensure(8); const v = this.buf.readDoubleLE(this.offset); this.offset += 8; return v; }
  bool(): boolean { return this.u8() !== 0; }

  string(): string {
    const len = this.u64();
    this.ensure(len);
    const s = this.buf.slice(this.offset, this.offset + len).toString('utf-8');
    this.offset += len;
    return s;
  }
}

function readValue(c: Cursor, type: number): unknown {
  switch (type) {
    case ValueType.UINT8:   return c.u8();
    case ValueType.INT8:    return c.i8();
    case ValueType.UINT16:  return c.u16();
    case ValueType.INT16:   return c.i16();
    case ValueType.UINT32:  return c.u32();
    case ValueType.INT32:   return c.i32();
    case ValueType.FLOAT32: return c.f32();
    case ValueType.BOOL:    return c.bool();
    case ValueType.STRING:  return c.string();
    case ValueType.UINT64:  return c.u64();
    case ValueType.INT64:   return c.i64();
    case ValueType.FLOAT64: return c.f64();
    case ValueType.ARRAY: {
      const elemType = c.u32();
      const count = c.u64();
      const arr: unknown[] = [];
      const skipLargeArrays = count > 4096;
      for (let i = 0; i < count; i++) {
        const v = readValue(c, elemType);
        if (!skipLargeArrays) arr.push(v);
      }
      return skipLargeArrays ? `[array<${count}>]` : arr;
    }
    default:
      throw new Error(`Unknown GGUF value type ${type}`);
  }
}

export function readGgufMetadata(filePath: string): GgufMetadata | null {
  let fd: number | null = null;
  try {
    const stat = fs.statSync(filePath);
    fd = fs.openSync(filePath, 'r');

    // Read first 1 MiB; Cursor will grow on demand.
    const initial = Buffer.alloc(Math.min(1024 * 1024, stat.size));
    fs.readSync(fd, initial, 0, initial.length, 0);

    const c = new Cursor(fd, initial, stat.size);

    const magic = c.u32();
    if (magic !== MAGIC) return null;

    const version = c.u32();
    const tensorCount = c.u64();
    const kvCount = c.u64();

    const raw: Record<string, unknown> = {};
    for (let i = 0; i < kvCount; i++) {
      const key = c.string();
      const type = c.u32();
      const value = readValue(c, type);
      raw[key] = value;
    }

    const arch = (raw['general.architecture'] as string) || 'unknown';
    const num = (k: string): number | null => {
      const v = raw[k];
      return typeof v === 'number' ? v : null;
    };
    const str = (k: string): string | null => {
      const v = raw[k];
      return typeof v === 'string' ? v : null;
    };

    return {
      architecture: arch,
      contextLength: num(`${arch}.context_length`),
      embeddingLength: num(`${arch}.embedding_length`),
      blockCount: num(`${arch}.block_count`),
      headCount: num(`${arch}.attention.head_count`),
      headCountKv: num(`${arch}.attention.head_count_kv`),
      feedForwardLength: num(`${arch}.feed_forward_length`),
      ropeScalingType: str(`${arch}.rope.scaling.type`),
      ropeScalingFactor: num(`${arch}.rope.scaling.factor`),
      ropeOriginalContextLength: num(`${arch}.rope.scaling.original_context_length`),
      ggufVersion: version,
      tensorCount,
      fileType: num('general.file_type'),
      raw,
    };
  } catch {
    return null;
  } finally {
    if (fd !== null) {
      try { fs.closeSync(fd); } catch { /* ignore */ }
    }
  }
}

const ARCH_TO_VLLM: Record<string, string> = {
  llama: 'LlamaForCausalLM',
  qwen2: 'Qwen2ForCausalLM',
  qwen3: 'Qwen3ForCausalLM',
  qwen35: 'Qwen2ForCausalLM',
  mistral: 'MistralForCausalLM',
  mixtral: 'MixtralForCausalLM',
  phi3: 'Phi3ForCausalLM',
  phi: 'PhiForCausalLM',
  gemma: 'GemmaForCausalLM',
  gemma2: 'Gemma2ForCausalLM',
  falcon: 'FalconForCausalLM',
  starcoder2: 'Starcoder2ForCausalLM',
  deepseek: 'LlamaForCausalLM',
  deepseek2: 'DeepseekV2ForCausalLM',
};

export function ggufArchToVllm(arch: string): string {
  return ARCH_TO_VLLM[arch.toLowerCase()] ?? `${arch[0].toUpperCase()}${arch.slice(1)}ForCausalLM`;
}

const FILE_TYPE_BPW: Record<number, number> = {
  0: 32, 1: 16, 2: 4.5, 3: 4.5, 7: 5.5, 8: 5.5, 10: 4.5, 12: 4.5, 14: 4.5,
  15: 4.5, 16: 5.5, 17: 5.5, 18: 6.5, 19: 5.5, 20: 4.5, 21: 4.5, 22: 4.5,
  23: 5.5, 24: 4.5, 25: 5.5, 26: 5.5, 27: 5.5, 28: 5.5, 29: 5.5, 30: 6.5,
  31: 6.5, 32: 6.5, 33: 4.5, 34: 4.5, 35: 5.5, 36: 5.5, 37: 6.5, 38: 6.5,
};

export function bitsPerWeightFromFileType(fileType: number | null): number {
  if (fileType === null) return 8;
  return FILE_TYPE_BPW[fileType] ?? 5.5;
}
