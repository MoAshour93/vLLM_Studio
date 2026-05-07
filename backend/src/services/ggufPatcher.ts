// GGUF architecture patcher — delegates to the Python gguf_patch.py script.
// The Python script handles the full GGUF v3 binary rewrite (arch rename +
// arch-prefixed key renames + tokenizer prefix) correctly by rebuilding the
// metadata section and streaming tensor data verbatim.

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SCRIPT = path.resolve(__dirname, 'gguf_patch.py');

function getPython(): string {
  return process.env.VLLM_PYTHON || 'python3';
}

export interface PatchResult {
  outputDir: string;
  outputFile: string;
  oldArch: string;
  newArch: string;
  bytesWritten: number;
}

export async function patchGgufArchitecture(args: {
  inputModelPath: string;
  inputGgufFile: string;
  newArchitecture: string;
  outputDir?: string;
  onProgress?: (bytesCopied: number, totalBytes: number) => void;
}): Promise<PatchResult> {
  if (!fs.existsSync(args.inputGgufFile)) {
    throw new Error(`Source GGUF not found: ${args.inputGgufFile}`);
  }
  const newArch = args.newArchitecture.trim();
  if (!newArch) throw new Error('newArchitecture must be non-empty');

  const outDir = args.outputDir
    ?? path.join(path.dirname(args.inputModelPath), `${path.basename(args.inputModelPath)}--patched-${newArch}`);
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  const baseName = path.basename(args.inputGgufFile).replace(/\.gguf$/i, `.${newArch}.gguf`);
  const outFile = path.join(outDir, baseName);

  const python = getPython();

  return new Promise<PatchResult>((resolve, reject) => {
    const proc = spawn(python, [SCRIPT, args.inputGgufFile, outFile, newArch], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let oldArch = '';
    let bytesWritten = 0;
    let stderr = '';

    proc.stdout?.on('data', (data: Buffer) => {
      for (const line of data.toString().split('\n').filter(Boolean)) {
        if (line.startsWith('INFO: old_arch=')) {
          const m = line.match(/old_arch=(\S+)/);
          if (m) oldArch = m[1];
        }
        if (line.startsWith('PROGRESS:')) {
          const [bytes, total] = line.replace('PROGRESS:', '').split(':').map(Number);
          args.onProgress?.(bytes, total);
        }
        if (line.startsWith('COMPLETE:')) {
          bytesWritten = parseInt(line.replace('COMPLETE:', ''));
        }
      }
    });

    proc.stderr?.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      if (code !== 0 || !fs.existsSync(outFile)) {
        reject(new Error(`GGUF patch failed with code ${code}: ${stderr.slice(-500)}`));
        return;
      }
      resolve({
        outputDir: outDir,
        outputFile: outFile,
        oldArch,
        newArch,
        bytesWritten: bytesWritten || fs.statSync(outFile).size,
      });
    });

    proc.on('error', (err) => {
      reject(new Error(`Failed to spawn Python patcher: ${err.message}`));
    });
  });
}

export function suggestTargetArch(oldArch: string): string | null {
  const a = oldArch.toLowerCase();
  if (a === 'qwen35' || a === 'qwen3.5' || a === 'qwen36' || a === 'qwen3.6') return 'qwen3';
  if (a === 'qwen25' || a === 'qwen2.5') return 'qwen2';
  if (a === 'mixtral') return 'llama';
  return null;
}
