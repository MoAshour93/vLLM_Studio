import { ChildProcess, spawn, execFileSync, execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { ServerStatus, GpuStats } from '../types/index.js';
import { getVllmCapabilities } from './vllmIntrospect.js';

const PID_FILE = path.resolve(process.env.DATA_DIR || './data', 'vllm.pid');
const MAX_LOG_LINES = 1000;
const MAX_RUNTIME_RETRIES = 1;
export const SERVED_MODEL_NAME = 'vllm-studio';

export type LoadStage = 'spawning' | 'downloading' | 'loading_weights' | 'compiling' | 'allocating_kv' | 'starting_api' | 'ready';

function getPython(): string {
  return process.env.VLLM_PYTHON || 'python3';
}

let vllmProcess: ChildProcess | null = null;
let serverStatus: ServerStatus = 'stopped';
let serverError: string | null = null;
let logBuffer: string[] = [];
let stderrTail: string[] = [];
let retryCount = 0;
let onLogCallback: ((line: string) => void) | null = null;
let onStatusChangeCallback: ((status: ServerStatus, error?: string, stage?: LoadStage) => void) | null = null;
let currentStage: LoadStage | null = null;

export function setOnLog(cb: (line: string) => void): void { onLogCallback = cb; }
export function setOnStatusChange(cb: (status: ServerStatus, error?: string, stage?: LoadStage) => void): void { onStatusChangeCallback = cb; }
export function getStatus(): ServerStatus { return serverStatus; }
export function getError(): string | null { return serverError; }
export function getLogs(lines: number = 200): string[] { return logBuffer.slice(-lines); }
export function getRetryCount(): number { return retryCount; }
export function getStage(): LoadStage | null { return currentStage; }

function setStatus(status: ServerStatus, error?: string, stage?: LoadStage): void {
  serverStatus = status;
  serverError = error ?? null;
  if (stage) currentStage = stage;
  if (status === 'running') currentStage = 'ready';
  if (status === 'stopped' || status === 'error') currentStage = null;
  onStatusChangeCallback?.(status, error, currentStage ?? undefined);
}

function addLog(line: string, isErr = false): void {
  logBuffer.push(line);
  if (logBuffer.length > MAX_LOG_LINES) logBuffer = logBuffer.slice(-MAX_LOG_LINES);
  if (isErr) {
    stderrTail.push(line);
    if (stderrTail.length > 50) stderrTail = stderrTail.slice(-50);
  }
  onLogCallback?.(line);
}

function detectStage(line: string): LoadStage | null {
  const l = line.toLowerCase();
  if (l.includes('downloading') || l.includes('huggingface_hub')) return 'downloading';
  if (l.includes('loading weights') || l.includes('loading model') || l.includes('loading safetensors') || l.includes('loading gguf')) return 'loading_weights';
  if (l.includes('torch.compile') || l.includes('cudagraph') || l.includes('capturing')) return 'compiling';
  if (l.includes('# gpu blocks') || l.includes('kv cache') || l.includes('memory profiling')) return 'allocating_kv';
  if (l.includes('starting vllm api server') || l.includes('uvicorn running')) return 'starting_api';
  if (l.includes('application startup complete')) return 'ready';
  return null;
}

export interface StartConfig {
  modelPath: string;            // Directory or .gguf file passed as --model
  port: number;
  host?: string;
  gpuMemoryUtilization: number;
  maxModelLen: number | null;
  quantization: string | null;
  tensorParallelSize: number;
  maxNumSeqs: number;
  dtype: string;
  additionalArgs: string[];
  loadFormat?: 'auto' | 'gguf' | 'safetensors';
  ggufFilePath?: string | null; // If set, --model points at this file with --load-format gguf.
  cpuOffloadGb?: number;
  maxNumBatchedTokens?: number;
  kvCacheDtype?: 'auto' | 'fp16' | 'fp8' | 'fp8_e4m3' | 'fp8_e5m2';
  enforceEager?: boolean;
  hfConfigPath?: string | null;   // HF id or local path to override config (esp. for unknown GGUF archs).
  tokenizer?: string | null;       // HF id or local path for the tokenizer (recommended for all GGUF loads).
  languageModelOnly?: boolean;     // For hybrid multimodal models (Llama-4, Mistral-3, Qwen3.5, Step3): skip vision/audio modules.
}

export async function startVllm(config: StartConfig): Promise<void> {
  if (vllmProcess) throw new Error('vLLM server is already running');

  // Free the port if a stale process is still bound to it.
  try {
    execSync(`fuser -k ${config.port}/tcp 2>/dev/null; true`, { timeout: 3000 });
  } catch { /* ignore */ }

  stderrTail = [];
  retryCount = 0;
  const host = config.host || '127.0.0.1';

  const isGguf = !!config.ggufFilePath || /\.gguf$/i.test(config.modelPath);
  const modelArg = config.ggufFilePath || config.modelPath;

  const args: string[] = [
    '-m', 'vllm.entrypoints.openai.api_server',
    '--model', modelArg,
    '--served-model-name', SERVED_MODEL_NAME,
    '--port', String(config.port),
    '--host', host,
    '--gpu-memory-utilization', String(config.gpuMemoryUtilization),
    '--max-num-seqs', String(config.maxNumSeqs),
  ];

  if (config.cpuOffloadGb && config.cpuOffloadGb > 0) {
    args.push('--cpu-offload-gb', String(config.cpuOffloadGb));
  }
  if (config.maxNumBatchedTokens && config.maxNumBatchedTokens > 0) {
    args.push('--max-num-batched-tokens', String(config.maxNumBatchedTokens));
  }
  if (config.kvCacheDtype && config.kvCacheDtype !== 'auto') {
    args.push('--kv-cache-dtype', config.kvCacheDtype);
  }
  if (config.enforceEager) {
    args.push('--enforce-eager');
  }
  if (config.hfConfigPath) {
    args.push('--hf-config-path', config.hfConfigPath);
  }
  if (config.tokenizer) {
    args.push('--tokenizer', config.tokenizer);
  }
  if (config.languageModelOnly) {
    args.push('--language-model-only');
  }

  if (config.loadFormat && config.loadFormat !== 'auto') {
    args.push('--load-format', config.loadFormat);
  } else if (isGguf) {
    args.push('--load-format', 'gguf');
  }

  if (!isGguf) {
    args.push('--dtype', config.dtype || 'auto');
  }

  if (config.maxModelLen && config.maxModelLen > 0) {
    args.push('--max-model-len', String(config.maxModelLen));
  }

  if (config.quantization && config.quantization !== 'none' && !isGguf) {
    args.push('--quantization', config.quantization);
  }

  if (config.tensorParallelSize > 1) {
    args.push('--tensor-parallel-size', String(config.tensorParallelSize));
  }

  if (config.additionalArgs.length > 0) {
    args.push(...config.additionalArgs);
  }

  const python = getPython();
  addLog(`Starting vLLM: ${python} ${args.join(' ')}`);
  setStatus('starting', undefined, 'spawning');

  try {
    // Ensure the venv bin directory is on PATH so that tools like ninja
    // (needed by FlashInfer JIT) are available to vLLM's EngineCore subprocess.
    const env = { ...process.env };
    const pythonResolved = fs.realpathSync(python);
    const pythonBinDir = path.dirname(pythonResolved);
    env.PATH = `${pythonBinDir}:${env.PATH || ''}`;
    vllmProcess = spawn(python, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env,
    });

    fs.writeFileSync(PID_FILE, String(vllmProcess.pid));

    const handleLine = (line: string, isErr: boolean) => {
      addLog(`[${isErr ? 'stderr' : 'stdout'}] ${line}`, isErr);
      const stage = detectStage(line);
      if (stage && serverStatus === 'starting') {
        currentStage = stage;
        onStatusChangeCallback?.('starting', undefined, stage);
      }
      if (
        serverStatus === 'starting' &&
        (line.includes('Application startup complete') || line.includes('Uvicorn running'))
      ) {
        setStatus('running');
      }
      // Surface common fatal errors immediately with the actual cause.
      if (isErr) {
        if (line.includes('CUDA out of memory') || line.includes('OutOfMemoryError')) {
          setStatus('error', `CUDA Out of Memory. ${line.trim()}`);
          terminateProcess();
        } else if (line.match(/ValueError: .*max_model_len/i)) {
          setStatus('error', `Context length too large for KV cache: ${line.trim()}`);
          terminateProcess();
        } else {
          const ggufArchMatch = line.match(/GGUF model with architecture (\S+) is not supported yet/);
          const unknownModelType = line.match(/model type `(\S+)` but Transformers does not recognize/i);
          const unknownGgufType = line.match(/Unknown gguf model_type:\s*(\S+)/i);
          const multimodalNeedTokenizer = line.match(/Loading a multimodal GGUF.*--tokenizer/);
          const shapeMismatch = line.match(/Attempted to load weight.*\(\[(\d+(?:,\s*\d+)*)\]\).*\(\[(\d+(?:,\s*\d+)*)\]\)/);
          const archNotRegistered = line.match(/Model architecture\s+(\w+)\s+is not supported|architectures must be one of/i);

          if (ggufArchMatch) {
            const arch = ggufArchMatch[1];
            const caps = getVllmCapabilities();
            const installedTr = caps.transformersGgufArchs.length
              ? caps.transformersGgufArchs.join(', ')
              : '(none detected)';
            setStatus('error',
              `GGUF architecture "${arch}" is not in the installed transformers' GGUF map. ` +
              `Installed can convert: ${installedTr}. ` +
              `Upgrade with: pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly`,
            );
            terminateProcess();
          } else if (unknownModelType) {
            const mt = unknownModelType[1];
            const caps = getVllmCapabilities();
            setStatus('error',
              `GGUF model type "${mt}" not recognised by installed transformers (vLLM ${caps.vllmVersion}). ` +
              `Upgrade with: pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly`,
            );
            terminateProcess();
          } else if (unknownGgufType) {
            setStatus('error',
              `GGUF model type "${unknownGgufType[1]}" not supported. ` +
              `Upgrade with: pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly`,
            );
            terminateProcess();
          } else if (multimodalNeedTokenizer) {
            setStatus('error',
              `This is a multimodal model — vLLM needs a tokenizer from the base model. ` +
              `Download the base HF model config files or specify --tokenizer when loading.`,
            );
            terminateProcess();
          } else if (shapeMismatch) {
            setStatus('error',
              `Weight shape mismatch: tried to load ${shapeMismatch[1]} into ${shapeMismatch[2]}. ` +
              `This model's GGUF weights are incompatible with the resolved HF model implementation. ` +
              `The architecture likely needs a newer vLLM: pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly`,
            );
            terminateProcess();
          } else if (archNotRegistered) {
            const arch = archNotRegistered[1] ?? 'this model';
            const caps = getVllmCapabilities();
            setStatus('error',
              `vLLM ${caps.vllmVersion} does not have ${arch} in its model registry. ` +
              `Upgrade with: pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly`,
            );
            terminateProcess();
          } else if (line.includes('not supported') && line.includes('architecture')) {
            const caps = getVllmCapabilities();
            setStatus('error', `${line.trim()} (vLLM ${caps.vllmVersion}; consider upgrading)`);
            terminateProcess();
          } else if (/EngineDeadError|Engine.*failed|RuntimeError: Failed to load/i.test(line)) {
            setStatus('error', line.trim());
            terminateProcess();
          }
        }
      }
    };

    const lineSplitter = (isErr: boolean) => {
      let buf = '';
      return (data: Buffer) => {
        buf += data.toString();
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (line.length > 0) handleLine(line, isErr);
        }
      };
    };

    vllmProcess.stdout?.on('data', lineSplitter(false));
    vllmProcess.stderr?.on('data', lineSplitter(true));

    vllmProcess.on('close', (code) => {
      addLog(`vLLM process exited with code ${code}`);
      if (fs.existsSync(PID_FILE)) {
        try { fs.unlinkSync(PID_FILE); } catch { /* ignore */ }
      }

      const wasRunning = serverStatus === 'running';
      const wasStarting = serverStatus === 'starting';
      vllmProcess = null;

      if (wasStarting) {
        // Startup failure: do NOT auto-retry; surface the real error.
        const tail = stderrTail.slice(-15).join('\n');
        const summary = tail || `vLLM exited during startup with code ${code}`;
        setStatus('error', summary);
        return;
      }

      if (wasRunning && retryCount < MAX_RUNTIME_RETRIES) {
        retryCount++;
        addLog(`Runtime crash — auto-retry ${retryCount}/${MAX_RUNTIME_RETRIES}`);
        setStatus('starting', undefined, 'spawning');
        startVllm(config).catch((err) => {
          addLog(`Auto-retry failed: ${err.message}`);
          setStatus('error', err.message);
        });
      } else if (wasRunning) {
        setStatus('error', `vLLM crashed (exit code ${code}). Auto-retry exhausted.`);
      }
    });

    vllmProcess.on('error', (err) => {
      addLog(`Failed to spawn vLLM: ${err.message}`, true);
      setStatus('error', `Failed to start vLLM: ${err.message}`);
      vllmProcess = null;
    });

    setTimeout(() => {
      if (serverStatus === 'starting') {
        addLog('Note: vLLM startup is taking longer than 60s — large models can take several minutes for the first load.');
      }
    }, 60000);

  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    addLog(`Error spawning vLLM: ${msg}`);
    setStatus('error', msg);
    throw err;
  }
}

function terminateProcess(): void {
  if (!vllmProcess) return;
  try { vllmProcess.kill('SIGTERM'); } catch { /* ignore */ }
  setTimeout(() => {
    if (vllmProcess) {
      try { vllmProcess.kill('SIGKILL'); } catch { /* ignore */ }
    }
  }, 3000);
}

export async function stopVllm(): Promise<void> {
  if (!vllmProcess) {
    setStatus('stopped');
    return;
  }
  setStatus('stopped');
  try {
    vllmProcess.kill('SIGTERM');
    await new Promise<void>((resolve) => {
      const timeout = setTimeout(() => {
        if (vllmProcess) try { vllmProcess.kill('SIGKILL'); } catch { /* ignore */ }
        resolve();
      }, 5000);
      vllmProcess?.on('close', () => { clearTimeout(timeout); resolve(); });
    });
  } catch (err) {
    addLog(`Error stopping vLLM: ${err}`);
  }
  vllmProcess = null;
  if (fs.existsSync(PID_FILE)) {
    try { fs.unlinkSync(PID_FILE); } catch { /* ignore */ }
  }
  setStatus('stopped');
}

export async function restartVllm(config: StartConfig): Promise<void> {
  await stopVllm();
  await new Promise(resolve => setTimeout(resolve, 1000));
  await startVllm(config);
}

export function getGpuStats(): GpuStats[] {
  try {
    const output = execSync(
      'nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits',
      { timeout: 5000 }
    ).toString().trim();
    const lines = output.split('\n');
    const stats: GpuStats[] = [];
    for (const line of lines) {
      const parts = line.split(',').map(s => s.trim());
      if (parts.length >= 7) {
        stats.push({
          index: parseInt(parts[0], 10),
          name: parts[1],
          totalMemoryMb: parseInt(parts[2], 10),
          usedMemoryMb: parseInt(parts[3], 10),
          freeMemoryMb: parseInt(parts[4], 10),
          utilizationPercent: parseInt(parts[5], 10),
          temperatureC: parts[6] !== '[Not Supported]' ? parseInt(parts[6], 10) : null,
        });
      }
    }
    return stats;
  } catch {
    return [];
  }
}

export function getVllmVersion(): string {
  try {
    const python = getPython();
    const output = execFileSync(python, ['-c', 'import vllm; print(vllm.__version__)'], { timeout: 5000, encoding: 'utf-8' }).trim();
    return output || 'unknown';
  } catch {
    return 'unknown';
  }
}

export function attachToExisting(): boolean {
  if (vllmProcess) return false;
  if (fs.existsSync(PID_FILE)) {
    try {
      const pid = parseInt(fs.readFileSync(PID_FILE, 'utf-8').trim(), 10);
      try {
        process.kill(pid, 0);
        addLog(`Re-attached to existing vLLM process (PID: ${pid})`);
        setStatus('running');
        return true;
      } catch {
        try { fs.unlinkSync(PID_FILE); } catch { /* ignore */ }
      }
    } catch { /* ignore */ }
  }
  return false;
}
