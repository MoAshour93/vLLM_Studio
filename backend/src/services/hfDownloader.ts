import { spawn, type ChildProcess } from 'child_process';

let currentDownloadProcess: ChildProcess | null = null;
let downloadState = {
  active: false,
  modelId: '' as string,
  progress: 0,
  stage: '' as string,
  targetDir: '' as string,
  error: null as string | null,
  startedAt: 0,
  speedBps: 0,
  etaSec: 0,
};

let onProgressCallback: ((state: typeof downloadState) => void) | null = null;

function getPython(): string {
  return process.env.VLLM_PYTHON || 'python3';
}

export function setOnDownloadProgress(cb: (state: typeof downloadState) => void): void {
  onProgressCallback = cb;
}

export function getDownloadState() {
  return { ...downloadState };
}

export async function startDownload(modelId: string, targetDir: string, quantization?: string): Promise<void> {
  if (currentDownloadProcess) {
    throw new Error('A download is already in progress');
  }

  const python = getPython();

  downloadState = {
    active: true,
    modelId,
    progress: 0,
    stage: 'starting',
    targetDir,
    error: null,
    startedAt: Date.now(),
    speedBps: 0,
    etaSec: 0,
  };
  emit();

  const configJson = JSON.stringify({ modelId, targetDir, quantization: quantization || null });
  const args = ['-u', '-c', DOWNLOAD_SCRIPT];

  return new Promise((resolve, reject) => {
    currentDownloadProcess = spawn(python, args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env },
    });

    // Send config as JSON via stdin
    if (currentDownloadProcess.stdin) {
      currentDownloadProcess.stdin.write(configJson);
      currentDownloadProcess.stdin.end();
    }

    let outputBuffer = '';
    let stderrBuffer = '';

    currentDownloadProcess.stdout?.on('data', (data: Buffer) => {
      const text = data.toString();
      outputBuffer += text;
      parseOutput(text);
    });

    currentDownloadProcess.stderr?.on('data', (data: Buffer) => {
      const text = data.toString();
      stderrBuffer += text;
      console.error('[hfDownloader stderr]', text);
      parseStderrProgress(text);
    });

    currentDownloadProcess.on('close', (code) => {
      currentDownloadProcess = null;

      if (code === 0 && downloadState.stage !== 'error') {
        downloadState.progress = 100;
        downloadState.stage = 'complete';
        downloadState.active = false;
        emit();
        resolve();
      } else {
        downloadState.error = downloadState.error || `Download failed with exit code ${code}`;
        if (stderrBuffer) {
          console.error('[hfDownloader] Full stderr:', stderrBuffer);
        }
        console.error('[hfDownloader] Download failed:', downloadState.error);
        downloadState.stage = 'error';
        downloadState.active = false;
        emit();
        reject(new Error(downloadState.error));
      }
    });

    currentDownloadProcess.on('error', (err) => {
      currentDownloadProcess = null;
      downloadState.error = err.message;
      console.error('[hfDownloader] Spawn error:', err.message);
      downloadState.stage = 'error';
      downloadState.active = false;
      emit();
      reject(err);
    });
  });
}

export function cancelDownload(): void {
  if (currentDownloadProcess) {
    currentDownloadProcess.kill('SIGTERM');
    currentDownloadProcess = null;
    downloadState.active = false;
    downloadState.stage = 'cancelled';
    emit();
  }
}

function parseOutput(text: string): void {
  const lines = text.split('\n').filter(Boolean);
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      if (parsed.stage) {
        if (parsed.stage === 'progress') {
          downloadState.stage = 'downloading';
          downloadState.progress = parsed.percent ?? downloadState.progress;
          downloadState.speedBps = parsed.speedBps ?? 0;
          downloadState.etaSec = parsed.etaSec ?? 0;
        } else if (parsed.stage === 'complete') {
          downloadState.stage = 'complete';
          downloadState.progress = 100;
        } else {
          downloadState.stage = parsed.stage;
          downloadState.progress = parseProgressText(text);
        }
        downloadState.error = parsed.error || null;
        downloadState.targetDir = parsed.targetDir || downloadState.targetDir;
        emit();
      }
    } catch {
      const pctMatch = text.match(/(\d+)%/);
      if (pctMatch) {
        downloadState.stage = 'downloading';
        downloadState.progress = Math.min(parseInt(pctMatch[1], 10), downloadState.progress + 1);
        emit();
      }
    }
  }
}

function parseStderrProgress(text: string): void {
  // Try to parse JSON error messages from stderr
  try {
    const parsed = JSON.parse(text.trim());
    if (parsed.stage === 'error') {
      downloadState.stage = 'error';
      downloadState.error = parsed.error || 'Unknown error';
      emit();
      return;
    }
  } catch { /* not JSON */ }

  const pctMatches = [...text.matchAll(/(\d+)%/g)];
  if (pctMatches.length > 0) {
    downloadState.stage = 'downloading';
    const maxProgress = Math.max(...pctMatches.map(m => parseInt(m[1], 10)));
    if (maxProgress > downloadState.progress) {
      downloadState.progress = maxProgress;
      emit();
    }
  }
}

function parseProgressText(text: string): number {
  const matches = [...text.matchAll(/(\d+)%/g)];
  if (matches.length === 0) return downloadState.progress;
  return Math.max(...matches.map(m => parseInt(m[1], 10)));
}

function emit(): void {
  if (onProgressCallback) {
    onProgressCallback({ ...downloadState });
  }
}

const DOWNLOAD_SCRIPT = `
import sys, os, json, threading, time

try:
    from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files, hf_hub_url
except ImportError:
    print(json.dumps({"stage": "error", "error": "huggingface_hub not installed."}), flush=True)
    sys.exit(1)

def _get_dir_size(path):
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
            elif entry.is_dir(follow_symlinks=False):
                total += _get_dir_size(entry.path)
    except OSError:
        pass
    return total

def _monitor_progress(target_dir, initial_size, total_size, stop_event):
    last_pct = -1
    first = True
    prev_bytes = 0
    prev_time = time.time()
    while not stop_event.is_set():
        now = time.time()
        current = _get_dir_size(target_dir)
        delta = max(0, current - initial_size)
        if total_size > 0:
            pct = min(int(delta * 100 / total_size), 99)
        else:
            pct = 0
        if pct > last_pct:
            last_pct = pct
            elapsed = now - prev_time
            bytes_since = delta - prev_bytes
            if elapsed > 0 and bytes_since > 0:
                speed_bps = bytes_since / elapsed
                remaining = total_size - delta
                eta = remaining / speed_bps if speed_bps > 0 else 0
                prev_bytes = delta
                prev_time = now
            else:
                speed_bps = 0
                eta = 0
            print(json.dumps({"stage": "progress", "percent": pct, "bytes": delta, "total": total_size, "speedBps": int(speed_bps), "etaSec": int(eta)}), flush=True)
        time.sleep(0.3 if first else 0.8)
        first = False

def _download_single_quant(model_id, target_dir, quant_name):
    print(json.dumps({"stage": "starting", "modelId": model_id, "targetDir": target_dir, "quant": quant_name}), flush=True)
    try:
        files = list_repo_files(model_id)
        gguf_file = None
        quant_upper = quant_name.upper()
        for f in files:
            f_upper = f.upper()
            if f_upper.endswith('.GGUF') and quant_upper in f_upper:
                gguf_file = f
                break
        if not gguf_file:
            for f in files:
                f_upper = f.upper()
                if f_upper.endswith('.GGUF'):
                    base = os.path.basename(f).upper().replace('.GGUF', '')
                    if quant_upper in base:
                        gguf_file = f
                        break
        if not gguf_file:
            ggufs = [f for f in files if f.upper().endswith('.GGUF')]
            print(json.dumps({"stage": "error", "error": f"No GGUF found for quant {quant_name}. Available: {ggufs}"}), flush=True)
            sys.exit(1)

        total_size = 0
        try:
            url = hf_hub_url(repo_id=model_id, filename=gguf_file)
            import urllib.request
            req = urllib.request.Request(url, method='HEAD')
            resp = urllib.request.urlopen(req, timeout=10)
            total_size = int(resp.headers.get('Content-Length', 0))
        except Exception:
            pass

        initial_size = _get_dir_size(target_dir)
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=_monitor_progress,
            args=(target_dir, initial_size, total_size, stop_event),
            daemon=True
        )
        monitor_thread.start()

        print(json.dumps({"stage": "downloading", "file": gguf_file, "size": total_size}), flush=True)
        hf_hub_download(repo_id=model_id, filename=gguf_file, local_dir=target_dir)

        stop_event.set()
        monitor_thread.join(timeout=2)

        essential = ["tokenizer_config.json", "tokenizer.json", "config.json",
                     "generation_config.json", "special_tokens_map.json",
                     "vocab.json", "merges.txt", "tokenizer.model", "added_tokens.json"]
        for ef in essential:
            if ef in files:
                try:
                    print(json.dumps({"stage": "downloading", "file": ef}), flush=True)
                    hf_hub_download(repo_id=model_id, filename=ef, local_dir=target_dir)
                except Exception:
                    pass

        # Ensure config.json exists (vLLM requires it)
        if not os.path.exists(os.path.join(target_dir, 'config.json')):
            _write_minimal_config(target_dir, gguf_file)

        print(json.dumps({"stage": "complete", "modelId": model_id, "targetDir": target_dir}), flush=True)
    except SystemExit:
        raise
    except Exception as e:
        print(json.dumps({"stage": "error", "error": str(e)}), flush=True)
        sys.exit(1)

def _write_minimal_config(target_dir, gguf_file):
    combined = gguf_file.lower()
    arch = 'LlamaForCausalLM'
    model_type = 'llama'
    if 'qwen' in combined: arch = 'Qwen2ForCausalLM'; model_type = 'qwen2'
    elif 'mistral' in combined and 'mixtral' not in combined: arch = 'MistralForCausalLM'; model_type = 'mistral'
    elif 'mixtral' in combined: arch = 'MixtralForCausalLM'; model_type = 'mixtral'
    elif 'phi-3' in combined or 'phi3' in combined: arch = 'Phi3ForCausalLM'; model_type = 'phi3'
    elif 'phi' in combined: arch = 'PhiForCausalLM'; model_type = 'phi'
    elif 'gemma-2' in combined: arch = 'Gemma2ForCausalLM'; model_type = 'gemma2'
    elif 'gemma' in combined: arch = 'GemmaForCausalLM'; model_type = 'gemma'
    elif 'falcon' in combined: arch = 'FalconForCausalLM'; model_type = 'falcon'
    cfg = {'architectures': [arch], 'model_type': model_type}
    with open(os.path.join(target_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f)
    print(json.dumps({"stage": "downloading", "file": "config.json (auto-generated)"}), flush=True)

def _download_full_repo(model_id, target_dir):
    print(json.dumps({"stage": "downloading", "modelId": model_id, "targetDir": target_dir}), flush=True)
    try:
        snapshot_download(repo_id=model_id, local_dir=target_dir,
                          ignore_patterns=["*.msgpack", "*.h5"])
        print(json.dumps({"stage": "complete", "modelId": model_id, "targetDir": target_dir}), flush=True)
    except Exception as e:
        print(json.dumps({"stage": "error", "error": str(e)}), flush=True)
        sys.exit(1)

# Read config from stdin
config = json.loads(sys.stdin.read())
model_id = config['modelId']
target_dir = config['targetDir']
quantization = config.get('quantization')

if not model_id or not target_dir:
    print(json.dumps({"stage": "error", "error": "modelId and targetDir are required"}), flush=True)
    sys.exit(1)

os.makedirs(target_dir, exist_ok=True)

if quantization:
    _download_single_quant(model_id, target_dir, quantization)
else:
    _download_full_repo(model_id, target_dir)
`;
