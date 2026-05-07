> A fast, Linux-native AI inference interface built on vLLM.

**VLLM Studio** is a production-grade local AI chat and inference management application purpose-built for the [vLLM](https://github.com/vllm-project/vllm) inference engine on Linux with NVIDIA CUDA GPUs. Think of it as LM Studio, but engineered specifically for the vLLM ecosystem — delivering PagedAttention memory efficiency, continuous batching throughput, and a polished desktop-like chat experience.

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/platform-Linux%20(CUDA)-orange?style=flat-square" alt="Platform">
  <img src="https://img.shields.io/badge/engine-vLLM-blueviolet?style=flat-square" alt="Engine">
</p>

---

## Table of Contents

- [Why VLLM Studio?](#why-vllm-studio)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [OpenAI-Compatible API](#openai-compatible-api)
- [Attaching Images & PDFs](#attaching-images--pdfs)
- [HuggingFace Hub Integration](#huggingface-hub-integration)
- [VRAM Estimation Engine](#vram-estimation-engine)
- [GGUF Architecture Support](#gguf-architecture-support)
- [Configuration Reference](#configuration-reference)
- [Data Storage & Backup](#data-storage--backup)
- [Troubleshooting](#troubleshooting)
- [Release Notes](#release-notes)
- [Roadmap](#roadmap)
- [About the Author](#about-the-author)
- [License](#license)

---

## Why VLLM Studio?

vLLM's PagedAttention algorithm achieves near-zero GPU memory waste by managing the KV cache like virtual memory pages. Combined with continuous batching, vLLM delivers up to **10x higher throughput** than standard HuggingFace Transformers serving. Yet vLLM lacked a polished local desktop management interface — until now.

| Feature | VLLM Studio | LM Studio | Ollama |
|---------|-------------|-----------|--------|
| **Inference Engine** | vLLM (PagedAttention) | llama.cpp (GGUF) | llama.cpp (GGUF) |
| **Throughput** | 10x+ over HF | Standard | Standard |
| **Vision Models** | Full GPT-4V-style | Partial | Limited |
| **Multi-GPU (TP)** | Full tensor parallelism | No | No |
| **Quantization** | AWQ, GPTQ, FP8, GGUF | GGUF only | GGUF only |
| **GPU Memory Efficiency** | Near-zero KV waste | Standard KV cache | Standard KV cache |
| **OpenAI API Compatible** | Native (vLLM) | No | Limited |
| **Streaming UI** | Token-by-token | Yes | Yes |
| **Model Format Support** | HF safetensors + GGUF | GGUF only | GGUF only |
| **Platform** | Linux (CUDA) | Win/Mac/Linux | macOS/Linux |

---

## Features

### Model Management
- Scan local directories for HuggingFace-format (safetensors) and GGUF models
- Auto-detect architecture, quantization (AWQ/GPTQ/FP8/GGUF), context length (RoPE scaling), vision capability
- 140+ supported model architectures with three-state compatibility tracking
- Smart model caching with directory mtime-based invalidation
- Model deletion with confirmation

### Chat Interface
- Streaming token-by-token output with blinking cursor
- Full markdown rendering: tables, syntax-highlighted code blocks, math
- Code blocks with language detection, line numbers, and copy button
- Image and PDF attachment support for vision-capable models
- Message regeneration, editing, and deletion
- Per-session system prompt and inference parameters
- Chat folders with drag-and-drop organization
- Session pinning and fuzzy search across titles and content
- Import/export chat history as JSON

### vLLM Server Management
- One-click start/stop/restart of vLLM OpenAI-compatible API server
- Live GPU monitoring: VRAM usage, utilization %, temperature (nvidia-smi)
- Live log streaming with intelligent startup stage detection
- Configurable: port, GPU memory utilization, max model length, quantization, tensor parallelism, enforce eager, KV cache dtype, CPU offload
- Auto-retry on runtime crash with smart error parsing (CUDA OOM, unsupported architecture, shape mismatch)

### HuggingFace Hub Browser
- Search HuggingFace Hub for models directory within the app
- Browse quantized GGUF variants with file sizes
- One-click download with real-time progress tracking
- Filter by vLLM compatibility
- Auto-detect base model repo for tokenizer configuration

### VRAM Estimation
- Pre-flight resource planning before loading a model
- Estimates: model weights, KV cache, activations, CUDA overhead
- Fit verdicts: full-gpu, partial-gpu, cpu-offload, too-large
- Recommended context length and CPU offload amount per GPU

### GGUF Architecture Support
- Pure TypeScript GGUF v3 binary metadata parser
- Read architecture, context length, embedding size, block count, attention heads, RoPE scaling
- One-click architecture patching for community models with non-standard labels
- Automatic transformers GGUF_CONFIG_MAPPING monkey-patching

### Release Notes & About
- Comprehensive release notes with search, section navigation, and roadmap
- Full about page with author bio, credentials, tech stack, AI collaborator credits
- Links to official vLLM documentation at docs.vllm.ai

---

## Prerequisites

| Dependency | Minimum Version | Check Command |
|------------|----------------|---------------|
| **Node.js** | 18.0+ | `node --version` |
| **npm** | 9.0+ | `npm --version` |
| **Python** | 3.10+ | `python3 --version` |
| **CUDA / NVIDIA GPU** | 12.0+ driver | `nvidia-smi` |

The `run.sh` launcher handles all remaining setup — it creates an isolated Python virtual environment, installs vLLM and huggingface_hub, and builds the application.

---

## Quick Start

```bash
git clone https://github.com/Moashour93/vllm-studio.git
cd vllm-studio
bash run.sh
```

The launcher performs 6 phases:

1. **System dependency checks** — Node.js, npm, Python, CUDA/nvidia-smi
2. **Python virtual environment** — isolated `venv/` with vLLM + huggingface_hub + ninja
3. **Node.js dependencies** — root, backend, and frontend packages
4. **Build** — TypeScript compilation + Vite frontend bundling
5. **Environment setup** — auto-generates `.env` from `.env.example`
6. **Launch** — starts backend at `http://localhost:3333`

On subsequent runs, the launcher detects existing installations and skips re-installation.

```bash
# Open in browser
http://localhost:3333

# Add model directories in Settings → Scan models
# Select a model → Start vLLM server → Start chatting!
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                       VLLM Studio                          │
├────────────────────────────────────────────────────────────┤
│  Browser (React SPA)                                       │
│  ┌──────────┬────────────────────┬──────────────┐         │
│  │ Sidebar  │    Chat Window     │ Config Panel │         │
│  │ Sessions │    Streaming MD    │ Models/Server│         │
│  │ Folders  │    Attachments     │ HF Downloads │         │
│  └──────────┴────────────────────┴──────────────┘         │
│         ↕ REST + WebSocket                                 │
├────────────────────────────────────────────────────────────┤
│  Express Backend (Node.js + TypeScript)                    │
│  ┌──────────────────────────────────────────────────┐     │
│  │  /api/models · /api/chat · /api/server           │     │
│  │  /api/sessions · /api/settings · /api/huggingface│     │
│  │  Services: vllmManager · modelScanner · database │     │
│  │  Subprocess: GGUF parser · HF downloader         │     │
│  └──────────────────────────────────────────────────┘     │
│         ↕ Subprocess (spawn)                               │
├────────────────────────────────────────────────────────────┤
│  vLLM Inference Server (Python)                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │  PagedAttention KV Cache                         │     │
│  │  Continuous Batching                              │     │
│  │  OpenAI-compatible /v1 endpoint                   │     │
│  └──────────────────────────────────────────────────┘     │
│         ↕ CUDA Driver                                      │
├────────────────────────────────────────────────────────────┤
│  NVIDIA GPU(s)                                             │
│  ┌──────────────────────────────────────────────────┐     │
│  │  VRAM  │  Tensor Cores  │  NVLink (multi-GPU)    │     │
│  └──────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18 · TypeScript 5.6 · Vite 5 · Zustand 5 · react-virtuoso · react-markdown · highlight.js · remark-gfm |
| **Backend** | Node.js 18+ · Express 4 · TypeScript 5.6 · better-sqlite3 · WebSocket (ws) · Zod · Multer · PDF-Parse · tsx |
| **Inference** | vLLM · Python 3.10+ · huggingface_hub · FlashInfer · PagedAttention · Continuous Batching · AWQ/GPTQ/FP8 |
| **Infrastructure** | SQLite (WAL mode) · nvidia-smi · isolated Python venv · run.sh all-in-one launcher · PID-based crash recovery |

---

## OpenAI-Compatible API

Once VLLM Studio has a model loaded, the vLLM OpenAI-compatible endpoint is available at:

**Base URL:** `http://localhost:8000/v1`

### Python (openai package)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

### curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

---

## Attaching Images & PDFs

### Supported Formats
- **Images:** JPG, PNG, WebP, GIF (max 20MB)
- **PDFs:** Server-side text extraction, injected as document context (max 50MB)

### Vision Model Support
Image attachments require a **vision-capable** (multimodal) model loaded in vLLM. Examples:
- `llava-hf/llava-v1.6-mistral-7b-hf`
- `llava-hf/llava-v1.6-vicuna-13b-hf`
- `internlm/internlm-xcomposer2-vl-7b`
- `microsoft/Phi-3-vision-128k-instruct`

> **Note:** Vision model support requires vLLM >= 0.6.0.

---

## HuggingFace Hub Integration

Browse and download models from HuggingFace Hub directly within VLLM Studio:

- **Search** HuggingFace for text-generation models with debounced queries
- **Filter** by vLLM compatibility to show only supported architectures
- **Sort** by downloads or likes
- **Browse** GGUF quantization variants per model
- **Download** quantized GGUF files with real-time progress tracking
- **Auto-config** tokenizer and base model path for downloaded GGUFs

---

## VRAM Estimation Engine

Before loading a model, VLLM Studio estimates memory requirements:
- **Weights memory** — based on file size or parameters × bits-per-weight
- **KV cache memory** — 2 × layers × context × KV heads × head dim × dtype bytes × batch
- **Activation memory** — 4 × prefill tokens × hidden size × 2
- **CUDA overhead** — 600MB baseline

Produces a **fit verdict** (full-gpu, partial-gpu, cpu-offload, too-large) with recommended context length and CPU offload amount per GPU configuration.

---

## GGUF Architecture Support

VLLM Studio includes a **pure TypeScript GGUF v3 binary metadata parser** that reads:
- Architecture string, context length, embedding length
- Block count, attention head count, KV head count, feed-forward length
- RoPE scaling parameters, file type/quantization
- All custom metadata key-value pairs

For community GGUF files with non-standard architecture labels, VLLM Studio offers **one-click architecture patching** that rewrites the GGUF binary header and patches transformers' `GGUF_CONFIG_MAPPING` to make the model loadable in vLLM.

---

## Configuration Reference

All settings are persisted in SQLite (`data/vllm-studio.db`) and configurable via the Settings modal in the UI.

| Setting | Default | Description |
|---------|---------|-------------|
| `modelScanDirs` | `['./data/models']` | Directories to scan for models |
| `defaultSystemPrompt` | `""` | Default system prompt for new sessions |
| `theme` | `"dark"` | UI theme: dark, light, or system |
| `defaultTemperature` | `0.7` | Default temperature |
| `defaultTopP` | `0.9` | Default top-p |
| `defaultMaxTokens` | `1024` | Default max output tokens |
| `backendPort` | `3333` | Backend server port |
| `vllmPort` | `8000` | vLLM server port |
| `autoStartVllm` | `false` | Auto-start vLLM on launch |
| `gpuMemoryUtilization` | `0.9` | Fraction of GPU memory for KV cache |
| `sendOnEnter` | `true` | Enter to send; Shift+Enter for newline |

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3333` | Backend server port |
| `HOST` | `127.0.0.1` | Backend bind address |
| `VLLM_PORT` | `8000` | vLLM server port |
| `DATA_DIR` | `./data` | Data directory path |
| `LOG_LEVEL` | `info` | Logging level |
| `VLLM_PYTHON` | `./venv/bin/python3` | Path to venv Python |

---

## Data Storage & Backup

```
vllm-studio/data/
├── vllm-studio.db        # SQLite (WAL mode) — sessions, messages, settings, model cache
├── attachments/          # Uploaded images and PDFs per session
├── chats/                # JSON chat exports
├── models/               # Downloaded HuggingFace models
└── logs/
    └── vllm-studio.log   # Application log
```

### Backup

```bash
cp -r vllm-studio/data ~/backups/vllm-studio-$(date +%Y%m%d)
```

### Export Chats

Use **Settings → Export All Chats (JSON)** to download a portable archive of all conversation history.

---

## Troubleshooting

### CUDA Out of Memory (OOM)

```
Error: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**
1. Reduce `gpu-memory-utilization` (e.g., 0.9 → 0.7)
2. Reduce `max-model-len` (context window)
3. Enable quantization (AWQ, GPTQ, FP8, or GGUF)
4. Use a smaller model variant
5. Enable CPU offload

### Port Conflicts

If port 3333 or 8000 is already in use:
- Change `backendPort` / `vllmPort` in Settings
- Or: `lsof -ti:3333 | xargs kill -9`

### Model Not Loading

Ensure model directory contains:
- `config.json` (model configuration)
- `tokenizer.json` or `tokenizer_config.json`
- Model weights (`.safetensors` or `.bin` files)

### GGUF Architecture Not Supported

Some community GGUF models use non-standard architecture labels. VLLM Studio will detect this and offer a one-click "Patch GGUF Architecture" button in the Model Selector. This rewrites the GGUF metadata and patches transformers' configuration mapping.

---

## Release Notes

### v1.0.0 — PagedAttention _(7 May 2026)_

**Initial release** of VLLM Studio. See the in-app Release Notes (accessible from the topbar book icon) for the full breakdown of:

- **What's New** — vLLM subprocess management, streaming chat, model scanner, vision support, chat folders, GPU monitoring, HuggingFace browser, VRAM estimator, GGUF parser
- **Improvements** — Virtualized rendering, SQLite WAL mode, auto architecture detection, intelligent startup stage detection
- **Bug Fixes** — vLLM orphan cleanup, GGUF parsing fixes, settings race condition
- **Known Issues** — Linux/CUDA only, multi-GPU requires manual config, some community GGUFs need patching
- **Dependencies** — vLLM 0.7.3+, React 18, Express 4, Zustand 5

---

## Roadmap

| Version | Status | Features |
|---------|--------|----------|
| **v1.1.0** | In Progress | Multi-GPU inference dashboard, chat message editing, download queue management |
| **v1.2.0** | Planned | Embeddings API, RAG pipeline integration, model benchmarking tools |
| **v2.0.0** | Under Consideration | Windows/macOS support, Docker deployment, remote API server, plugin system |

---

## About the Author

<p align="left">
  <strong>Mohamed Ashour</strong><br>
  MRICS (Chartered Quantity Surveyor) · MBCS (Chartered Data Analyst) · Civil Engineer<br>
  Digital Transformation Manager
</p>

Mohamed is a chartered surveyor and data analyst with a decade of combined experience spanning construction engineering, digital transformation, and AI/LLM development. He has worked on major UK and international infrastructure programmes and specialises in building AI-powered tools for the construction and cost intelligence sector. VLLM Studio was built out of a genuine frustration with existing local inference tools on Linux — and a desire for something faster, cleaner, and purpose-built for vLLM.

- **GitHub:** [github.com/Moashour93](https://github.com/Moashour93)
- **LinkedIn:** [linkedin.com/in/mohamed-ashour-0727](https://www.linkedin.com/in/mohamed-ashour-0727/)
- **Email:** [mo_ashour1@outlook.com](mailto:mo_ashour1@outlook.com)

### AI Collaborator

**DeepSeek V4 (Pro)** — Primary AI pair programmer used throughout the architecture, backend, and frontend development of VLLM Studio. All design decisions, architecture choices, and final implementation were directed and reviewed by the author.

---

## Important Note

LLM model support in VLLM Studio is bound by what the vLLM inference engine supports. For the latest list of supported model architectures, quantization methods, and configuration options, refer to the official vLLM documentation at **[docs.vllm.ai](https://docs.vllm.ai)**.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run type-checker (`npm run typecheck`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## License

Apache 2.0 — see the [LICENSE](LICENSE) file for details.

---

**Built with:** React · TypeScript · vLLM · Express · SQLite · NVIDIA CUDA
