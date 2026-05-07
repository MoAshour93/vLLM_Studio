#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# VLLM Studio Launcher
# Single-entry bootstrap script for the entire application.
# Checks dependencies, creates a Python virtual environment for vLLM,
# installs all packages, builds, and launches the full stack.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/data/logs"
LOG_FILE="$LOG_DIR/vllm-studio.log"
PID_FILE="$SCRIPT_DIR/data/vllm.pid"
VENV_DIR="$SCRIPT_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python3"

# Ensure directories exist
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$SCRIPT_DIR/data/chats"
mkdir -p "$SCRIPT_DIR/data/attachments"
mkdir -p "$SCRIPT_DIR/data/models"

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
log() {
  local timestamp
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$timestamp] $*" | tee -a "$LOG_FILE"
}

# ANSI colour codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

success() { echo -e "${GREEN}✓${NC} $*"; }
warn()   { echo -e "${YELLOW}⚠${NC} $*"; }
error()  { echo -e "${RED}✗${NC} $*" >&2; }
info()   { echo -e "${BLUE}ℹ${NC} $*"; }

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
log ""
log "=============================================="
log " VLLM Studio — Launcher"
log "=============================================="
log ""

echo -e "${BOLD}${CYAN}╔══════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║        VLLM Studio Launcher           ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════╝${NC}"
echo ""

# -----------------------------------------------------------------------------
# Phase 1: System dependency checks
# -----------------------------------------------------------------------------
info "Phase 1: Checking system dependencies..."
echo ""

MISSING_DEPS=0

check_dependency() {
  local name="$1"
  local check_cmd="$2"
  local install_hint="$3"
  local min_version="$4"
  local version_cmd="$5"

  if ! eval "$check_cmd" &>/dev/null; then
    error "$name is NOT installed."
    echo -e "  ${CYAN}Install:${NC} $install_hint" >&2
    MISSING_DEPS=$((MISSING_DEPS + 1))
    return 1
  fi

  if [[ -n "$min_version" && -n "$version_cmd" ]]; then
    local installed_version
    installed_version="$(eval "$version_cmd" 2>/dev/null | grep -oP '[0-9]+(\.[0-9]+)*' | head -1 || echo "0")"
    if [[ -z "$installed_version" ]]; then
      warn "Could not determine version of $name, skipping version check."
    else
      local min_major min_minor installed_major installed_minor
      IFS='.' read -r min_major min_minor _ <<< "$min_version"
      IFS='.' read -r installed_major installed_minor _ <<< "$installed_version"
      min_major=${min_major:-0}; min_minor=${min_minor:-0}
      installed_major=${installed_major:-0}; installed_minor=${installed_minor:-0}
      if (( installed_major < min_major )) || { (( installed_major == min_major )) && (( installed_minor < min_minor )); }; then
        error "$name version $installed_version is too old (need ≥ $min_version)."
        echo -e "  ${CYAN}Update:${NC} $install_hint" >&2
        MISSING_DEPS=$((MISSING_DEPS + 1))
        return 1
      fi
    fi
  fi

  success "$name found."
  return 0
}

check_dependency "Node.js" \
  "node --version" \
  "curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt-get install -y nodejs" \
  "18.0.0" \
  "node --version"

check_dependency "npm" \
  "npm --version" \
  "sudo apt-get install -y npm" \
  "" \
  ""

check_dependency "Python 3" \
  "python3 --version" \
  "sudo apt-get install -y python3 python3-pip python3-venv" \
  "3.10.0" \
  "python3 --version"

check_dependency "CUDA / nvidia-smi" \
  "nvidia-smi" \
  "Install NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads" \
  "" \
  ""

echo ""
if [[ $MISSING_DEPS -gt 0 ]]; then
  error "$MISSING_DEPS dependency/dependencies missing. Please install them and re-run this script."
  exit 1
fi

success "All system dependencies satisfied."
echo ""

# -----------------------------------------------------------------------------
# Phase 2: Python virtual environment + vLLM
# -----------------------------------------------------------------------------
info "Phase 2: Setting up Python virtual environment for vLLM..."
echo ""

VENV_READY=false

if [[ -d "$VENV_DIR" ]] && [[ -f "$VENV_PYTHON" ]]; then
  info "Existing virtual environment found at $VENV_DIR"

  # Verify the venv is healthy and has vLLM
  if "$VENV_PYTHON" -c "import vllm" &>/dev/null; then
    vllm_ver="$("$VENV_PYTHON" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")"
    success "Virtual environment is healthy. vLLM version: $vllm_ver"

    if ! "$VENV_PYTHON" -c "import huggingface_hub" &>/dev/null; then
      info "Installing huggingface_hub for model downloads..."
      "$VENV_PYTHON" -m pip install "huggingface_hub[hf_transfer]" 2>&1 | tee -a "$LOG_FILE" || \
        "$VENV_PYTHON" -m pip install huggingface_hub 2>&1 | tee -a "$LOG_FILE"
      "$VENV_PYTHON" -c "import huggingface_hub" &>/dev/null && \
        success "huggingface_hub installed." || \
        warn "Failed to install huggingface_hub."
    fi

    VENV_READY=true
  else
    warn "Virtual environment exists but vLLM is not installed. Re-installing..."
  fi
else
  info "No virtual environment found. Creating one..."
fi

if [[ "$VENV_READY" != "true" ]]; then
  if [[ -e "$VENV_DIR" ]]; then
    info "Removing broken virtual environment..."
    rm -rf "$VENV_DIR"
  fi
  info "Creating Python virtual environment..."
  python3 -m venv "$VENV_DIR"

  if [[ ! -f "$VENV_PYTHON" ]]; then
    error "Failed to create virtual environment at $VENV_DIR"
    exit 1
  fi

  # Upgrade pip and setuptools inside venv
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel 2>&1 | tee -a "$LOG_FILE"

  info "Installing vLLM into virtual environment (this may take several minutes)..."
  log "Running: $VENV_PYTHON -m pip install vllm"

  if "$VENV_PYTHON" -m pip install vllm 2>&1 | tee -a "$LOG_FILE"; then
    vllm_ver="$("$VENV_PYTHON" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")"
    success "vLLM $vllm_ver installed successfully in virtual environment."
  else
    error "Failed to install vLLM. Check the log for details: $LOG_FILE"
    error ""
    error "Manual install:"
    error "  source $VENV_DIR/bin/activate"
    error "  pip install vllm"
    error ""
    exit 1
  fi

  info "Installing huggingface_hub for model downloads..."

  "$VENV_PYTHON" -m pip install "huggingface_hub[hf_transfer]" 2>&1 | tee -a "$LOG_FILE" || \
    "$VENV_PYTHON" -m pip install huggingface_hub 2>&1 | tee -a "$LOG_FILE"

  if "$VENV_PYTHON" -c "import huggingface_hub" &>/dev/null; then
    success "huggingface_hub installed."
  else
    warn "Failed to install huggingface_hub. Model download features will be unavailable."
  fi
fi

# Export the venv Python path so the backend knows which Python to use
export VLLM_PYTHON="$VENV_PYTHON"
log "VLLM_PYTHON=$VLLM_PYTHON"

  success "vLLM virtual environment ready."

  # Ensure ninja-build is available on PATH (needed by FlashInfer JIT in vLLM's EngineCore)
  if [[ ! -f "$VENV_DIR/bin/ninja" ]]; then
    info "Installing ninja (required by FlashInfer)..."
    "$VENV_PYTHON" -m pip install ninja 2>&1 | tee -a "$LOG_FILE"
  fi
  # Make ninja available system-wide for EngineCore subprocesses
  mkdir -p "$HOME/.local/bin"
  ln -sf "$VENV_DIR/bin/ninja" "$HOME/.local/bin/ninja"
  success "ninja available."

echo ""

# -----------------------------------------------------------------------------
# Phase 3: Node.js dependencies
# -----------------------------------------------------------------------------
info "Phase 3: Installing Node.js dependencies..."
echo ""

if [[ ! -d "$SCRIPT_DIR/node_modules" ]]; then
  log "Root node_modules missing. Running npm install..."
  npm install --prefix "$SCRIPT_DIR" 2>&1 | tee -a "$LOG_FILE"
  success "Root dependencies installed."
else
  success "Root dependencies already present."
fi

if [[ ! -d "$SCRIPT_DIR/backend/node_modules" ]]; then
  log "Backend node_modules missing. Running npm install..."
  npm install --prefix "$SCRIPT_DIR/backend" 2>&1 | tee -a "$LOG_FILE"
  success "Backend dependencies installed."
else
  success "Backend dependencies already present."
fi

if [[ ! -d "$SCRIPT_DIR/frontend/node_modules" ]]; then
  log "Frontend node_modules missing. Running npm install..."
  npm install --prefix "$SCRIPT_DIR/frontend" 2>&1 | tee -a "$LOG_FILE"
  success "Frontend dependencies installed."
else
  success "Frontend dependencies already present."
fi

echo ""

# -----------------------------------------------------------------------------
# Phase 4: Build steps
# -----------------------------------------------------------------------------
info "Phase 4: Building application..."
echo ""

# Build frontend if dist missing or package.json/src changed
NEEDS_FRONTEND_BUILD=false
if [[ ! -d "$SCRIPT_DIR/frontend/dist" ]]; then
  NEEDS_FRONTEND_BUILD=true
else
  # Check if any source file is newer than the built output
  DIST_MTIME=$(stat -c %Y "$SCRIPT_DIR/frontend/dist/index.html" 2>/dev/null || echo 0)
  NEWEST_SRC=$(find "$SCRIPT_DIR/frontend/src" "$SCRIPT_DIR/frontend/index.html" "$SCRIPT_DIR/frontend/package.json" "$SCRIPT_DIR/frontend/tsconfig.json" "$SCRIPT_DIR/frontend/vite.config.ts" -type f -exec stat -c %Y {} \; 2>/dev/null | sort -rn | head -1)
  if [[ ${NEWEST_SRC:-0} -gt ${DIST_MTIME:-0} ]]; then
    NEEDS_FRONTEND_BUILD=true
  fi
fi

if [[ "$NEEDS_FRONTEND_BUILD" == "true" ]]; then
  log "Frontend needs rebuilding. Building..."
  rm -rf "$SCRIPT_DIR/frontend/dist"
  npm run build --prefix "$SCRIPT_DIR/frontend" 2>&1 | tee -a "$LOG_FILE"
  success "Frontend built."
else
  success "Frontend up to date."
fi

# Compile backend if dist missing or any TS source changed
NEEDS_BACKEND_BUILD=false
if [[ ! -f "$SCRIPT_DIR/backend/dist/index.js" ]]; then
  NEEDS_BACKEND_BUILD=true
else
  DIST_MTIME=$(stat -c %Y "$SCRIPT_DIR/backend/dist/index.js" 2>/dev/null || echo 0)
  NEWEST_SRC=$(find "$SCRIPT_DIR/backend/src" "$SCRIPT_DIR/backend/tsconfig.json" "$SCRIPT_DIR/backend/package.json" -type f -exec stat -c %Y {} \; 2>/dev/null | sort -rn | head -1)
  if [[ ${NEWEST_SRC:-0} -gt ${DIST_MTIME:-0} ]]; then
    NEEDS_BACKEND_BUILD=true
  fi
fi

if [[ "$NEEDS_BACKEND_BUILD" == "true" ]]; then
  log "Backend needs recompiling..."
  rm -rf "$SCRIPT_DIR/backend/dist"
  npm run build --prefix "$SCRIPT_DIR/backend" 2>&1 | tee -a "$LOG_FILE"
  success "Backend compiled."
else
  success "Backend up to date."
fi

echo ""

# -----------------------------------------------------------------------------
# Phase 5: Generate .env file from .env.example if missing
# -----------------------------------------------------------------------------
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
  if [[ -f "$SCRIPT_DIR/.env.example" ]]; then
    info "Creating .env from .env.example..."
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo "VLLM_PYTHON=$VENV_PYTHON" >> "$SCRIPT_DIR/.env"
    success ".env file created with venv Python path."
  fi
else
  # Ensure the .env has the current VLLM_PYTHON value
  if grep -q "^VLLM_PYTHON=" "$SCRIPT_DIR/.env" 2>/dev/null; then
    sed -i "s|^VLLM_PYTHON=.*|VLLM_PYTHON=$VENV_PYTHON|" "$SCRIPT_DIR/.env"
  else
    echo "VLLM_PYTHON=$VENV_PYTHON" >> "$SCRIPT_DIR/.env"
  fi
fi

echo ""

# -----------------------------------------------------------------------------
# Port cleanup helper — kills any process bound to the given ports
# -----------------------------------------------------------------------------
free_ports() {
  for port in "$@"; do
    fuser -k "$port/tcp" 2>/dev/null && log "Freed port $port" || true
  done
}

# -----------------------------------------------------------------------------
# Cleanup handler
# -----------------------------------------------------------------------------
cleanup() {
  log "Received shutdown signal. Cleaning up..."
  warn "Shutting down VLLM Studio..."

  if [[ -f "$PID_FILE" ]]; then
    local vllm_pid
    vllm_pid="$(cat "$PID_FILE" 2>/dev/null || echo "")"
    if [[ -n "$vllm_pid" ]]; then
      log "Stopping vLLM process (PID: $vllm_pid)..."
      kill -TERM "$vllm_pid" 2>/dev/null || true
      sleep 2
      kill -KILL "$vllm_pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi

  # Kill any remaining vLLM engine-core subprocesses
  for pid in $(pgrep -f "EngineCore" 2>/dev/null); do
    kill -KILL "$pid" 2>/dev/null || true
  done

  if [[ -n "${BACKEND_PID:-}" ]]; then
    log "Stopping backend server (PID: $BACKEND_PID)..."
    kill -TERM "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi

  # Free all known ports
  free_ports 3333 8000 8001 8002 8003

  log "VLLM Studio shutdown complete."
  exit 0
}

trap cleanup SIGINT SIGTERM

# Also register an EXIT trap so that if the script is killed via SIGKILL
# or any other unhandled signal, we at least try to clean up ports.
# (SIGKILL cannot be trapped, but this helps for normal exit paths.)
trap 'free_ports 3333 8000 8001 8002 8003' EXIT

# -----------------------------------------------------------------------------
# Pre-launch GPU cleanup: kill leftover vLLM processes and free ports
# -----------------------------------------------------------------------------
info "Cleaning up stale processes..."
STALE_COUNT=0
for pid in $(pgrep -f "vllm.entrypoints\|EngineCore" 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null && STALE_COUNT=$((STALE_COUNT + 1))
done
if [[ $STALE_COUNT -gt 0 ]]; then
  warn "Killed $STALE_COUNT stale vLLM process(es). GPU memory freed."
else
  success "No stale vLLM processes found."
fi

# Free all ports used by the app before starting
free_ports 3333 8000 8001 8002 8003

echo ""

# -----------------------------------------------------------------------------
# Phase 6: Launch
# -----------------------------------------------------------------------------
info "Phase 6: Starting VLLM Studio..."
echo ""

log "Launching backend server with VLLM_PYTHON=$VLLM_PYTHON..."

VLLM_PYTHON="$VENV_PYTHON" node "$SCRIPT_DIR/backend/dist/index.js" &
BACKEND_PID=$!

log "Backend server started with PID $BACKEND_PID"
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║  VLLM Studio is running!                          ║${NC}"
echo -e "${GREEN}${BOLD}║                                                    ║${NC}"
echo -e "${GREEN}${BOLD}║  Open your browser at:                             ║${NC}"
echo -e "${GREEN}${BOLD}║  → http://localhost:3333                           ║${NC}"
echo -e "${GREEN}${BOLD}║                                                    ║${NC}"
echo -e "${GREEN}${BOLD}║  Virtual env: $VENV_DIR                ║${NC}"
echo -e "${GREEN}${BOLD}║  Press Ctrl+C to stop.                             ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""

if command -v xdg-open &>/dev/null; then
  xdg-open "http://localhost:3333" &>/dev/null || true
elif command -v open &>/dev/null; then
  open "http://localhost:3333" &>/dev/null || true
fi

wait "$BACKEND_PID" 2>/dev/null || true
