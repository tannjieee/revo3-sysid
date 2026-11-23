#!/usr/bin/env bash
set -euo pipefail

# One-command installer for REVO3-SysID (prefers uv; falls back cleanly to python3 -m venv).
# Usage:
#   ./scripts/install.sh              # Core + REVO3 hardware SDK (recommended)
#   ./scripts/install.sh --sim-only   # Simulation / SysID / asset export only (no SDK)

EXTRAS=".[hardware]"
if [[ "${1:-}" == "--sim-only" ]]; then
  EXTRAS="."
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== REVO3-SysID Installer ($ROOT_DIR) ==="

if command -v uv >/dev/null 2>&1; then
  echo "[1/3] Found uv ($(uv --version)). Creating .venv without requiring python3-venv..."
  rm -rf .venv
  uv venv .venv
  echo "[2/3] Installing $EXTRAS via uv..."
  uv pip install --python .venv/bin/python -e "$EXTRAS"
else
  PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  echo "[1/3] Creating .venv with python3 ($PY_VER)..."
  rm -rf .venv
  if ! python3 -m venv .venv; then
    rm -rf .venv
    echo ""
    echo "[ERROR] python3 -m venv failed (missing ensurepip / python3-venv on Debian/Ubuntu)."
    echo "Please choose one of the two fixes and re-run ./scripts/install.sh:"
    echo "  Option A (recommended, no sudo): curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Option B (apt):                  sudo apt update && sudo apt install -y python${PY_VER}-venv"
    exit 1
  fi
  echo "[2/3] Installing $EXTRAS via pip..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/pip install -e "$EXTRAS"
fi

echo "[3/3] Running zero-hardware verification..."
.venv/bin/revo3-sysid demo

echo ""
echo "=== Installation complete! ==="
echo "Activate and launch the 5-step guided GUI with:"
echo "  source .venv/bin/activate"
echo "  revo3-sysid gui"
