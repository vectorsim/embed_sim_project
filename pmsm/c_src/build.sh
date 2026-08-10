#!/usr/bin/env bash
# =============================================================================
# build.sh  —  pmsm/c_src
# =============================================================================
# Compile embedsim_control_wrapper Cython extension on Linux.
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " Building EmbedSim Control Wrapper (Sensor-Based)"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$PARENT_DIR")"

echo ""
echo "[1/5] Cleaning previous builds..."
rm -rf build
rm -f embedsim_control_wrapper*.so
rm -f embedsim_control_wrapper*.c
rm -f "$PARENT_DIR"/embedsim_control_wrapper*.so
echo "       Clean complete"

echo ""
echo "[2/5] Setting up Python environment..."

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
    echo "       Using .venv Python: $VENV_PYTHON"
else
    PYTHON="python3"
    echo "       .venv not found — using system Python"
fi

"$PYTHON" --version

echo "       Checking dependencies..."
if ! "$PYTHON" -c "import setuptools" &>/dev/null; then
    echo "       setuptools not found. Installing..."
    "$PYTHON" -m pip install --upgrade pip setuptools || { echo "       Failed to install setuptools"; exit 1; }
else
    echo "       setuptools OK"
fi

if ! "$PYTHON" -c "import Cython" &>/dev/null; then
    echo "       Cython not found. Installing..."
    "$PYTHON" -m pip install --upgrade pip cython || { echo "       Failed to install Cython"; exit 1; }
else
    echo "       Cython OK"
fi

if ! "$PYTHON" -c "import numpy" &>/dev/null; then
    echo "       NumPy not found. Installing..."
    "$PYTHON" -m pip install numpy || { echo "       Failed to install NumPy"; exit 1; }
else
    echo "       NumPy OK"
fi

echo ""
echo "[3/5] Checking source files..."
MISSING=0
for src in embedsim_control_wrapper.pyx embed_sim_control.c embed_sim_dfc_controller.c embed_sim_coordinate_transform.c embed_sim_sv_pwm.c embed_sim_matrix.c; do
    if [ ! -f "$src" ]; then
        echo "       ERROR: Missing $src"
        MISSING=1
    else
        echo "       Found: $src"
    fi
done

if [ $MISSING -ne 0 ]; then
    echo "       ERROR: Missing source files. Aborting."
    exit 1
fi

echo ""
echo "[4/5] Building embedsim_control_wrapper..."
"$PYTHON" embedsim_control_wrapper.py build_ext --inplace
if [ $? -ne 0 ]; then
    echo "       ERROR: Build failed"
    exit 1
fi
echo "       OK - embedsim_control_wrapper compiled"

echo ""
echo "[5/5] Copying .so to parent directory..."
PYD_FOUND=0
for f in "$SCRIPT_DIR"/embedsim_control_wrapper*.so; do
    [ -e "$f" ] || continue
    PYD_FOUND=1
    echo "       Found: $(basename "$f")"
    cp -f "$f" "$PARENT_DIR/embedsim_control_wrapper.so" || {
        echo "       ERROR: copy failed — check permissions on $PARENT_DIR"
        exit 1
    }
    echo "       Copied to $PARENT_DIR/embedsim_control_wrapper.so"
done

if [ "$PYD_FOUND" -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo " WARNING: Build succeeded but no .so found in $SCRIPT_DIR"
    echo "============================================================"
    echo ""
    echo "Run:  find . -name 'embedsim_control_wrapper*.so'"
    echo "to locate the output and copy manually."
    exit 1
fi

echo ""
echo "============================================================"
echo " EmbedSim Control Wrapper built successfully!"
echo "============================================================"
echo ""
echo "   c_src/                    : embedsim_control_wrapper*.so  (ABI-tagged)"
echo "   pmsm/  : embedsim_control_wrapper.so   (plain name)"
echo ""
echo "Import with:"
echo "   from embedsim_control_wrapper import control_init, control_step"
echo ""