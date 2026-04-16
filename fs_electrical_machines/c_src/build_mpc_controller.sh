#!/usr/bin/env bash
# =============================================================================
# build_mpc_controller.sh  —  fs_electrical_machines/c_src
# =============================================================================
# Compile mpc_controller_wrapper Cython extension on Linux.
# Output .so is copied to fs_electrical_machines/ for easy importing.
#
# MISRA C:2012 compliant code generation via Cython.
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " Building MPC Controller C extension  (Linux)"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$PARENT_DIR")"

echo ""
echo "[1/5] Cleaning previous builds..."
rm -rf build
rm -f mpc_controller_wrapper*.so
rm -f mpc_controller_wrapper.c
rm -f mpc_controller_wrapper.html
rm -f "$PARENT_DIR"/mpc_controller_wrapper*.so
echo "       Clean complete"

echo ""
echo "[2/5] Checking required source files..."
MISSING_FILES=0
for f in "mpc_controller_wrapper.pyx" \
         "embed_sim_mpc_controller.c" \
         "embed_sim_mpc_controller.h" \
         "embed_sim_mpc_gains.h" \
         "embed_sim_coordinate_transform.c" \
         "embed_sim_matrix.c"; do
    if [ ! -f "$SCRIPT_DIR/$f" ]; then
        echo "       MISSING: $f"
        MISSING_FILES=1
    fi
done

if [ $MISSING_FILES -eq 1 ]; then
    echo ""
    echo "============================================================"
    echo " ERROR: Missing source files!"
    echo "============================================================"
    echo ""
    echo "Ensure all required MPC source files are present in:"
    echo "   $SCRIPT_DIR"
    echo ""
    exit 1
fi
echo "       All source files present"

echo ""
echo "[3/5] Setting up Python environment..."

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
echo "[4/5] Building mpc_controller_wrapper..."
"$PYTHON" setup_mpc_controller.py build_ext --inplace

if [ $? -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo " ERROR: Build failed!"
    echo "============================================================"
    echo ""
    echo "Common causes:"
    echo "   1. Missing C compiler (gcc/clang)"
    echo "   2. Missing Python development headers"
    echo "   3. C syntax errors in MPC source files"
    echo ""
    exit 1
fi
echo "       OK - mpc_controller_wrapper compiled"

echo ""
echo "[5/5] Copying .so to parent directory..."
PYD_FOUND=0
for f in "$SCRIPT_DIR"/mpc_controller_wrapper*.so; do
    [ -e "$f" ] || continue
    PYD_FOUND=1
    echo "       Found: $(basename "$f")"
    cp -f "$f" "$PARENT_DIR/mpc_controller_wrapper.so" || { echo "       ERROR: copy failed — check permissions on $PARENT_DIR"; exit 1; }
    echo "       Copied to $PARENT_DIR/mpc_controller_wrapper.so"
done

if [ "$PYD_FOUND" -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo " WARNING: Build succeeded but no .so found in $SCRIPT_DIR"
    echo "============================================================"
    echo ""
    echo "Run:  find . -name 'mpc_controller_wrapper*.so'"
    echo "to locate the output and copy manually."
    exit 1
fi

echo ""
echo "============================================================"
echo " MPC Controller built successfully!"
echo "============================================================"
echo ""
echo "   c_src/                   : mpc_controller_wrapper*.so  (ABI-tagged)"
echo "   fs_electrical_machines/  : mpc_controller_wrapper.so   (plain name)"
echo ""
echo "Import with:"
echo "   from fs_electrical_machines.mpc_controller_wrapper import MPCControllerWrapper"
echo ""
echo "Test with:"
echo "   python -c \"from fs_electrical_machines.mpc_controller_wrapper import MPCControllerWrapper; print('OK')\""
echo ""