#!/usr/bin/env bash
# =============================================================================
# build_coordinate_transform.sh  —  fs_electrical_machines/c_src
# =============================================================================
# Compile coordinate_transform_wrapper Cython extension on Linux.
# Output .so is copied to fs_electrical_machines/ for easy importing.
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " Building Coordinate Transform C extension  (Linux)"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$PARENT_DIR")"

echo ""
echo "[1/4] Cleaning previous builds..."
rm -rf build
rm -f coordinate_transform_wrapper*.so
rm -f "$PARENT_DIR"/coordinate_transform_wrapper*.so
echo "       Clean complete"

echo ""
echo "[2/4] Setting up Python environment..."

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
echo "[3/4] Building coordinate_transform_wrapper..."
"$PYTHON" setup_coordinate_transform.py build_ext --inplace
echo "       OK - coordinate_transform_wrapper compiled"

echo ""
echo "[4/4] Copying .so to parent directory..."
PYD_FOUND=0
for f in "$SCRIPT_DIR"/coordinate_transform_wrapper*.so; do
    [ -e "$f" ] || continue
    PYD_FOUND=1
    echo "       Found: $(basename "$f")"
    cp -f "$f" "$PARENT_DIR/coordinate_transform_wrapper.so" || { echo "       ERROR: copy failed — check permissions on $PARENT_DIR"; exit 1; }
    echo "       Copied to $PARENT_DIR/coordinate_transform_wrapper.so"
done

if [ "$PYD_FOUND" -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo " WARNING: Build succeeded but no .so found in $SCRIPT_DIR"
    echo "============================================================"
    echo ""
    echo "Run:  find . -name 'coordinate_transform_wrapper*.so'"
    echo "to locate the output and copy manually."
    exit 1
fi

echo ""
echo "============================================================"
echo " Coordinate Transform built successfully!"
echo "============================================================"
echo ""
echo "   c_src/                   : coordinate_transform_wrapper*.so  (ABI-tagged)"
echo "   fs_electrical_machines/  : coordinate_transform_wrapper.so   (plain name)"
echo ""
echo "Import with:"
echo "   from fs_electrical_machines.coordinate_transform_wrapper import ClarkeTransformBlock"
echo ""
