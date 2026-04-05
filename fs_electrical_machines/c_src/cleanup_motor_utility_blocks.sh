#!/usr/bin/env bash
# =============================================================================
# cleanup_motor_utility_blocks.sh  —  fs_electrical_machines/c_src/
# =============================================================================
# Removes all build artefacts produced by build_motor_utility_blocks.sh.
#
# DELETED:
#   build/                                  Cython compile output
#   motor_utility_blocks_wrapper.c          Cython-generated C transpile
#   motor_utility_blocks_wrapper.html       Cython annotation file
#   motor_utility_blocks_wrapper*.so        Versioned .so in c_src/
#   motor_utility_blocks_wrapper.so         Plain copy in c_src/
#   ../motor_utility_blocks_wrapper.so      Promoted copy in fs_electrical_machines/
#
# KEPT (never touched):
#   motor_utility_blocks.c / .h             Hand-written C source
#   Sys_Types.h                             Shared type header
#   motor_utility_blocks_wrapper.pyx        Cython wrapper source
#   setup_motor_utility_blocks.py           Build script
#   build_motor_utility_blocks.sh           Build script
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " motor_utility_blocks -- Clean-up"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

echo " c_src : $SCRIPT_DIR"
echo " parent: $PARENT_DIR"
echo ""

# -- 1. build/ directory ------------------------------------------------------
echo "[1/5] Removing build/ directory..."
if [[ -d build ]]; then
    rm -rf build
    echo "        build/ removed"
else
    echo "        (nothing to do)"
fi

# -- 2. Cython-generated wrapper .c ------------------------------------------
echo ""
echo "[2/5] Removing Cython-generated motor_utility_blocks_wrapper.c..."
if [[ -f motor_utility_blocks_wrapper.c ]]; then
    rm -f motor_utility_blocks_wrapper.c
    echo "        motor_utility_blocks_wrapper.c removed"
else
    echo "        (nothing to do)"
fi

# -- 3. Cython HTML annotation ------------------------------------------------
echo ""
echo "[3/5] Removing Cython HTML annotation file..."
if [[ -f motor_utility_blocks_wrapper.html ]]; then
    rm -f motor_utility_blocks_wrapper.html
    echo "        motor_utility_blocks_wrapper.html removed"
else
    echo "        (nothing to do)"
fi

# -- 4. .so files in c_src/ --------------------------------------------------
echo ""
echo "[4/5] Removing .so files from c_src/..."
shopt -s nullglob
so_files=(motor_utility_blocks_wrapper*.so)
if [[ ${#so_files[@]} -gt 0 ]]; then
    for f in "${so_files[@]}"; do
        rm -f "$f"
        echo "        $f removed"
    done
else
    echo "        (nothing to do)"
fi
shopt -u nullglob

# -- 5. Promoted copy in fs_electrical_machines/ -----------------------------
echo ""
echo "[5/5] Removing promoted copy from parent directory..."
if [[ -f "$PARENT_DIR/motor_utility_blocks_wrapper.so" ]]; then
    rm -f "$PARENT_DIR/motor_utility_blocks_wrapper.so"
    echo "        $PARENT_DIR/motor_utility_blocks_wrapper.so removed"
else
    echo "        (nothing to do)"
fi

echo ""
echo "============================================================"
echo " motor_utility_blocks clean complete"
echo "============================================================"
echo ""