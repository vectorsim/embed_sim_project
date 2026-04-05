#!/usr/bin/env bash
# =============================================================================
# cleanup_svpwm.sh  —  foc_generator/c_src/
# =============================================================================
# Removes all build artefacts produced by build_svpwm.sh.
#
# DELETED:
#   build/                      Cython compile output
#   svpwm_wrapper.c             Cython-generated C transpile
#   svpwm_wrapper.html          Cython annotation file
#   svpwm_wrapper*.so           Versioned .so in c_src/
#   svpwm_wrapper.so            Plain copy in c_src/
#   ../svpwm_wrapper.so         Promoted copy in foc_generator/
#
# KEPT (never touched):
#   svpwm.c / svpwm.h           Hand-written SVPWM source
#   Sys_Types.h                 Shared type header
#   svpwm_wrapper.pyx           Cython wrapper source
#   setup_svpwm.py              Build script
#   build_svpwm.sh              Build script
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " SVPWM -- Clean-up"
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
echo "[2/5] Removing Cython-generated svpwm_wrapper.c..."
if [[ -f svpwm_wrapper.c ]]; then
    rm -f svpwm_wrapper.c
    echo "        svpwm_wrapper.c removed"
else
    echo "        (nothing to do)"
fi

# -- 3. Cython HTML annotation ------------------------------------------------
echo ""
echo "[3/5] Removing Cython HTML annotation file..."
if [[ -f svpwm_wrapper.html ]]; then
    rm -f svpwm_wrapper.html
    echo "        svpwm_wrapper.html removed"
else
    echo "        (nothing to do)"
fi

# -- 4. .so files in c_src/ --------------------------------------------------
echo ""
echo "[4/5] Removing .so files from c_src/..."
shopt -s nullglob
so_files=(svpwm_wrapper*.so)
if [[ ${#so_files[@]} -gt 0 ]]; then
    for f in "${so_files[@]}"; do
        rm -f "$f"
        echo "        $f removed"
    done
else
    echo "        (nothing to do)"
fi
shopt -u nullglob

# -- 5. Promoted copy in foc_generator/ ---------------------------------------
echo ""
echo "[5/5] Removing promoted copy from parent directory..."
if [[ -f "$PARENT_DIR/svpwm_wrapper.so" ]]; then
    rm -f "$PARENT_DIR/svpwm_wrapper.so"
    echo "        $PARENT_DIR/svpwm_wrapper.so removed"
else
    echo "        (nothing to do)"
fi

echo ""
echo "============================================================"
echo " SVPWM clean complete"
echo "============================================================"
echo ""