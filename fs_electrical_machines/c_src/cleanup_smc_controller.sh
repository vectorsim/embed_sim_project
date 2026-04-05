#!/usr/bin/env bash
# =============================================================================
# cleanup_smc_controller.sh  —  fs_electrical_machines/c_src
# =============================================================================
# Clean up build artifacts from smc_controller_wrapper compilation.
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " Cleaning SMC Controller build artifacts"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "Removing build directory..."
rm -rf build

echo "Removing .so files..."
rm -f smc_controller_wrapper*.so

echo "Removing from parent directory..."
rm -f "$PARENT_DIR"/smc_controller_wrapper*.so

echo "Removing .c files from Cython..."
rm -f smc_controller_wrapper.c
rm -f smc_controller_wrapper.html

echo ""
echo "============================================================"
echo " Clean complete!"
echo "============================================================"