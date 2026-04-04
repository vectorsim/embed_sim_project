#!/usr/bin/env bash
# =============================================================================
# cleanup_dfc_controller.sh  —  fs_electrical_machines/c_src
# =============================================================================
# Clean DFC controller build artifacts.
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " Cleaning DFC Controller build artifacts"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "Removing build directory..."
rm -rf build

echo "Removing compiled extensions..."
rm -f dfc_controller_wrapper*.so
rm -f dfc_controller_wrapper*.c

echo "Removing from parent directory..."
rm -f "$PARENT_DIR"/dfc_controller_wrapper*.so

echo "Removing temp directories..."
rm -rf dfc_controller_wrapper_*/

echo ""
echo "============================================================"
echo " Cleanup complete!"
echo "============================================================"