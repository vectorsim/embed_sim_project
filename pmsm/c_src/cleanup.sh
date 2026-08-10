#!/usr/bin/env bash
# =============================================================================
# cleanup.sh  —  pmsm/c_src
# =============================================================================
# Clean EmbedSim Control Wrapper build artifacts.
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " Cleaning EmbedSim Control Wrapper build artifacts"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"          # pmsm/

echo ""
echo "Removing build directory..."
rm -rf build

echo "Removing compiled extensions from c_src/..."
rm -f embedsim_control_wrapper*.so
rm -f embedsim_control_wrapper*.c

echo "Removing from parent directory (pmsm/)..."
rm -f "$PARENT_DIR"/embedsim_control_wrapper*.so

echo "Removing temp directories..."
rm -rf embedsim_control_wrapper_*/

echo ""
echo "============================================================"
echo " Cleanup complete!"
echo "============================================================"