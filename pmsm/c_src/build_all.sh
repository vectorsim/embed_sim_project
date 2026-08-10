#!/usr/bin/env bash
# =============================================================================
# build_all.sh  —  pmsm/c_src/
# =============================================================================
# Build ALL Cython wrappers in one script:
#   1. embedsim_control_wrapper (pmsm/c_src/)
#   2. coordinate_transform_wrapper (fs_electrical_machines/c_src/)
#   3. svpwm_wrapper (fs_electrical_machines/c_src/)
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " Building ALL EmbedSim Wrappers"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"   # EMProject/

echo "Project root: $PROJECT_ROOT"

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
    echo "Using .venv Python: $VENV_PYTHON"
else
    PYTHON="python3"
    echo ".venv not found — using system Python"
fi

"$PYTHON" --version
echo ""

# --------------------------------------------------------------------------
# 1. Build embedsim_control_wrapper (pmsm/c_src/)
# --------------------------------------------------------------------------
echo "============================================================"
echo " [1/3] Building embedsim_control_wrapper"
echo "============================================================"
cd "$PROJECT_ROOT/pmsm/c_src"
"$PYTHON" embedsim_control_wrapper.py build_ext --inplace
cp embedsim_control_wrapper*.so ../embedsim_control_wrapper.so 2>/dev/null || true
echo "✅ embedsim_control_wrapper built"
echo ""

# --------------------------------------------------------------------------
# 2. Build coordinate_transform_wrapper (fs_electrical_machines/c_src/)
# --------------------------------------------------------------------------
echo "============================================================"
echo " [2/3] Building coordinate_transform_wrapper"
echo "============================================================"
cd "$PROJECT_ROOT/fs_electrical_machines/c_src"
"$PYTHON" setup_coordinate_transform.py build_ext --inplace
cp coordinate_transform_wrapper*.so ../coordinate_transform_wrapper.so 2>/dev/null || true
echo "✅ coordinate_transform_wrapper built"
echo ""

# --------------------------------------------------------------------------
# 3. Build svpwm_wrapper (fs_electrical_machines/c_src/)
# --------------------------------------------------------------------------
echo "============================================================"
echo " [3/3] Building svpwm_wrapper"
echo "============================================================"
cd "$PROJECT_ROOT/fs_electrical_machines/c_src"
"$PYTHON" setup_svpwm.py build_ext --inplace
cp svpwm_wrapper*.so ../svpwm_wrapper.so 2>/dev/null || true
echo "✅ svpwm_wrapper built"
echo ""

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
echo "============================================================"
echo " ALL wrappers built successfully!"
echo "============================================================"
echo ""
echo "  pmsm/embedsim_control_wrapper.so"
echo "  fs_electrical_machines/coordinate_transform_wrapper.so"
echo "  fs_electrical_machines/svpwm_wrapper.so"
echo ""
echo "Import with:"
echo "  from embedsim_control_wrapper import control_init, control_step"
echo "  from coordinate_transform_wrapper import ClarkeWrapper, ParkWrapper, InvParkWrapper, InvClarkeWrapper"
echo "  from svpwm_wrapper import EmbedSimSVPWM"
echo "============================================================"
