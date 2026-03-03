#!/usr/bin/env bash
# =============================================================================
# build_all.sh
# =============================================================================
# Compile all pmsm_blocks Cython extensions in one shot.
#
# Run from the pmsm_blocks/c_src directory:
#   cd pmsm_blocks/c_src
#   chmod +x build_all.sh
#   ./build_all.sh
#
# After building, copy the .so files to where Python can find them:
#   cp *.so ../          (so 'from pmsm_blocks import ...' works)
# =============================================================================

set -e   # abort on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Building pmsm_blocks C extensions"
echo " Directory: $SCRIPT_DIR"
echo "============================================================"

# ── PMSM motor ──────────────────────────────────────────────────────────────
echo ""
echo "[1/3]  pmsm_motor_wrapper"
python setup_pmsm_motor.py build_ext --inplace
echo "       ✓  pmsm_motor_wrapper compiled"

# ── Transforms ──────────────────────────────────────────────────────────────
echo ""
echo "[2/3]  transforms_wrapper  (Clarke, InvClarke, Park, InvPark)"
python setup_transforms.py build_ext --inplace
echo "       ✓  transforms_wrapper compiled"

# ── PI controller ───────────────────────────────────────────────────────────
echo ""
echo "[3/3]  pi_controller_wrapper"
python setup_pi_controller.py build_ext --inplace
echo "       ✓  pi_controller_wrapper compiled"

# ── Copy .so files one level up (into the pmsm_blocks package) ──────────────
echo ""
echo "Copying .so / .pyd files to parent package directory..."
find . -maxdepth 2 \( -name "*.so" -o -name "*.pyd" \) \
    ! -path "./build/*" \
    -exec cp {} .. \;

echo ""
echo "============================================================"
echo " All extensions built successfully."
echo " .so / .pyd files copied to pmsm_blocks/"
echo "============================================================"
echo ""
echo "Test with:"
echo "  python -c \"from pmsm_blocks import PMSMMotorBlock; print('OK')\""
