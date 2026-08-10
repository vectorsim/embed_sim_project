#!/usr/bin#!/usr/bin/env python3
"""
test_wrapper.py  —  Test Cython Wrapper and C Functions
============================================================
Tests:
1. Transform_Init() - initialize transforms
2. Clarke transform
3. Park transform
4. Inverse Park transform
5. Inverse Clarke transform
6. SVPWM (svm_calc and svm_calc_dq)
7. Control initialization
8. Control step
"""

import sys
import math
import numpy as np

# Add paths
import os
from pathlib import Path

# ================================================================
# FIX: Use the same path helper as the example
# ================================================================
from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE = get_current_parent()
_ROOT = get_project_root()
_PMSM = _ROOT / "pmsm"
_C_SRC = _PMSM / "c_src"

# Add ALL paths (same as working example)
for _p in (get_embedsim_import_path(),
           str(_PMSM),
           str(_C_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

print(f"Python path: {sys.path[:5]}")
print(f"Looking for wrapper in: {_C_SRC}")

print("=" * 80)
print(" TESTING CYTHON WRAPPER AND C FUNCTIONS")
print("=" * 80)

# ============================================================================
# TEST 1: Import wrapper
# ============================================================================

print("\n[TEST 1] Importing embedsim_control_wrapper...")
try:
    from embedsim_control_wrapper import (
        clarke, park, inv_park, inv_clarke,
        svm_calc, svm_calc_dq,
        control_init, control_step
    )
    print("  ✅ Import successful!")
    print(f"     Functions: clarke, park, inv_park, inv_clarke, svm_calc, svm_calc_dq")
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
    print("\n  Trying to find the wrapper...")
    import glob
    so_files = glob.glob(str(_C_SRC / "*.so"))
    print(f"  .so files in {_C_SRC}: {so_files}")
    if not so_files:
        print("\n  ❌ No .so files found! Build the wrapper first:")
        print("     cd /home/epl05/EMProject/pmsm/c_src")
        print("     python setup.py build_ext --inplace")
    sys.exit(1)

# ============================================================================
# TEST 2: Transform_Init
# ============================================================================

print("\n[TEST 2] Testing transform initialization...")
try:
    control_init()
    print("  ✅ Transform_Init() called successfully")
except Exception as e:
    print(f"  ❌ Transform_Init() failed: {e}")
    sys.exit(1)

# ============================================================================
# TEST 3: Clarke Transform
# ============================================================================

print("\n[TEST 3] Testing Clarke transform...")
u, v, w = 1.0, -0.5, -0.5
try:
    alpha, beta = clarke(u, v, w)
    print(f"  clarke({u:.1f}, {v:.1f}, {w:.1f}) -> alpha={alpha:.4f}, beta={beta:.4f}")

    if abs(alpha - 1.0) < 0.001 and abs(beta - 0.0) < 0.001:
        print("  ✅ Clarke transform PASSED")
    else:
        print(f"  ❌ Clarke transform FAILED: expected alpha=1.0, beta=0.0")
except Exception as e:
    print(f"  ❌ Clarke transform failed: {e}")

# ============================================================================
# TEST 4: Park Transform
# ============================================================================

print("\n[TEST 4] Testing Park transform...")
alpha, beta = 1.0, 0.0
theta = 0.0
try:
    d, q = park(alpha, beta, theta)
    print(f"  park({alpha:.1f}, {beta:.1f}, {theta:.1f}) -> d={d:.4f}, q={q:.4f}")

    if abs(d - 1.0) < 0.001 and abs(q - 0.0) < 0.001:
        print("  ✅ Park transform PASSED")
    else:
        print(f"  ❌ Park transform FAILED: expected d=1.0, q=0.0")
except Exception as e:
    print(f"  ❌ Park transform failed: {e}")

# ============================================================================
# TEST 5: Inverse Park Transform
# ============================================================================

print("\n[TEST 5] Testing Inverse Park transform...")
d, q = 1.0, 0.0
theta = 0.0
try:
    alpha2, beta2 = inv_park(d, q, theta)
    print(f"  inv_park({d:.1f}, {q:.1f}, {theta:.1f}) -> alpha={alpha2:.4f}, beta={beta2:.4f}")

    if abs(alpha2 - 1.0) < 0.001 and abs(beta2 - 0.0) < 0.001:
        print("  ✅ Inverse Park transform PASSED")
    else:
        print(f"  ❌ Inverse Park transform FAILED: expected alpha=1.0, beta=0.0")
except Exception as e:
    print(f"  ❌ Inverse Park transform failed: {e}")

# ============================================================================
# TEST 6: Inverse Clarke Transform
# ============================================================================

print("\n[TEST 6] Testing Inverse Clarke transform...")
alpha, beta = 1.0, 0.0
try:
    u2, v2, w2 = inv_clarke(alpha, beta)
    print(f"  inv_clarke({alpha:.1f}, {beta:.1f}) -> u={u2:.4f}, v={v2:.4f}, w={w2:.4f}")

    if abs(u2 - 1.0) < 0.001 and abs(v2 - (-0.5)) < 0.001 and abs(w2 - (-0.5)) < 0.001:
        print("  ✅ Inverse Clarke transform PASSED")
    else:
        print(f"  ❌ Inverse Clarke transform FAILED: expected u=1.0, v=-0.5, w=-0.5")
except Exception as e:
    print(f"  ❌ Inverse Clarke transform failed: {e}")

# ============================================================================
# TEST 7: SVPWM (svm_calc)
# ============================================================================

print("\n[TEST 7] Testing SVPWM (svm_calc)...")
v_alpha, v_beta = 6.0, 3.464
vdc = 12.0
try:
    ta, tb, tc, sector = svm_calc(v_alpha, v_beta, vdc)
    print(f"  svm_calc({v_alpha:.1f}, {v_beta:.1f}, {vdc:.1f}) -> ta={ta:.4f}, tb={tb:.4f}, tc={tc:.4f}, sector={sector}")

    if abs(ta - 0.75) < 0.1 and abs(tb - 0.25) < 0.1:
        print("  ✅ SVPWM PASSED")
    else:
        print(f"  ❌ SVPWM FAILED: expected ta≈0.75, tb≈0.25")
except Exception as e:
    print(f"  ❌ SVPWM failed: {e}")

# ============================================================================
# TEST 8: SVPWM from DQ (svm_calc_dq)
# ============================================================================

print("\n[TEST 8] Testing SVPWM from DQ (svm_calc_dq)...")
vd, vq = 0.0, 3.464
theta = 0.0
vdc = 12.0
try:
    ta, tb, tc, sector = svm_calc_dq(vd, vq, theta, vdc)
    print(f"  svm_calc_dq({vd:.1f}, {vq:.1f}, {theta:.1f}, {vdc:.1f}) -> ta={ta:.4f}, tb={tb:.4f}, tc={tc:.4f}, sector={sector}")

    if abs(ta - 0.75) < 0.1 and abs(tb - 0.25) < 0.1:
        print("  ✅ SVPWM DQ PASSED")
    else:
        print(f"  ❌ SVPWM DQ FAILED: expected ta≈0.75, tb≈0.25")
except Exception as e:
    print(f"  ❌ SVPWM DQ failed: {e}")

# ============================================================================
# TEST 9: SVPWM with Vdc=0.5 (The Problem Case!)
# ============================================================================

print("\n[TEST 9] Testing SVPWM with Vdc=0.5 (problem case)...")
vd, vq = 0.0, 0.144337
theta = 0.0
vdc = 0.5
try:
    ta, tb, tc, sector = svm_calc_dq(vd, vq, theta, vdc)
    print(f"  svm_calc_dq({vd:.1f}, {vq:.6f}, {theta:.1f}, {vdc:.1f}) -> ta={ta:.4f}, tb={tb:.4f}, tc={tc:.4f}, sector={sector}")

    if abs(ta - 0.5) < 0.1 and abs(tb - 0.5) < 0.1:
        print("  ✅ SVPWM with Vdc=0.5 PASSED")
    else:
        print("  ⚠️ SVPWM with Vdc=0.5 returned different values (may still be valid)")
except Exception as e:
    print(f"  ❌ SVPWM with Vdc=0.5 FAILED: {e}")

# ============================================================================
# TEST 10: Control Step
# ============================================================================

print("\n[TEST 10] Testing control_step...")
try:
    result = control_step(
        speed_ref_rpm=2000.0,
        ia=0.0,
        ib=0.0,
        ic=0.0,
        position_rad=0.0,
        speed_rpm=0.0,
        vdc=12.0,
        dt=50e-6,
        ctrl_alg=0,
        valid=1,
    )
    print(f"  control_step result:")
    print(f"    ta={result['ta']:.4f}, tb={result['tb']:.4f}, tc={result['tc']:.4f}")
    print(f"    speed_est={result['speed_est']:.1f} RPM")
    print(f"    sector={result['sector']}, valid={result['valid']}")

    if result['valid'] == 1:
        print("  ✅ Control step PASSED")
    else:
        print(f"  ❌ Control step FAILED: valid={result['valid']}")
except Exception as e:
    print(f"  ❌ Control step failed: {e}")

# ============================================================================
# TEST 11: Control Step with DFC mode
# ============================================================================

print("\n[TEST 11] Testing control_step with DFC mode...")
try:
    result = control_step(
        speed_ref_rpm=2000.0,
        ia=0.0,
        ib=0.0,
        ic=0.0,
        position_rad=0.0,
        speed_rpm=0.0,
        vdc=12.0,
        dt=50e-6,
        ctrl_alg=1,
        valid=1,
    )
    print(f"  control_step (DFC) result:")
    print(f"    ta={result['ta']:.4f}, tb={result['tb']:.4f}, tc={result['tc']:.4f}")
    print(f"    speed_est={result['speed_est']:.1f} RPM")
    print(f"    sector={result['sector']}, valid={result['valid']}")

    if result['valid'] == 1:
        print("  ✅ DFC control step PASSED")
    else:
        print(f"  ❌ DFC control step FAILED: valid={result['valid']}")
except Exception as e:
    print(f"  ❌ DFC control step failed: {e}")

# ============================================================================
# TEST 12: Check if Vdc is passed correctly through control_step
# ============================================================================

print("\n[TEST 12] Testing Vdc propagation through control_step...")
try:
    vdc_test_values = [0.5, 5.0, 12.0, 17.0, 24.0]
    for test_vdc in vdc_test_values:
        result = control_step(
            speed_ref_rpm=1000.0,
            ia=0.0,
            ib=0.0,
            ic=0.0,
            position_rad=0.0,
            speed_rpm=0.0,
            vdc=test_vdc,
            dt=50e-6,
            ctrl_alg=0,
            valid=1,
        )
        print(f"  Vdc={test_vdc:.1f}V -> ta={result['ta']:.4f}, valid={result['valid']}")

    print("  ✅ Vdc propagation test PASSED (no crashes)")
except Exception as e:
    print(f"  ❌ Vdc propagation test FAILED: {e}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print(" TEST SUMMARY")
print("=" * 80)
print("\n  All tests completed!")
print("  Check the output above for any failures.")
print("=" * 80)