"""
test_svpwm.py  —  Test the compiled SVPWM module
==================================================
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fs_electrical_machines.svpwm_wrapper import EmbedSimSVPWM, svpwm_step

    print("=" * 60)
    print("SVPWM Module Test")
    print("=" * 60)

    # Test 1: Create instance
    print("\n1. Creating EmbedSimSVPWM instance...")
    svpwm = EmbedSimSVPWM(timer_period=10000)
    print(f"   ✓ Timer period: {svpwm.timer_period}")

    # Test 2: Calculate with alpha-beta (0.8, 0.0)
    print("\n2. Calculating with alpha=0.8, beta=0.0...")
    svpwm.calculate(0.8, 0.0)
    print(f"   Phase A: {svpwm.ta:.4f}")
    print(f"   Phase B: {svpwm.tb:.4f}")
    print(f"   Phase C: {svpwm.tc:.4f}")
    print(f"   Sector: {svpwm.sector}")
    print(f"   Magnitude: {svpwm.magnitude:.4f}")
    print(f"   Angle: {svpwm.angle:.4f} rad")

    # Test 3: Calculate with complex voltage
    print("\n3. Calculating with complex voltage (0.6+0.6j)...")
    svpwm.calculate_complex(0.6 + 0.6j)
    print(f"   Phase A: {svpwm.ta:.4f}")
    print(f"   Phase B: {svpwm.tb:.4f}")
    print(f"   Phase C: {svpwm.tc:.4f}")
    print(f"   Sector: {svpwm.sector}")

    # Test 4: Get numpy array output
    print("\n4. Getting numpy array output...")
    outputs = svpwm.get_outputs()
    print(f"   Outputs array: {outputs}")
    print(f"   [ta, tb, tc, sector, status, magnitude, angle]")

    # Test 5: Convenience function
    print("\n5. Testing convenience function svpwm_step...")
    ta, tb, tc, sector = svpwm_step(0.8, 0.0)
    print(f"   Step result: ta={ta:.4f}, tb={tb:.4f}, tc={tc:.4f}, sector={sector}")

    # Test 6: Complex convenience function
    print("\n6. Testing svpwm_step_complex...")
    from fs_electrical_machines.svpwm_wrapper import svpwm_step_complex

    ta, tb, tc, sector = svpwm_step_complex(0.6 + 0.6j)
    print(f"   Complex step: ta={ta:.4f}, tb={tb:.4f}, tc={tc:.4f}, sector={sector}")

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)

except ImportError as e:
    print(f"✗ Failed to import svpwm_wrapper: {e}")
    print("\nMake sure the build completed successfully.")
except Exception as e:
    print(f"✗ Test failed: {e}")