#!/usr/bin/env python
"""
test_dfc_controller.py - Test the DFC controller wrapper
"""

import numpy as np
import time

try:
    from fs_electrical_machines.dfc_controller_wrapper import DFCControllerWrapper, dfc_step

    print("✅ DFC controller wrapper imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you built the extension with: ./build_dfc_controller.sh")
    exit(1)

print("\n" + "=" * 60)
print("Testing DFC Controller")
print("=" * 60)

# Create controller instance
controller = DFCControllerWrapper(
    v_dc=17.0,
    p_poles=4,
    R_s=0.285,
    L_d=0.0003675,
    L_q=0.0003675,
    lambda_pm=0.0014,
    i_max=3.57,
    dt_s=50e-6,
    kp_speed=0.119,
    kp_id=2.0,
    kp_iq=2.0
)

print("\n✅ Controller created successfully")
print(f"   Type: {type(controller)}")
print(f"   Name: {controller}")

# Test individual input setting
print("\n📝 Testing set_inputs_individual()...")
controller.set_inputs_individual(
    omega_ref_mech=100.0,  # 100 rad/s ~ 955 RPM
    theta_m=0.0,
    ia=0.0,
    ib=0.0,
    ic=0.0
)
print("   ✅ set_inputs_individual() works")

# Test array input setting
print("\n📝 Testing set_inputs()...")
inputs = np.array([100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
controller.set_inputs(inputs)
print("   ✅ set_inputs() works")

# Test compute
print("\n📝 Testing compute()...")
dt = 50e-6  # 20 kHz
controller.compute(dt)
print("   ✅ compute() works")

# Test get_outputs
print("\n📝 Testing get_outputs()...")
v_alpha, v_beta = controller.get_outputs()
print(f"   v_alpha = {v_alpha:.6f} V")
print(f"   v_beta  = {v_beta:.6f} V")

# Test get_diagnostics
print("\n📝 Testing get_diagnostics()...")
diag = controller.get_diagnostics()
print(f"   Diagnostics: [speed_est, iq_ref, iq_meas, id_meas, speed_ref_rpm, alpha, omega_e]")
print(f"   {diag}")

# Test gain setting
print("\n📝 Testing set_gains()...")
controller.set_gains(kp_speed=0.150, kp_id=2.5, kp_iq=2.5)
print("   ✅ set_gains() works")

# Test reset
print("\n📝 Testing reset()...")
controller.reset()
print("   ✅ reset() works")

print("\n" + "=" * 60)
print("🎉 All tests passed! DFC controller is ready to use.")
print("=" * 60)

# Simple simulation test
print("\n📊 Running short simulation...")
controller.reset()
t = 0.0
dt = 50e-6
duration = 0.01  # 10 ms
steps = int(duration / dt)

v_alpha_log = []
v_beta_log = []
speed_log = []

for i in range(steps):
    # Simulate motor at standstill initially
    theta_m = 0.0  # No rotation
    ia = ib = ic = 0.0  # No current

    controller.set_inputs_individual(100.0, theta_m, ia, ib, ic)
    controller.compute(dt)

    v_alpha, v_beta = controller.get_outputs()
    v_alpha_log.append(v_alpha)
    v_beta_log.append(v_beta)

    # Get estimated speed
    diag = controller.get_diagnostics()
    speed_log.append(diag[0])  # speed_est

    t += dt

print(f"   Simulated {steps} steps at {1 / dt / 1000:.0f} kHz")
print(f"   Final v_alpha = {v_alpha_log[-1]:.6f} V")
print(f"   Final v_beta  = {v_beta_log[-1]:.6f} V")
print(f"   Speed estimate = {speed_log[-1]:.2f} rad/s")

# Plot results if matplotlib is available
try:
    import matplotlib.pyplot as plt

    time_array = np.arange(0, duration, dt)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(time_array, v_alpha_log, label='v_alpha')
    axes[0].plot(time_array, v_beta_log, label='v_beta')
    axes[0].set_ylabel('Voltage [V]')
    axes[0].set_xlabel('Time [s]')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('DFC Output Voltages')

    axes[1].plot(time_array, speed_log)
    axes[1].set_ylabel('Speed [rad/s]')
    axes[1].set_xlabel('Time [s]')
    axes[1].grid(True)
    axes[1].set_title('Estimated Speed')

    plt.tight_layout()
    plt.savefig('dfc_test_results.png', dpi=150)
    print(f"   📊 Plot saved to dfc_test_results.png")

except ImportError:
    print("   ⚠️ matplotlib not available - skipping plots")

print("\n✅ DFC controller is ready to use in your simulations!")