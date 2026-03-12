#!/usr/bin/env python3
"""
Test script for PI Buck Controller wrapper
"""

import numpy as np
import matplotlib.pyplot as plt
import pi_buck_wrapper


def test_pi_buck_controller():
    """Test the PI Buck controller with a step response"""

    # Create controller instance
    ctrl = pi_buck_wrapper.PI_BuckWrapper()

    # Set parameters for a typical buck converter
    # L=100µH, C=100µF, Vin=24V, Vout=12V, fsw=100kHz
    ctrl.set_params(
        Kp=0.1,  # Proportional gain
        Ki=5.0,  # Integral gain
        duty_max=0.95,  # Max duty cycle
        duty_min=0.05,  # Min duty cycle
        Ts=1e-4  # 10 kHz control loop
    )

    # Simulation parameters
    dt = 1e-4  # 100 µs time step
    t_end = 0.01  # 10 ms simulation
    t = np.arange(0, t_end, dt)

    # Reference step: 0V -> 12V at t=2ms
    V_ref = np.zeros_like(t)
    V_ref[int(0.002 / dt):] = 12.0

    # Initialize
    V_meas = 0.0
    duty_history = []
    V_meas_history = []
    integrator_history = []

    # Simulation loop
    for k, ref in enumerate(V_ref):
        # Compute controller output
        duty = ctrl.compute(ref, V_meas, dt)

        # Simulate buck converter (simplified model)
        # For a real test, you'd connect to your BuckConverterBlock
        if duty > 0:
            # Very simple model: Vout = duty * Vin * (1 - exp(-t/RC))
            # This is just for testing
            V_meas = V_meas + dt * (duty * 24.0 - V_meas) * 1000

        # Store data
        duty_history.append(duty)
        V_meas_history.append(V_meas)
        integrator_history.append(ctrl.integrator)

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    ax1.plot(t * 1000, V_ref, 'r--', label='Reference')
    ax1.plot(t * 1000, V_meas_history, 'b-', label='Measured')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('PI Buck Controller Step Response')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t * 1000, duty_history, 'g-')
    ax2.set_ylabel('Duty Cycle')
    ax2.set_xlabel('Time (ms)')
    ax2.grid(True)

    ax3.plot(t * 1000, integrator_history, 'm-')
    ax3.set_ylabel('Integrator')
    ax3.set_xlabel('Time (ms)')
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('pi_buck_test.png', dpi=150)
    plt.show()

    print(f"Final voltage: {V_meas:.2f} V")
    print(f"Final duty: {duty:.3f}")
    print(f"Final integrator: {ctrl.integrator:.3f}")

    return ctrl


if __name__ == "__main__":
    ctrl = test_pi_buck_controller()

    # Test parameter getter
    print("\nController parameters:")
    for key, value in ctrl.params.items():
        print(f"  {key}: {value}")