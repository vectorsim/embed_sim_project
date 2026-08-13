"""
embedsim_generic_control.py
===========================

Generic control block that can run in either C backend mode (optimized)
or Python mode (user-editable). Provides transformation utilities for
FOC but no default control logic.

Usage:
    # Python mode - edit compute_py() freely
    ctrl = GenericControlBlock(name="ctrl", use_c_backend=False)

    # C backend mode - faster but requires compilation
    ctrl = GenericControlBlock(name="ctrl", use_c_backend=True)
"""

import math
import sys
from pathlib import Path

import numpy as np
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE

# Try to import the base class
try:
    from embedsim_control_block import EmbedSimControlBlock, SIM_CTRL_OPEN_LOOP, SIM_CTRL_DFC
    from embedsim_control_wrapper import clarke, park, inv_park, inv_clarke
except ImportError:
    from embedsim.embedsim_control_block import EmbedSimControlBlock, SIM_CTRL_OPEN_LOOP, SIM_CTRL_DFC
    from embedsim.embedsim_control_wrapper import clarke, park, inv_park, inv_clarke


class GenericControlBlock(EmbedSimControlBlock):
    """
    Generic control block with Python and C backend support.

    Provides FOC transformation utilities but NO default control logic.
    Users must implement their own control algorithm in compute_py().

    Transformations available:
        - clarke(ia, ib, ic) -> (alpha, beta)
        - park(alpha, beta, theta) -> (id, iq)
        - inv_park(id, iq, theta) -> (alpha, beta)
        - inv_clarke(alpha, beta) -> (va, vb, vc)
    """

    def __init__(self, name="ctrl", dt_s=50e-6, ctrl_alg=SIM_CTRL_DFC,
                 vdc_nom=12.0, use_c_backend=False, dtype=None, **kwargs):
        """
        Initialize the generic control block.

        Args:
            name: Block name
            dt_s: Control sample time in seconds
            ctrl_alg: Control algorithm (SIM_CTRL_DFC or SIM_CTRL_OPEN_LOOP)
            vdc_nom: Nominal DC bus voltage
            use_c_backend: False = Python mode, True = C backend
            dtype: Numeric data type
        """
        # Initialize the parent class
        super().__init__(name=name, dt_s=dt_s, ctrl_alg=ctrl_alg,
                         vdc_nom=vdc_nom, use_c_backend=use_c_backend, dtype=dtype)

        # ---- Python controller states (users can add their own) ----
        self._last_py_print = -1.0
        self._print_counter = 0

    # =====================================================================
    # TRANSFORMATION METHODS (using C wrapper implementations)
    # =====================================================================

    def clarke(self, ia, ib, ic):
        """
        Clarke transform: 3-phase -> Alpha-Beta stationary frame.

        Args:
            ia: Phase A current [A]
            ib: Phase B current [A]
            ic: Phase C current [A]

        Returns:
            tuple: (alpha, beta) [A]
        """
        return clarke(ia, ib, ic)

    def park(self, alpha, beta, theta):
        """
        Park transform: Alpha-Beta stationary -> D-Q rotating frame.

        Args:
            alpha: Alpha-axis [A or V]
            beta: Beta-axis [A or V]
            theta: Rotor electrical angle [rad]

        Returns:
            tuple: (d, q) [A or V]
        """
        return park(alpha, beta, theta)

    def inv_park(self, d, q, theta):
        """
        Inverse Park transform: D-Q rotating -> Alpha-Beta stationary.

        Args:
            d: D-axis [A or V]
            q: Q-axis [A or V]
            theta: Rotor electrical angle [rad]

        Returns:
            tuple: (alpha, beta) [A or V]
        """
        return inv_park(d, q, theta)

    def inv_clarke(self, alpha, beta):
        """
        Inverse Clarke transform: Alpha-Beta -> 3-phase.

        Args:
            alpha: Alpha-axis [A or V]
            beta: Beta-axis [A or V]

        Returns:
            tuple: (a, b, c) [A or V]
        """
        return inv_clarke(alpha, beta)

    # =====================================================================
    # MAIN COMPUTE METHODS
    # =====================================================================

    def compute(self, t, dt, input_values=None):
        """
        Route to either C backend or Python implementation.
        """
        # ---- Mode 1: C backend ----
        if self.use_c_backend:
            return super().compute(t, dt, input_values)

        # ---- Mode 2: Python implementation ----
        return self.compute_py(t, dt, input_values)

    def compute_py(self, t, dt, input_values=None):
        """
        YOUR CONTROL ALGORITHM GOES HERE - EDIT FREELY!

        This is a template with no default control logic.
        Implement your own speed control, FOC, or any other algorithm.

        Input vector (10 elements):
            [0] speed_ref_rpm      - Speed reference in RPM
            [1] ia                 - Phase A current [A]
            [2] ib                 - Phase B current [A]
            [3] ic                 - Phase C current [A]
            [4] speed_sensor_rpm   - Measured speed in RPM
            [5] sample_time        - Sample time [s]
            [6] position_sensor_rad - Rotor position [rad mechanical]
            [7] valid_in           - Input validity flag
            [8] reserved
            [9] vdc                - DC bus voltage [V]

        Output vector (4 elements):
            [0] duty_u  - Phase U duty cycle [0..1]
            [1] duty_v  - Phase V duty cycle [0..1]
            [2] duty_w  - Phase W duty cycle [0..1]
            [3] valid   - Output validity flag
        """
        u = input_values[0].value

        # Unpack inputs
        speed_ref_rpm = float(u[0])
        ia = float(u[1])
        ib = float(u[2])
        ic = float(u[3])
        speed_sensor_rpm = float(u[4])
        sample_time = float(u[5])
        position_sensor_rad = float(u[6])
        valid_in = int(u[7])
        vdc = float(u[9])

        # Debug prints (rate-limited to 0.2 s)
        if (t - self._last_py_print) >= 0.2:
            self._last_py_print = t
            self._print_counter += 1
            print(f"\n[GenericControl t={t:.2f}s]")
            print(f"  speed_ref={speed_ref_rpm:.1f}  speed_sensor={speed_sensor_rpm:.1f}")
            print(f"  ia={ia:.3f}  ib={ib:.3f}  ic={ic:.3f}  vdc={vdc:.2f}")

        # ================================================================
        #  !!! YOUR CONTROL ALGORITHM STARTS HERE !!!
        #  Edit freely - this is just a template example
        # ================================================================

        # --- EXAMPLE: Simple open-loop control (replace with your own) ---
        # Convert mechanical position to electrical angle (for 4-pole motor)
        pole_pairs = 2
        theta_elec = position_sensor_rad * pole_pairs

        # Get dq currents (if you need FOC)
        # ialpha, ibeta = self.clarke(ia, ib, ic)
        # id, iq = self.park(ialpha, ibeta, theta_elec)

        # Your control logic here...
        # Example: simple speed controller (uncomment and modify)
        # error = speed_ref_rpm - speed_sensor_rpm
        # duty = np.clip(error * 0.01, 0.0, 1.0)

        # For now, just output fixed duty cycle
        duty_u = 0.5
        duty_v = 0.5
        duty_w = 0.5
        valid_out = 1

        # ================================================================
        #  !!! YOUR CONTROL ALGORITHM ENDS HERE !!!
        # ================================================================

        # Print duties (same tick as debug print)
        if (t - self._last_py_print) <= 0.001:
            print(f"  duties -> u={duty_u:.4f} v={duty_v:.4f} w={duty_w:.4f}")

        out = np.array([duty_u, duty_v, duty_w, float(valid_out)], dtype=DEFAULT_DTYPE)
        self.output = VectorSignal(out, self.name)
        return self.output

    def reset(self):
        """Reset the controller state."""
        super().reset()
        self._last_py_print = -1.0
        self._print_counter = 0

    def __repr__(self):
        mode = "Python" if not self.use_c_backend else "C"
        return f"GenericControlBlock('{self.name}', mode={mode})"


__all__ = ["GenericControlBlock", "SIM_CTRL_OPEN_LOOP", "SIM_CTRL_DFC"]