"""
embedsim_connections.py
========================

Utility connection blocks for PMSM simulation.

Classes:
    CtrlPacker:       Packs speed reference and motor feedback into control input vector
    LoadAdapter:      Adapts duty cycles to motor load inputs
    MotorVectorDelay: Vector delay specifically for motor feedback signals
"""

import math
import numpy as np
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.simulation_engine import VectorDelay


# =============================================================================
# CtrlPacker
# =============================================================================

class CtrlPacker(VectorBlock):
    """
    Packs speed reference and motor feedback into a 10-element control vector.

    Inputs:
        port 0: speed reference (1 element)
        port 1: motor feedback (8 elements: rpm, ia, ib, ic, theta_m, Tem, id, iq)

    Output:
        10-element array: [rpm_ref, ia, ib, ic, rpm_sensor, dt, theta_m, valid, 0, Vdc]
    """
    TOPO_CATEGORY = "utility"
    C_CODEGEN_EXCLUDE = True
    NUM_INPUTS = 2
    output_label = "ctrl_inputs[10]"

    def __init__(self, name="ctrl_packer", vdc=12.0, valid_flag=1):
        super().__init__(name)
        self.vector_size = 10
        self._vdc = float(vdc)
        self._valid_flag = int(valid_flag)

    def compute_py(self, t, dt, input_values=None):
        speed_ref_rpm = float(input_values[0].value[0])
        motor_vals = input_values[1].value

        speed_sensor_rpm = float(motor_vals[0])
        ia = float(motor_vals[1])
        ib = float(motor_vals[2])
        ic = float(motor_vals[3])
        theta_m = float(motor_vals[4]) % (2.0 * math.pi)

        output_array = np.array([
            speed_ref_rpm,
            ia,
            ib,
            ic,
            speed_sensor_rpm,
            dt,
            theta_m,
            self._valid_flag,
            0.0,
            self._vdc,
        ], dtype=DEFAULT_DTYPE)

        self.output = VectorSignal(output_array, self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# LoadAdapter
# =============================================================================

class LoadAdapter(VectorBlock):
    """
    Adapts duty cycle outputs to motor load inputs.

    Input:
        [duty_u, duty_v, duty_w, valid]

    Output:
        [ta, tb, tc, Vdc, Tload]
    """
    TOPO_CATEGORY = "utility"
    C_CODEGEN_EXCLUDE = True
    output_label = "[ta,tb,tc,Vdc,Tload]"

    def __init__(self, name="load_adapter", vdc=12.0, tload=0.0):
        super().__init__(name)
        self.vector_size = 5
        self._vdc = float(vdc)
        self._tload = float(tload)

    def compute_py(self, t, dt, input_values=None):
        v = input_values[0].value
        ta = float(v[0])
        tb = float(v[1])
        tc = float(v[2])

        self.output = VectorSignal(
            np.array([ta, tb, tc, self._vdc, self._tload], dtype=DEFAULT_DTYPE),
            self.name
        )
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# MotorVectorDelay
# =============================================================================

class MotorVectorDelay(VectorDelay):
    """
    A VectorDelay specifically for motor feedback signals.
    Stores and outputs a full vector of motor states.

    Inherits from the built-in VectorDelay, so it is automatically
    a loop breaker (is_loop_breaker = True).

    Expected format: [rpm, ia, ib, ic, theta_m, Tem, id, iq]
    """
    def __init__(self, name="motor_delay", initial=None, vector_size=8):
        """
        Initialize the motor vector delay.

        Args:
            name: Block name
            initial: Initial state vector (list or array). If None, zeros.
            vector_size: Size of the vector. Default: 8 (motor outputs)
        """
        if initial is None:
            initial = [0.0] * vector_size
        super().__init__(name, initial=initial[0])
        self._state = np.array(initial, dtype=DEFAULT_DTYPE)
        self.vector_size = len(self._state)
        self.output = VectorSignal(self._state.copy(), self.name)

    def compute(self, t, dt, input_values=None):
        # 1. Output the currently stored state
        if self._state is not None:
            out = self._state.copy()
        else:
            # Fallback: zeros if no state (should not happen after init)
            out = np.zeros(self.vector_size, dtype=DEFAULT_DTYPE)

        # 2. Update storage with the new input for the next step
        if input_values and len(input_values) > 0:
            new_val = input_values[0].value
            if len(new_val) == self.vector_size:
                self._state = new_val.copy()
            else:
                # handle size mismatch (e.g., raise ValueError)
                pass

        self.output = VectorSignal(out, self.name)
        return self.output

    def reset(self):
        """Reset the delay state to zeros."""
        self._state = np.zeros(self.vector_size, dtype=DEFAULT_DTYPE)
        self.output = VectorSignal(self._state.copy(), self.name)


class SignalPrinter(VectorBlock):
    """
    A debug block that prints the input signal with descriptive labels and passes it through unchanged.
    """
    C_CODEGEN_EXCLUDE = True

    def __init__(self, name="signal_printer", fields=None, print_prefix="", every_n=1):
        super().__init__(name)
        self.fields = fields or []
        self.print_prefix = print_prefix
        self.every_n = every_n
        self._counter = 0

    def compute_py(self, t, dt, input_values=None):
        self._counter += 1
        val = input_values[0].value

        if self._counter % self.every_n == 0:
            parts = []
            if self.fields:
                for i, f in enumerate(self.fields):
                    parts.append(f"{f}={val[i]:.6f}" if i < len(val) else f"{f}=N/A")
            else:
                parts = [f"[{i}]={v:.6f}" for i, v in enumerate(val)]

            print(f"[{self.name}] t={t:.6f} | {self.print_prefix}{', '.join(parts)}")

        self.output = VectorSignal(val.copy(), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# Module Metadata
# =============================================================================

__all__ = [
    'CtrlPacker',
    'LoadAdapter',
    'MotorVectorDelay',
    'SignalPrinter',
]

__version__ = '1.0.0'