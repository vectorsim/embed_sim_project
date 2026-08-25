"""
embedsim_control_block.py  —  C Backend Only
"""

import os
import sys
from pathlib import Path

import numpy as np

from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_C_SRC = _HERE / "c_src"

for _p in (str(_HERE), str(_C_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ===========================================================================
# Import from pyx wrapper
# ===========================================================================

from embedsim_control_wrapper import (
    control_init,
    control_step as c_control_step,
    get_motor_state,
)

# ===========================================================================
# Constants
# ===========================================================================

SIM_CTRL_OPEN_LOOP = 0
SIM_CTRL_DFC = 1

# ===========================================================================
# Debug flags – Toggle these independently
# ===========================================================================

# Debug input/output: prints control inputs and outputs (duties)
DEBUG_IO = False  # <--- Toggle: True = print I/O, False = silent

# Debug state: prints full motor state from C
DEBUG_STATE = False   # <--- Toggle: True = print state, False = silent

# Debug print interval (seconds)
DEBUG_INTERVAL = 0.3 # Print every 0.1 seconds


# ===========================================================================
# EmbedSimControlBlock - C Backend
# ===========================================================================

class EmbedSimControlBlock(VectorBlock):
    NUM_INPUTS = 1
    OUTPUT_SIZE = 4

    def __init__(self, name="ctrl", dt_s=50e-6, ctrl_alg=SIM_CTRL_DFC,
                 vdc_nom=12.0, use_c_backend=True, dtype=None):
        super().__init__(name, use_c_backend=True, dtype=dtype)

        self.dt_s = float(dt_s)
        self.ctrl_alg = int(ctrl_alg)
        self.vdc_nom = float(vdc_nom)
        self.vector_size = 4
        self.output_label = "[duty_u,duty_v,duty_w,valid]"
        self.is_dynamic = False
        self._last_print_t = -1.0   # for rate‑limited prints

        control_init()

        mode_name = "DFC" if ctrl_alg == SIM_CTRL_DFC else "OPEN_LOOP"
        print(f"[Control] C backend '{name}'")
        print(f"[Control]   DT: {dt_s*1e6:.0f} us")
        print(f"[Control]   Mode: {mode_name}")
        print(f"[Control]   Vdc: {vdc_nom:.1f} V")
        print(f"[Control]   Debug I/O: {'ENABLED' if DEBUG_IO else 'DISABLED'}")
        print(f"[Control]   Debug State: {'ENABLED' if DEBUG_STATE else 'DISABLED'}")

    def compute(self, t, dt, input_values=None):
        u = input_values[0].value

        speed_ref_rpm = float(u[0])
        ia = float(u[1])
        ib = float(u[2])
        ic = float(u[3])
        speed_sensor_rpm = float(u[4])
        sample_time = float(u[5])
        position_sensor_rad = float(u[6])
        valid_in = int(u[7])
        vdc = float(u[9])

        # ---- Debug: Print Inputs (if enabled) ----
        if DEBUG_IO and (t - self._last_print_t >= DEBUG_INTERVAL):
            print(f"\n{'='*70}")
            print(f"[Ctrl I/O t={t:.3f}s]")
            print(f"  INPUTS:")
            print(f"    speed_ref     = {speed_ref_rpm:8.1f} RPM")
            print(f"    speed_sensor  = {speed_sensor_rpm:8.1f} RPM")
            print(f"    position      = {position_sensor_rad:8.4f} rad")
            print(f"    ia            = {ia:8.3f} A")
            print(f"    ib            = {ib:8.3f} A")
            print(f"    ic            = {ic:8.3f} A")
            print(f"    vdc           = {vdc:8.2f} V")
            print(f"    valid_in      = {valid_in:8d}")

        # ---- Execute control step ----
        result = c_control_step(
            ia=ia,
            ib=ib,
            ic=ic,
            rotor_position_rad=position_sensor_rad,
            rotor_velocity_rpm=speed_sensor_rpm,
            speed_ref_rpm=speed_ref_rpm,
            vdc=vdc,
            sample_time=sample_time,
            ctrl_alg=self.ctrl_alg,
            valid_in=valid_in,
        )

        # ---- Debug: Print Outputs (if enabled) ----
        if DEBUG_IO and (t - self._last_print_t >= DEBUG_INTERVAL):
            print(f"  OUTPUTS:")
            print(f"    duty_u        = {result['pwm_u']:8.4f}")
            print(f"    duty_v        = {result['pwm_v']:8.4f}")
            print(f"    duty_w        = {result['pwm_w']:8.4f}")
            print(f"    valid_out     = {result['valid_out']:8d}")

        # ---- Debug: Print Motor State (if enabled) ----
        if DEBUG_STATE and (t - self._last_print_t >= DEBUG_INTERVAL):
            try:
                state = get_motor_state()
                if state and state.get('valid', 0):
                    mode = "CLOSED" if state.get('closed_loop', 0) else "OPEN"
                    print(f"  MOTOR STATE:")
                    print(f"    mode          = {mode}")
                    print(f"    speed_rpm     = {state.get('speed_rpm', 0):8.1f} RPM")
                    print(f"    speed_ref_rpm = {state.get('speed_ref_rpm', 0):8.1f} RPM")
                    print(f"    id            = {state.get('id', 0):8.3f} A")
                    print(f"    iq            = {state.get('iq', 0):8.3f} A")
                    print(f"    torque_total  = {state.get('torque_total', 0):8.4f} Nm")
                    print(f"    closed_loop   = {state.get('closed_loop', 0):8d}")
                    print(f"    control_reinit= {state.get('control_reinit', 0):8d}")
                    print(f"    spinning_cnt  = {state.get('spinning_counter', 0):8d}")
                    print(f"    is_spinning   = {state.get('is_spinning', 0):8d}")
                    print(f"    is_stopped    = {state.get('is_stopped', 0):8d}")
                    print(f"    speed_error   = {state.get('speed_error_rpm', 0):8.1f} RPM")
                    print(f"    modulation_idx= {state.get('modulation_index', 0):8.4f}")
                    print(f"    duty_u        = {state.get('duty_u', 0):8.4f}")
                    print(f"    duty_v        = {state.get('duty_v', 0):8.4f}")
                    print(f"    duty_w        = {state.get('duty_w', 0):8.4f}")
            except Exception as e:
                print(f"  ⚠️ Error getting motor state: {e}")

        # ---- Print separator ----
        if DEBUG_IO or DEBUG_STATE:
            if t - self._last_print_t >= DEBUG_INTERVAL:
                print(f"{'='*70}")
                self._last_print_t = t

        # ---- Build output array ----
        output_array = np.array([
            float(result['pwm_u']),
            float(result['pwm_v']),
            float(result['pwm_w']),
            float(result['valid_out']),
        ], dtype=DEFAULT_DTYPE)

        self.output = VectorSignal(output_array, self.name)
        return self.output

    def reset(self):
        super().reset()
        self._last_print_t = -1.0

    def __repr__(self):
        mode_name = "DFC" if self.ctrl_alg == SIM_CTRL_DFC else "OPEN_LOOP"
        return f"EmbedSimControlBlock('{self.name}', dt={self.dt_s*1e6:.0f}us, mode={mode_name})"


__all__ = ["EmbedSimControlBlock", "SIM_CTRL_OPEN_LOOP", "SIM_CTRL_DFC"]