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
)

# ===========================================================================
# Constants
# ===========================================================================

SIM_CTRL_OPEN_LOOP = 0
SIM_CTRL_DFC = 1

# ===========================================================================
# Debug flag – set to True to enable console prints
# ===========================================================================
DEBUG_CONTROL = False # <--- Toggle this: True = print, False = silent


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
        if DEBUG_CONTROL:
            print("[Control] Debug prints ENABLED (every 0.2 s)")

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

        # ---- Debug print every 0.2 seconds (if enabled) ----
        if DEBUG_CONTROL and (t - self._last_print_t >= 0.2):
            self._last_print_t = t
            print(f"\n[Ctrl t={t:.2f}s]")
            print(f"  pos_rad (mech) = {position_sensor_rad:.4f}")
            print(f"  speed_ref = {speed_ref_rpm:.1f}  speed_sensor = {speed_sensor_rpm:.1f}")
            print(f"  ia={ia:.3f}  ib={ib:.3f}  ic={ic:.3f}  vdc={vdc:.2f}  valid_in={valid_in}")

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

        # ---- Debug print duties (if enabled) ----
        if DEBUG_CONTROL and (t - self._last_print_t <= 0.001):  # same tick
            print(f"  duties -> u={result['pwm_u']:.4f} v={result['pwm_v']:.4f} w={result['pwm_w']:.4f}  valid_out={result['valid_out']}")

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