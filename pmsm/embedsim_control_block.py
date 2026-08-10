"""
embedsim_control_block.py  —  C Backend Only
"""

import os
import sys
import math
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
    clarke, park, inv_park, inv_clarke,
    svm_calc_dq,
    control_init,
    control_step as c_control_step,
)

# ===========================================================================
# Constants
# ===========================================================================

VALID_FLAG = 1
INVALID_FLAG = 0
SIM_CTRL_OPEN_LOOP = 0
SIM_CTRL_DFC = 1
HARDWARE_VDC = 12.0


# ===========================================================================
# EmbedSimControlBlock - C Backend Only
# ===========================================================================

class EmbedSimControlBlock(VectorBlock):
    NUM_INPUTS = 1
    OUTPUT_SIZE = 7

    INPUT_NAMES = ["speed_ref_rpm", "ia", "ib", "ic", "position_rad", "speed_rpm", "vdc"]
    INPUT_KEEP = [0, 1, 2, 3, 4, 5, 6]

    OUTPUT_NAMES = ["ta", "tb", "tc", "speed_est_rpm", "position_est_rad", "sector", "valid"]
    OUTPUT_KEEP = [0, 1, 2, 3, 4, 5, 6]

    def __init__(self, name="ctrl", dt_s=50e-6, ctrl_alg=SIM_CTRL_OPEN_LOOP,
                 vdc_nom=17.0, use_c_backend=True, dtype=None):
        super().__init__(name, use_c_backend=True, dtype=dtype)

        self.dt_s = float(dt_s)
        self.ctrl_alg = int(ctrl_alg)
        self.vdc_nom = float(vdc_nom)
        self.vector_size = 7
        self.output_label = "[ta,tb,tc,speed_est,pos_est,sector,valid]"
        self.is_dynamic = False

        # Call control_init ONCE
        control_init()

        print(f"[Control] C backend '{name}'")
        print(f"[Control]   DT: {dt_s*1e6:.0f} us")
        print(f"[Control]   Mode: {'DFC' if ctrl_alg else 'OPEN_LOOP'}")
        print(f"[Control]   Vdc: {vdc_nom:.1f} V")

    def compute(self, t, dt, input_values=None):
        """C backend only"""
        safe = np.zeros(7, dtype=np.float32)
        safe[0] = safe[1] = safe[2] = 0.5

        if not input_values or input_values[0] is None:
            self.output = VectorSignal(safe.copy(), self.name)
            return self.output

        u = input_values[0].value
        if len(u) < 7:
            self.output = VectorSignal(safe.copy(), self.name)
            return self.output

        vdc = HARDWARE_VDC

        result = c_control_step(
            speed_ref_rpm=float(u[0]),
            ia=float(u[1]),
            ib=float(u[2]),
            ic=float(u[3]),
            position_rad=float(u[8]),
            speed_rpm=float(u[7]),
            vdc=vdc,
            dt=float(dt),
            ctrl_alg=self.ctrl_alg,
            valid=VALID_FLAG,
        )

        self.output = VectorSignal(
            np.array([
                result['ta'], result['tb'], result['tc'],
                result['speed_est'], result['position_est'],
                float(result['sector']), float(result['valid']),
            ], dtype=DEFAULT_DTYPE),
            self.name
        )
        return self.output

    def reset(self):
        super().reset()

    def __repr__(self):
        return f"EmbedSimControlBlock('{self.name}', dt={self.dt_s*1e6:.0f}us)"


__all__ = ["EmbedSimControlBlock"]