"""
pmsm_python_plant.py  -  Python PMSM Motor Plant Model
                       USES C TRANSFORMS VIA CYTHON INTERFACE
"""

from __future__ import annotations
import sys

from pathlib import Path
import math
import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_C_SRC = _HERE / "c_src"

for _p in (str(_HERE), str(_C_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)



from embedsim.core_blocks import (
    VectorSignal,
    DEFAULT_DTYPE,
    VectorBlock,
)

# Import C transform functions from the Cython wrapper
from embedsim_control_wrapper import (
    clarke,
    inv_clarke,
    park,
    inv_park,
)


class PMSM_Python_Plant(VectorBlock):
    """
    PMSM Motor Plant Model - USES C TRANSFORMS VIA CYTHON.

    All transforms call the C implementation directly.
    """

    def __init__(
        self,
        name: str = "motor",
        R: float = 0.19,
        L_d: float = 0.125e-3,
        L_q: float = 0.125e-3,
        lambda_pm: float = 0.0014,
        J: float = 2.4e-6,
        B_fric: float = 1.0e-6,
        p: float = 4.0,
        v_dc: float = 12.0,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.R = R
        self.L_d = L_d
        self.L_q = L_q
        self.lambda_pm = lambda_pm
        self.J = J
        self.B_fric = B_fric
        self.p = p
        self.v_dc = v_dc

        # State variables
        self.omega_m = 0.0      # Mechanical speed [rad/s]
        self.theta_m = 0.0      # Mechanical position [rad]
        self.id = 0.0           # d-axis current [A]
        self.iq = 0.0           # q-axis current [A]

        self.vector_size = 8

        # Constants
        self.SQRT3 = math.sqrt(3.0)
        self.INV_SQRT3 = 1.0 / self.SQRT3
        self.TWO_INV_SQRT3 = 2.0 / self.SQRT3
        self.HALF = 0.5
        self.HALF_SQRT3 = 0.5 * self.SQRT3

        print(f"[PMSM_Python_Plant] '{name}' R={R} Ld={L_d} Lq={L_q} "
              f"lpm={lambda_pm} J={J} B={B_fric} p={p} Vdc={v_dc}")

    # ==================================================================
    # Coordinate Transforms - CALL C IMPLEMENTATION
    # ==================================================================

    def _clarke(self, u: float, v: float, w: float) -> Tuple[float, float]:
        """
        Clarke transform - CALLS C IMPLEMENTATION via Cython.
        """
        return clarke(u, v, w)

    def _inv_clarke(self, alpha: float, beta: float) -> Tuple[float, float, float]:
        """
        Inverse Clarke transform - CALLS C IMPLEMENTATION via Cython.
        """
        return inv_clarke(alpha, beta)

    def _park(self, alpha: float, beta: float, theta_elec: float) -> Tuple[float, float]:
        """
        Park transform - CALLS C IMPLEMENTATION via Cython.
        """
        return park(alpha, beta, theta_elec)

    def _inv_park(self, vd: float, vq: float, theta_elec: float) -> Tuple[float, float]:
        """
        Inverse Park transform - CALLS C IMPLEMENTATION via Cython.
        """
        return inv_park(vd, vq, theta_elec)

    def _currents_to_phase(self, theta_elec: float) -> Tuple[float, float, float]:
        """
        Convert dq currents to phase currents.
        Uses inverse Park + inverse Clarke - CALLS C.
        """
        # Inverse Park (dq -> alpha-beta) - CALLS C
        ialpha, ibeta = self._inv_park(self.id, self.iq, theta_elec)

        # Inverse Clarke (alpha-beta -> abc) - CALLS C
        ia, ib, ic = self._inv_clarke(ialpha, ibeta)

        return ia, ib, ic

    # ==================================================================
    # Motor Dynamics
    # ==================================================================

    def _compute_electrical(self, vd: float, vq: float, omega_e: float, dt: float):
        """
        Compute electrical dynamics.
        """
        did_dt = (vd - self.R * self.id + omega_e * self.L_q * self.iq) / self.L_d
        diq_dt = (vq - self.R * self.iq - omega_e * (self.L_d * self.id + self.lambda_pm)) / self.L_q

        self.id += did_dt * dt
        self.iq += diq_dt * dt

    # ==================================================================
    # Main Block Method
    # ==================================================================

    def compute(self, t: float, dt: float, input_values=None) -> VectorSignal:
        """
        Compute one step of the motor model.

        Input vector (from controller):
        [0] duty_u
        [1] duty_v
        [2] duty_w
        [3] vdc
        [4] T_load (optional)

        Output vector:
        [0] omega_m   [RPM]
        [1] ia        [A]
        [2] ib        [A]
        [3] ic        [A]
        [4] theta_m   [rad]
        [5] T_em      [N.m]
        [6] id        [A]
        [7] iq        [A]
        """
        if input_values is None:
            return self.output

        u = input_values[0].value

        duty_u = float(u[0])
        duty_v = float(u[1])
        duty_w = float(u[2])
        vdc = float(u[3])
        T_load = float(u[4]) if len(u) > 4 else 0.0

        # Limit duty cycles
        duty_u = max(0.0, min(1.0, duty_u))
        duty_v = max(0.0, min(1.0, duty_v))
        duty_w = max(0.0, min(1.0, duty_w))

        # Convert duty cycles to phase voltages
        v_u = (duty_u - 0.5) * vdc
        v_v = (duty_v - 0.5) * vdc
        v_w = (duty_w - 0.5) * vdc

        # --- CRITICAL: Ensure balanced 3-phase voltages ---
        # The Clarke transform assumes u + v + w = 0
        v_offset = (v_u + v_v + v_w) / 3.0
        v_u -= v_offset
        v_v -= v_offset
        v_w -= v_offset

        # --- Convert to alpha-beta using C Clarke ---
        valpha, vbeta = self._clarke(v_u, v_v, v_w)

        # --- Electrical angle ---
        theta_elec = self.theta_m * self.p

        # --- Convert to dq using C Park ---
        vd, vq = self._park(valpha, vbeta, theta_elec)

        # --- Electrical speed ---
        omega_e = self.omega_m * self.p

        # --- Update electrical dynamics ---
        self._compute_electrical(vd, vq, omega_e, dt)

        # --- Compute electromagnetic torque ---
        T_em = 1.5 * self.p * self.lambda_pm * self.iq

        # --- Update mechanical dynamics ---
        domega_dt = (T_em - self.B_fric * self.omega_m - T_load) / self.J
        self.omega_m += domega_dt * dt
        self.theta_m += self.omega_m * dt
        self.theta_m = self.theta_m % (2.0 * math.pi)

        # --- Convert currents to phase currents using C transforms ---
        ia, ib, ic = self._currents_to_phase(theta_elec)

        # --- Convert speed to RPM ---
        rpm = self.omega_m * 60.0 / (2.0 * math.pi)

        out = np.array([
            float(rpm),           # 0: Speed [RPM]
            float(ia),            # 1: Phase A current [A]
            float(ib),            # 2: Phase B current [A]
            float(ic),            # 3: Phase C current [A]
            float(self.theta_m),  # 4: Mechanical position [rad]
            float(T_em),          # 5: Electromagnetic torque [N.m]
            float(self.id),       # 6: d-axis current [A]
            float(self.iq),       # 7: q-axis current [A]
        ], dtype=DEFAULT_DTYPE)

        self.output = VectorSignal(out, self.name)
        return self.output

    def reset(self):
        """Reset motor state."""
        super().reset()
        self.omega_m = 0.0
        self.theta_m = 0.0
        self.id = 0.0
        self.iq = 0.0
        self.ia = 0.0
        self.ib = 0.0
        self.ic = 0.0