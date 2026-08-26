"""
pmsm_python_plant.py  -  Python PMSM Motor Plant Model
                       USES C TRANSFORMS VIA CYTHON INTERFACE
                       NOW WITH 4th-ORDER RUNGE-KUTTA INTEGRATION
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
    States are integrated using 4th-order Runge-Kutta (RK4).
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
        """Clarke transform - CALLS C IMPLEMENTATION via Cython."""
        return clarke(u, v, w)

    def _inv_clarke(self, alpha: float, beta: float) -> Tuple[float, float, float]:
        """Inverse Clarke transform - CALLS C IMPLEMENTATION via Cython."""
        return inv_clarke(alpha, beta)

    def _park(self, alpha: float, beta: float, theta_elec: float) -> Tuple[float, float]:
        """Park transform - CALLS C IMPLEMENTATION via Cython."""
        return park(alpha, beta, theta_elec)

    def _inv_park(self, vd: float, vq: float, theta_elec: float) -> Tuple[float, float]:
        """Inverse Park transform - CALLS C IMPLEMENTATION via Cython."""
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
    # Derivatives for RK4
    # ==================================================================

    def _derivatives(
        self,
        id_: float,
        iq_: float,
        omega_m_: float,
        theta_m_: float,
        vd: float,
        vq: float,
        T_load: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute time derivatives of the state variables at a given operating point.

        Returns:
            (did_dt, diq_dt, domega_dt, dtheta_dt)
        """
        omega_e = omega_m_ * self.p

        # Electrical derivatives (PMSM dq equations)
        did_dt = (vd - self.R * id_ + omega_e * self.L_q * iq_) / self.L_d
        diq_dt = (vq - self.R * iq_ - omega_e * (self.L_d * id_ + self.lambda_pm)) / self.L_q

        # Electromagnetic torque
        T_em = 1.5 * self.p * self.lambda_pm * iq_

        # Mechanical derivatives
        domega_dt = (T_em - self.B_fric * omega_m_ - T_load) / self.J
        dtheta_dt = omega_m_

        return did_dt, diq_dt, domega_dt, dtheta_dt

    # ==================================================================
    # Main Block Method
    # ==================================================================

    def compute(self, t: float, dt: float, input_values=None) -> VectorSignal:
        """
        Compute one step of the motor model using RK4 integration.

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

        # Ensure balanced 3-phase voltages (sum=0) for Clarke transform
        v_offset = (v_u + v_v + v_w) / 3.0
        v_u -= v_offset
        v_v -= v_offset
        v_w -= v_offset

        # Convert phase voltages to alpha-beta using C Clarke
        valpha, vbeta = self._clarke(v_u, v_v, v_w)

        # Electrical angle (mechanical * pole pairs)
        theta_elec = self.theta_m * self.p

        # Convert alpha-beta voltages to dq using C Park
        vd, vq = self._park(valpha, vbeta, theta_elec)

        # ------------------------------------------------------------
        # RK4 integration of states (id, iq, omega_m, theta_m)
        # ------------------------------------------------------------
        id_curr = self.id
        iq_curr = self.iq
        omega_curr = self.omega_m
        theta_curr = self.theta_m

        # Stage 1
        k1_id, k1_iq, k1_omega, k1_theta = self._derivatives(
            id_curr, iq_curr, omega_curr, theta_curr, vd, vq, T_load
        )

        # Stage 2
        id2 = id_curr + 0.5 * dt * k1_id
        iq2 = iq_curr + 0.5 * dt * k1_iq
        omega2 = omega_curr + 0.5 * dt * k1_omega
        theta2 = theta_curr + 0.5 * dt * k1_theta
        k2_id, k2_iq, k2_omega, k2_theta = self._derivatives(
            id2, iq2, omega2, theta2, vd, vq, T_load
        )

        # Stage 3
        id3 = id_curr + 0.5 * dt * k2_id
        iq3 = iq_curr + 0.5 * dt * k2_iq
        omega3 = omega_curr + 0.5 * dt * k2_omega
        theta3 = theta_curr + 0.5 * dt * k2_theta
        k3_id, k3_iq, k3_omega, k3_theta = self._derivatives(
            id3, iq3, omega3, theta3, vd, vq, T_load
        )

        # Stage 4
        id4 = id_curr + dt * k3_id
        iq4 = iq_curr + dt * k3_iq
        omega4 = omega_curr + dt * k3_omega
        theta4 = theta_curr + dt * k3_theta
        k4_id, k4_iq, k4_omega, k4_theta = self._derivatives(
            id4, iq4, omega4, theta4, vd, vq, T_load
        )

        # Update states with weighted average
        self.id = id_curr + (dt / 6.0) * (k1_id + 2.0*k2_id + 2.0*k3_id + k4_id)
        self.iq = iq_curr + (dt / 6.0) * (k1_iq + 2.0*k2_iq + 2.0*k3_iq + k4_iq)
        self.omega_m = omega_curr + (dt / 6.0) * (k1_omega + 2.0*k2_omega + 2.0*k3_omega + k4_omega)
        self.theta_m = theta_curr + (dt / 6.0) * (k1_theta + 2.0*k2_theta + 2.0*k3_theta + k4_theta)

        # Wrap mechanical angle to [0, 2π)
        self.theta_m = self.theta_m % (2.0 * math.pi)

        # Compute electromagnetic torque from updated iq
        T_em = 1.5 * self.p * self.lambda_pm * self.iq

        # Convert updated dq currents to phase currents using updated theta_elec
        theta_elec_new = self.theta_m * self.p
        ia, ib, ic = self._currents_to_phase(theta_elec_new)

        # Convert speed to RPM
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
        # Note: ia, ib, ic are not stored as states; they are derived.