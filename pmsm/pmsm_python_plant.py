# pmsm_python_plant.py
"""
Pure-Python PMSM Plant Block for EmbedSim
==========================================
Textbook dq-frame PMSM.  No FMU, no DASSL, no surprises.

EmbedSim calls compute(t, dt, inputs) every step.  This block owns its
four state variables [i_d, i_q, omega_m, theta_e] and advances them
using RK4 internally — 4th-order accuracy at the 50 us step rate.

All Clarke / Park / InvPark / InvClarke calculations are delegated to the
canonical C functions in embed_sim_coordinate_transform.c, surfaced by the
compiled dfc_controller_wrapper (clarke / park / inv_park / inv_clarke).
This is the SAME transform code that runs on the AURIX target — there is no
parallel Python implementation and no inline transform math in this file.

dq voltage equations (Krishnan, "PMSM and BLDC Motor Drives", Ch. 4):
    L_d * di_d/dt = v_d - R*i_d + omega_e*L_q*i_q
    L_q * di_q/dt = v_q - R*i_q - omega_e*(L_d*i_d + lambda_pm)

Torque:
    T_em = 1.5 * p * (lambda_pm*i_q + (L_d - L_q)*i_d*i_q)

Mechanical:
    J * domega_m/dt = T_em - B*omega_m - T_load
    dtheta_e/dt     = omega_e = p * omega_m

Block interface
---------------
Input bus [0] : [ta, tb, tc, v_dc, T_load]   (from SVPWM / cg_end)
Output bus    : [rpm, ia, ib, ic, theta_m, T_em, id, iq]
                  [0]  [1][2][3]    [4]     [5]  [6][7]
"""

import math
import sys
from pathlib import Path

import numpy as np
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE

# Frame transforms come from the canonical C implementation
# (embed_sim_coordinate_transform.c), surfaced by the compiled DFC wrapper.
# This is the SAME code that runs on the AURIX — no parallel Python mirror.
_C_SRC = Path(__file__).resolve().parent / "c_src"
if str(_C_SRC) not in sys.path:
    sys.path.insert(0, str(_C_SRC))
from dfc_controller_wrapper import clarke, park, inv_park, inv_clarke


class PMSM_Python_Plant(VectorBlock):
    """
    Pure-Python PMSM plant with RK4 internal integration.

    Parameters
    ----------
    R         : Stator resistance [Ohm]
    L_d, L_q  : dq inductances [H]
    lambda_pm : PM flux linkage [Wb]
    J         : Rotor inertia [kg.m2]
    B_fric    : Viscous friction coefficient [N.m.s/rad]
    p         : Pole pairs (integer)
    v_dc      : Nominal DC bus voltage [V]
    """

    TOPO_CATEGORY     = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[rpm,ia,ib,ic,theta_m,Tem,id,iq]"

    def __init__(self, name: str = "pmsm",
                 R: float         = 0.19,
                 L_d: float       = 0.125e-3,
                 L_q: float       = 0.125e-3,
                 lambda_pm: float = 0.0014,
                 J: float         = 2.4e-6,
                 B_fric: float    = 1e-6,
                 p: float         = 4.0,
                 v_dc: float      = 17.0,
                 **kwargs):
        super().__init__(name, **kwargs)

        self.R         = float(R)
        self.L_d       = float(L_d)
        self.L_q       = float(L_q)
        self.lambda_pm = float(lambda_pm)
        self.J         = float(J)
        self.B_fric    = float(B_fric)
        self.p         = int(p)
        self._v_dc_nom = float(v_dc)

        # State vector: [i_d, i_q, omega_m, theta_e]
        # Using float64 for RK4 precision
        self._x = np.zeros(4, dtype=np.float64)

        # Latched inputs (zero-order hold between compute calls)
        self._ta    = 0.5
        self._tb    = 0.5
        self._tc    = 0.5
        self._v_dc  = float(v_dc)
        self._tload = 0.0

        # Diagnostics
        self._t_last_print = -1.0
        self._nprint       = 0

        # ── Frame transforms — canonical C via the DFC wrapper ───────────────
        # duties -> Clarke -> Park -> (vd,vq)     [in _vdq_from_duties]
        # (id,iq) -> InvPark -> InvClarke -> abc  [in _abc_from_idiq]
        # No sub-blocks and no inline math: the module-level clarke/park/
        # inv_park/inv_clarke functions call embed_sim_coordinate_transform.c
        # directly (Transform_Init() runs once when the wrapper is imported).

        print(f"[PMSM_Python_Plant] '{name}'  "
              f"R={R} Ld={L_d} Lq={L_q} lpm={lambda_pm} "
              f"J={J} B={B_fric} p={p} Vdc={v_dc}  "
              f"[transforms -> embed_sim_coordinate_transform.c]")

    # ------------------------------------------------------------------ reset
    def reset(self):
        super().reset()
        self._x[:]  = 0.0
        self._ta    = 0.5
        self._tb    = 0.5
        self._tc    = 0.5
        self._v_dc  = self._v_dc_nom
        self._tload = 0.0
        self._t_last_print = -1.0
        self._nprint = 0

    # ------------------------------------------------------------ transforms
    # Delegate to the canonical C transforms (embed_sim_coordinate_transform.c)
    # exposed by the DFC wrapper — no inline math, no Python mirror.

    def _vdq_from_duties(self, ta, tb, tc, v_dc, theta_e):
        """
        Duties -> (v_d, v_q).

        Star voltages -> Clarke_Transform_Matrix -> Park_Transform_Matrix (C).
        """
        # Star phase voltages (neutral = average of legs)
        va_leg = ta * v_dc
        vb_leg = tb * v_dc
        vc_leg = tc * v_dc
        vn = (va_leg + vb_leg + vc_leg) / 3.0
        va = va_leg - vn
        vb = vb_leg - vn
        vc = vc_leg - vn

        # Clarke: va,vb,vc -> v_alpha, v_beta   (C: Clarke_Transform_Matrix)
        v_alpha, v_beta = clarke(va, vb, vc)

        # Park: v_alpha,v_beta,theta_e -> v_d, v_q   (C: Park_Transform_Matrix)
        v_d, v_q = park(v_alpha, v_beta, theta_e)
        return float(v_d), float(v_q)

    def _abc_from_idiq(self, i_d, i_q, theta_e):
        """
        (i_d, i_q) -> (ia, ib, ic).

        C: InvPark_Transform_Matrix -> InvClarke_Transform_Matrix.
        """
        # Inverse Park: i_d,i_q,theta_e -> i_alpha, i_beta
        i_alpha, i_beta = inv_park(i_d, i_q, theta_e)

        # Inverse Clarke: i_alpha,i_beta -> ia, ib, ic
        ia, ib, ic = inv_clarke(i_alpha, i_beta)
        return float(ia), float(ib), float(ic)

    # -------------------------------------------------------------- ODE rhs
    def _ode(self, x, ta, tb, tc, v_dc, T_load):
        """
        dx/dt = f(x, u)   x = [i_d, i_q, omega_m, theta_e]
        Inputs held constant (zero-order hold) across the RK4 stages.
        """
        i_d, i_q, omega_m, theta_e = x
        omega_e = self.p * omega_m
        vd, vq  = self._vdq_from_duties(ta, tb, tc, v_dc, theta_e)

        did_dt     = (vd - self.R*i_d + omega_e*self.L_q*i_q) / self.L_d
        diq_dt     = (vq - self.R*i_q - omega_e*(self.L_d*i_d + self.lambda_pm)) / self.L_q

        T_em       = 1.5 * self.p * (self.lambda_pm*i_q
                                      + (self.L_d - self.L_q)*i_d*i_q)
        domega_dt  = (T_em - self.B_fric*omega_m - T_load) / self.J
        dtheta_dt  = omega_e

        return np.array([did_dt, diq_dt, domega_dt, dtheta_dt],
                        dtype=np.float64)

    # --------------------------------------------------------------- RK4
    def _rk4(self, x, dt, ta, tb, tc, v_dc, T_load):
        """Classic RK4 with zero-order hold on inputs."""
        k1 = self._ode(x,               ta, tb, tc, v_dc, T_load)
        k2 = self._ode(x + 0.5*dt*k1,   ta, tb, tc, v_dc, T_load)
        k3 = self._ode(x + 0.5*dt*k2,   ta, tb, tc, v_dc, T_load)
        k4 = self._ode(x +     dt*k3,   ta, tb, tc, v_dc, T_load)
        return x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    # ----------------------------------------------------------- compute
    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        # Latch inputs
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                self._ta = float(max(0.0, min(1.0, v[0])))
                self._tb = float(max(0.0, min(1.0, v[1])))
                self._tc = float(max(0.0, min(1.0, v[2])))
            if len(v) >= 4 and float(v[3]) > 0.0:
                self._v_dc = float(v[3])
            if len(v) >= 5:
                self._tload = float(v[4])

        # Advance states by one RK4 step
        self._x = self._rk4(self._x, float(dt),
                             self._ta, self._tb, self._tc,
                             self._v_dc, self._tload)

        i_d, i_q, omega_m, theta_e = self._x

        # Derived outputs
        T_em       = 1.5 * self.p * (self.lambda_pm*i_q
                                      + (self.L_d - self.L_q)*i_d*i_q)
        ia, ib, ic = self._abc_from_idiq(i_d, i_q, theta_e)
        theta_m    = theta_e / self.p      # mechanical angle (unwrapped)
        speed_rpm  = omega_m * 60.0 / (2.0 * math.pi)

        # Periodic console print
        if t - self._t_last_print >= 0.2 and self._nprint < 20:
            print(f"[PMSM t={t:.2f}s]  rpm={speed_rpm:+8.1f}  "
                  f"theta_e={theta_e:.4f}rad  "
                  f"id={i_d:+.4f}A  iq={i_q:+.4f}A  "
                  f"T_em={T_em*1e3:+.3f}mN.m  "
                  f"T_load={self._tload*1e3:.1f}mN.m")
            self._t_last_print = t
            self._nprint += 1

        self.output = VectorSignal(np.array([
            speed_rpm,    # [0] RPM
            ia, ib, ic,   # [1-3] phase currents [A]
            theta_m,      # [4] mechanical angle [rad]  <- CtrlPacker index [4]
            T_em,         # [5] electromagnetic torque [N.m]
            i_d,          # [6] d-axis current [A]
            i_q,          # [7] q-axis current [A]
        ], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)
