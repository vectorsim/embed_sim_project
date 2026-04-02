# pmsm_python_plant.py
"""
Pure-Python PMSM Plant Block for EmbedSim
==========================================
Textbook dq-frame PMSM.  No FMU, no DASSL, no surprises.

EmbedSim calls compute(t, dt, inputs) every step.  This block owns its
four state variables [i_d, i_q, omega_m, theta_e] and advances them
using RK4 internally — 4th-order accuracy at the 50 us step rate.

All Clarke / Park / InvPark / InvClarke calculations are delegated to
coordinate_transform_blocks.py — the Python mirrors of the canonical C
functions in embed_sim_coordinate_transform.c.  There is no inline
transform math in this file.

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
import numpy as np
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from coordinate_transform_blocks import (
    ClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
    InvClarkeTransformBlock,
)


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
                 R: float         = 0.285,      # DELTA wiring
                 L_d: float       = 0.3675e-3,  # DELTA wiring
                 L_q: float       = 0.3675e-3,  # DELTA wiring
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

        # ── Transform block instances — canonical, no inline math ────────────
        # Imported at top of file from coordinate_transform_blocks.py.
        # Four blocks cover the complete voltage and current transform chain:
        #   duties -> Clarke -> Park -> (vd,vq)  [in _vdq_from_duties]
        #   (id,iq) -> InvPark -> InvClarke -> abc  [in _abc_from_idiq]
        self._ct_clarke     = ClarkeTransformBlock("_plant_clarke",     use_c_backend=False)
        self._ct_park       = ParkTransformBlock("_plant_park",         use_c_backend=False)
        self._ct_inv_park   = InvParkTransformBlock("_plant_inv_park",  use_c_backend=False)
        self._ct_inv_clarke = InvClarkeTransformBlock("_plant_inv_clarke", use_c_backend=False)

        print(f"[PMSM_Python_Plant] '{name}'  "
              f"R={R} Ld={L_d} Lq={L_q} lpm={lambda_pm} "
              f"J={J} B={B_fric} p={p} Vdc={v_dc}  "
              f"[transforms -> coordinate_transform_blocks]")

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
        self._ct_clarke.reset()
        self._ct_park.reset()
        self._ct_inv_park.reset()
        self._ct_inv_clarke.reset()

    # ------------------------------------------------------------ transforms
    # Delegate to coordinate_transform_blocks — no inline math here.
    # Block instances created once in __init__ and reused every RK4 stage.

    def _vdq_from_duties(self, ta, tb, tc, v_dc, theta_e):
        """
        Duties -> (v_d, v_q).

        Star voltages -> ClarkeTransformBlock -> ParkTransformBlock.
        Identical pipeline to what the SMC controller uses on the current
        feedback path, ensuring consistent frame alignment.
        """
        # Star phase voltages (neutral = average of legs)
        va_leg = ta * v_dc
        vb_leg = tb * v_dc
        vc_leg = tc * v_dc
        vn = (va_leg + vb_leg + vc_leg) / 3.0
        va = va_leg - vn
        vb = vb_leg - vn
        vc = vc_leg - vn

        # Clarke: va,vb,vc -> v_alpha, v_beta
        inp = VectorSignal(np.array([va, vb, vc], dtype=np.float32), "_plant_clarke")
        ab  = self._ct_clarke.compute_py(0.0, 0.0, [inp])
        v_alpha, v_beta = float(ab.value[0]), float(ab.value[1])

        # Park: v_alpha,v_beta,theta_e -> v_d, v_q
        ab_sig = VectorSignal(np.array([v_alpha, v_beta], dtype=np.float32), "_plant_park")
        th_sig = VectorSignal(np.array([theta_e],         dtype=np.float32), "_plant_park")
        dq     = self._ct_park.compute_py(0.0, 0.0, [ab_sig, th_sig])
        return float(dq.value[0]), float(dq.value[1])

    def _abc_from_idiq(self, i_d, i_q, theta_e):
        """
        (i_d, i_q) -> (ia, ib, ic).

        InvParkTransformBlock -> InvClarkeTransformBlock.
        """
        # Inverse Park: i_d,i_q,theta_e -> i_alpha, i_beta
        dq_sig = VectorSignal(np.array([i_d, i_q], dtype=np.float32), "_plant_invpark")
        th_sig = VectorSignal(np.array([theta_e],  dtype=np.float32), "_plant_invpark")
        ab     = self._ct_inv_park.compute_py(0.0, 0.0, [dq_sig, th_sig])

        # Inverse Clarke: i_alpha,i_beta -> ia, ib, ic
        ab_sig = VectorSignal(np.array([ab.value[0], ab.value[1]], dtype=np.float32),
                              "_plant_invclarke")
        abc    = self._ct_inv_clarke.compute_py(0.0, 0.0, [ab_sig])
        return float(abc.value[0]), float(abc.value[1]), float(abc.value[2])

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
