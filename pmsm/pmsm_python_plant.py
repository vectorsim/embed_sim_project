# pmsm_python_plant.py
"""
Pure-Python PMSM Plant Block for EmbedSim
==========================================
Textbook dq-frame PMSM.  No FMU, no DASSL, no surprises.

SVPWM Interface Convention (AURIX embed_sim_sv_pwm.c):
----------------------------------------------------
The SVPWM outputs duty cycles in [0, 1] range where:
- 0.5 = zero voltage (all phase voltages equal)
- 0.0 = minimum voltage (all low-side switches on)
- 1.0 = maximum voltage (all high-side switches on)

This is verified in SVM_CalculateDutyFromTimes():
  For ModIndex = 0: T1 = T2 = 0, t0 = 0.5
  ta = tb = tc = 0.5 → zero phase-to-neutral voltage

The PMSM plant uses the standard inverter model:
  v_phase_leg = duty * Vdc
  v_neutral = (va_leg + vb_leg + vc_leg) / 3
  v_phase_to_neutral = v_phase_leg - v_neutral

All Clarke / Park transforms use the canonical C implementation
from embed_sim_coordinate_transform.c.

Block interface
---------------
Input bus [0] : [ta, tb, tc, v_dc, T_load]   (from SVPWM)
                  |   |   |    |      |
                  |   |   |    |      +-- Load torque [N.m]
                  |   |   |    +--------- DC bus voltage [V]
                  |   |   +-------------- Duty cycle phase C [0.0, 1.0]
                  |   +------------------ Duty cycle phase B [0.0, 1.0]
                  +---------------------- Duty cycle phase A [0.0, 1.0]

Note: ta=tb=tc=0.5 → zero voltage (verified from SVPWM implementation)

Output bus    : [rpm, ia, ib, ic, theta_e, T_em, id, iq]
                  [0]  [1][2][3]    [4]     [5]  [6][7]
"""

import math
import sys
from pathlib import Path

import numpy as np
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE

# Frame transforms from canonical C implementation
_C_SRC = Path(__file__).resolve().parent / "c_src"
if str(_C_SRC) not in sys.path:
    sys.path.insert(0, str(_C_SRC))

try:
    from embedsim_control_wrapper import clarke, park, inv_park, inv_clarke
except ImportError:
    try:
        from dfc_controller_wrapper import clarke, park, inv_park, inv_clarke
    except ImportError:
        raise ImportError(
            "No wrapper found with transform functions. "
            "Build the wrapper first:\n"
            "  cd pmsm/c_src && ./build.sh"
        )


class PMSM_Python_Plant(VectorBlock):
    """
    Pure-Python PMSM plant with RK4 internal integration.

    SVPWM Duty Convention: [0, 1] where 0.5 = zero voltage
    (Matches AURIX embed_sim_sv_pwm.c implementation)

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
                 v_dc: float      = 12.0,
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
        self._x = np.zeros(4, dtype=np.float64)

        # Latched inputs (zero-order hold between compute calls)
        # Initialized to 0.5 = zero voltage (matches SVPWM)
        self._ta    = 0.5
        self._tb    = 0.5
        self._tc    = 0.5
        self._v_dc  = float(v_dc)
        self._tload = 0.0

        # Diagnostics
        self._t_last_print = -1.0

        print(f"[PMSM_Python_Plant] '{name}'  "
              f"R={R} Ld={L_d} Lq={L_q} lpm={lambda_pm} "
              f"J={J} B={B_fric} p={p} Vdc={v_dc}  "
              f"[SVPWM convention: 0.5 = zero voltage]")

    # ------------------------------------------------------------------
    def reset(self):
        super().reset()
        self._x[:]  = 0.0
        self._ta    = 0.5  # Zero voltage (SVPWM convention)
        self._tb    = 0.5
        self._tc    = 0.5
        self._v_dc  = self._v_dc_nom
        self._tload = 0.0
        self._t_last_print = -1.0

    # ------------------------------------------------------------ transforms
    def _vdq_from_duties(self, ta, tb, tc, v_dc, theta_e):
        """
        SVPWM duties [0,1] -> (v_d, v_q).

        SVPWM convention (from embed_sim_sv_pwm.c):
        - Duty = 0.5 → zero voltage (all phase voltages equal)
        - Duty = 0.0 → minimum voltage
        - Duty = 1.0 → maximum voltage

        Phase voltages:
        - Leg voltage: v_leg = duty * Vdc
        - Neutral: vn = (va_leg + vb_leg + vc_leg) / 3
        - Phase-to-neutral: v_phase = v_leg - vn

        For ta=tb=tc=0.5: va=vb=vc=0.5*Vdc, vn=0.5*Vdc, va-vn=0 ✓
        """
        # Clamp duties to [0, 1] for safety
        ta = float(max(0.0, min(1.0, ta)))
        tb = float(max(0.0, min(1.0, tb)))
        tc = float(max(0.0, min(1.0, tc)))

        # Phase leg voltages (relative to DC-)
        va_leg = ta * v_dc
        vb_leg = tb * v_dc
        vc_leg = tc * v_dc

        # Neutral point voltage (floating motor neutral)
        vn = (va_leg + vb_leg + vc_leg) / 3.0

        # Phase-to-neutral voltages
        va = va_leg - vn
        vb = vb_leg - vn
        vc = vc_leg - vn

        # Clarke transform: va,vb,vc -> v_alpha, v_beta
        v_alpha, v_beta = clarke(va, vb, vc)

        # Park transform: v_alpha,v_beta,theta_e -> v_d, v_q
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

        # dq voltage equations (Krishnan, Ch. 4)
        did_dt     = (vd - self.R*i_d + omega_e*self.L_q*i_q) / self.L_d
        diq_dt     = (vq - self.R*i_q - omega_e*(self.L_d*i_d + self.lambda_pm)) / self.L_q

        # Electromagnetic torque
        T_em       = 1.5 * self.p * (self.lambda_pm*i_q
                                      + (self.L_d - self.L_q)*i_d*i_q)

        # Mechanical dynamics
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
                # SVPWM duties in [0,1] from AURIX (0.5 = zero voltage)
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
        speed_rpm  = omega_m * 60.0 / (2.0 * math.pi)

        theta_m = (theta_e / self.p) % (2*math.pi)

        # Periodic console print (every 0.2s)
        if t - self._t_last_print >= 0.2:
            # Show duty offset from zero (0.5)
            da_offset = (self._ta - 0.5) * 100.0
            db_offset = (self._tb - 0.5) * 100.0
            dc_offset = (self._tc - 0.5) * 100.0
            print(f"[PMSM t={t:.2f}s]  rpm={speed_rpm:+8.1f}  "
                  f"theta_m={theta_m:.4f}rad  "
                  f"id={i_d:+.4f}A  iq={i_q:+.4f}A  "
                  f"T_em={T_em*1e3:+.3f}mN.m  "
                  f"duty=[{self._ta:.3f}({da_offset:+5.1f}%) "
                  f"{self._tb:.3f}({db_offset:+5.1f}%) "
                  f"{self._tc:.3f}({dc_offset:+5.1f}%)]")
            self._t_last_print = t

        # Output bus
        self.output = VectorSignal(np.array([
            speed_rpm,    # [0] RPM
            ia, ib, ic,   # [1-3] phase currents [A]
            theta_m,      # [4] mechanical angle [rad]
            T_em,         # [5] electromagnetic torque [N.m]
            i_d,          # [6] d-axis current [A]
            i_q,          # [7] q-axis current [A]
        ], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)