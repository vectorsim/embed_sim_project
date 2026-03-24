# pi_foc_block.py
"""
PIFOCBlock — Closed-Loop PI FOC for NANOTEC DB42S02
====================================================

Identical internal algorithm to ClosedLoopPIFOC from
db42s02_closed_loop_pi_foc.py — that file is the proven reference.

The only additions over ClosedLoopPIFOC:
  • C metadata (C_SOURCES, C_HEADERS, state_struct, init_func, C_INIT_ARGS)
  • C_CUSTOM_EMIT — emits the full Clarke→Park→PI→InvPark chain using the
    same C function signatures as coordinate_transform_blocks.py
  • dt_s constructor parameter (replaces module-level DT constant)
  • Diagnostic log at 1 kHz for scope data extraction

Signal contract (unchanged from reference):
  Input  port 0  (6): [omega_ref, omega_m, theta_e, ia, ib, ic]
  Output         (3): [v_alpha, v_beta, vdc]   ← vdc pass-through for SVPWMPackBlock

Gains (pole-zero cancellation, contraction metric):
  Current  ωc_i = 2π×500 Hz:  Kp_i = L·ωc_i   Ki_i = R·ωc_i
  Speed    ωc_ω = 2π×15  Hz:  Kp_ω = J·ωc_ω   Ki_ω = B·ωc_ω

C_CUSTOM_EMIT variable names match coordinate_transform_blocks.py:
  y_Clarke[2], y_Park[2], THETA_E, y_inv_park[2]
  PI_FOC_State_T struct members: int_id, int_iq, int_spd
"""

from __future__ import annotations

import math, sys
import numpy as np
from pathlib import Path
from typing import List, Optional

_HERE  = Path(__file__).resolve().parent
_C_SRC = _HERE / "c_src"

for _p in (str(_HERE.parent), str(_C_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE


# =============================================================================
# Motor constants
# =============================================================================

class _DB42S02:
    p      = 4
    R_s    = 0.19
    L_d    = 0.125e-3
    L_q    = 0.125e-3
    lam_pm = 0.0014
    J      = 2.4e-6
    B      = 1e-6     # N·m·s/rad  — from PMSM_Motor.mo (NOT 7e-5)
    I_max  = 3.57
    V_dc   = 17.0
    V_max  = V_dc / math.sqrt(3.0)

    KT     = 1.5 * p * lam_pm  # 0.00840 N·m/A

    WC_I   = 2.0 * math.pi * 500.0    # 500 Hz — z=0.843, proven stable
    KP_I   = L_d * WC_I               # 0.3927 V/A
    KI_I   = R_s * WC_I               # 596.9  V/(A·s)

    # Speed PI — Kp = J·ωc/KT
    # Ki = Kp/Ti  where Ti=1s (practical integral, ~1s load recovery)
    # Ki=2.538 (Ti=10ms) caused integrator windup → ±I_max oscillation
    # With B=1e-6 the mechanical τ=2.4s — Ti=1s is already aggressive
    WC_W   = 2.0 * math.pi * 15.0
    KP_SPD = J * WC_W / KT            # 0.02693 A·s/rad
    KI_SPD = KP_SPD / 0.1             # 0.2693  A/rad  (Ti = 0.1 s)


# =============================================================================
# PIFOCBlock
# =============================================================================

class PIFOCBlock(VectorBlock):
    """
    PI FOC controller — single EmbedSim block node.

    Input  port 0  (6): [omega_ref, omega_m, theta_e, ia, ib, ic]
    Output         (3): [v_alpha, v_beta, vdc]

    The vdc pass-through (output[2]) lets SVPWMPackBlock read port 0
    directly without any packer block — identical to ClosedLoopPIFOC.

    C metadata — StepGenerator emits PI_FOC_Step() which internally calls:
      Clarke_Step → Park_Step → speed_PI → current_PI → InvPark_Step
    All state in PI_FOC_T struct.
    """

    # ── C metadata ────────────────────────────────────────────────────────────
    C_SOURCES    = ["PI_FOC.c", "Coordinate_Transform.c", "Matrix.c"]
    C_HEADERS    = ["PI_FOC.h", "Coordinate_Transform.h"]
    step_func    = "PI_FOC_Step"
    init_func    = "PI_FOC_Init"
    state_struct = "PI_FOC_T"
    NUM_INPUTS   = 1
    OUTPUT_SIZE  = 3
    OUTPUT_NAMES = ["v_alpha", "v_beta", "vdc"]
    OUTPUT_KEEP  = [0, 1]        # only v_alpha, v_beta go to EmbedSim_Output_T
    C_INIT_ARGS  = ["v_dc", "p_poles", "R_s", "L_d", "L_q",
                    "lambda_pm", "J_rotor", "B_friction", "i_max", "dt_s"]

    # C_CUSTOM_EMIT — variable names match coordinate_transform_blocks.py exactly
    # so the generated embedsim_step.c compiles without modification.
    C_CUSTOM_EMIT = """\
    /* --- pi_foc (PIFOCBlock) --- */
    real32_T y_pi_foc[3];
    {
        /* Clarke: ia,ib,ic → i_alpha, i_beta */
        MatrixFloat clarke_alpha, clarke_beta;
        Clarke_Step(&Clarke_state,
                    u_pi_foc[3],  /* ia */
                    u_pi_foc[4],  /* ib */
                    u_pi_foc[5],  /* ic */
                    &clarke_alpha, &clarke_beta);
        real32_T y_Clarke[2];
        y_Clarke[0] = (real32_T)clarke_alpha;
        y_Clarke[1] = (real32_T)clarke_beta;

        /* Park: i_alpha, i_beta, theta_e → id, iq */
        real32_T THETA_E = u_pi_foc[2];
        MatrixFloat park_d, park_q;
        Park_Step(&Park_state,
                  y_Clarke[0], y_Clarke[1],
                  THETA_E,
                  &park_d, &park_q);
        real32_T y_Park[2];
        y_Park[0] = (real32_T)park_d;
        y_Park[1] = (real32_T)park_q;

        /* PI FOC: omega_ref, omega_m, id, iq → vd, vq */
        real32_T vd_pi, vq_pi;
        PI_FOC_Step(&pi_foc_state,
                    u_pi_foc[0],  /* omega_ref */
                    u_pi_foc[1],  /* omega_m   */
                    y_Park[0],    /* id_meas   */
                    y_Park[1],    /* iq_meas   */
                    dt,
                    &vd_pi, &vq_pi);

        /* InvPark: vd, vq, theta_e → v_alpha, v_beta */
        MatrixFloat invpark_alpha, invpark_beta;
        InvPark_Step(&inv_park_state,
                     vd_pi, vq_pi,
                     THETA_E,
                     &invpark_alpha, &invpark_beta);
        real32_T y_inv_park[2];
        y_inv_park[0] = (real32_T)invpark_alpha;
        y_inv_park[1] = (real32_T)invpark_beta;

        y_pi_foc[0] = y_inv_park[0];  /* v_alpha */
        y_pi_foc[1] = y_inv_park[1];  /* v_beta  */
        y_pi_foc[2] = PI_FOC_VDC;     /* vdc pass-through */
    }"""

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(self, name: str,
                 v_dc:          float = _DB42S02.V_dc,
                 p_poles:       int   = _DB42S02.p,
                 R_s:           float = _DB42S02.R_s,
                 L_d:           float = _DB42S02.L_d,
                 L_q:           float = _DB42S02.L_q,
                 lambda_pm:     float = _DB42S02.lam_pm,
                 J_rotor:       float = _DB42S02.J,
                 B_friction:    float = _DB42S02.B,
                 i_max:         float = _DB42S02.I_max,
                 dt_s:          float = 50e-6) -> None:
        super().__init__(name)

        # Instance attrs for C_INIT_ARGS
        self.v_dc       = float(v_dc)
        self.p_poles    = int(p_poles)
        self.R_s        = float(R_s)
        self.L_d        = float(L_d)
        self.L_q        = float(L_q)
        self.lambda_pm  = float(lambda_pm)
        self.J_rotor    = float(J_rotor)
        self.B_friction = float(B_friction)
        self.i_max      = float(i_max)
        self.dt_s       = float(dt_s)

        self.output_label = "[v_α,v_β,vdc]"
        self.is_dynamic   = False

        # ── Gains — computed once here, used in compute_py ───────────────────
        # Current loop (pole-zero cancellation):  vd/vq output
        wc_i         = 2.0 * math.pi * 500.0   # 500 Hz — z=0.843, proven stable
        self._kp_i   = self.L_d * wc_i                # 0.3927 V/A
        self._ki_i   = self.R_s * wc_i                # 596.9  V/(A·s)
        self._v_max  = self.v_dc / math.sqrt(3.0)     # 9.815 V
        self._v_lim  = self._v_max / self._ki_i       # integrator anti-windup

        # Speed loop:  iq_ref [A]
        # Kp = J·ωc/KT,  Ki = Kp/Ti  (Ti=1s — avoids windup with τ_mech=2.4s)
        KT           = 1.5 * self.p_poles * self.lambda_pm
        wc_spd       = 2.0 * math.pi * 15.0
        self._kp_spd = self.J_rotor * wc_spd / KT     # 0.02693 A·s/rad
        self._ki_spd = self._kp_spd / 0.1             # 0.2693  A/rad  (Ti=0.1s)
        self._iq_lim = self.i_max / self._ki_spd       # anti-windup limit

        # ── Integrator state ──────────────────────────────────────────────────
        self._int_id  = 0.0
        self._int_iq  = 0.0
        self._int_spd = 0.0

        # ── Diagnostic log (1 kHz) ─────────────────────────────────────────────
        self._log_t:    list = []
        self._log_spd:  list = []
        self._log_sref: list = []
        self._log_iqr:  list = []
        self._log_iq:   list = []
        self._log_id:   list = []
        self._log_next: float = 0.0

    # ── Static transform helpers (identical to ClosedLoopPIFOC) ──────────────

    @staticmethod
    def _clarke(ia, ib, ic):
        return ((2.0/3.0)*ia - (1.0/3.0)*ib - (1.0/3.0)*ic,
                (ib - ic) / math.sqrt(3.0))

    @staticmethod
    def _park(a, b, th):
        c, s = math.cos(th), math.sin(th)
        return a*c + b*s, -a*s + b*c

    @staticmethod
    def _inv_park(vd, vq, th):
        c, s = math.cos(th), math.sin(th)
        return vd*c - vq*s, vd*s + vq*c

    # ── compute_py (identical algorithm to ClosedLoopPIFOC.compute_py) ────────

    def compute_py(self, t: float, dt: float,
                   input_values: Optional[List[VectorSignal]] = None
                   ) -> VectorSignal:
        _z = VectorSignal(np.zeros(3, dtype=DEFAULT_DTYPE), self.name)
        if not input_values or input_values[0] is None:
            self.output = _z; return _z
        u = input_values[0].value
        if len(u) < 6:
            self.output = _z; return _z

        omega_ref = float(u[0])
        omega_m   = float(u[1])
        theta_e   = float(u[2])
        ia, ib, ic = float(u[3]), float(u[4]), float(u[5])

        i_alpha, i_beta  = self._clarke(ia, ib, ic)
        id_meas, iq_meas = self._park(i_alpha, i_beta, theta_e)

        e_spd = omega_ref - omega_m
        self._int_spd = max(-self._iq_lim,
                            min(self._iq_lim, self._int_spd + e_spd * dt))
        iq_ref = max(-self.i_max,
                     min(self.i_max,
                         self._kp_spd * e_spd + self._ki_spd * self._int_spd))

        we   = float(self.p_poles) * omega_m
        e_id = 0.0 - id_meas
        e_iq = iq_ref - iq_meas

        self._int_id = max(-self._v_lim, min(self._v_lim,
                                             self._int_id + e_id * dt))
        self._int_iq = max(-self._v_lim, min(self._v_lim,
                                             self._int_iq + e_iq * dt))

        vd = (self._kp_i * e_id + self._ki_i * self._int_id
              - we * self.L_q * iq_meas)
        vq = (self._kp_i * e_iq + self._ki_i * self._int_iq
              + we * self.L_d * id_meas + we * self.lambda_pm)

        v_mag = math.sqrt(vd*vd + vq*vq)
        if v_mag > self._v_max:
            s = self._v_max / v_mag
            vd *= s; vq *= s

        v_alpha, v_beta = self._inv_park(vd, vq, theta_e)

        # Diagnostic log
        if t >= self._log_next:
            self._log_t.append(t)
            self._log_spd.append(omega_m * 60.0 / (2.0 * math.pi))
            self._log_sref.append(omega_ref * 60.0 / (2.0 * math.pi))
            self._log_iqr.append(iq_ref)
            self._log_iq.append(iq_meas)
            self._log_id.append(id_meas)
            self._log_next += 1e-3

        self.output = VectorSignal(
            np.array([v_alpha, v_beta, self.v_dc], dtype=DEFAULT_DTYPE),
            self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)

    def reset(self):
        super().reset()
        self._int_id = self._int_iq = self._int_spd = 0.0
        self._log_t.clear();   self._log_spd.clear()
        self._log_sref.clear(); self._log_iqr.clear()
        self._log_iq.clear();  self._log_id.clear()
        self._log_next = 0.0

    @property
    def log_data(self) -> dict:
        return {
            "t":         np.array(self._log_t,    dtype=np.float32),
            "speed":     np.array(self._log_spd,  dtype=np.float32),
            "speed_ref": np.array(self._log_sref, dtype=np.float32),
            "iq_ref":    np.array(self._log_iqr,  dtype=np.float32),
            "iq":        np.array(self._log_iq,   dtype=np.float32),
            "id":        np.array(self._log_id,   dtype=np.float32),
        }


__all__ = ["PIFOCBlock", "_DB42S02"]
