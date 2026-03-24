"""
pmsm_lqr_foc.py
===============================================================================
EmbedSim — PMSM FOC with LQR Current Controller
NANOTEC DB42S02  |  AURIX TC3xx target
===============================================================================

Architecture
------------
Speed loop  : PI  (outer, 15 Hz bandwidth)  →  iq_ref
Current loop: LQR state-feedback on augmented state [id, iq, xi_d, xi_q]
              xi = integrators for zero steady-state current error
Plant       : DB42S02PlantBlock  (PMSM_Python_Plant, no FMU required)

Signal flow
-----------
  omega_ref ──► [SpeedRefPacker]
  motor_delay ─► [FeedbackUnpackBlock]
                    │ omega_m, theta_e, id, iq
  [SpeedRefPacker] ──► [PISpeedBlock] ──► iq_ref
  [FeedbackUnpackBlock] + iq_ref ──► [LQRCurrentBlock] ──► [v_alpha, v_beta]
  [v_alpha, v_beta] ──► [InvClarkeBlock] ──► [va, vb, vc]
  [va, vb, vc] ──► [DB42S02PlantBlock] ──► motor_delay ──► sink

LQR Design
----------
  PMSM dq model (linearised):
    did/dt = -Rs/Ld·id + ωe·iq  + vd/Ld
    diq/dt = -ωe·id   - Rs/Lq·iq - ωe·λpm/Lq + vq/Lq

  Augmented with integrators on current error:
    dxi_d/dt = 0 - id         (id_ref = 0  MTPA)
    dxi_q/dt = iq_ref - iq

  State x4 = [id, iq, xi_d, xi_q],  input u2 = [vd, vq]
  Control law: u = -K·x  (LQR gain K solved from algebraic Riccati)
  Feedforward: vd += -ωe·Lq·iq,  vq += ωe·(Ld·id + λpm)

Motor: NANOTEC DB42S02
  Rs=0.19Ω  Ld=Lq=0.125mH  λpm=0.0014Wb  p=4
  J=2.4e-6kg·m²  B_fric=7e-5N·m·s  Vdc=17V
"""

from __future__ import annotations

import math
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE
for _candidate in (_HERE, _HERE.parent, _HERE.parent.parent):
    if (_candidate / "embedsim").is_dir():
        _ROOT = _candidate
        break
for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── fs_electrical_machines path ────────────────────────────────────────────────
_FS = _ROOT / "fs_electrical_machines"
if _FS.is_dir() and str(_FS) not in sys.path:
    sys.path.insert(0, str(_FS))

# ── EmbedSim core ──────────────────────────────────────────────────────────────
from embedsim.core_blocks       import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks     import VectorStep
from embedsim.dynamic_blocks    import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver, VectorDelay

# ── PMSM plant ─────────────────────────────────────────────────────────────────
from fs_electrical_machines.pmsm_python_plant import PMSM_Python_Plant

# ==============================================================================
#  Motor constants — NANOTEC DB42S02
# ==============================================================================
class _DB42S02:
    p         = 4
    R_s       = 0.19        # Ω
    L_d       = 0.125e-3    # H
    L_q       = 0.125e-3    # H
    lam_pm    = 0.0014      # Wb
    J         = 2.4e-6      # kg·m²
    B         = 7e-5        # N·m·s/rad
    I_max     = 3.57        # A  peak rated current
    V_dc      = 17.0        # V
    V_max     = 17.0 / 2.0  # 8.5 V  peak phase voltage (half-bridge: duty*Vdc - Vdc/2)

    # Speed PI  (bandwidth 30 Hz — faster than before)
    WC_W      = 2.0 * math.pi * 30.0
    KP_SPD    = J * WC_W                # 4.52e-4 A·s/rad
    KI_SPD    = B * WC_W                # 1.32e-2 A/rad

    # LQR operating point (400 RPM → ωe_op)
    RPM_OP    = 400.0
    WE_OP     = RPM_OP / 60.0 * 2.0 * math.pi * p    # electrical rad/s


def _solve_lqr(we_op: float) -> np.ndarray:
    """
    LQR on augmented state  x = [id, iq, theta_e, xi_d, xi_q]  (5 states)

    Plant equations (linearised at we_op, id_ref=0, iq_ref=iq_op):
      did/dt    = -Rs/Ld·id  + we·iq      + vd/Ld
      diq/dt    = -we·id     - Rs/Lq·iq   + vq/Lq   (back-EMF FF removed)
      dtheta_e/dt = we  = p·(3/2·lam_pm·iq - B·omega_m) / J  ≈  p·KT/J·iq
                   linearised:  dtheta_e/dt ≈ (p·KT/J) · iq   (omega integrates iq torque)

    Integrators on current error (zero ss current error):
      dxi_d/dt  = e_d = id  - 0
      dxi_q/dt  = e_q = iq  - iq_ref

    Full A (5×5), B (5×2):
      rows: [id, iq, theta_e, xi_d, xi_q]

    Q weights:
      id, iq       — moderate  (current tracking)
      theta_e      — HIGH  (position is the primary objective)
      xi_d, xi_q   — moderate  (integrators)

    R: normalised to voltage budget V_max
    """
    Rs      = _DB42S02.R_s       # 0.19 Ω
    Ld      = _DB42S02.L_d       # 125 µH
    Lq      = _DB42S02.L_q       # 125 µH
    lam_pm  = _DB42S02.lam_pm    # 0.0014 Wb
    J       = _DB42S02.J         # 2.4e-6 kg·m²
    B_fric  = _DB42S02.B         # 7e-5 N·m·s
    p       = float(_DB42S02.p)  # 4 pole pairs
    I_max   = _DB42S02.I_max     # 3.57 A
    V_max   = _DB42S02.V_max     # 8.5 V

    # Torque constant  KT = 3/2 · p · lam_pm  [N·m/A]
    KT      = 1.5 * p * lam_pm   # 0.0084 N·m/A

    # dtheta_e/dt linearised gain:  p·KT/J  (position sensitivity to iq)
    k_pos   = p * KT / J          # rad/(s·A)  ≈  14000

    # ── A matrix (5×5) ────────────────────────────────────────────────────────
    #        id          iq        theta_e   xi_d  xi_q
    A = np.array([
        [-Rs/Ld,    we_op,      0.0,       0.0,  0.0],   # did/dt
        [-we_op,   -Rs/Lq,      0.0,       0.0,  0.0],   # diq/dt
        [  0.0,    k_pos,       0.0,       0.0,  0.0],   # dtheta_e/dt  ← iq drives position
        [  1.0,    0.0,         0.0,       0.0,  0.0],   # dxi_d/dt = id
        [  0.0,    1.0,         0.0,       0.0,  0.0],   # dxi_q/dt = iq
    ], dtype=float)

    # ── B matrix (5×2) ────────────────────────────────────────────────────────
    B = np.zeros((5, 2))
    B[0, 0] = 1.0 / Ld    # vd → id
    B[1, 1] = 1.0 / Lq    # vq → iq

    # ── Q matrix — position gets highest weight ────────────────────────────────
    # Normalise: Q_ii = 1 / x_i_max²
    theta_max = 2.0 * math.pi / p   # one electrical revolution = 1 mechanical rev / p
    xi_max    = I_max / (Rs / Ld)   # integrator saturation ~ I_max / bandwidth

    q_i      = 1.0 / (I_max    ** 2)    # current weight     ≈ 0.079
    q_theta  = 50.0 / (theta_max ** 2)  # position weight — 50× larger than current
    q_xi     = 1.0 / (xi_max   ** 2)    # integrator weight

    Q = np.diag([q_i, q_i, q_theta, q_xi, q_xi])

    # ── R matrix — normalised to voltage budget ────────────────────────────────
    R = np.diag([1.0 / (V_max ** 2),
                 1.0 / (V_max ** 2)])

    # ── Solve Riccati ─────────────────────────────────────────────────────────
    try:
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
    except ImportError:
        P = _dare_iterative(A, B, Q, R, dt=50e-6, n=40000)

    K = np.linalg.solve(R, B.T @ P)   # K (2×5)

    # ── Closed-loop eigenvalues ───────────────────────────────────────────────
    A_cl  = A - B @ K
    eigs  = sorted(np.linalg.eigvals(A_cl), key=lambda z: z.real)
    print(f"\n  LQR  state = [id, iq, theta_e, xi_d, xi_q]")
    print(f"  Q diag = [{q_i:.2e}, {q_i:.2e}, {q_theta:.2e}, {q_xi:.2e}, {q_xi:.2e}]")
    print(f"  K (vd row): {K[0]}")
    print(f"  K (vq row): {K[1]}")
    print(f"  Closed-loop eigenvalues:")
    for i, ev in enumerate(eigs):
        bw = -ev.real / (2 * math.pi)
        print(f"    λ{i+1} = {ev.real:+.1f}{ev.imag:+.1f}j   BW≈{bw:.0f} Hz")
    assert all(ev.real < 0 for ev in eigs), "UNSTABLE — all eigenvalues must be negative!"

    return K   # (2×5)


def _dare_iterative(A, B, Q, R, dt=50e-6, n=20000) -> np.ndarray:
    """Discretise and iterate DARE when scipy is not available."""
    I  = np.eye(A.shape[0])
    Ad = I + A * dt
    Bd = B * dt
    Rd_inv = np.linalg.inv(R)
    P = Q.copy()
    for _ in range(n):
        K  = Rd_inv @ Bd.T @ P
        P  = Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K + Q
    return P


# ==============================================================================
#  Block: DB42S02PlantBlock
# ==============================================================================
class DB42S02PlantBlock(PMSM_Python_Plant):
    """
    NANOTEC DB42S02 — pure-Python plant (no FMU).

    Input  port 0: [va, vb, vc]   three-phase voltages (V)
    Output       : [rpm, ia, ib, ic, theta_m, T_em, id, iq]
    """
    TOPO_CATEGORY     = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[rpm,ia,ib,ic,theta_m,Tem,id,iq]"

    def __init__(self, name: str) -> None:
        super().__init__(
            name      = name,
            R         = _DB42S02.R_s,
            L_d       = _DB42S02.L_d,
            L_q       = _DB42S02.L_q,
            lambda_pm = _DB42S02.lam_pm,
            J         = _DB42S02.J,
            B_fric    = _DB42S02.B,
            p         = float(_DB42S02.p),
            v_dc      = _DB42S02.V_dc,
        )

    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        va = vb = vc = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                va, vb, vc = float(v[0]), float(v[1]), float(v[2])
        # PMSM_Python_Plant expects [ta, tb, tc, v_dc, T_load]
        # ta = va/v_dc + 0.5  (duty cycle from voltage)
        vdc = _DB42S02.V_dc
        ta  = np.clip(va / vdc + 0.5, 0.0, 1.0)
        tb  = np.clip(vb / vdc + 0.5, 0.0, 1.0)
        tc  = np.clip(vc / vdc + 0.5, 0.0, 1.0)
        augmented = [VectorSignal(
            np.array([ta, tb, tc, vdc, 0.0], dtype=DEFAULT_DTYPE))]
        return super().compute_py(t, dt, augmented)

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# ==============================================================================
#  Block: SpeedRefPackerBlock
# ==============================================================================
class SpeedRefPackerBlock(VectorBlock):
    """
    Pack [omega_ref] and motor feedback [rpm,...] into
    a single bus [omega_ref_rads, omega_m_rads] for the speed PI.

    port 0: [omega_ref_rpm]     scalar step
    port 1: [rpm,ia,ib,ic,...]  motor output bus
    """
    def __init__(self, name: str = "spd_packer") -> None:
        super().__init__(name)
        self.output_label = "[omega_ref,omega_m]"
        self.is_dynamic   = False
        self.vector_size  = 2

    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        omega_ref = omega_m = 0.0
        if input_values:
            if input_values[0] is not None:
                omega_ref = float(input_values[0].value[0]) * math.pi / 30.0   # rpm→rad/s
            if len(input_values) > 1 and input_values[1] is not None:
                omega_m = float(input_values[1].value[0]) * math.pi / 30.0     # rpm[0]→rad/s
        self.output = VectorSignal(
            np.array([omega_ref, omega_m], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# ==============================================================================
#  Block: PISpeedBlock
# ==============================================================================
class PISpeedBlock(VectorBlock):
    """
    Speed PI  — bandwidth 15 Hz.

    port 0: [omega_ref_rads, omega_m_rads]
    output: [iq_ref]
    """
    def __init__(self, name: str = "pi_speed",
                 kp: float = _DB42S02.KP_SPD,
                 ki: float = _DB42S02.KI_SPD,
                 i_max: float = _DB42S02.I_max) -> None:
        super().__init__(name)
        self.kp    = float(kp)
        self.ki    = float(ki)
        self.i_max = float(i_max)
        self.output_label = "[iq_ref]"
        self.is_dynamic   = False
        self.vector_size  = 1
        self._int_lim = i_max / ki if ki > 0 else 1e6
        self._integ   = 0.0

    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        omega_ref = omega_m = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 2:
                omega_ref, omega_m = float(v[0]), float(v[1])
        e = omega_ref - omega_m
        self._integ = float(np.clip(self._integ + e * dt,
                                    -self._int_lim, self._int_lim))
        iq_ref = float(np.clip(self.kp * e + self.ki * self._integ,
                               -self.i_max, self.i_max))
        self.output = VectorSignal(
            np.array([iq_ref], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)

    def reset(self):
        super().reset()
        self._integ = 0.0


# ==============================================================================
#  Block: FeedbackUnpackBlock
# ==============================================================================
class FeedbackUnpackBlock(VectorBlock):
    """
    Unpack motor output bus → dq feedback + theta_e.

    Motor output: [rpm, ia, ib, ic, theta_m, T_em, id, iq]
    Output bus  : [id, iq, theta_e, omega_m_rads]
    """
    def __init__(self, name: str = "fb_unpack") -> None:
        super().__init__(name)
        self.output_label = "[id,iq,theta_e,omega_m]"
        self.is_dynamic   = False
        self.vector_size  = 4

    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        id_m = iq_m = theta_e = omega_m = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            # [0]=rpm [1]=ia [2]=ib [3]=ic [4]=theta_m [5]=T_em [6]=id [7]=iq
            if len(v) >= 8:
                rpm     = float(v[0])
                omega_m = rpm * math.pi / 30.0
                theta_m = float(v[4])
                theta_e = float((theta_m * _DB42S02.p) % (2.0 * math.pi))
                id_m    = float(v[6])
                iq_m    = float(v[7])
        self.output = VectorSignal(
            np.array([id_m, iq_m, theta_e, omega_m], dtype=DEFAULT_DTYPE),
            self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# ==============================================================================
#  Block: LQRCurrentBlock
# ==============================================================================
class LQRCurrentBlock(VectorBlock):
    """
    LQR current controller with integrator augmentation.

    State:  x = [id, iq, xi_d, xi_q]
    Input:  u = -K·x  + decoupling feedforward
    Output: [v_alpha, v_beta]  (stationary αβ frame via InvPark)

    port 0: [id, iq, theta_e, omega_m]   from FeedbackUnpackBlock
    port 1: [iq_ref]                      from PISpeedBlock
    output: [v_alpha, v_beta]
    """
    C_CODEGEN_EXCLUDE = True   # LQR C code to be generated separately

    def __init__(self, name: str = "lqr_current",
                 we_op: float = _DB42S02.WE_OP,
                 v_max: float = _DB42S02.V_max) -> None:
        super().__init__(name)
        self.v_max       = float(v_max)
        self.output_label = "[v_alpha,v_beta]"
        self.is_dynamic   = False
        self.vector_size  = 2
        # Integrator states
        self._xi_d        = 0.0
        self._xi_q        = 0.0
        self._theta_e_ref = 0.0   # accumulated reference position
        # Solve LQR at operating point
        self._K = _solve_lqr(we_op)
        print(f"  LQR gain K =\n{self._K}")

    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        id_m = iq_m = theta_e = omega_m = iq_ref = theta_e_ref = 0.0

        if input_values:
            if input_values[0] is not None:
                v = input_values[0].value
                if len(v) >= 4:
                    id_m, iq_m, theta_e, omega_m = (
                        float(v[0]), float(v[1]), float(v[2]), float(v[3]))
            if len(input_values) > 1 and input_values[1] is not None:
                iq_ref = float(input_values[1].value[0])

        id_ref = 0.0   # MTPA

        # theta_e_ref: integrate omega_ref * p over time
        # We use omega_m as proxy — the reference trajectory is implicit
        # (LQR drives theta_e error to zero via the position state)
        # theta_e_ref = 0 means we penalise absolute theta_e deviation
        # which is correct since the LQR was designed around the linearised
        # operating point — position error = theta_e - theta_e_ref
        # Here we track theta_e_ref accumulated from iq-driven torque
        we_ref = iq_ref * (1.5 * float(_DB42S02.p) * _DB42S02.lam_pm / _DB42S02.J) * dt
        self._theta_e_ref += we_ref * float(_DB42S02.p)
        # Wrap both to [0, 2pi) and compute shortest-path error
        e_theta = theta_e - self._theta_e_ref
        # Wrap to [-pi, pi]
        e_theta = (e_theta + math.pi) % (2.0 * math.pi) - math.pi

        # Error states
        id_err = id_m - id_ref
        iq_err = iq_m - iq_ref

        # Integrators with anti-windup
        xi_lim = self.v_max * _DB42S02.L_d / _DB42S02.R_s   # ~5.6 A·s
        self._xi_d = float(np.clip(self._xi_d + id_err * dt, -xi_lim, xi_lim))
        self._xi_q = float(np.clip(self._xi_q + iq_err * dt, -xi_lim, xi_lim))

        # LQR state vector: [id_err, iq_err, theta_e_err, xi_d, xi_q]
        x = np.array([id_err, iq_err, e_theta, self._xi_d, self._xi_q], dtype=float)
        u = -self._K @ x   # [vd, vq]  — K is (2×5)
        vd = float(u[0])
        vq = float(u[1])

        # Feedforward decoupling — cancel back-EMF cross-coupling
        we  = omega_m * float(_DB42S02.p)
        vd += -we * _DB42S02.L_q * iq_m
        vq +=  we * (_DB42S02.L_d * id_m + _DB42S02.lam_pm)

        # Circular voltage limit
        v_norm = math.sqrt(vd**2 + vq**2)
        if v_norm > self.v_max:
            scale = self.v_max / v_norm
            vd   *= scale
            vq   *= scale

        # Inverse Park: dq → αβ
        c, s    = math.cos(theta_e), math.sin(theta_e)
        v_alpha = vd * c - vq * s
        v_beta  = vd * s + vq * c

        self.output = VectorSignal(
            np.array([v_alpha, v_beta], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)

    def reset(self):
        super().reset()
        self._xi_d        = 0.0
        self._xi_q        = 0.0
        self._theta_e_ref = 0.0


# ==============================================================================
#  Block: InvClarkeVoltageBlock
# ==============================================================================
class InvClarkeVoltageBlock(VectorBlock):
    """
    Inverse Clarke:  [v_alpha, v_beta] → [va, vb, vc]

    port 0: [v_alpha, v_beta]
    output: [va, vb, vc]
    """
    def __init__(self, name: str = "inv_clarke") -> None:
        super().__init__(name)
        self.output_label = "[va,vb,vc]"
        self.is_dynamic   = False
        self.vector_size  = 3

    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        alpha = beta = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 2:
                alpha, beta = float(v[0]), float(v[1])
        _sq3_2 = math.sqrt(3.0) / 2.0
        va =  alpha
        vb = -0.5 * alpha + _sq3_2 * beta
        vc = -0.5 * alpha - _sq3_2 * beta
        self.output = VectorSignal(
            np.array([va, vb, vc], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# ==============================================================================
#  Build & run simulation
# ==============================================================================
def main() -> None:
    print("=" * 65)
    print("  EmbedSim — PMSM FOC with LQR  |  DB42S02")
    print("=" * 65)

    DT       = 50e-6     # s   20 kHz
    T_SIM    = 2.0       # s   — enough to see full acceleration to 400 RPM
    RPM_REF  = 400.0     # target speed

    # ── Blocks ─────────────────────────────────────────────────────────────────
    omega_ref  = VectorStep("omega_ref",
                             step_time=0.05,
                             before_value=0.0,
                             after_value=float(RPM_REF))

    spd_pack   = SpeedRefPackerBlock("spd_pack")
    pi_speed   = PISpeedBlock("pi_speed")
    fb_unpack  = FeedbackUnpackBlock("fb_unpack")
    lqr_cur    = LQRCurrentBlock("lqr_cur")
    inv_clarke = InvClarkeVoltageBlock("inv_clarke")
    motor      = DB42S02PlantBlock("motor")
    # One-step delay breaks the algebraic loop: motor → delay → controllers → motor
    motor_dly  = VectorDelay("motor_dly", initial=[0.0]*8)   # [rpm,ia,ib,ic,theta_m,Tem,id,iq]
    sink       = VectorEnd("sink")

    # ── Wiring ──────────────────────────────────────────────────────────────────
    #  Speed packer receives reference and delayed motor feedback
    omega_ref  >> spd_pack
    motor_dly  >> spd_pack      # port 1: motor rpm feedback (delayed 1 step)

    spd_pack   >> pi_speed      # [omega_ref, omega_m] → iq_ref

    motor_dly  >> fb_unpack     # [rpm,ia,ib,ic,theta_m,Tem,id,iq] → [id,iq,theta_e,omega_m]

    fb_unpack  >> lqr_cur       # port 0: dq state + theta_e
    pi_speed   >> lqr_cur       # port 1: iq_ref

    lqr_cur    >> inv_clarke    # [v_alpha, v_beta] → [va, vb, vc]
    inv_clarke >> motor         # [va, vb, vc] → plant
    motor      >> motor_dly     # close feedback loop — VectorDelay holds t-1 value
    motor      >> sink

    # ── Simulation ─────────────────────────────────────────────────────────────
    sim = EmbedSim(sinks=[sink], T=T_SIM, dt=DT, solver=ODESolver.EULER)

    print(f"\n  Target: {RPM_REF:.0f} RPM  |  dt={DT*1e6:.0f} µs  |  T={T_SIM:.1f}s")
    print()
    sim.topo.print_console()

    # motor output: [0]=rpm [1]=ia [2]=ib [3]=ic [4]=theta_m [5]=T_em [6]=id [7]=iq
    sim.scope.add(motor,     indices=[0, 1, 2, 3, 4],  label="Motor")
    sim.scope.add(omega_ref, indices=[0],                label="SpeedRef")
    sim.scope.add(pi_speed,  indices=[0],                label="PISpeed")

    print(f"\nRunning simulation ({T_SIM}s @ {1/DT:.0f} Hz)...")
    sim.run()
    print(f"  Completed: {len(sim.scope.t)} steps")

    # ── Extract signals ────────────────────────────────────────────────────────
    sc  = sim.scope
    t   = np.array(sc.t, dtype=np.float32)

    def _get(label, pos):
        s = sc.get_signal(label, pos)
        return s if s is not None else np.zeros(len(t), dtype=np.float32)

    rpm_meas  = _get("Motor", 0)
    ia        = _get("Motor", 1)
    ib        = _get("Motor", 2)
    ic        = _get("Motor", 3)
    theta_m   = _get("Motor", 4)          # mechanical angle [rad] — accumulated
    rpm_ref   = _get("SpeedRef", 0)
    iq_ref    = _get("PISpeed",  0)

    # Electrical angle and reference position
    theta_e_meas = (theta_m * float(_DB42S02.p)) % (2.0 * math.pi)
    # Reference theta_e: integrate omega_ref (RPM → rad/s → electrical)
    omega_ref_rads = rpm_ref * math.pi / 30.0
    theta_e_ref    = np.cumsum(omega_ref_rads * DT) * float(_DB42S02.p)
    theta_e_ref    = theta_e_ref % (2.0 * math.pi)

    # ── Plot: 4 subplots ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle("EmbedSim — PMSM FOC LQR  |  DB42S02  400 RPM\n"
                 "Position-weighted LQR  |  state=[id, iq, θe, ξd, ξq]",
                 fontsize=13, fontweight="bold")

    # 1. Speed
    ax = axes[0]
    ax.plot(t, rpm_ref,  "k--", lw=1.2, label="ω_ref  [RPM]")
    ax.plot(t, rpm_meas, "C0",  lw=1.8, label="ω_meas [RPM]")
    ax.set_ylabel("Speed  [RPM]", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.35)

    # 2. Three-phase currents ia, ib, ic — full trace
    ax = axes[1]
    ax.plot(t, ia, "C3", lw=0.8, label="ia  [A]")
    ax.plot(t, ib, "C2", lw=0.8, label="ib  [A]")
    ax.plot(t, ic, "C0", lw=0.8, label="ic  [A]")
    ax.set_ylabel("Phase current  [A]", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.35)

    # 3. Zoomed abc currents — last 20 ms to see sinusoidal shape clearly
    t_zoom_start = max(t[-1] - 0.02, t[0])
    mask = t >= t_zoom_start
    ax = axes[2]
    ax.plot(t[mask], ia[mask], "C3", lw=1.4, label="ia  [A]")
    ax.plot(t[mask], ib[mask], "C2", lw=1.4, label="ib  [A]")
    ax.plot(t[mask], ic[mask], "C0", lw=1.4, label="ic  [A]")
    ax.set_ylabel("abc currents — last 20 ms  [A]", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.35)

    # 4. Position: theta_e actual vs reference (electrical angle)
    ax = axes[3]
    ax.plot(t, theta_e_ref,  "k--", lw=1.0, alpha=0.7, label="θe_ref  [rad]")
    ax.plot(t, theta_e_meas, "C4",  lw=1.2, label="θe_meas [rad]")
    ax.set_ylabel("Electrical angle  [rad]", fontsize=12)
    ax.set_xlabel("Time  [s]", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.35)

    plt.tight_layout()
    out_png = _HERE / "pmsm_lqr_foc_results.png"
    plt.savefig(str(out_png), dpi=150)
    print(f"\n💾  Saved: {out_png}")
    print("=" * 65)


if __name__ == "__main__":
    main()
