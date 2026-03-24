"""
pmsm_lqr_foc.py
===============================================================================
EmbedSim — PMSM FOC  |  LQR Speed Controller  |  DB42S02  |  AURIX TC3xx
===============================================================================

Aligned to:
  Xu, Wen-Jun (2012) "Permanent Magnet Synchronous Motor with Linear Quadratic
  Speed Controller", Energy Procedia 14, pp 364-369.

State model (paper eq. 12-14):
  id = 0 enforced (field-oriented, id=0 method, paper §2)
  x  = [iq,  ω_r,  θ_r]           (3 states, paper eq. 13)
  u  = vq                           (scalar, paper eq. 13)
  w  = T_L                          (disturbance)

  A = | -R/L      λP/L    0 |       (paper eq. 14)
      | 1.5Pλ/J  -D/J     0 |
      |  0         1      0 |

  B = [1/L,  0,  0]^T
  E = [0, -1/J, 0]^T
  C = [0, 1, 0]   → output is ω_r

LQR design (paper §3):
  J = ∫ (x^T Q x + u^T R u) dt
  Q = diag(100, 1, 1)   R = 1          (paper §3)
  → K = [7.9117, 0.7249, 1.0000]       (paper §3, reproduced via scipy CARE)

Control law:
  u = -K·x_err   where x_err = x - x_ref
  x_ref = [0, ω_ref, θ_integrated]

Output chain:
  vq (LQR)  → Inverse Park → v_alpha, v_beta
  vd = 0    (id = 0 strategy)
  v_alpha, v_beta → SVPWM → [da, db, dc]  (duty cycles 0..1)
  da, db, dc → SWPWMBlock → [sw_a, sw_b, sw_c]  (0/1 switch states)

Motor: NANOTEC DB42S02
  Rs=0.19 Ω  Ld=Lq=125 µH  λpm=0.0014 Wb  p=4  J=2.4e-6 kg·m²
  B=7e-5 N·m·s/rad  Vdc=17 V

Note on paper parameters vs DB42S02:
  The paper uses R=2.875 Ω, L=8.5 mH, P=2, λ=0.175 Wb.
  The DB42S02 parameters above are substituted — the LQR is re-solved
  numerically for those values; the gain structure is identical to the paper.
"""

from __future__ import annotations
import math
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# ── Path bootstrap ──────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE
for _c in (_HERE, _HERE.parent, _HERE.parent.parent):
    if (_c / "embedsim").is_dir():
        _ROOT = _c
        break
for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_FS = _ROOT / "fs_electrical_machines"
if _FS.is_dir() and str(_FS) not in sys.path:
    sys.path.insert(0, str(_FS))

from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep
from embedsim.dynamic_blocks import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver, VectorDelay
from fs_electrical_machines.pmsm_python_plant import PMSM_Python_Plant


# ==============================================================================
#  Motor constants — NANOTEC DB42S02
# ==============================================================================
class _DB42S02:
    p       = 4
    R_s     = 0.19
    L_d     = 0.125e-3
    L_q     = 0.125e-3
    lam_pm  = 0.0014
    J       = 2.4e-6
    B_fric  = 7e-5          # damping D [N·m·s/rad]
    I_max   = 3.57
    V_dc    = 17.0
    V_max   = 17.0 / math.sqrt(3.0)   # phase-voltage limit
    KT      = 1.5 * p * lam_pm        # 0.0084 N·m/A


# ==============================================================================
#  Paper §2: Linearised State-Space (3-state, id=0 enforced)
# ==============================================================================
#
#  x = [iq, ω_r, θ_r]^T     u = vq     w = T_L
#
#  A = | -R/L      λP/L    0 |
#      | 1.5Pλ/J  -D/J     0 |
#      |  0         1      0 |
#
#  B = [1/L, 0, 0]^T
#  E = [0,  -1/J, 0]^T
#  C = [0, 1, 0]
# ==============================================================================

def build_paper_matrices(m=_DB42S02):
    """Build A, B, E, C matrices exactly as in Xu (2012) eq. (14)."""
    L   = m.L_q          # L_d = L_q = L  (surface-mount, paper assumption)
    R   = m.R_s
    P   = float(m.p)
    lam = m.lam_pm
    J   = m.J
    D   = m.B_fric

    A = np.array([
        [-R / L,          lam * P / L,   0.0],   # diq/dt
        [1.5 * P * lam / J, -D / J,      0.0],   # dω/dt
        [0.0,             1.0,           0.0],   # dθ/dt
    ], dtype=float)

    B = np.array([[1.0 / L], [0.0], [0.0]], dtype=float)  # (3×1)

    E = np.array([[0.0], [-1.0 / J], [0.0]], dtype=float)  # disturbance

    C = np.array([[0.0, 1.0, 0.0]], dtype=float)           # output = ω_r

    return A, B, E, C


# ==============================================================================
#  Paper §3: LQR Design
#  Q = diag(100, 1, 1)   R = 1   →   K = [7.9117, 0.7249, 1.0000]
# ==============================================================================

def solve_lqr(A, B, Q, R_scalar):
    """
    Solve continuous-time CARE:  A^T P + P A - P B R^{-1} B^T P + Q = 0
    Returns feedback gain K = R^{-1} B^T P   (paper eq. 16-17)
    """
    R_mat = np.atleast_2d(float(R_scalar))
    try:
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R_mat)
        K = np.linalg.solve(R_mat, B.T @ P)
        return K, P
    except Exception as exc:
        print(f"  [LQR] scipy CARE failed ({exc}), using iterative method")
        return _lqr_iterative(A, B, Q, R_mat)


def _lqr_iterative(A, B, Q, R_mat, dt=50e-6, max_iter=20000):
    """Forward-Euler iterative solution of discrete ARE as fallback."""
    n = A.shape[0]
    Ad = np.eye(n) + A * dt
    Bd = B * dt
    P  = Q.copy()
    for _ in range(max_iter):
        K_tmp = np.linalg.solve(R_mat + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        P_new = Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K_tmp + Q
        if np.max(np.abs(P_new - P)) < 1e-8:
            break
        P = P_new
    K = np.linalg.solve(R_mat + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


class LQRDesign:
    """
    LQR design aligned to Xu (2012) §3.

    State: x = [iq, ω_r, θ_r]
    Q = diag(100, 1, 1)    (paper: "Q = diag[100 1 1]")
    R = 1                   (paper: "R = 1")
    Paper result: K = [7.9117, 0.7249, 1.0000]
    """

    Q_PAPER = np.diag([100.0, 1.0, 1.0])
    R_PAPER = 1.0

    def __init__(self):
        self.A, self.B, self.E, self.C = build_paper_matrices()
        self.K, self.P = solve_lqr(self.A, self.B, self.Q_PAPER, self.R_PAPER)

        # Closed-loop eigenvalues
        A_cl = self.A - self.B @ self.K
        self.eigs = np.linalg.eigvals(A_cl)

    def print_summary(self):
        print("=" * 62)
        print("  LQR Design — aligned to Xu (2012)")
        print("  Q = diag(100, 1, 1)   R = 1")
        np.set_printoptions(precision=4, suppress=True)
        print(f"  K (paper)  = [7.9117,  0.7249,  1.0000]")
        print(f"  K (solved) = {np.round(self.K.flatten(), 4)}")
        print("  Closed-loop eigenvalues:")
        for i, ev in enumerate(self.eigs):
            bw = abs(ev.real) / (2 * math.pi)
            sym = "✅" if ev.real < -1e-6 else "⚠️"
            print(f"    λ{i+1} = {ev.real:+.2f}{ev.imag:+.2f}j   |BW|≈{bw:.0f} Hz  {sym}")
        print("=" * 62)


# ==============================================================================
#  SVPWM Block  (Space-Vector PWM)
#  Input:  [v_alpha, v_beta]   (stationary-frame voltages)
#  Output: [da, db, dc]        (duty cycles  0 … 1)
# ==============================================================================

class SVPWMBlock(VectorBlock):
    """
    Space-Vector PWM modulator.

    Converts stationary-frame voltage references to three duty cycles
    using the standard 7-segment SVPWM algorithm.

    Input:  VectorSignal([v_alpha, v_beta])
    Output: VectorSignal([da, db, dc])   duty cycles ∈ [0, 1]
    """

    def __init__(self, name: str = "svpwm", v_dc: float = _DB42S02.V_dc):
        super().__init__(name)
        self.v_dc = float(v_dc)
        self.vector_size = 3
        self.output_label = "[da,db,dc]"
        self.is_dynamic = False

    @staticmethod
    def _svpwm(v_alpha: float, v_beta: float, v_dc: float):
        """
        Standard 7-segment SVPWM.
        Returns duty cycles (da, db, dc) in [0, 1].

        Reference voltages are normalised to the hexagon:
            V_ref_max = (2/3) * v_dc  for linear modulation
        Phase voltages:
            va =  v_alpha
            vb = -v_alpha/2 + √3/2 · v_beta
            vc = -v_alpha/2 - √3/2 · v_beta
        Sector detection and dwell times follow standard SVM.
        """
        inv_vdc = 1.0 / max(v_dc, 1e-9)
        s3h = math.sqrt(3.0) * 0.5

        # Inverse Clarke → three reference voltages (un-normalised)
        va =  v_alpha
        vb = -0.5 * v_alpha + s3h * v_beta
        vc = -0.5 * v_alpha - s3h * v_beta

        # Normalise to [−1, +1] relative to half DC link
        va_n = va * inv_vdc
        vb_n = vb * inv_vdc
        vc_n = vc * inv_vdc

        # Clamp to avoid over-modulation (optional; keeps linear range)
        v_max = max(va_n, vb_n, vc_n)
        v_min = min(va_n, vb_n, vc_n)
        v_ofs = -0.5 * (v_max + v_min)   # midpoint injection (SVPWM offset)

        da = 0.5 + va_n + v_ofs
        db = 0.5 + vb_n + v_ofs
        dc = 0.5 + vc_n + v_ofs

        # Hard clamp [0, 1]
        da = max(0.0, min(1.0, da))
        db = max(0.0, min(1.0, db))
        dc = max(0.0, min(1.0, dc))

        return da, db, dc

    def compute_py(self, t, dt, input_values=None):
        v_alpha = v_beta = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 2:
                v_alpha, v_beta = float(v[0]), float(v[1])

        da, db, dc = self._svpwm(v_alpha, v_beta, self.v_dc)
        self.output = VectorSignal(
            np.array([da, db, dc], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# ==============================================================================
#  SW_PWM Block  (Duty-cycle → binary switch states)
#  Input:  [da, db, dc]   duty cycles  0 … 1
#  Output: [sw_a, sw_b, sw_c]   0 or 1  (comparator against sawtooth carrier)
# ==============================================================================

class SWPWMBlock(VectorBlock):
    """
    Switching PWM comparator block.

    Compares three duty-cycle references against a triangular/sawtooth
    carrier at the switching frequency to produce binary gate signals.

    Input:  VectorSignal([da, db, dc])   duty cycles ∈ [0, 1]
    Output: VectorSignal([sw_a, sw_b, sw_c])   float32 {0.0, 1.0}

    The carrier counter increments by dt/T_sw each step, wrapping 0→1.
    sw_x = 1.0  if  d_x > carrier,  else 0.0
    """

    def __init__(self, name: str = "sw_pwm", f_sw: float = 20e3):
        super().__init__(name)
        self.f_sw   = float(f_sw)
        self.T_sw   = 1.0 / self.f_sw
        self._carrier: float = 0.0       # sawtooth phase ∈ [0, 1)
        self.vector_size  = 3
        self.output_label = "[sw_a,sw_b,sw_c]"
        self.is_dynamic   = False

    def compute_py(self, t, dt, input_values=None):
        da = db = dc = 0.5
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                da, db, dc = float(v[0]), float(v[1]), float(v[2])

        # Advance sawtooth carrier
        self._carrier = (self._carrier + dt / self.T_sw) % 1.0
        c = self._carrier

        sw_a = 1.0 if da > c else 0.0
        sw_b = 1.0 if db > c else 0.0
        sw_c = 1.0 if dc > c else 0.0

        self.output = VectorSignal(
            np.array([sw_a, sw_b, sw_c], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)

    def reset(self):
        super().reset()
        self._carrier = 0.0


# ==============================================================================
#  LQR Controller Block — aligned to paper
# ==============================================================================
class LQRBlock(VectorBlock):
    """
    LQR speed controller aligned to Xu (2012) §3.

    State vector (paper eq. 13):  x = [iq, ω_r, θ_r]
    Control law:  vq = -K · x_err        (paper eq. 18)
                  vd = 0                  (id = 0 strategy, paper §2)

    The error state includes integrator action on θ:
        x_err[0] = iq      - 0           (current error, i_d=0 so iq is free)
        x_err[1] = ω_r     - ω_ref       (speed error)
        x_err[2] = θ_r     - θ_ref       (angle error — provides integral action
                                          via state feedback on position)

    Inputs:
      port 0: motor state bus  [rpm, ia, ib, ic, theta_m, Tem, id, iq]
              (from DB42S02PlantBlock, same bus as before)
      port 1: omega_ref [RPM]

    Output: VectorSignal([v_alpha, v_beta])
    """

    C_CODEGEN_EXCLUDE = True

    def __init__(self, name: str = "lqr", v_max: float = _DB42S02.V_max):
        super().__init__(name)
        self.v_max = float(v_max)
        self.vector_size  = 2
        self.output_label = "[v_alpha,v_beta]"
        self.is_dynamic   = False

        # ── Solve LQR exactly per paper ─────────────────────────────────────
        design = LQRDesign()
        design.print_summary()
        # K shape (1×3): [k_iq, k_omega, k_theta]
        self.K = design.K.flatten()   # length-3 vector

        # Integrating theta reference  (θ_ref = ∫ ω_ref dt)
        self._theta_ref: float = 0.0
        self._omega_ref_prev: float = 0.0

    def compute_py(self, t, dt, input_values=None):
        # ── Parse inputs ────────────────────────────────────────────────────
        rpm_meas = iq_meas = theta_m = 0.0
        omega_ref_rpm = 0.0

        if input_values:
            if input_values[0] is not None:
                bus = input_values[0].value
                # bus = [rpm, ia, ib, ic, theta_m, Tem, id, iq]
                if len(bus) >= 8:
                    rpm_meas = float(bus[0])
                    theta_m  = float(bus[4])
                    iq_meas  = float(bus[7])

            if len(input_values) > 1 and input_values[1] is not None:
                omega_ref_rpm = float(input_values[1].value[0])

        omega_m   = rpm_meas * math.pi / 30.0          # [rad/s]
        omega_ref = omega_ref_rpm * math.pi / 30.0     # [rad/s]
        theta_e   = float(_DB42S02.p) * theta_m        # electrical angle

        # ── Integrate theta_ref = ∫ ω_ref dt  (Tustin) ──────────────────────
        self._theta_ref += 0.5 * dt * (omega_ref + self._omega_ref_prev)
        self._omega_ref_prev = omega_ref

        # ── Error state (paper eq. 13, id=0 so iq is the only current state)
        #   x_err = [iq - 0,  ω_r - ω_ref,  θ_r - θ_ref]
        # ────────────────────────────────────────────────────────────────────
        x_err = np.array([
            iq_meas,                          # iq error (ref = 0 for id=0)
            omega_m - omega_ref,              # speed error
            theta_m - self._theta_ref,        # position error (integral action)
        ], dtype=float)

        # ── LQR control law: u = -K · x_err   (paper eq. 18) ───────────────
        vq = float(-self.K @ x_err)

        # vd = 0  (id = 0 enforced, paper §2)
        vd = 0.0

        # ── Feedforward decoupling (standard FOC, not in paper but improves
        #    tracking — can be disabled by setting lam_pm = 0 mentally)
        we  = float(_DB42S02.p) * omega_m
        vd += -we * _DB42S02.L_q * 0.0            # id = 0  → no cross term
        vq += we * _DB42S02.lam_pm                 # back-EMF compensation

        # ── Voltage limit ───────────────────────────────────────────────────
        v_norm = math.sqrt(vd ** 2 + vq ** 2)
        if v_norm > self.v_max and v_norm > 0.0:
            scale = self.v_max / v_norm
            vd *= scale
            vq *= scale

        # ── Inverse Park: dq → αβ  ──────────────────────────────────────────
        c = math.cos(theta_e)
        s = math.sin(theta_e)
        v_alpha = vd * c - vq * s
        v_beta  = vd * s + vq * c

        self.output = VectorSignal(
            np.array([v_alpha, v_beta], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)

    def reset(self):
        super().reset()
        self._theta_ref      = 0.0
        self._omega_ref_prev = 0.0


# ==============================================================================
#  Plant Block — DB42S02
# ==============================================================================
class DB42S02PlantBlock(PMSM_Python_Plant):
    """
    DB42S02 PMSM plant.
    Input:  [sw_a, sw_b, sw_c]   switch states {0,1}
    Output: [rpm, ia, ib, ic, theta_m, Tem, id, iq]
    """
    TOPO_CATEGORY = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label = "[rpm,ia,ib,ic,theta_m,Tem,id,iq]"

    def __init__(self, name: str = "motor"):
        super().__init__(
            name=name,
            R=_DB42S02.R_s, L_d=_DB42S02.L_d, L_q=_DB42S02.L_q,
            lambda_pm=_DB42S02.lam_pm, J=_DB42S02.J, B_fric=_DB42S02.B_fric,
            p=float(_DB42S02.p), v_dc=_DB42S02.V_dc)

    def compute_py(self, t, dt, input_values=None):
        """
        Accepts [sw_a, sw_b, sw_c] switch states.
        Reconstructs phase voltages via ideal VSI model then calls parent.
        """
        sw_a = sw_b = sw_c = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                sw_a, sw_b, sw_c = float(v[0]), float(v[1]), float(v[2])

        vdc = _DB42S02.V_dc
        # Ideal 2-level VSI: phase voltage = (sw - 1/3*(sw_a+sw_b+sw_c)) * Vdc
        s_avg = (sw_a + sw_b + sw_c) / 3.0
        va = (sw_a - s_avg) * vdc
        vb = (sw_b - s_avg) * vdc
        vc = (sw_c - s_avg) * vdc

        # Convert to duty cycles for parent block  (parent expects [da,db,dc,vdc,0])
        ta = np.clip(va / vdc + 0.5, 0.0, 1.0)
        tb = np.clip(vb / vdc + 0.5, 0.0, 1.0)
        tc = np.clip(vc / vdc + 0.5, 0.0, 1.0)

        return super().compute_py(t, dt, [VectorSignal(
            np.array([ta, tb, tc, vdc, 0.0], dtype=DEFAULT_DTYPE))])

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# ==============================================================================
#  Main simulation
# ==============================================================================
def main():
    print("=" * 65)
    print("  EmbedSim — LQR FOC  |  DB42S02  |  Xu (2012) aligned")
    print("=" * 65)

    DT      = 50e-6          # 50 µs  (20 kHz)
    T_SIM   = 2.0            # 2 s
    RPM_REF = 400.0          # target speed [RPM]
    F_SW    = 20e3           # switching frequency [Hz]

    # ── Blocks ──────────────────────────────────────────────────────────────
    omega_ref = VectorStep("omega_ref",
                           step_time=0.05,
                           before_value=0.0,
                           after_value=float(RPM_REF))

    motor   = DB42S02PlantBlock("motor")
    lqr     = LQRBlock("lqr")
    svpwm   = SVPWMBlock("svpwm",  v_dc=_DB42S02.V_dc)
    sw_pwm  = SWPWMBlock("sw_pwm", f_sw=F_SW)

    motor_dly = VectorDelay("motor_dly", initial=[0.0] * 8)
    sink      = VectorEnd("sink")

    # ── Wiring ───────────────────────────────────────────────────────────────
    #
    #  omega_ref ──────────────────────────────────► lqr [port 1]
    #  motor ──► motor_dly ──► lqr [port 0]
    #                      └──► sink
    #  lqr ──► svpwm ──► sw_pwm ──► motor
    #
    motor     >> motor_dly
    motor_dly >> lqr        # port 0: motor state bus
    omega_ref >> lqr        # port 1: speed reference [RPM]
    motor_dly >> sink

    lqr    >> svpwm          # [v_alpha, v_beta]
    svpwm  >> sw_pwm         # [da, db, dc]  → duty cycles
    sw_pwm >> motor          # [sw_a, sw_b, sw_c] → switch states

    # ── Simulation ───────────────────────────────────────────────────────────
    sim = EmbedSim(sinks=[sink], T=T_SIM, dt=DT, solver=ODESolver.EULER)

    print(f"\n  Target: {RPM_REF:.0f} RPM  |  dt={DT*1e6:.0f} µs  |"
          f"  T={T_SIM:.1f} s  |  f_sw={F_SW/1e3:.0f} kHz")
    sim.topo.print_console()

    sim.scope.add(motor,     indices=[0, 1, 2, 3, 4, 6, 7], label="Motor")
    sim.scope.add(omega_ref, indices=[0],                     label="SpeedRef")
    sim.scope.add(svpwm,     indices=[0, 1, 2],               label="Duty")
    sim.scope.add(sw_pwm,    indices=[0, 1, 2],               label="SW")

    print(f"\nRunning ({T_SIM} s @ {1/DT:.0f} Hz)…")
    sim.run()
    print(f"  Steps: {len(sim.scope.t)}")

    # ── Results ──────────────────────────────────────────────────────────────
    sc = sim.scope
    t  = np.array(sc.t, dtype=np.float32)

    def _g(lbl, pos):
        s = sc.get_signal(lbl, pos)
        return s if s is not None else np.zeros(len(t), dtype=np.float32)

    rpm_meas = _g("Motor",    0)
    ia       = _g("Motor",    1)
    ib       = _g("Motor",    2)
    ic       = _g("Motor",    3)
    id_meas  = _g("Motor",    5)
    iq_meas  = _g("Motor",    6)
    rpm_ref  = _g("SpeedRef", 0)
    da       = _g("Duty",     0)
    db       = _g("Duty",     1)
    dc       = _g("Duty",     2)
    sw_a     = _g("SW",       0)
    sw_b     = _g("SW",       1)
    sw_c     = _g("SW",       2)

    # ── Performance metrics ──────────────────────────────────────────────────
    i_settle = int(0.5 * len(t))
    settling_time = t[-1]
    for i in range(len(t)):
        if abs(rpm_meas[i] - RPM_REF) < 0.02 * RPM_REF:
            end = min(i + 200, len(t))
            if np.all(np.abs(rpm_meas[i:end] - RPM_REF) < 0.02 * RPM_REF):
                settling_time = t[i]
                break

    overshoot = max(0.0, (float(rpm_meas.max()) - RPM_REF) / RPM_REF * 100.0)
    ss_error  = float(np.mean(np.abs(rpm_meas[i_settle:] - RPM_REF)))

    print(f"\n  ── Performance (Xu 2012 alignment check) ──")
    print(f"  Settling time (2%%): {settling_time:.4f} s  (paper: 0.0075 s)")
    print(f"  Overshoot:          {overshoot:.1f}%%         (paper: 10%%)")
    print(f"  SS speed error:     {ss_error:.2f} RPM")

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(13, 18), sharex=True)
    fig.suptitle(
        "EmbedSim — PMSM LQR FOC  |  DB42S02  |  Aligned to Xu (2012)\n"
        "x = [iq, ω, θ]   Q = diag(100,1,1)   R = 1   → K = R⁻¹B^T P",
        fontsize=13, fontweight="bold")

    # 1. Speed
    axes[0].plot(t, rpm_ref,  "k--", lw=1.3, label="ω_ref [RPM]")
    axes[0].plot(t, rpm_meas, "C0",  lw=1.6, label="ω_meas [RPM]")
    axes[0].set_ylabel("Speed [RPM]", fontsize=12)
    axes[0].legend(fontsize=10); axes[0].grid(alpha=0.35)
    axes[0].set_title(f"Settling: {settling_time:.4f} s   Overshoot: {overshoot:.1f}%   SS error: {ss_error:.2f} RPM",
                      fontsize=10)

    # 2. Phase currents
    axes[1].plot(t, ia, "C3", lw=0.7, alpha=0.7, label="ia [A]")
    axes[1].plot(t, ib, "C2", lw=0.7, alpha=0.7, label="ib [A]")
    axes[1].plot(t, ic, "C0", lw=0.7, alpha=0.7, label="ic [A]")
    axes[1].set_ylabel("Phase currents [A]", fontsize=12)
    axes[1].legend(fontsize=10); axes[1].grid(alpha=0.35)

    # 3. dq currents (id should stay ~0 confirming paper assumption)
    axes[2].plot(t, id_meas, "C3", lw=1.2, label="id [A]  (should ≈ 0)")
    axes[2].plot(t, iq_meas, "C2", lw=1.2, label="iq [A]  (torque current)")
    axes[2].axhline(0, color="k", lw=0.6, alpha=0.4)
    axes[2].set_ylabel("dq currents [A]", fontsize=12)
    axes[2].legend(fontsize=10); axes[2].grid(alpha=0.35)

    # 4. Duty cycles (SVPWM output)
    axes[3].plot(t, da, "C0", lw=0.8, alpha=0.8, label="da")
    axes[3].plot(t, db, "C1", lw=0.8, alpha=0.8, label="db")
    axes[3].plot(t, dc, "C2", lw=0.8, alpha=0.8, label="dc")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_ylabel("SVPWM duty [0…1]", fontsize=12)
    axes[3].legend(fontsize=10); axes[3].grid(alpha=0.35)

    # 5. Switch states (sw_pwm output)  — plot a short window
    t_zoom = min(0.01, t[-1])        # 10 ms zoom to see switching
    mask   = t <= t_zoom
    axes[4].step(t[mask], sw_a[mask], "C3", lw=1.0, where="post", label="sw_a")
    axes[4].step(t[mask], sw_b[mask] + 1.1, "C2", lw=1.0, where="post", label="sw_b (+1.1)")
    axes[4].step(t[mask], sw_c[mask] + 2.2, "C0", lw=1.0, where="post", label="sw_c (+2.2)")
    axes[4].set_ylabel("sw_pwm {0,1}", fontsize=12)
    axes[4].set_xlabel(f"Time [s]  (first {t_zoom*1e3:.0f} ms)", fontsize=12)
    axes[4].legend(fontsize=10); axes[4].grid(alpha=0.35)

    plt.tight_layout()
    out = _HERE / "pmsm_lqr_foc_results.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out}")
    print("=" * 65)


if __name__ == "__main__":
    main()