"""
mpc_controller_block.py
========================
Model Predictive Control FOC Controller block — NANOTEC DB42S02.

EmbedSim VectorBlock wrapper for the C implementation in:
    embed_sim_mpc_controller.c / embed_sim_mpc_controller.h

ARCHITECTURE
============
True 3-state receding-horizon MPC.  At each ISR tick the controller:

    1. Estimates mechanical speed from encoder angle (IIR finite-difference).
    2. Runs the SMO back-EMF observer (αβ frame, tanh sliding surface).
    3. Performs Clarke abc→αβ and Park αβ→dq transforms.
    4. Projects the d-axis BEMF estimate through Park for feedforward.
    5. Clamps BEMF to the physical maximum (prevents SMO saturation artefacts).
    6. Solves the analytical MPC problem in closed form (O(N), no iteration).
    7. Applies soft-start vq limit and speed-error integral correction.
    8. Performs inverse Park dq→αβ and normalises for SVPWM.

STATE VECTOR / COST FUNCTION
==============================
    x = [id, iq, omega_m]
    u = [vd, vq]

    J = Σ_{k=1}^{N} [ Q_id   · id_k²
                     + Q_iq   · iq_k²
                     + Q_omega · (omega_k − omega_ref)²
                     + R_vd   · vd²  +  R_vq · vq² ]

ANALYTICAL CLOSED-FORM SOLUTION (O(N), no iteration)
======================================================
Free-run trajectory (u = 0, BEMF handled separately by feedforward):

    id_free(k+1) = a·id_free + dt·ωe·iq_free        (cross-coupling)
    iq_free(k+1) = a·iq_free − dt·ωe·id_free        (cross-coupling)
    ω_free(k+1)  = ω_free   + (dt/J)·(KT·iq_free − B·ω_free)

    a  = 1 − dt·R_S/L                               (current decay factor)
    b  = dt/L                                        (voltage-to-current gain)

Step-response coefficients for unit vq input:
    bk  = bk·a + b                                   (iq response at step k)
    ek += (dt/J)·KT·bk                               (ω  response: running sum)

Analytical optimal inputs (gradient of J set to zero):
    vd_mpc = Q_id·Σbk·(0 − id_free)   / (Q_id·Σbk² + R_vd)
    vq_mpc = (Q_omega·Σek·(ωref−ω_free) + Q_iq·Σbk·(0−iq_free))
             / (Q_omega·Σek² + Q_iq·Σbk² + R_vq)

BEMF feedforward (exact cancellation at every step):
    vd = vd_mpc + ed_hat         vq = vq_mpc + eq_hat

BEMF clamp (prevents SMO saturation artefacts):
    |ed_hat|, |eq_hat| ≤ ωe·λ_pm  (physical maximum back-EMF)
    At startup (ωe ≈ 0) the clamp → 0 and feedforward vanishes (correct physics).

SIGNAL BUS (input_values[0], 5 elements)
=========================================
    u[0]  omega_ref_mech  [rad/s]   Mechanical speed reference
    u[1]  theta_m         [rad]     Encoder mechanical angle (accumulating)
    u[2]  ia              [A]       Phase-A current
    u[3]  ib              [A]       Phase-B current
    u[4]  ic              [A]       Phase-C current

OUTPUT (2 elements)
====================
    y[0]  v_alpha   [-]   Normalised alpha voltage for SVPWM  (÷ SVPWM_GAIN)
    y[1]  v_beta    [-]   Normalised beta voltage for SVPWM   (÷ SVPWM_GAIN)

MOTOR PARAMETERS (DELTA-winding star-equivalent, DB42S02)
==========================================================
    R_star = R_delta / 3  = 0.855 Ω / 3   = 0.285 Ω
    L_star = L_delta / 3  = 1.1025 mH / 3 = 0.3675 mH
    Confirmed during SMC/DFC hardware bring-up.  Prior values (R=0.19 Ω,
    L=0.125 mH) were incorrect star values and caused an id limit-cycle
    oscillation (b = dt/L was 3× too large).

C ALIGNMENT
===========
Every class, method, and constant in this file mirrors the corresponding
C construct by name.  Inline ``# C:`` comments give the exact C line.
All default parameter values are numerically identical to the C #defines
in embed_sim_mpc_controller.h.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from coordinate_transform_blocks import (
    ClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_HERE  = Path(__file__).resolve().parent
_C_SRC = _HERE / "c_src"


# ===========================================================================
# MPC_EncSpeed
# Mirrors: MPC_EncSpeed_T  +  MPC_EncSpeed_Update()
# ===========================================================================

class MPC_EncSpeed:
    """
    Mechanical speed estimator from encoder angle.

    Algorithm (matching MPC_EncSpeed_Update()):

        delta      = theta_m − theta_m_prev   [unwrapped to (−π, +π]]
        omega_raw  = delta / dt               [rad/s mech]
        omega_filt += IIR·(omega_raw − omega_filt)

    IIR coefficient: MPC_ENC_IIR = 0.20 → 4-sample time constant.
    Continuous angle unwrapping prevents 2π wrap-around spikes from
    corrupting the speed estimate — critical for stable MPC prediction.

    Parameters
    ----------
    iir : float
        IIR filter coefficient [-].  C: MPC_ENC_IIR = 0.20.
        Effective time constant ≈ dt·(1−iir)/iir = 200 µs at 20 kHz.
    """

    def __init__(self, iir: float = 0.20) -> None:
        self.iir = iir   # [-]  C: MPC_ENC_IIR

        # ---- Persistent state  (mirrors MPC_EncSpeed_T) --------------------
        self._theta_m_prev:      float = 0.0  # [rad]    previous encoder angle
        self._theta_m_unwrapped: float = 0.0  # [rad]    continuously unwrapped angle
        self._omega_filt:        float = 0.0  # [rad/s]  IIR-filtered speed (output)

    # -----------------------------------------------------------------------
    # update  —  C: MPC_EncSpeed_Update()
    # -----------------------------------------------------------------------
    def update(self, theta_m: float, dt: float) -> float:
        """
        Execute one encoder speed estimation step.

        Parameters
        ----------
        theta_m : float
            Encoder mechanical angle [rad] (accumulating, never wraps).
        dt : float
            Step period [s].

        Returns
        -------
        omega_m : float
            IIR-filtered mechanical speed [rad/s].

        C counterpart: MPC_EncSpeed_Update() in embed_sim_mpc_controller.c.
        """
        if dt <= 0.0:
            return self._omega_filt

        # ---- Unwrap angle delta to (−π, +π] ---------------------------------
        # C: delta = theta_m - enc->theta_m_prev;
        #    while (delta >  MPC_PI_F) delta -= MPC_TWO_PI_F;
        #    while (delta < -MPC_PI_F) delta += MPC_TWO_PI_F;
        delta: float = theta_m - self._theta_m_prev
        delta -= 2.0 * math.pi * math.floor((delta + math.pi) / (2.0 * math.pi))
        self._theta_m_unwrapped += delta

        # ---- Finite-difference speed ----------------------------------------
        # C: omega_raw = delta / dt;
        omega_raw: float = delta / dt   # [rad/s mech]

        # ---- IIR smoothing --------------------------------------------------
        # C: enc->omega_filt = (1-MPC_ENC_IIR)*enc->omega_filt + MPC_ENC_IIR*omega_raw;
        self._omega_filt = (1.0 - self.iir) * self._omega_filt + self.iir * omega_raw

        # ---- Persist state --------------------------------------------------
        self._theta_m_prev = theta_m

        return self._omega_filt

    def reset(self) -> None:
        """Reset all state to zero.  C counterpart: init block in MPC_Controller_Init()."""
        self._theta_m_prev      = 0.0
        self._theta_m_unwrapped = 0.0
        self._omega_filt        = 0.0


# ===========================================================================
# MPC_SMO
# Mirrors: MPC_SMO_T  +  MPC_SMO_Step()
# ===========================================================================

class MPC_SMO:
    """
    Sliding Mode Observer — back-EMF estimation in the stationary αβ frame.

    Estimates filtered back-EMF (e_alpha_filt, e_beta_filt) for use as
    additive disturbance feedforward in the MPC cost function.

    Algorithm (matching MPC_SMO_Step()):

        Current observer (Forward Euler):
            i_hat[k+1] = i_hat[k] + dt/L·(v − R·i_hat − K·tanh((i−i_hat)/0.01))

        Back-EMF LPF (exponential IIR at corner frequency fc):
            alpha_lpf  = wc·dt / (1 + wc·dt)     (bilinear discretisation)
            e_hat[k+1] = e_hat[k] + alpha_lpf·(K·tanh(err/0.01) − e_hat[k])

    tanh boundary layer width = 0.01 A ≈ 0.28 % of I_MAX.
    Smooth sliding eliminates chattering in the current observer.

    Parameters
    ----------
    L : float
        Stator inductance [H].  C: MPC_L = 0.3675e-3.
    R_S : float
        Stator resistance [Ω].  C: MPC_R_S = 0.285.
    k_smo : float
        Switching gain [V].  C: MPC_SMO_K = 4.68.
        Must exceed |e_max| = ωe_max·λpm = 838·0.0014 = 1.17 V.
        4× margin gives robust convergence with the tanh boundary layer.
    fc : float
        Back-EMF LPF corner frequency [Hz].  C: MPC_SMO_FC = 1000.0.
    """

    def __init__(
        self,
        L:     float = 0.3675e-3,   # [H]
        R_S:   float = 0.285,        # [Ω]   C: MPC_R_S
        k_smo: float = 4.68,         # [V]   C: MPC_SMO_K
        fc:    float = 1000.0,       # [Hz]  C: MPC_SMO_FC
    ) -> None:
        self.L     = L
        self.R_S   = R_S
        self.k_smo = k_smo
        self.fc    = fc

        # ---- Persistent state  (mirrors MPC_SMO_T) -------------------------
        self._i_alpha_hat:  float = 0.0   # [A]  estimated alpha current
        self._i_beta_hat:   float = 0.0   # [A]  estimated beta current
        self._e_alpha_filt: float = 0.0   # [V]  LPF back-EMF alpha (output)
        self._e_beta_filt:  float = 0.0   # [V]  LPF back-EMF beta  (output)
        self._alpha_lpf:    float = 0.0   # [-]  LPF coefficient (set by set_dt)

    # -----------------------------------------------------------------------
    # set_dt  —  pre-computes alpha_lpf; called once from MPCControllerBlock.__init__
    # -----------------------------------------------------------------------
    def set_dt(self, dt: float) -> None:
        """
        Pre-compute the LPF coefficient from fc and dt.

        C: smo->alpha_lpf = (MPC_TWO_PI_F*MPC_SMO_FC*dt)
                            / (1.0f + MPC_TWO_PI_F*MPC_SMO_FC*dt);
        """
        wc = 2.0 * math.pi * self.fc
        self._alpha_lpf = wc * dt / (1.0 + wc * dt)   # [-]

    # -----------------------------------------------------------------------
    # step  —  C: MPC_SMO_Step()
    # -----------------------------------------------------------------------
    def step(
        self,
        v_alpha: float,   # [V]  applied alpha voltage (z-1)
        v_beta:  float,   # [V]  applied beta voltage  (z-1)
        i_alpha: float,   # [A]  measured alpha current
        i_beta:  float,   # [A]  measured beta current
        dt:      float,   # [s]  step period
    ) -> Tuple[float, float]:
        """
        Execute one SMO step.

        Returns
        -------
        e_alpha_filt, e_beta_filt : float
            LPF-smoothed back-EMF estimates [V].

        C counterpart: MPC_SMO_Step() in embed_sim_mpc_controller.c.
        """
        if dt <= 0.0:
            return self._e_alpha_filt, self._e_beta_filt

        inv_L: float = 1.0 / self.L                # C: inv_L = 1.0f / MPC_L;
        k:     float = self.k_smo                  # C: k = MPC_SMO_K;
        alpha: float = self._alpha_lpf             # C: alpha = smo->alpha_lpf;

        # ---- Current estimation errors --------------------------------------
        # C: err_alpha = i_alpha - smo->i_hat_alpha;
        err_alpha: float = i_alpha - self._i_alpha_hat
        err_beta:  float = i_beta  - self._i_beta_hat

        # ---- Smooth switching  (tanh, boundary layer = 0.01 A) --------------
        # C: sw_alpha = k * tanhf(err_alpha / 0.01f);
        sw_alpha: float = k * math.tanh(err_alpha / 0.01)
        sw_beta:  float = k * math.tanh(err_beta  / 0.01)

        # ---- Current observer (Forward Euler) --------------------------------
        # C: smo->i_hat_alpha += dt*inv_L*(v_alpha - MPC_R_S*smo->i_hat_alpha - sw_alpha);
        self._i_alpha_hat += dt * inv_L * (v_alpha - self.R_S * self._i_alpha_hat - sw_alpha)
        self._i_beta_hat  += dt * inv_L * (v_beta  - self.R_S * self._i_beta_hat  - sw_beta)

        # ---- Back-EMF LPF ---------------------------------------------------
        # C: smo->e_alpha_filt += alpha*(sw_alpha - smo->e_alpha_filt);
        self._e_alpha_filt += alpha * (sw_alpha - self._e_alpha_filt)
        self._e_beta_filt  += alpha * (sw_beta  - self._e_beta_filt)

        return self._e_alpha_filt, self._e_beta_filt

    def reset(self) -> None:
        """Reset all state to zero.  C counterpart: init block in MPC_Controller_Init()."""
        self._i_alpha_hat  = 0.0
        self._i_beta_hat   = 0.0
        self._e_alpha_filt = 0.0
        self._e_beta_filt  = 0.0


# ===========================================================================
# MPC_State  (dataclass — mirrors MPC_State_T)
# ===========================================================================

@dataclass
class MPC_State:
    """
    MPC prediction initial state.  Mirrors MPC_State_T in embed_sim_mpc_controller.h.

    Fields
    ------
    id    : float   D-axis current [A].
    iq    : float   Q-axis current [A].
    omega : float   Mechanical angular speed [rad/s].
    """
    id:    float = 0.0   # [A]      C: MPC_State_T.id
    iq:    float = 0.0   # [A]      C: MPC_State_T.iq
    omega: float = 0.0   # [rad/s]  C: MPC_State_T.omega


# ===========================================================================
# _DB42S02  —  motor parameter namespace
# Mirrors: compile-time #defines in embed_sim_mpc_controller.h
# ===========================================================================

class _DB42S02:
    """
    NANOTEC DB42S02 motor parameters — DELTA-winding star-equivalent.

    All values confirmed during SMC/DFC hardware bring-up.
    DELTA-to-star conversion:
        R_star = R_delta / 3  = 0.855 Ω / 3   = 0.285 Ω
        L_star = L_delta / 3  = 1.1025 mH / 3 = 0.3675 mH

    Prior values R=0.19 Ω, L=0.125 mH were incorrect and caused the
    id limit-cycle oscillation (b = dt/L was 3× too large).
    """
    P_POLES    = 4                       # [-]     C: MPC_P_POLES
    R_S        = 0.285                   # [Ω]     C: MPC_R_S        (was 0.19)
    L_D        = 0.3675e-3               # [H]     C: MPC_L_D        (was 0.125e-3)
    L_Q        = 0.3675e-3               # [H]     C: MPC_L_Q        (was 0.125e-3)
    LAMBDA_PM  = 0.0014                  # [Wb]    C: MPC_LAMBDA_PM
    J_ROTOR    = 2.4e-6                  # [kg·m²] C: MPC_J_ROTOR
    B_FRICTION = 1e-6                    # [N·m·s] C: MPC_B_FRICTION
    I_MAX      = 3.57                    # [A]     C: MPC_I_MAX
    V_DC       = 17.0                    # [V]     C: MPC_V_DC
    V_MAX      = 17.0 / math.sqrt(3.0)  # [V]     C: MPC_V_MAX  (hexagon = V_DC/√3)
    SVPWM_GAIN = 17.0 / 2.0             # [V]     C: MPC_SVPWM_GAIN (= V_DC/2)
    KT         = 1.5 * 4 * 0.0014       # [N·m/A] C: MPC_KT  (= 1.5·p·λpm = 0.0084)

    SMO_K      = 4.68                    # [V]     C: MPC_SMO_K
    SMO_FC     = 1000.0                  # [Hz]    C: MPC_SMO_FC


# ===========================================================================
# MPCControllerBlock
# Mirrors: MPC_Controller_T  +  MPC_Controller_Step() / Init() / Reset()
# ===========================================================================

class MPCControllerBlock(VectorBlock):
    """
    EmbedSim VectorBlock wrapping the Model Predictive FOC Controller.

    Accepts the 5-element input bus produced by CtrlPacker (identical to the
    DFControllerBlock interface) and emits a 2-element normalised
    [v_alpha, v_beta] output for SVPWM (values in [-1, +1]).

    Two execution backends are available:
        Python  — full numerical replication of the C implementation.
                  Use for simulation, gain tuning, and surrogate optimisation.
        C       — calls the compiled Cython wrapper around the AURIX C code.
                  Use for hardware-in-the-loop verification.

    All default parameter values are numerically identical to the compile-time
    #define constants in embed_sim_mpc_controller.h.
    """

    # ---- EmbedSim CodeGen interface ----------------------------------------
    NUM_INPUTS  = 1
    OUTPUT_SIZE = 2

    # Input bus layout — matches MPC_Input_T and DFC_Input_T field order
    INPUT_NAMES = ["omega_ref_mech", "theta_m", "ia", "ib", "ic"]
    INPUT_KEEP  = [0, 1, 2, 3, 4]

    # C struct field comments emitted by CtrlPacker code generation
    C_FIELD_COMMENTS = {
        "omega_ref_mech": "Mechanical speed reference [rad/s]; range [0, ~314] for 0-3000 RPM",
        "theta_m":        "Mechanical rotor angle [rad]; accumulating (NOT wrapped), from encoder",
        "ia":             "Phase-A current from ADC [A]; range [-MPC_I_MAX, +MPC_I_MAX]",
        "ib":             "Phase-B current from ADC [A]; range [-MPC_I_MAX, +MPC_I_MAX]",
        "ic":             "Phase-C current from ADC [A]; range [-MPC_I_MAX, +MPC_I_MAX]",
    }

    # ---- C code generation linkage -----------------------------------------
    state_struct = "MPC_Controller_T"       # C: MPC_Controller_T
    step_func    = "MPC_Controller_Step"    # C: MPC_Controller_Step()
    init_func    = "MPC_Controller_Init"    # C: MPC_Controller_Init()
    C_SOURCES    = ["embed_sim_mpc_controller.c"]
    C_HEADERS    = ["embed_sim_mpc_controller.h"]

    # Custom C snippet emitted into embedsim_loop.c by the code generator
    C_CUSTOM_EMIT = (
        "    /* --- mpc (MPCControllerBlock) --- */\n"
        "    {\n"
        "        MPC_Input_T   u_mpc;\n"
        "        MPC_Output_T  y_mpc_out;\n"
        "        real32_T      y_mpc[2];\n"
        "\n"
        "        u_mpc.omega_ref_mech = in->omega_ref_mech;\n"
        "        u_mpc.theta_m        = in->theta_m;\n"
        "        u_mpc.ia             = in->ia;\n"
        "        u_mpc.ib             = in->ib;\n"
        "        u_mpc.ic             = in->ic;\n"
        "\n"
        "        MPC_Controller_Step(&mpc_state, &u_mpc, dt, &y_mpc_out);\n"
        "\n"
        "        y_mpc[0] = y_mpc_out.v_alpha;\n"
        "        y_mpc[1] = y_mpc_out.v_beta;\n"
        "    }"
    )

    # Diagnostic log density: every N steps at 20 kHz
    DIAG_STEPS: int = 20   # → 1 kHz log rate  C: MPC_DIAG_STEPS

    # -----------------------------------------------------------------------
    # Constructor  —  C: MPC_Controller_Init()
    # -----------------------------------------------------------------------
    def __init__(
        self,
        name:          str   = "mpc",
        # ---- Motor parameters (C: MPC_MotorParams defgroup) ----------------
        P_POLES:       int   = _DB42S02.P_POLES,    # [-]     C: MPC_P_POLES
        R_S:           float = _DB42S02.R_S,         # [Ω]     C: MPC_R_S
        L:             float = _DB42S02.L_D,         # [H]     C: MPC_L  (L_D = L_Q, SPMSM)
        LAMBDA_PM:     float = _DB42S02.LAMBDA_PM,   # [Wb]    C: MPC_LAMBDA_PM
        J:             float = _DB42S02.J_ROTOR,     # [kg·m²] C: MPC_J_ROTOR
        B:             float = _DB42S02.B_FRICTION,  # [N·m·s] C: MPC_B_FRICTION
        I_MAX:         float = _DB42S02.I_MAX,        # [A]     C: MPC_I_MAX
        V_DC:          float = _DB42S02.V_DC,         # [V]     C: MPC_V_DC
        # V_MAX / SVPWM_GAIN auto-derived from V_DC; pass explicit values to override
        V_MAX:         float = None,
        SVPWM_GAIN:    float = None,
        # ---- MPC weight constants (C: embed_sim_mpc_gains.h) ----------------
        N:             int   = 10,      # [-]     C: MPC_N          prediction horizon
        Q_id:          float = 10.0,    # [-]     C: MPC_Q_ID       d-axis state cost
        Q_iq:          float = 0.1,     # [-]     C: MPC_Q_IQ       q-axis regulariser
        Q_omega:       float = 500.0,   # [-]     C: MPC_Q_OMEGA    speed tracking cost
        R_vd:          float = 0.01,    # [-]     C: MPC_R_VD       vd effort weight
        R_vq:          float = 0.01,    # [-]     C: MPC_R_VQ       vq effort weight
        dt_s:          float = 50e-6,   # [s]     nominal ISR period
        # ---- SMO parameters (C: MPC_SMOParams defgroup) --------------------
        SMO_K:         float = _DB42S02.SMO_K,    # [V]  C: MPC_SMO_K
        SMO_FC:        float = _DB42S02.SMO_FC,   # [Hz] C: MPC_SMO_FC
        # ---- Integral correction (C: MPC_IntegralParams defgroup) -----------
        KI_v:          float = 0.03,    # [V/(rad/s·s)] C: MPC_KI_V
        SOFTSTART_T:   float = 0.1,     # [s]           C: MPC_SOFTSTART_T
        # ---- Backend selection ---------------------------------------------
        use_c_backend: bool  = False,
        dtype                = None,
    ) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        # ---- Motor parameters ----------------------------------------------
        self.P_POLES   = int(P_POLES)     # [-]
        self.R_S       = float(R_S)       # [Ω]
        self.L         = float(L)         # [H]
        self.LAMBDA_PM = float(LAMBDA_PM) # [Wb]
        self.J         = float(J)         # [kg·m²]
        self.B         = float(B)         # [N·m·s]
        self.I_MAX     = float(I_MAX)     # [A]
        self.V_DC      = float(V_DC)      # [V]
        # C: KT = 1.5f * (float)MPC_P_POLES * MPC_LAMBDA_PM;
        self.KT = 1.5 * float(P_POLES) * float(LAMBDA_PM)   # [N·m/A]

        # Voltage limits
        # C: MPC_V_MAX      = MPC_V_DC / sqrtf(3.0f);
        # C: MPC_SVPWM_GAIN = MPC_V_DC / 2.0f;
        self.V_MAX      = float(V_MAX)      if V_MAX      is not None else V_DC / math.sqrt(3.0)
        self.SVPWM_GAIN = float(SVPWM_GAIN) if SVPWM_GAIN is not None else V_DC / 2.0

        # ---- MPC weight constants ------------------------------------------
        self.N       = int(N)         # [-]
        self.Q_id    = float(Q_id)    # [-]
        self.Q_iq    = float(Q_iq)    # [-]
        self.Q_omega = float(Q_omega) # [-]
        self.R_vd    = float(R_vd)    # [-]
        self.R_vq    = float(R_vq)    # [-]
        self._dt     = float(dt_s)    # [s]

        # ---- Integral correction parameters --------------------------------
        self.KI_v        = float(KI_v)        # [V/(rad/s·s)]
        self.SOFTSTART_T = float(SOFTSTART_T) # [s]

        # ---- VectorBlock metadata ------------------------------------------
        self.vector_size  = 2
        self.output_label = "[v_alpha, v_beta]"
        self.is_dynamic   = False

        # ---- Coordinate transform sub-blocks (always Python) ---------------
        self._ct_clarke   = ClarkeTransformBlock("_mpc_clarke",    use_c_backend=False)
        self._ct_park     = ParkTransformBlock("_mpc_park",        use_c_backend=False)
        self._ct_inv_park = InvParkTransformBlock("_mpc_inv_park", use_c_backend=False)

        # ---- Encoder speed estimator (mirrors MPC_EncSpeed_T) --------------
        self._enc = MPC_EncSpeed(iir=0.20)   # C: MPC_ENC_IIR = 0.20f

        # ---- Back-EMF SMO (mirrors MPC_SMO_T inside MPC_Controller_T) ------
        self._smo = MPC_SMO(
            L     = self.L,
            R_S   = self.R_S,
            k_smo = float(SMO_K),
            fc    = float(SMO_FC),
        )
        self._smo.set_dt(self._dt)   # pre-compute alpha_lpf

        # ---- Internal state (mirrors MPC_Controller_T scalar fields) -------
        self._v_alpha_prev:       float = 0.0  # [V]   C: s->v_alpha_prev
        self._v_beta_prev:        float = 0.0  # [V]   C: s->v_beta_prev
        self._iq_limit:           float = 0.0  # [A]   C: s->iq_limit  (soft-start)
        self._speed_err_integral: float = 0.0  # [rad] C: s->speed_err_integral

        # ---- Diagnostic log (mirrors MPC_Controller_GetDiagnostics() keys) -
        self.log_data: dict = {
            "t":         [],   # [s]    simulation time
            "speed_ref": [],   # [RPM]  C: s->log_speed_ref
            "speed":     [],   # [RPM]  C: s->log_speed
            "id":        [],   # [A]    C: s->log_id
            "iq":        [],   # [A]    C: s->log_iq
            "vd":        [],   # [V]    C: s->log_vd
            "vq":        [],   # [V]    C: s->log_vq
        }
        self._log_next: float = 0.0   # [s]  next log timestamp

        # ---- C backend wrapper ---------------------------------------------
        self._wrapper = None
        if use_c_backend:
            self._load_wrapper()

        # ---- Startup diagnostics -------------------------------------------
        print(f"\n[MPC Controller] Initialized  (3-state speed-tracking MPC)")
        print(f"  Prediction horizon : N={self.N}")
        print(f"  Weights            : Q_id={Q_id:.1f}  Q_iq={Q_iq:.1f}  Q_omega={Q_omega:.1f}")
        print(f"  Control weights    : R_vd={R_vd:.4f}  R_vq={R_vq:.4f}")
        print(f"  Voltage limits     : V_MAX={self.V_MAX:.3f}V (hexagon)  "
              f"SVPWM_GAIN={self.SVPWM_GAIN:.3f}V")
        print(f"  Integral gain      : KI_v={KI_v:.4f}")
        print(f"  Soft-start         : {SOFTSTART_T * 1000:.0f}ms ramp")
        print(f"  SMO                : k={SMO_K:.2f} V  fc={SMO_FC:.0f} Hz")
        print(f"  Motor              : R={R_S:.4f} Ω  L={L*1e3:.4f} mH  "
              f"λpm={LAMBDA_PM:.4f} Wb  p={P_POLES}")
        print(f"  Backend            : {'C (Cython)' if use_c_backend else 'Python'}")

    # -----------------------------------------------------------------------
    # _load_wrapper  —  C backend initialisation
    # -----------------------------------------------------------------------
    def _load_wrapper(self) -> None:
        """
        Load the Cython extension wrapping the C MPC controller.

        Raises
        ------
        ImportError
            If the .so / .pyd extension has not been built.
        RuntimeError
            If the wrapper object cannot be instantiated.
        """
        try:
            from mpc_controller_wrapper import MPCControllerWrapper
            self._wrapper = MPCControllerWrapper(
                self.V_DC, self.P_POLES,
                self.R_S, self.L, self.LAMBDA_PM,
                self.I_MAX, self._dt,
                self.N, self.Q_id, self.Q_iq, self.Q_omega,
                self.R_vd, self.R_vq,
                self._smo.k_smo, self._smo.fc,
                self.KI_v, self.SOFTSTART_T,
            )
        except ImportError as exc:
            raise ImportError(
                "mpc_controller_wrapper not found. Build with:\n"
                "  cd fs_electrical_machines/c_src\n"
                "  python setup_mpc_controller.py build_ext --inplace"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"MPCControllerWrapper instantiation failed: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Transform helpers — always delegate to coordinate_transform_blocks
    # -----------------------------------------------------------------------

    def _clarke(self, ia: float, ib: float, ic: float) -> Tuple[float, float]:
        """
        Clarke abc → αβ.  Returns (i_alpha [A], i_beta [A]).
        C counterpart: Clarke_Step() in embed_sim_coordinate_transform.c.
        """
        inp = VectorSignal(np.array([ia, ib, ic], dtype=np.float32), "_clarke")
        out = self._ct_clarke.compute_py(0.0, 0.0, [inp])
        return float(out.value[0]), float(out.value[1])

    def _park(
        self, i_alpha: float, i_beta: float, theta_e: float
    ) -> Tuple[float, float]:
        """
        Park αβ → dq.  Returns (id_meas [A], iq_meas [A]).
        C counterpart: Park_Step() in embed_sim_coordinate_transform.c.
        """
        ab  = VectorSignal(np.array([i_alpha, i_beta], dtype=np.float32), "_park")
        th  = VectorSignal(np.array([theta_e],         dtype=np.float32), "_park")
        out = self._ct_park.compute_py(0.0, 0.0, [ab, th])
        return float(out.value[0]), float(out.value[1])

    def _inv_park(
        self, vd: float, vq: float, theta_e: float
    ) -> Tuple[float, float]:
        """
        Inverse Park dq → αβ.  Returns (v_alpha [V], v_beta [V]).
        C counterpart: InvPark_Step() in embed_sim_coordinate_transform.c.
        """
        dq  = VectorSignal(np.array([vd, vq],  dtype=np.float32), "_inv_park")
        th  = VectorSignal(np.array([theta_e], dtype=np.float32), "_inv_park")
        out = self._ct_inv_park.compute_py(0.0, 0.0, [dq, th])
        return float(out.value[0]), float(out.value[1])

    # -----------------------------------------------------------------------
    # _clamp  —  C: MPC_Clamp()
    # -----------------------------------------------------------------------
    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        """
        Saturate x to [lo, hi].
        C counterpart: MPC_Clamp() in embed_sim_mpc_controller.c.
        """
        return max(lo, min(hi, x))   # C: return fmaxf(lo, fminf(hi, x));

    # -----------------------------------------------------------------------
    # _solve_mpc  —  C: MPC_SolveMPC()
    # -----------------------------------------------------------------------
    def _solve_mpc(
        self,
        x0:        MPC_State,   # initial state [id A, iq A, omega rad/s]
        omega_ref: float,       # [rad/s] mechanical speed reference
        ed_hat:    float,       # [V]     d-axis BEMF feedforward
        eq_hat:    float,       # [V]     q-axis BEMF feedforward
        dt:        float,       # [s]     step period
    ) -> Tuple[float, float]:
        """
        Analytical 3-state MPC solver.

        Returns (vd [V], vq [V]) including BEMF feedforward.

        Free-run trajectory computed WITHOUT ed/eq_hat so BEMF is handled by
        additive feedforward after the solve; this keeps the free-run prediction
        numerically well-conditioned at all speeds (no BEMF accumulation error).

        C counterpart: MPC_SolveMPC() in embed_sim_mpc_controller.c.
        """
        # ---- Pre-compute constants ------------------------------------------
        # C: omega_e = (float)MPC_P_POLES * x0.omega;
        omega_e: float = float(self.P_POLES) * x0.omega   # [rad/s elec]
        inv_L:   float = 1.0 / self.L                     # [1/H]
        a:       float = 1.0 - dt * self.R_S * inv_L      # [-]   current decay
        b:       float = dt * inv_L                        # [A/V] voltage→current
        dt_J:    float = dt / self.J                       # [rad/s per N·m]

        # ---- Free-run trajectory (u = 0) ------------------------------------
        id_free:    float = self._clamp(x0.id, -self.I_MAX, self.I_MAX)
        iq_free:    float = self._clamp(x0.iq, -self.I_MAX, self.I_MAX)
        omega_free: float = x0.omega

        # ---- Step-response / gradient accumulators --------------------------
        bk:           float = 0.0   # iq step-response at step k
        ek:           float = 0.0   # ω  step-response (running sum of KT·bk)
        sum_bk_err_d: float = 0.0   # Σ bk·(0    − id_free)  → vd numerator
        sum_bk_err_q: float = 0.0   # Σ bk·(0    − iq_free)  → vq numerator (Q_iq)
        sum_ek_err:   float = 0.0   # Σ ek·(ωref − ω_free)   → vq numerator (Q_omega)
        sum_bk2:      float = 0.0   # Σ bk²  → denominator parts
        sum_ek2:      float = 0.0   # Σ ek²  → denominator parts

        for _ in range(self.N):
            # ---- Cross-coupling disturbances --------------------------------
            # C: f_d = dt*inv_L*(omega_e*MPC_L*iq_free);
            f_d:     float = dt * inv_L * ( omega_e * self.L * iq_free)
            f_q:     float = dt * inv_L * (-omega_e * self.L * id_free)
            f_omega: float = dt_J * (self.KT * iq_free - self.B * omega_free)

            # ---- Propagate free-run states ----------------------------------
            # C: id_free = a*id_free + f_d;
            id_free    = a * id_free    + f_d
            iq_free    = a * iq_free    + f_q
            omega_free = omega_free     + f_omega
            id_free    = self._clamp(id_free, -self.I_MAX, self.I_MAX)
            iq_free    = self._clamp(iq_free, -self.I_MAX, self.I_MAX)

            # ---- Step-response update ----------------------------------------
            # C: bk = bk*a + b;
            bk  = bk * a + b               # iq response to unit vq
            ek += dt_J * self.KT * bk      # ω  response (accumulated)

            # ---- Gradient accumulation ---------------------------------------
            sum_bk_err_d += bk * (0.0      - id_free)
            sum_bk_err_q += bk * (0.0      - iq_free)
            sum_ek_err   += ek * (omega_ref - omega_free)
            sum_bk2      += bk * bk
            sum_ek2      += ek * ek

        # ---- Analytical optimal inputs (closed-form) -------------------------
        # C: denom_d = MPC_Q_ID*sum_bk2 + MPC_R_VD;
        denom_d: float = self.Q_id * sum_bk2 + self.R_vd
        denom_q: float = self.Q_omega * sum_ek2 + self.Q_iq * sum_bk2 + self.R_vq

        vd_mpc: float = (self.Q_id * sum_bk_err_d) / denom_d if denom_d > 1e-30 else 0.0
        vq_mpc: float = (
            (self.Q_omega * sum_ek_err + self.Q_iq * sum_bk_err_q) / denom_q
            if denom_q > 1e-30 else 0.0
        )

        # ---- BEMF feedforward + hexagon clamp --------------------------------
        # C: vd = MPC_Clamp(vd_mpc + ed_hat, -MPC_V_MAX, MPC_V_MAX);
        vd: float = self._clamp(vd_mpc + ed_hat, -self.V_MAX, self.V_MAX)
        vq: float = self._clamp(vq_mpc + eq_hat, -self.V_MAX, self.V_MAX)

        return vd, vq

    # -----------------------------------------------------------------------
    # _log_step  —  C: MPC_Controller_LogDiag()
    # -----------------------------------------------------------------------
    def _log_step(
        self,
        t:              float,
        omega_ref_mech: float,
        omega_m:        float,
        id_meas:        float,
        iq_meas:        float,
        vd:             float,
        vq:             float,
        v_alpha:        float,
        v_beta:         float,
    ) -> None:
        """
        Append one diagnostic sample and emit console progress.

        Log rate: DIAG_STEPS = 20 → 1 kHz at 20 kHz ISR rate.
        Console: every 10 ms for t < 0.5 s, then every 500 ms.

        C counterpart: MPC_Controller_LogDiag() in embed_sim_mpc_controller.c.
        """
        if t < self._log_next:
            return

        # C: s->log_speed_ref[idx] = omega_ref * 60.0f / MPC_TWO_PI_F;
        self.log_data["t"].append(t)
        self.log_data["speed_ref"].append(omega_ref_mech * 60.0 / (2.0 * math.pi))
        self.log_data["speed"].append(omega_m  * 60.0 / (2.0 * math.pi))
        self.log_data["id"].append(id_meas)
        self.log_data["iq"].append(iq_meas)
        self.log_data["vd"].append(vd)
        self.log_data["vq"].append(vq)
        self._log_next += self.DIAG_STEPS * self._dt

        n = len(self.log_data["t"])
        print_interval = 10 if t < 0.5 else 500
        if n % print_interval == 1:
            v_mag = math.hypot(v_alpha, v_beta)
            print(
                f"[MPC t={t:.3f}s]  "
                f"rpm={omega_m * 60.0 / (2.0 * math.pi):+8.1f}  "
                f"id={id_meas:+7.3f}A  iq={iq_meas:+7.3f}A  "
                f"vd={vd:+6.3f}V  vq={vq:+6.3f}V  "
                f"|Vab|={v_mag:.3f}V  limit={self._iq_limit:.2f}A"
            )

    # -----------------------------------------------------------------------
    # compute_py  —  C: MPC_Controller_Step()  (Python backend)
    # -----------------------------------------------------------------------
    def compute_py(
        self,
        t:            float,
        dt:           float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """
        Execute one MPC step — Python backend.

        Implements MPC_Controller_Step() exactly, in the same step order
        as the C function.  All variable names match the C local variables.

        Returns
        -------
        VectorSignal
            value = [v_alpha_norm [-], v_beta_norm [-]]  (÷ SVPWM_GAIN).

        C counterpart: MPC_Controller_Step() in embed_sim_mpc_controller.c.
        """
        zero = np.zeros(2, dtype=np.float32)

        if not input_values or not input_values[0]:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        u = input_values[0].value
        if len(u) < 5:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        _dt: float = dt if dt > 0.0 else self._dt

        # ---- Step 1: Unpack input bus ----------------------------------------
        # C: omega_ref_mech = in->omega_ref_mech;
        omega_ref_mech: float = float(u[0])   # [rad/s]
        theta_m:        float = float(u[1])   # [rad]
        ia:             float = float(u[2])   # [A]
        ib:             float = float(u[3])   # [A]
        ic:             float = float(u[4])   # [A]

        # ---- Step 2: Electrical angle from encoder --------------------------
        # C: theta_e = (float)MPC_P_POLES * theta_m;
        theta_e: float = float(self.P_POLES) * theta_m   # [rad elec]

        # ---- Step 3: Encoder speed estimate ---------------------------------
        # C: omega_m = MPC_EncSpeed_Update(&s->enc, theta_m, dt);
        omega_m: float = self._enc.update(theta_m, _dt)   # [rad/s mech]

        # ---- Step 4: Clarke transform  abc → αβ ----------------------------
        # C: Clarke_Step(&s->clarke, ia, ib, ic, &i_alpha, &i_beta);
        i_alpha, i_beta = self._clarke(ia, ib, ic)

        # ---- Step 5: SMO back-EMF observation --------------------------------
        # C: MPC_SMO_Step(&s->smo, v_alpha_prev, v_beta_prev,
        #                 i_alpha, i_beta, dt, &e_alpha_filt, &e_beta_filt);
        e_alpha_filt, e_beta_filt = self._smo.step(
            self._v_alpha_prev, self._v_beta_prev, i_alpha, i_beta, _dt
        )

        # ---- Step 6: Park transform  αβ → dq --------------------------------
        # C: Park_Step(&s->park, i_alpha, i_beta, theta_e, &id_meas, &iq_meas);
        id_meas, iq_meas = self._park(i_alpha, i_beta, theta_e)

        # ---- Step 7: Park transform on back-EMF for feedforward -------------
        # C: Park_Step(&s->park_emf, e_alpha_filt, e_beta_filt, theta_e, &ed_raw, &eq_raw);
        ed_hat_raw, eq_hat_raw = self._park(e_alpha_filt, e_beta_filt, theta_e)

        # ---- Step 8: Physical BEMF clamp ------------------------------------
        # C: bemf_max = fabsf(omega_e) * MPC_LAMBDA_PM;
        omega_e:  float = float(self.P_POLES) * omega_m
        bemf_max: float = abs(omega_e) * self.LAMBDA_PM
        ed_hat: float = self._clamp(ed_hat_raw, -bemf_max, bemf_max)
        eq_hat: float = self._clamp(eq_hat_raw, -bemf_max, bemf_max)

        # ---- Step 9: Soft-start ramp ----------------------------------------
        # C: s->iq_limit = fminf(MPC_I_MAX, s->iq_limit + MPC_I_MAX*dt/MPC_SOFTSTART_T);
        self._iq_limit = min(self.I_MAX,
                             self._iq_limit + self.I_MAX * _dt / self.SOFTSTART_T)

        # ---- Step 10: MPC solver ---------------------------------------------
        # C: MPC_SolveMPC(&s->solver, x0, omega_ref_mech, ed_hat, eq_hat, dt, &vd, &vq);
        id0: float = self._clamp(id_meas, -self.I_MAX, self.I_MAX)
        iq0: float = self._clamp(iq_meas, -self.I_MAX, self.I_MAX)
        x0 = MPC_State(id0, iq0, omega_m)
        vd, vq = self._solve_mpc(x0, omega_ref_mech, ed_hat, eq_hat, _dt)

        # ---- Step 11: Soft-start vq limit ------------------------------------
        # C: vq_lim = (s->iq_limit / MPC_I_MAX) * MPC_V_MAX;
        vq_lim: float = (self._iq_limit / self.I_MAX) * self.V_MAX
        vq = self._clamp(vq, -vq_lim, vq_lim)

        # ---- Step 12: Speed-error integral correction (anti-windup) ---------
        # C: s->speed_err_integral += (omega_ref_mech - omega_m) * dt;
        speed_err: float = omega_ref_mech - omega_m
        self._speed_err_integral += speed_err * _dt
        head    = self._clamp(self.V_MAX - abs(vq), 0.0, self.V_MAX)
        int_max = head / (self.KI_v + 1e-30)
        self._speed_err_integral = self._clamp(
            self._speed_err_integral, -int_max, int_max
        )
        vq = self._clamp(
            vq + self.KI_v * self._speed_err_integral, -self.V_MAX, self.V_MAX
        )

        # ---- Step 13: Inverse Park  dq → αβ ---------------------------------
        # C: InvPark_Step(&s->inv_park, vd, vq, theta_e, &v_alpha, &v_beta);
        v_alpha, v_beta = self._inv_park(vd, vq, theta_e)

        # ---- Step 14: Latch voltages for next step's SMO (z-1) --------------
        # C: s->v_alpha_prev = v_alpha;  s->v_beta_prev = v_beta;
        self._v_alpha_prev = v_alpha
        self._v_beta_prev  = v_beta

        # ---- Step 15: Normalise for SVPWM  [V] → [-1, +1] ------------------
        # C: y->v_alpha = MPC_Clamp(v_alpha / MPC_SVPWM_GAIN, -1.0f, 1.0f);
        v_alpha_out: float = self._clamp(v_alpha / self.SVPWM_GAIN, -1.0, 1.0)
        v_beta_out:  float = self._clamp(v_beta  / self.SVPWM_GAIN, -1.0, 1.0)

        # ---- Step 16: Diagnostic log ----------------------------------------
        self._log_step(t, omega_ref_mech, omega_m, id_meas, iq_meas,
                       vd, vq, v_alpha, v_beta)

        self.output = VectorSignal(
            np.array([v_alpha_out, v_beta_out], dtype=np.float32), self.name
        )
        return self.output

    # -----------------------------------------------------------------------
    # compute_c  —  C backend via Cython wrapper
    # -----------------------------------------------------------------------
    def compute_c(
        self,
        t:            float,
        dt:           float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """
        Execute one MPC step — C backend (Cython wrapper).

        Calls the compiled MPC_Controller_Step() directly.  All state lives
        inside the C MPC_Controller_T struct managed by the wrapper;
        Python-side SMO/EncSpeed state is not used.

        Returns
        -------
        VectorSignal
            value = [v_alpha [-], v_beta [-]]  (normalised, ready for SVPWM).
        """
        zero = np.zeros(2, dtype=np.float32)

        if not input_values or not input_values[0]:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        u = input_values[0].value
        if len(u) < 5:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        # Pack input bus — MPC_Input_T field order
        inputs    = np.zeros(5, dtype=np.float32)
        inputs[0] = float(u[0])   # omega_ref_mech [rad/s]
        inputs[1] = float(u[1])   # theta_m        [rad]
        inputs[2] = float(u[2])   # ia             [A]
        inputs[3] = float(u[3])   # ib             [A]
        inputs[4] = float(u[4])   # ic             [A]

        self._wrapper.set_inputs(inputs)
        self._wrapper.compute(float(dt))
        outputs = self._wrapper.get_outputs()   # [v_alpha [-], v_beta [-]]

        self.output = VectorSignal(outputs, self.name)
        return self.output

    # -----------------------------------------------------------------------
    # compute  —  dispatcher
    # -----------------------------------------------------------------------
    def compute(
        self,
        t:            float,
        dt:           float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """Route to compute_c() or compute_py() based on use_c_backend."""
        if self.use_c_backend and self._wrapper is not None:
            return self.compute_c(t, dt, input_values)
        return self.compute_py(t, dt, input_values)

    # -----------------------------------------------------------------------
    # reset  —  C: MPC_Controller_Reset()
    # -----------------------------------------------------------------------
    def reset(self) -> None:
        """
        Reset all controller state to zero.

        Mirrors MPC_Controller_Reset() → MPC_Controller_Init() in C:
        a single canonical path that zeroes everything, preventing residuals
        from diverging when Reset and Init are maintained separately.

        C counterpart: MPC_Controller_Reset() in embed_sim_mpc_controller.c.
        """
        super().reset()

        # Internal state (mirrors MPC_Controller_T scalar fields)
        self._v_alpha_prev       = 0.0
        self._v_beta_prev        = 0.0
        self._iq_limit           = 0.0
        self._speed_err_integral = 0.0
        self._log_next           = 0.0

        # Sub-blocks
        self._ct_clarke.reset()
        self._ct_park.reset()
        self._ct_inv_park.reset()
        self._enc.reset()
        self._smo.reset()

        # Diagnostic log
        self.log_data = {k: [] for k in self.log_data}

        # C backend wrapper
        if self._wrapper is not None:
            self._wrapper.reset()

    # -----------------------------------------------------------------------
    # get_diagnostics  —  C: MPC_Controller_GetDiagnostics()
    # -----------------------------------------------------------------------
    def get_diagnostics(self) -> dict:
        """
        Return the current diagnostic snapshot.

        Mirrors MPC_Controller_GetDiagnostics() in embed_sim_mpc_controller.c.
        All keys and units match the C log_* fields in MPC_Controller_T.

        Returns
        -------
        dict
            speed_ref_rpm : float   Speed reference [RPM]
            speed_rpm     : float   Actual speed estimate [RPM]
            id_meas       : float   D-axis current [A]
            iq_meas       : float   Q-axis current [A]
            vd            : float   D-axis voltage command [V]
            vq            : float   Q-axis voltage command [V]
        """
        def _last(key: str) -> float:
            lst = self.log_data.get(key, [])
            return lst[-1] if lst else 0.0

        return {
            "speed_ref_rpm": _last("speed_ref"),
            "speed_rpm":     _last("speed"),
            "id_meas":       _last("id"),
            "iq_meas":       _last("iq"),
            "vd":            _last("vd"),
            "vq":            _last("vq"),
        }

    # -----------------------------------------------------------------------
    # Diagnostic properties
    # -----------------------------------------------------------------------

    @property
    def smo_e_alpha(self) -> float:
        """SMO filtered alpha back-EMF estimate [V].  Diagnostic."""
        return self._smo._e_alpha_filt

    @property
    def smo_e_beta(self) -> float:
        """SMO filtered beta back-EMF estimate [V].  Diagnostic."""
        return self._smo._e_beta_filt

    @property
    def enc_omega_m(self) -> float:
        """Encoder IIR-filtered mechanical speed [rad/s].  MPC speed feedback."""
        return self._enc._omega_filt

    def __repr__(self) -> str:
        backend = "C" if (self.use_c_backend and self._wrapper) else "Python"
        return (
            f"MPCControllerBlock('{self.name}', backend={backend}, "
            f"N={self.N}, Q_id={self.Q_id}, Q_iq={self.Q_iq}, "
            f"Q_omega={self.Q_omega}, R_vd={self.R_vd}, R_vq={self.R_vq})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "MPCControllerBlock",
    "MPC_EncSpeed",
    "MPC_SMO",
    "MPC_State",
    "_DB42S02",
]
