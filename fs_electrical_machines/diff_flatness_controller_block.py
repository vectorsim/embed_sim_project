"""
diff_flatness_controller_block.py
==================================
Differential Flatness Controller — PMSM trajectory tracking.

Drop-in replacement for SMCControllerBlock in the EmbedSim closed-loop.
Accepts the identical 5-element CtrlPacker / SMC_Input_T bus so simulation
wiring and CodeGen boundary are unchanged.

Input bus  (SMC_Input_T — 5 elements):
    u[0]  omega_ref_mech  [rad/s]   ramp-filtered speed reference
    u[1]  theta_m         [rad]     mechanical angle, accumulating
    u[2]  ia              [A]       phase-A current
    u[3]  ib              [A]       phase-B current
    u[4]  ic              [A]       phase-C current

Output:  [v_alpha, v_beta]  — αβ voltages for SVPWMPackBlock

Internal architecture (all hidden behind the CodeGen boundary):
    1. Clarke         ia,ib,ic  → iα,iβ
    2. SMO            iα,iβ,v_prev → θ̂_e, ω̂_e
    3. SpeedFusion    θ_m,ω̂_e  → θ_e (encoder), ω_e (complementary blend)
    4. Speed P-loop   ω_ref,ω_e → iq_ref  (clamped ±I_MAX)
    5. θ_ref integr.  ω_ref     → θ_ref
    6. Park           iα,iβ,θ_e → id,iq
    7. DFC law        id_ref=0,iq_ref,did=0,diq/dt,id,iq,ω_e → vd,vq
    8. Inv-Park       vd,vq,θ_e → vα,vβ

CodeGen attributes (canonical EmbedSim pattern):
    step_func    = "DFC_Controller_Step"
    state_struct = "DFC_State_T"
    init_func    = "DFC_Controller_Init"
    C_SOURCES    = ["embed_sim_dfc_controller.c",
                    "embed_sim_coordinate_transform.c",
                    "embed_sim_matrix.c"]
    C_HEADERS    = ["embed_sim_dfc_controller.h",
                    "embed_sim_coordinate_transform.h",
                    "embed_sim_matrix.h"]
"""

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from embedsim.core_blocks import VectorBlock, VectorSignal
from coordinate_transform_blocks import (
    ClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
)

_HERE = Path(__file__).resolve().parent
_C_SRC = _HERE / "c_src"


# =============================================================================
# SpeedFusion
# =============================================================================

class SpeedFusion:
    """
    Speed-dependent complementary filter.

    theta_e = p * theta_m                           encoder always
    omega_e = (1-α)*omega_enc  +  α*omega_smo

    α rises from 0→1 via normalised sigmoid across [omega_lo, omega_hi].
    Encoder IIR coefficient adapts with speed (heavier smoothing at low speed).
    """

    def __init__(self,
                 P_POLES: int = 4,
                 omega_lo: float = 50.0,
                 omega_hi: float = 250.0,
                 gamma: float = 2.0,
                 iir_lo: float = 0.05,
                 iir_hi: float = 0.30):
        self.p = float(P_POLES)
        self.omega_lo = omega_lo
        self.omega_hi = omega_hi
        self.gamma = gamma
        self.iir_lo = iir_lo
        self.iir_hi = iir_hi
        self._mid = (omega_lo + omega_hi) * 0.5
        self._raw_lo = 1.0 / (1.0 + math.exp(gamma))
        self._raw_hi = 1.0 / (1.0 + math.exp(-gamma))
        # State
        self._theta_m_prev: float = 0.0
        self._omega_enc_filt: float = 0.0
        self._omega_e_prev: float = 0.0
        # Diagnostics
        self.alpha: float = 0.0
        self.omega_enc: float = 0.0

    def _alpha(self, omega_abs: float) -> float:
        if omega_abs <= self.omega_lo:
            return 0.0
        if omega_abs >= self.omega_hi:
            return 1.0
        arg = self.gamma * (omega_abs - self._mid) / self._mid
        raw = 1.0 / (1.0 + math.exp(-arg))
        return (raw - self._raw_lo) / (self._raw_hi - self._raw_lo)

    def update(self, theta_m: float, omega_smo: float,
               dt: float) -> Tuple[float, float]:
        """Returns (theta_e [rad], omega_e [rad/s elec])."""
        theta_e = self.p * theta_m
        a = self._alpha(abs(self._omega_e_prev))
        # Encoder differentiation + adaptive IIR
        delta = theta_m - self._theta_m_prev
        delta -= 2.0 * math.pi * math.floor((delta + math.pi) / (2.0 * math.pi))
        omega_raw = delta / dt if dt > 0.0 else 0.0
        iir = self.iir_lo + a * (self.iir_hi - self.iir_lo)
        self._omega_enc_filt = (1.0 - iir) * self._omega_enc_filt + iir * omega_raw
        omega_enc_e = self.p * self._omega_enc_filt
        omega_e = (1.0 - a) * omega_enc_e + a * omega_smo
        self._theta_m_prev = theta_m
        self._omega_e_prev = omega_e
        self.alpha = a
        self.omega_enc = omega_enc_e
        return theta_e, omega_e

    def reset(self) -> None:
        self._theta_m_prev = self._omega_enc_filt = self._omega_e_prev = 0.0
        self.alpha = self.omega_enc = 0.0


# =============================================================================
# SlidingModeObserver
# =============================================================================

class SlidingModeObserver:
    """
    Classical SMO — BEMF estimation in αβ frame.
    omega_e_hat feeds SpeedFusion. theta_e_hat is diagnostic only.
    """

    def __init__(self,
                 L_S: float = 0.3675e-3, R_S: float = 0.285,
                 k_smo: float = 6.0, tau_e: float = 2e-3,
                 sigmoid_w: float = 5.0):
        self.L_S, self.R_S = L_S, R_S
        self.k_smo, self.tau_e, self.sigmoid_w = k_smo, tau_e, sigmoid_w
        self._i_hat_alpha = self._i_hat_beta = 0.0
        self._e_hat_alpha = self._e_hat_beta = 0.0
        self.theta_e_hat = self.omega_e_hat = 0.0
        self._theta_e_prev = 0.0

    def _inject(self, e: float) -> float:
        return (math.tanh(self.sigmoid_w * e) if self.sigmoid_w > 0.0
                else (math.copysign(1.0, e) if e != 0.0 else 0.0))

    def step(self, v_alpha: float, v_beta: float,
             i_alpha: float, i_beta: float, dt: float) -> Tuple[float, float]:
        L, R = self.L_S, self.R_S
        ea = self.k_smo * self._inject(i_alpha - self._i_hat_alpha)
        eb = self.k_smo * self._inject(i_beta - self._i_hat_beta)
        self._i_hat_alpha += dt * (v_alpha - R * self._i_hat_alpha + ea) / L
        self._i_hat_beta += dt * (v_beta - R * self._i_hat_beta + eb) / L
        lpf = dt / (self.tau_e + dt)
        self._e_hat_alpha += lpf * (ea - self._e_hat_alpha)
        self._e_hat_beta += lpf * (eb - self._e_hat_beta)
        th = math.atan2(-self._e_hat_alpha, self._e_hat_beta)
        dth = th - self._theta_e_prev
        dth -= 2.0 * math.pi * math.floor((dth + math.pi) / (2.0 * math.pi))
        self.omega_e_hat = dth / dt if dt > 0.0 else 0.0
        self._theta_e_prev = th
        self.theta_e_hat = th
        return self.theta_e_hat, self.omega_e_hat

    def reset(self) -> None:
        self._i_hat_alpha = self._i_hat_beta = 0.0
        self._e_hat_alpha = self._e_hat_beta = 0.0
        self.theta_e_hat = self.omega_e_hat = 0.0
        self._theta_e_prev = 0.0


# =============================================================================
# DFControllerBlock
# =============================================================================

class DFControllerBlock(VectorBlock):
    """
    Differential Flatness FOC Controller — drop-in for SMCControllerBlock.

    Accepts the 5-element SMC_Input_T bus from CtrlPacker unchanged.
    All outer-loop logic (speed P-loop, theta_ref integration, SpeedFusion)
    lives here so the CodeGen boundary and simulation wiring are identical
    to the SMC simulation.
    """

    # ── EmbedSim CodeGen attributes ──────────────────────────────────────────
    NUM_INPUTS = 1
    OUTPUT_SIZE = 2

    # Matches SMC_Input_T exactly — StepGenerator reads these
    INPUT_NAMES = ["omega_ref_mech", "theta_m", "ia", "ib", "ic"]
    INPUT_KEEP = [0, 1, 2, 3, 4]

    C_FIELD_COMMENTS = {
        "omega_ref_mech": "Mechanical speed reference [rad/s]; range [0, ~314] for 0-3000 RPM",
        "theta_m": "Mechanical rotor angle [rad]; accumulating (NOT wrapped), from encoder",
        "ia": "Phase-A current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
        "ib": "Phase-B current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
        "ic": "Phase-C current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
    }

    # CodeGen C linkage
    step_func = "DFC_Controller_Step"
    state_struct = "DFC_State_T"
    init_func = "DFC_Controller_Init"
    C_INIT_ARGS = ["dt_s"]
    C_SOURCES = [
        "embed_sim_dfc_controller.c",
    ]
    C_HEADERS = [
        "embed_sim_dfc_controller.h",
    ]

    # Cython wrapper metadata
    PYX_FILE = str(_C_SRC / "dfc_controller_wrapper.pyx")

    # Custom C code emission for code generation
    C_CUSTOM_EMIT = """\
        /* --- dfc_controller (DFControllerBlock) --- */
        /* DFC_Controller_Step() outputs physical voltages [V].                */
        /* The SVPWM block expects normalised [-1,+1] references.              */
        /* The caller must divide by SVPWM_GAIN = V_DC/2 before SVPWM.         */
        DFC_Input_T   u_dfc;
        DFC_Output_T  y_dfc_out;
        real32_T      y_dfc[2];

        u_dfc.omega_ref_mech = in->omega_ref_mech;
        u_dfc.theta_m        = in->theta_m;
        u_dfc.ia             = in->ia;
        u_dfc.ib             = in->ib;
        u_dfc.ic             = in->ic;

        DFC_Controller_Step(&dfc_state, &u_dfc, dt, &y_dfc_out);

        /* Convert physical voltages to normalised for SVPWM */
        y_dfc[0] = y_dfc_out.v_alpha / DFC_V_MAX;
        y_dfc[1] = y_dfc_out.v_beta / DFC_V_MAX;"""

    DIAG_STEPS: int = 200 if os.environ.get("DFC_DBG") == "1" else 20
    _SQRT3 = math.sqrt(3.0)

    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(self,
                 name: str = "dfc",
                 # Motor parameters
                 P_POLES: int = 4,
                 R_S: float = 0.285,
                 L_D: float = 0.3675e-3,
                 L_Q: float = 0.3675e-3,
                 LAMBDA_PM: float = 0.0014,
                 V_DC: float = 17.0,
                 I_MAX: float = 3.57,
                 dt_s: float = 50e-6,
                 # DFC current feedback gains
                 Kp_id: float = 2.0,
                 Kp_iq: float = 2.0,
                 # Outer speed P-loop
                 Kp_speed: float = 0.119,
                 # SMO parameters
                 smo_k: float = 6.0,
                 smo_tau: float = 2e-3,
                 smo_sigmoid_w: float = 5.0,
                 # SpeedFusion parameters
                 fusion_omega_lo: float = 50.0,
                 fusion_omega_hi: float = 250.0,
                 fusion_gamma: float = 2.0,
                 fusion_iir_lo: float = 0.05,
                 fusion_iir_hi: float = 0.30,
                 use_c_backend: bool = False,
                 dtype=None):
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        self.P_POLES = P_POLES
        self.R_S = R_S
        self.L_D = L_D
        self.L_Q = L_Q
        self.LAMBDA_PM = LAMBDA_PM
        self.V_DC = V_DC
        self.I_MAX = I_MAX
        self.V_MAX = V_DC / self._SQRT3
        self.dt_s = dt_s
        self.Kp_id = Kp_id
        self.Kp_iq = Kp_iq
        self.Kp_speed = Kp_speed

        self.vector_size = 2
        self.output_label = "[v_alpha, v_beta]"
        self.is_dynamic = False

        # Coordinate transforms (Python implementations)
        self._ct_clarke = ClarkeTransformBlock("_dfc_clarke", use_c_backend=False)
        self._ct_park = ParkTransformBlock("_dfc_park", use_c_backend=False)
        self._ct_inv_park = InvParkTransformBlock("_dfc_inv_park", use_c_backend=False)

        # SMO (Python implementation)
        self._smo = SlidingModeObserver(
            L_S=0.5 * (L_D + L_Q),
            R_S=R_S,
            k_smo=smo_k,
            tau_e=smo_tau,
            sigmoid_w=smo_sigmoid_w,
        )

        # SpeedFusion (Python implementation)
        self.fusion = SpeedFusion(
            P_POLES=P_POLES,
            omega_lo=fusion_omega_lo,
            omega_hi=fusion_omega_hi,
            gamma=fusion_gamma,
            iir_lo=fusion_iir_lo,
            iir_hi=fusion_iir_hi,
        )

        # Internal state — mirrored in DFC_State_T on AURIX
        self._v_alpha_prev: float = 0.0
        self._v_beta_prev: float = 0.0
        self._theta_ref: float = 0.0
        self._iq_ref_prev: float = 0.0
        self._diq_filt: float = 0.0
        self._smo_warmup: int = 0

        # Log data — same key structure as SMCControllerBlock.log_data
        self.log_data: dict = {
            "t": [],
            "speed_ref": [],
            "iq_ref": [],
            "id": [],
            "iq": [],
            "alpha": [],
            "omega_e": [],
        }

        # C backend
        self._wrapper = None
        if use_c_backend:
            self._load_wrapper()

        print(f"[DFC] Differential Flatness Controller initialized")
        print(f"[DFC] Speed gains: Kp_speed={self.Kp_speed:.4f} A/(rad/s)")
        print(f"[DFC] Current gains: Kp_id={self.Kp_id:.2f} V/A, Kp_iq={self.Kp_iq:.2f} V/A")
        print(f"[DFC] SMO: k={smo_k:.1f} V, tau={smo_tau * 1000:.1f} ms")
        print(f"[DFC] C backend: {'ENABLED' if use_c_backend else 'disabled (using Python)'}")

    # ── C backend loader ──────────────────────────────────────────────────────

    def _load_wrapper(self) -> None:
        """Load the C extension wrapper for DFC controller."""
        try:
            from dfc_controller_wrapper import DFCControllerWrapper
            self._wrapper = DFCControllerWrapper(
                self.V_DC, self.P_POLES,
                self.R_S, self.L_D, self.L_Q,
                self.LAMBDA_PM, self.I_MAX, self.dt_s,
                self.Kp_speed, self.Kp_id, self.Kp_iq)
        except ImportError as exc:
            raise ImportError(
                "dfc_controller_wrapper not found. Build with:\n"
                "  cd fs_electrical_machines/c_src\n"
                "  python setup_dfc_controller.py build_ext --inplace"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"DFCControllerWrapper instantiation failed: {exc}") from exc

    # ── Transform helpers — always delegate to coordinate_transform_blocks ────

    def _clarke(self, ia: float, ib: float, ic: float) -> Tuple[float, float]:
        """Clarke abc→αβ via ClarkeTransformBlock.compute_py()."""
        inp = VectorSignal(np.array([ia, ib, ic], dtype=np.float32), "_clarke")
        out = self._ct_clarke.compute_py(0.0, 0.0, [inp])
        return float(out.value[0]), float(out.value[1])

    def _park(self, i_alpha: float, i_beta: float, theta_e: float) -> Tuple[float, float]:
        """Park αβ→dq via ParkTransformBlock.compute_py()."""
        ab = VectorSignal(np.array([i_alpha, i_beta], dtype=np.float32), "_park")
        th = VectorSignal(np.array([theta_e], dtype=np.float32), "_park")
        out = self._ct_park.compute_py(0.0, 0.0, [ab, th])
        return float(out.value[0]), float(out.value[1])

    def _inv_park(self, vd: float, vq: float, theta_e: float) -> Tuple[float, float]:
        """Inverse Park dq→αβ via InvParkTransformBlock.compute_py()."""
        dq = VectorSignal(np.array([vd, vq], dtype=np.float32), "_inv_park")
        th = VectorSignal(np.array([theta_e], dtype=np.float32), "_inv_park")
        out = self._ct_inv_park.compute_py(0.0, 0.0, [dq, th])
        return float(out.value[0]), float(out.value[1])

    # ── flatness voltage law ──────────────────────────────────────────────────

    def _df_control(self,
                    id_ref: float, iq_ref: float,
                    did_dt: float, diq_dt: float,
                    id_meas: float, iq_meas: float,
                    omega_e: float) -> Tuple[float, float]:
        """
        PMSM flatness voltage law:
            vd = R·id_ref + Ld·did/dt − ω_e·Lq·iq_ref + Kp_id·(id_ref−id)
            vq = R·iq_ref + Lq·diq/dt + ω_e·Ld·id_ref + ω_e·λ_pm
                 + Kp_iq·(iq_ref−iq)
        """
        vd = (self.R_S * id_ref
              + self.L_D * did_dt
              - omega_e * self.L_Q * iq_ref
              + self.Kp_id * (id_ref - id_meas))

        vq = (self.R_S * iq_ref
              + self.L_Q * diq_dt
              + omega_e * self.L_D * id_ref
              + omega_e * self.LAMBDA_PM
              + self.Kp_iq * (iq_ref - iq_meas))

        # Circular voltage saturation
        mag = math.sqrt(vd * vd + vq * vq)
        if mag > self.V_MAX:
            vd *= self.V_MAX / mag
            vq *= self.V_MAX / mag

        return vd, vq

    # ── compute_py (Python implementation) ────────────────────────────────────

    def compute_py(self,
                   t: float, dt: float,
                   input_values: Optional[List[VectorSignal]] = None
                   ) -> VectorSignal:

        zero = np.zeros(2, dtype=np.float32)
        if not input_values or len(input_values[0].value) < 5:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        u = input_values[0].value
        _dt = dt if dt > 0.0 else self.dt_s

        omega_ref_mech = float(u[0])
        theta_m = float(u[1])
        ia, ib, ic = float(u[2]), float(u[3]), float(u[4])

        # ── 1. Clarke ─────────────────────────────────────────────────────────
        i_alpha, i_beta = self._clarke(ia, ib, ic)

        # ── 2. SMO (causal: uses previous-step voltages) ──────────────────────
        _, omega_e_smo_raw = self._smo.step(
            self._v_alpha_prev, self._v_beta_prev,
            i_alpha, i_beta, _dt)

        # Gate SMO speed until BEMF has converged (~400 steps = 20ms at 20kHz).
        self._smo_warmup += 1
        omega_e_smo = omega_e_smo_raw if self._smo_warmup > 400 else 0.0

        # ── 3. SpeedFusion → theta_e (encoder), omega_e (complementary) ──────
        theta_e, omega_e = self.fusion.update(theta_m, omega_e_smo, _dt)

        # ── 4. Outer speed P-loop → iq_ref ───────────────────────────────────
        omega_meas_mech = self.fusion._omega_enc_filt
        speed_err = omega_ref_mech - omega_meas_mech
        iq_ref = max(-self.I_MAX, min(self.I_MAX, self.Kp_speed * speed_err))
        id_ref = 0.0

        # ── 5. Reference angle integration ────────────────────────────────────
        self._theta_ref += omega_ref_mech * _dt

        # ── 6. Current derivatives (feedforward) ──────────────────────────────
        diq_raw = (iq_ref - self._iq_ref_prev) / _dt if _dt > 0.0 else 0.0
        _tau_diq = 1e-3
        _lpf = _dt / (_tau_diq + _dt) if _dt > 0.0 else 0.0
        self._diq_filt = (1.0 - _lpf) * self._diq_filt + _lpf * diq_raw
        diq_ref_dt = self._diq_filt
        did_ref_dt = 0.0
        self._iq_ref_prev = iq_ref

        # ── 7. Park ───────────────────────────────────────────────────────────
        id_meas, iq_meas = self._park(i_alpha, i_beta, theta_e)

        # ── 8. Flatness voltage law ───────────────────────────────────────────
        vd, vq = self._df_control(
            id_ref, iq_ref, did_ref_dt, diq_ref_dt,
            id_meas, iq_meas, omega_e)

        # ── 9. Inverse Park → αβ ──────────────────────────────────────────────
        v_alpha, v_beta = self._inv_park(vd, vq, theta_e)

        # Store voltages for SMO next step (z-1)
        self._v_alpha_prev = v_alpha
        self._v_beta_prev = v_beta

        # ── Log (same structure as SMCControllerBlock.log_data) ───────────────
        self.log_data["t"].append(t)
        self.log_data["speed_ref"].append(omega_ref_mech * 60.0 / (2.0 * math.pi))
        self.log_data["iq_ref"].append(iq_ref)
        self.log_data["id"].append(id_meas)
        self.log_data["iq"].append(iq_meas)
        self.log_data["alpha"].append(self.fusion.alpha)
        self.log_data["omega_e"].append(omega_e)

        self.output = VectorSignal(
            np.array([v_alpha, v_beta], dtype=np.float32), self.name)
        return self.output

    # ── compute_c (C backend implementation) ──────────────────────────────────

    def compute_c(self, t: float, dt: float,
                  input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        C backend compute method for DFControllerBlock.

        Uses the compiled C extension for high-performance execution.
        """
        zero = np.array([0.0, 0.0], dtype=np.float32)
        if not input_values or not input_values[0]:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        u = input_values[0].value
        if len(u) < 5:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        # Copy inputs to float32 array for C wrapper
        inputs = np.zeros(5, dtype=np.float32)
        inputs[0] = float(u[0])  # omega_ref_mech
        inputs[1] = float(u[1])  # theta_m
        inputs[2] = float(u[2])  # ia
        inputs[3] = float(u[3])  # ib
        inputs[4] = float(u[4])  # ic

        # Call C wrapper
        self._wrapper.set_inputs(inputs)
        self._wrapper.compute(float(dt))
        outputs = self._wrapper.get_outputs()

        self.output = VectorSignal(outputs, self.name)
        return self.output

    # ── compute (dispatcher) ──────────────────────────────────────────────────

    def compute(self, t: float, dt: float,
                input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Dispatch to either Python or C implementation based on use_c_backend.
        """
        # Note: use_c_backend is an attribute from the base VectorBlock class
        if self.use_c_backend and self._wrapper is not None:
            return self.compute_c(t, dt, input_values)
        else:
            return self.compute_py(t, dt, input_values)

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the controller state."""
        super().reset()
        # Reset Python state
        self._v_alpha_prev = 0.0
        self._v_beta_prev = 0.0
        self._theta_ref = 0.0
        self._iq_ref_prev = 0.0
        self._diq_filt = 0.0
        self._smo_warmup = 0
        # Reset transform blocks
        self._ct_clarke.reset()
        self._ct_park.reset()
        self._ct_inv_park.reset()
        # Reset SMO and SpeedFusion
        self._smo.reset()
        self.fusion.reset()
        # Clear log data
        self.log_data = {k: [] for k in self.log_data}
        # Reset C wrapper if present
        if self._wrapper is not None:
            self._wrapper.reset()

    # ── diagnostics ───────────────────────────────────────────────────────────

    @property
    def smo_theta_e(self) -> float:
        """SMO estimated electrical angle [rad] (diagnostic only)."""
        return self._smo.theta_e_hat

    @property
    def smo_omega_e(self) -> float:
        """SMO estimated electrical speed [rad/s] (diagnostic only)."""
        return self._smo.omega_e_hat

    def __repr__(self) -> str:
        return f"DFControllerBlock('{self.name}')"


__all__ = ["DFControllerBlock", "SpeedFusion", "SlidingModeObserver"]