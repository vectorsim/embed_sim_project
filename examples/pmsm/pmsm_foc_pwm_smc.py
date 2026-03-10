# =============================================================================
# pmsm_foc_pwm_smc.py (ULTIMATE FIX - ENSURE MOTOR IS EXECUTED)
# =============================================================================

import sys
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.path.insert(0, _HERE)
    from _path_utils import get_project_root

    _PROJECT_ROOT = str(get_project_root())
    _ELEC_BLOCKS = os.path.join(_PROJECT_ROOT, "electrical_blocks")
except ImportError:
    _PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
    _ELEC_BLOCKS = os.path.abspath(os.path.join(_HERE, "..", "..", "electrical_blocks"))

for _p in [_PROJECT_ROOT, _ELEC_BLOCKS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# EmbedSim core
from embedsim.code_generator import SimBlockBase, CodeGenStart, CodeGenEnd
from embedsim.core_blocks import VectorBlock, VectorSignal
from embedsim.dynamic_blocks import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver, LoopBreaker, VectorDelay
from embedsim.source_blocks import VectorConstant, GaussianNoiseBlock

# Your existing blocks
from coordinate_transform_blocks import (
    ClarkeTransformBlock,
    InvClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
)
from speed_pi_block import SpeedPIBlock
from smc_block import SMCBlock
from svpwm_block import SVPWMBlock

# NEW motor model
from PMSM_Motor_WithSensorsBlock import PMSM_Motor_WithSensorsBlock

# ---------------------------------------------------------------------------
# FMU path for the new motor model
# ---------------------------------------------------------------------------
_FMU_PATH = os.path.join(_ELEC_BLOCKS, "modelica", "PMSM_Motor_WithSensors.fmu")

# =============================================================================
# DEBUG FLAG
# =============================================================================
DEBUG = False

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================
V_DC = 48.0
POLE_PAIRS = 2
R_S = 0.5
L_D = 0.005
L_Q = 0.006
LAMBDA_PM = 0.175
T_LOAD = 0.0

SPEED_RPM = 600.0   # Max achievable ~756 RPM on 48V with this motor (V_MAX=27.7V, lambda_pm=0.175Wb)
SPEED_RAD_S = SPEED_RPM * 2.0 * np.pi / 60.0

T_END = 1.0
DT = 1e-4

V_MAX = V_DC / np.sqrt(3.0)

# =============================================================================
# SPEED PI GAINS
# =============================================================================
KP_SPEED = 1.0
KI_SPEED = 10.0
IQ_MAX = 15.0   # Limit iq to avoid overcurrent

# =============================================================================
# SMC GAINS
# =============================================================================
# SMC gains for this motor: R=0.5Ω, L_d=5mH, L_q=6mH, lambda_pm=0.175Wb
# K_sw MUST equal out_max (= V_MAX) so sat()=1 delivers full voltage.
# phi = boundary layer [A]: smooths control within ±phi of the sliding surface.
# lambda matches current-loop bandwidth R/L_q = 0.5/0.006 ~ 83 rad/s.
LAMBDA_D = 80.0
K_SW_D = V_MAX   # = 27.71 V — full voltage at |error| >= phi
PHI_D = 2.0      # A — boundary layer width

LAMBDA_Q = 80.0
K_SW_Q = V_MAX   # = 27.71 V
PHI_Q = 2.0      # A

V_SMC_MAX = V_MAX

# =============================================================================
# SENSOR NOISE PARAMETERS
# =============================================================================
# All noise is additive Gaussian white noise (zero-mean) injected at the
# sensor output, before the FOC control chain.
#
# Current sensors (ia, ib, ic):
#   Typical shunt/Hall-effect current sensor LSB ≈ 10-50 mA.
#   STD = 0.02 A  ≈ 20 mA 1-σ  → realistic for a 15 A peak FOC drive.
#
# Speed sensor (omega_m):
#   Resolver or encoder after velocity estimation.
#   STD = 0.5 rad/s  ≈ 4.8 RPM 1-σ.
#
# Angle sensor (theta_m mechanical → theta_e electrical):
#   12-bit resolver / sin-cos encoder.
#   STD = 0.005 rad  ≈ 0.29° 1-σ on the ELECTRICAL angle.
#
# Set any STD to 0.0 to disable that noise channel.
# Seeds are fixed so runs are reproducible; set None for Monte-Carlo.
NOISE_CURRENT_STD = 0.02    # A   — three independent channels (ia, ib, ic)
NOISE_SPEED_STD   = 0.5     # rad/s
NOISE_ANGLE_STD   = 0.005   # rad  (electrical angle)
NOISE_SEED_BASE   = 42      # seeds: current=42, speed=43, angle=44

# =============================================================================
# OPEN-LOOP STARTUP PARAMETERS
# =============================================================================
OPEN_LOOP_TIME = 0.2  # 200ms of open-loop
OPEN_LOOP_VOLTAGE = 5.0  # Voltage amplitude
OPEN_LOOP_FREQ = 10.0  # Hz


# =============================================================================
# Simple blocks
# =============================================================================

class OpenLoopStartupBlock(VectorBlock):
    """Generate open-loop voltages directly for the motor"""

    def __init__(self, name: str, open_loop_time: float, v_start: float, f_start: float):
        super().__init__(name)
        self.open_loop_time = open_loop_time
        self.v_start = v_start
        self.f_start = f_start
        self.vector_size = 3  # Output three-phase duties directly
        self.output_label = "[da_open, db_open, dc_open]"
        self.theta = 0.0
        self.last_output = np.array([0.5, 0.5, 0.5])

    def compute(self, t, dt, input_values=None):
        if t < self.open_loop_time:
            # Generate rotating voltage vector in alpha-beta
            self.theta += 2.0 * np.pi * self.f_start * dt
            v_alpha = self.v_start * np.cos(self.theta)
            v_beta = self.v_start * np.sin(self.theta)

            # Convert to three-phase voltages
            v_a = v_alpha
            v_b = -0.5 * v_alpha + 0.866 * v_beta
            v_c = -0.5 * v_alpha - 0.866 * v_beta

            # Convert to duties (0-1 range, centered at 0.5)
            duties = np.array([0.5 + v_a / V_DC, 0.5 + v_b / V_DC, 0.5 + v_c / V_DC])
            self.last_output = duties

            if DEBUG:
                print(f"OpenLoop t={t:.3f}: duties=({duties[0]:.3f},{duties[1]:.3f},{duties[2]:.3f})")
        else:
            # After open-loop, use SVPWM outputs
            if input_values and input_values[0] is not None:
                self.last_output = input_values[0].value.copy()
            else:
                self.last_output = np.array([0.5, 0.5, 0.5])

        self.output = VectorSignal(self.last_output.copy(), self.name)
        return self.output

    def reset(self):
        super().reset()
        self.theta = 0.0
        self.last_output = np.array([0.5, 0.5, 0.5])


class MotorOutputBlock(VectorBlock):
    """Read motor outputs and print them for debugging"""

    def __init__(self, name: str):
        super().__init__(name)
        self.vector_size = 5  # [i_a, i_b, i_c, theta_m, omega_m]
        self.output_label = "[ia,ib,ic,θm,ωm]"

    # FMU OUTPUT_VARS index map (matches PMSM_Motor_WithSensorsBlock.OUTPUT_VARS):
    # 0:i_a  1:i_b  2:i_c  3:theta_m  4:omega_m_out  5:emf_a  6:emf_b  7:emf_c
    # 8:speed_rpm  9:T_em_out  10:i_d  11:i_q
    _FMU_IDX = {'i_a': 0, 'i_b': 1, 'i_c': 2, 'theta_m': 3, 'omega_m_out': 4}

    def compute(self, t, dt, input_values=None):
        out = np.zeros(5, dtype=np.float32)
        if input_values and input_values[0] is not None:
            raw = input_values[0].value

            if isinstance(raw, dict):
                # FMU returned a named dict
                out[0] = float(raw.get('i_a', 0.0))
                out[1] = float(raw.get('i_b', 0.0))
                out[2] = float(raw.get('i_c', 0.0))
                out[3] = float(raw.get('theta_m', 0.0))
                out[4] = float(raw.get('omega_m_out', 0.0))
            elif hasattr(raw, '__len__') and len(raw) >= 5:
                # FMU returned flat numpy array aligned with OUTPUT_VARS order
                out[0] = float(raw[0])   # i_a
                out[1] = float(raw[1])   # i_b
                out[2] = float(raw[2])   # i_c
                out[3] = float(raw[3])   # theta_m
                out[4] = float(raw[4])   # omega_m_out
            elif hasattr(raw, '__len__') and len(raw) == 5:
                # Already packed by a previous MotorOutputBlock pass (VectorDelay)
                out = raw.copy().astype(np.float32)
            # else: size-1 placeholder from engine — leave as zeros(5)

        self.output = VectorSignal(out, self.name)
        return self.output


class MotorCurrentsBlock(VectorBlock):
    """Extract currents from motor outputs"""

    def __init__(self, name: str):
        super().__init__(name)
        self.vector_size = 3
        self.output_label = "[ia,ib,ic]"

    def compute(self, t, dt, input_values=None):
        currents = np.zeros(3, dtype=np.float32)
        if input_values and input_values[0] is not None:
            motor_out = input_values[0].value
            if hasattr(motor_out, '__len__') and len(motor_out) >= 3:
                currents = np.array(motor_out[:3], dtype=np.float32)
        self.output = VectorSignal(currents, self.name)
        return self.output


class MotorAngleBlock(VectorBlock):
    """Extract angle from motor outputs and convert to electrical angle"""

    def __init__(self, name: str, pole_pairs: int):
        super().__init__(name)
        self.pole_pairs = pole_pairs
        self.vector_size = 1
        self.output_label = "θe"

    def compute(self, t, dt, input_values=None):
        theta_e = 0.0
        if input_values and input_values[0] is not None:
            motor_out = input_values[0].value
            # motor_out is packed [i_a, i_b, i_c, theta_m, omega_m_out] by
            # MotorOutputBlock.  Index 3 is theta_m (mechanical angle [rad]).
            # The FMU also exposes theta_e directly, but we reconstruct it here
            # from theta_m to avoid depending on internal FMU state variables.
            if hasattr(motor_out, '__len__') and len(motor_out) >= 4:
                theta_m = motor_out[3]
                theta_e = float(theta_m) * self.pole_pairs  # theta_e = p * theta_m
        self.output = VectorSignal(np.array([theta_e], dtype=np.float32), self.name)
        return self.output


class MotorSpeedBlock(VectorBlock):
    """Extract speed from motor outputs"""

    def __init__(self, name: str):
        super().__init__(name)
        self.vector_size = 1
        self.output_label = "ωm"

    def compute(self, t, dt, input_values=None):
        omega_m = 0.0
        if input_values and input_values[0] is not None:
            motor_out = input_values[0].value
            # Index 4 is omega_m_out as packed by MotorOutputBlock [rad/s].
            if hasattr(motor_out, '__len__') and len(motor_out) >= 5:
                omega_m = float(motor_out[4])
        self.output = VectorSignal(np.array([omega_m], dtype=np.float32), self.name)
        return self.output


class NoisySensorBlock(VectorBlock):
    """
    Additive Gaussian white-noise wrapper for any scalar/vector sensor.

    Receives a clean signal on port 0 and adds i.i.d. N(0, std²) noise to
    every element, modelling ADC quantisation + analogue front-end noise.

    This block is intentionally thin — all noise statistics live in the
    NOISE_* parameters at the top of the file so they are easy to sweep.

    Args:
        name:  Block identifier.
        std:   Noise standard deviation (same units as input signal).
        dim:   Output vector length — must match the upstream signal.
        seed:  RNG seed for reproducibility (None = non-deterministic).

    Signal flow (example — current path):
        motor_fb → motor_currents → noisy_currents → clarke
    """

    def __init__(self, name: str, std: float, dim: int, seed: int = None, **kwargs):
        super().__init__(name, **kwargs)
        self.noise_std   = std
        self.vector_size = dim
        self._rng        = np.random.default_rng(seed)

    def compute(self, t, dt, input_values=None):
        if input_values and input_values[0] is not None:
            clean = np.asarray(input_values[0].value, dtype=np.float32)
        else:
            clean = np.zeros(self.vector_size, dtype=np.float32)
        noise = self._rng.normal(0.0, self.noise_std, size=self.vector_size).astype(np.float32)
        self.output = VectorSignal(clean + noise, self.name)
        return self.output


class DQRefBlock(VectorBlock):
    """DQ reference (id_ref=0, iq_ref from constant)"""

    def __init__(self, name: str):
        super().__init__(name)
        self.vector_size = 2
        self.output_label = "[id_ref,iq_ref]"

    def compute(self, t, dt, input_values=None):
        iq_ref_val = input_values[0].value[1] if (input_values and len(input_values[0].value) >= 2) else 0.0
        self.output = VectorSignal(np.array([0.0, iq_ref_val], dtype=np.float32), self.name)
        return self.output


# =============================================================================
# Build simulation - ULTIMATE FIX
# =============================================================================

def build_simulation():
    """Instantiate all blocks, wire signals — full open-loop + closed-loop FOC."""

    # =========================================================================
    # Motor plant with realistic sensors
    # =========================================================================
    motor = PMSM_Motor_WithSensorsBlock(
        name="motor",
        fmu_path=_FMU_PATH,
        R=R_S,
        L_d=L_D,
        L_q=L_Q,
        lambda_pm=LAMBDA_PM,
        J=0.002,
        B=0.001,
        p=float(POLE_PAIRS),
    )

    # =========================================================================
    # Motor output monitor + signal extractors
    # =========================================================================
    motor_out = MotorOutputBlock("motor_out")
    motor >> motor_out

    # Break the algebraic loop: motor_out feeds back into the FOC chain which
    # drives motor inputs — a zero-delay cycle.  VectorDelay holds the previous
    # step's motor outputs, giving the solver a one-step lag on the feedback
    # signals (acceptable at dt=0.1 ms for a 10 kHz FOC loop).
    motor_fb = VectorDelay("motor_fb", initial=[0.0, 0.0, 0.0, 0.0, 0.0])  # [ia,ib,ic,theta_m,omega_m_out]
    motor_out >> motor_fb

    motor_currents = MotorCurrentsBlock("motor_currents")
    motor_angle    = MotorAngleBlock("motor_angle", POLE_PAIRS)
    motor_speed    = MotorSpeedBlock("motor_speed")

    motor_fb >> motor_currents
    motor_fb >> motor_angle
    motor_fb >> motor_speed

    # =========================================================================
    # Sensor noise injection  (Feature — additive Gaussian white noise)
    # =========================================================================
    # Noise is inserted AFTER the signal extractors and BEFORE the FOC chain,
    # matching where real ADC/resolver noise enters a physical drive.
    #
    # Disable individual channels by setting the corresponding STD to 0.0 in
    # the NOISE_* parameters at the top of this file.

    noisy_currents = NoisySensorBlock(
        "noisy_currents",
        std=NOISE_CURRENT_STD,
        dim=3,
        seed=NOISE_SEED_BASE,       # seed 42
    )
    motor_currents >> noisy_currents

    noisy_speed = NoisySensorBlock(
        "noisy_speed",
        std=NOISE_SPEED_STD,
        dim=1,
        seed=NOISE_SEED_BASE + 1,   # seed 43
    )
    motor_speed >> noisy_speed

    noisy_angle = NoisySensorBlock(
        "noisy_angle",
        std=NOISE_ANGLE_STD,
        dim=1,
        seed=NOISE_SEED_BASE + 2,   # seed 44
    )
    motor_angle >> noisy_angle

    # Ensure motor is always in the execution graph
    motor_sink = VectorEnd("motor_sink")
    motor >> motor_sink

    # =========================================================================
    # Closed-loop FOC controller chain
    # ─────────────────────────────────────────────────────────────────────────
    # Feature 05121967 — PYXInspector / CodeGen region
    #
    # Everything between cg_start and cg_end is the "C region": after the
    # simulation runs, calling cg_end.generate_loop(cg_start, ...) will walk
    # these blocks in topological order and emit embedsim_loop.c +
    # embedsim_loop.h targeting the Infineon Aurix TC3xx (TriCore).
    #
    # The two marker blocks are transparent passthroughs at runtime — they
    # concatenate their inputs into a single flat vector and pass it on,
    # so they have zero effect on numerical results.
    #
    # cg_start is NOT wired into the FOC chain here.  It is a standalone
    # marker that records the boundary; the actual FOC inputs (noisy_speed,
    # noisy_currents, noisy_angle) connect directly to the FOC blocks.
    # =========================================================================
    # =========================================================================
    # Feature 05121967 — CodeGen region  (cg_start … cg_end)
    # ─────────────────────────────────────────────────────────────────────────
    # cg_start is a transparent passthrough (forwards input_values[0] unchanged).
    # It must be INLINE — at least one FOC block must receive its output —
    # so the execution engine visits it and generate_loop() can stop here.
    #
    # Runtime wiring:   sensor sources → cg_start → speed_pi (port 0: ω_ref)
    #                   sensor sources also wire DIRECTLY to the other FOC blocks
    #                   (clarke, park, inv_park) for correct simulation values.
    #
    # CodeGen wiring:   C_INPUT_MAP on every FOC block references "cg_start"
    #                   ports so generate_loop() stops at the boundary and
    #                   never emits the Python-only sensor blocks.
    #
    # cg_start input layout (flat, for C_INPUT_MAP reference):
    #   [0]    ω_ref          (speed_ref,      1 element)
    #   [1]    ω_m measured   (noisy_speed,    1 element)
    #   [2..4] ia, ib, ic     (noisy_currents, 3 elements)
    #   [5]    θe measured    (noisy_angle,    1 element)
    # =========================================================================

    # --- cg_start: inline input boundary ---
    speed_ref = VectorConstant("speed_ref", value=[SPEED_RAD_S])
    cg_start  = CodeGenStart("cg_start")
    speed_ref      >> cg_start   # [0]   ω_ref       (port 0 — forwarded at runtime)
    noisy_speed    >> cg_start   # [1]   ω_m         (recorded for C_INPUT_MAP)
    noisy_currents >> cg_start   # [2..4] ia,ib,ic   (recorded for C_INPUT_MAP)
    noisy_angle    >> cg_start   # [5]   θe          (recorded for C_INPUT_MAP)

    # 1. Speed PI: ω_ref × ω_m → [id_ref=0, iq_ref]
    speed_pi = SpeedPIBlock(
        "speed_pi",
        Kp=KP_SPEED,
        Ki=KI_SPEED,
        i_max=IQ_MAX,
        t_enable=OPEN_LOOP_TIME,
    )
    cg_start    >> speed_pi   # runtime: cg_start forwards speed_ref (ω_ref)
    noisy_speed >> speed_pi   # runtime: ω_m measured (port 1)
    # CodeGen: ω_ref=cg_start[0], ω_m=cg_start[1]
    speed_pi.C_INPUT_MAP = [("cg_start", 0), ("cg_start", 1)]

    # 2. DQ reference (id_ref=0, iq_ref from speed PI) — Python-only helper
    dq_ref = DQRefBlock("dq_ref")
    speed_pi >> dq_ref

    # 3. Clarke: [ia,ib,ic] → [iα, iβ]
    clarke = ClarkeTransformBlock("clarke")
    noisy_currents >> clarke   # runtime: currents direct from sensor
    # CodeGen: ia=cg_start[2], ib=cg_start[3], ic=cg_start[4]
    clarke.C_INPUT_MAP = [("cg_start", 2), ("cg_start", 3), ("cg_start", 4)]

    # 4. Park: [iα, iβ] × θe → [id, iq]
    park = ParkTransformBlock("park")
    clarke      >> park   # port 0: alpha-beta currents
    noisy_angle >> park   # port 1: θe direct from sensor (runtime)
    # CodeGen: θe=cg_start[5]
    park.C_INPUT_MAP = [("clarke", 0), ("clarke", 1), ("cg_start", 5)]

    # 5. SMC: [id_ref, iq_ref] × [id, iq] → [vd, vq]
    smc = SMCBlock(
        "smc",
        lambda_d=LAMBDA_D, K_sw_d=K_SW_D, phi_d=PHI_D,
        lambda_q=LAMBDA_Q, K_sw_q=K_SW_Q, phi_q=PHI_Q,
        out_min=-V_SMC_MAX, out_max=V_SMC_MAX,
        t_enable=0.0,
    )
    dq_ref >> smc   # port 0: [id_ref, iq_ref]
    park   >> smc   # port 1: [id, iq]
    # CodeGen: dq_ref is Python-only; SMC_Input_T flattens to 4 scalars.
    smc.C_CUSTOM_EMIT = (
        "\n"
        "    /* --- smc (SMCBlock) --- */\n"
        "    real32_T y_smc[2];\n"
        "    {\n"
        "        SMC_Input_T _smc_in;\n"
        "        _smc_in.ref_d  = 0.0f;           /* id_ref = 0 (MTPA) */\n"
        "        _smc_in.ref_q  = y_speed_pi[1];  /* iq_ref from speed PI */\n"
        "        _smc_in.meas_d = y_park[0];       /* id measured */\n"
        "        _smc_in.meas_q = y_park[1];       /* iq measured */\n"
        "        SMC_Output_T _smc_out;\n"
        "        SMC_Compute(&smc_state, &_smc_in, dt, &_smc_out);\n"
        "        y_smc[0] = _smc_out.v_d;\n"
        "        y_smc[1] = _smc_out.v_q;\n"
        "    }\n"
    )

    # 6. Inverse Park: [vd, vq] × θe → [vα, vβ]
    inv_park = InvParkTransformBlock("inv_park")
    smc         >> inv_park   # port 0: [vd, vq]
    noisy_angle >> inv_park   # port 1: θe direct from sensor (runtime)
    # CodeGen: θe=cg_start[5]
    inv_park.C_INPUT_MAP = [("smc", 0), ("smc", 1), ("cg_start", 5)]

    # 7. Inverse Clarke: [vα, vβ] → [va, vb, vc]
    inv_clarke = InvClarkeTransformBlock("inv_clarke")
    inv_park >> inv_clarke

    # 8. SVPWM: [va, vb, vc] → [duty_a, duty_b, duty_c]
    svpwm = SVPWMBlock("svpwm", v_dc=V_DC)
    inv_clarke >> svpwm
    # CodeGen: all 3 phase voltages from inv_clarke
    svpwm.C_INPUT_MAP = [("inv_clarke", 0), ("inv_clarke", 1), ("inv_clarke", 2)]

    # --- cg_end: inline output boundary ---
    cg_end = CodeGenEnd("cg_end")
    svpwm >> cg_end

    # Motor inputs — duty cycles flow out of cg_end to the plant
    vdc_src   = VectorConstant("vdc",   value=[V_DC])
    tload_src = VectorConstant("tload", value=[T_LOAD])

    cg_end    >> motor   # port 0: duty cycles (svpwm → cg_end → motor)
    vdc_src   >> motor   # port 1: v_dc
    tload_src >> motor   # port 2: T_load

    # =========================================================================
    # Sinks
    # =========================================================================
    sink = VectorEnd("sink")
    motor_out >> sink

    # =========================================================================
    # Simulation engine
    # =========================================================================
    sim = EmbedSim(
        sinks=[sink, motor_sink],
        T=T_END,
        dt=DT,
        solver=ODESolver.RK4,
    )

    # Scope channels
    sim.scope.add(motor_speed,    label="speed")
    sim.scope.add(noisy_speed,    label="noisy_speed")
    sim.scope.add(motor_currents, label="currents")
    sim.scope.add(noisy_currents, label="noisy_currents")
    sim.scope.add(motor_angle,    label="angle")
    sim.scope.add(noisy_angle,    label="noisy_angle")
    sim.scope.add(svpwm,          label="duties")
    sim.scope.add(park,           label="dq_currents")
    sim.scope.add(smc,            label="vdq")

    # ── Attach objects that main() needs after the sim runs ───────────────────
    # sim.motor    — FMU handle (already set above for initialize_fmu)
    # sim.cg_start — CodeGenStart marker (input boundary of C region)
    # sim.cg_end   — CodeGenEnd   marker (output boundary; owns generate_loop)
    sim.motor    = motor
    sim.cg_start = cg_start
    sim.cg_end   = cg_end
    return sim


# =============================================================================
# Plotting
# =============================================================================

def _s(sim, label, idx=0):
    key = f"{label}[{idx}]"
    d = sim.scope.data.get(key)
    return np.array(d) if d is not None else None


def plot_results(sim, out_path: str):
    t = np.array(sim.scope.t)
    if len(t) == 0:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # Speed — clean vs noisy
    speed       = _s(sim, "speed",       0)
    noisy_spd   = _s(sim, "noisy_speed", 0)
    if speed is not None:
        rpm = speed * 60.0 / (2.0 * np.pi)
        ax1.plot(t, rpm, 'b-', lw=1.5, label="Clean ωm")
    if noisy_spd is not None:
        rpm_n = noisy_spd * 60.0 / (2.0 * np.pi)
        ax1.plot(t, rpm_n, color='cornflowerblue', lw=0.8, alpha=0.6, label="Noisy ωm")
    ax1.axhline(SPEED_RPM, color='r', linestyle='--', lw=1, label=f"Target {SPEED_RPM} RPM")
    ax1.axvline(OPEN_LOOP_TIME, color='k', linestyle=':', lw=1, alpha=0.5, label='OL→CL')
    ax1.set_ylabel('RPM')
    ax1.set_title('Motor Speed  (clean vs noisy)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Angle — clean vs noisy (show sin for readability)
    angle     = _s(sim, "angle",       0)
    noisy_ang = _s(sim, "noisy_angle", 0)
    if angle is not None:
        ax2.plot(t, np.sin(angle), 'g-', lw=1.5, label="Clean θe")
    if noisy_ang is not None:
        ax2.plot(t, np.sin(noisy_ang), color='lightgreen', lw=0.8, alpha=0.6, label="Noisy θe")
    ax2.axvline(OPEN_LOOP_TIME, color='k', linestyle=':', lw=1, alpha=0.5)
    ax2.set_ylabel('sin(θe)')
    ax2.set_title('Electrical Angle  (clean vs noisy)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Phase currents — clean vs noisy (ia only for clarity, then noisy ia)
    for i, (color, lbl) in enumerate(zip(['r', 'g', 'b'], ['ia', 'ib', 'ic'])):
        c = _s(sim, "currents",       i)
        n = _s(sim, "noisy_currents", i)
        if c is not None:
            ax3.plot(t, c, color=color, lw=1.2, label=lbl, alpha=0.85)
        if n is not None:
            ax3.plot(t, n, color=color, lw=0.6, alpha=0.35, linestyle='-')
    ax3.axvline(OPEN_LOOP_TIME, color='k', linestyle=':', lw=1, alpha=0.5)
    ax3.set_ylabel('Current (A)')
    ax3.set_title('Phase Currents  (solid=clean, faint=noisy)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # SVPWM duty cycles
    for i, (color, lbl) in enumerate(zip(['r', 'g', 'b'], ['d_a', 'd_b', 'd_c'])):
        duty = _s(sim, "duties", i)
        if duty is not None:
            ax4.plot(t, duty, color=color, lw=1, label=lbl, alpha=0.7)
    ax4.axhline(0.5, color='k', linestyle='--', lw=0.5, alpha=0.5)
    ax4.axvline(OPEN_LOOP_TIME, color='k', linestyle=':', lw=1, alpha=0.5)
    ax4.set_ylabel('Duty')
    ax4.set_title('PWM Duties')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Info panel
    ax5.axis('off')
    ax6.axis('off')

    info_text = (
        f"PMSM FOC — Sensor Noise + CodeGen\n"
        f"{'='*38}\n"
        f"Target Speed : {SPEED_RPM} RPM\n"
        f"DC Voltage   : {V_DC} V\n"
        f"dt           : {DT*1000:.2f} ms\n"
        f"\nSensor Noise (1-σ):\n"
        f"  Current    : {NOISE_CURRENT_STD*1000:.0f} mA  (ia,ib,ic)\n"
        f"  Speed      : {NOISE_SPEED_STD:.2f} rad/s"
        f"  ({NOISE_SPEED_STD*60/(2*np.pi):.1f} RPM)\n"
        f"  Angle      : {NOISE_ANGLE_STD*1000:.1f} mrad  (θe)\n"
        f"  Seeds      : {NOISE_SEED_BASE}–{NOISE_SEED_BASE+2}\n"
        f"\nFeature 05121967:\n"
        f"  CodeGenStart/End markers active\n"
        f"  FOC chain → embedsim_loop.c\n"
    )
    ax5.text(0.05, 0.95, info_text, fontsize=9, va='top',
             fontfamily='monospace', transform=ax5.transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    # ─────────────────────────────────────────────────────────────────────────
    # Banner
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("PMSM FOC — Sensor Noise + Topology + Feature 05121967 Aurix CodeGen")
    print("=" * 70)
    print(f"Target Speed  : {SPEED_RPM} RPM")
    print(f"DC Voltage    : {V_DC} V")
    print(f"dt            : {DT * 1000:.2f} ms  ({1.0/DT:.0f} Hz sample rate)")
    print(f"Open-loop     : {OPEN_LOOP_TIME * 1000:.0f} ms at "
          f"{OPEN_LOOP_VOLTAGE}V, {OPEN_LOOP_FREQ}Hz")
    print(f"Noise — ia/ib/ic : σ={NOISE_CURRENT_STD*1000:.0f} mA  seed={NOISE_SEED_BASE}")
    print(f"Noise — ωm       : σ={NOISE_SPEED_STD:.3f} rad/s  seed={NOISE_SEED_BASE+1}")
    print(f"Noise — θe       : σ={NOISE_ANGLE_STD*1000:.1f} mrad  seed={NOISE_SEED_BASE+2}")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # Build the block diagram
    # ─────────────────────────────────────────────────────────────────────────
    # build_simulation() instantiates all VectorBlocks, wires them with >>,
    # and returns the fully-configured EmbedSim object.
    # The FMU motor plant is NOT yet running at this point.
    # ─────────────────────────────────────────────────────────────────────────
    sim = build_simulation()

    # ─────────────────────────────────────────────────────────────────────────
    # TOPOLOGY PRINT  (console)
    # ─────────────────────────────────────────────────────────────────────────
    # sim.topo is a TopologyPrinter automatically attached to every EmbedSim
    # instance by embedsim/__init__.py.  It replaces the old
    # sim.print_topology_sources2sink() call.
    #
    # print_console() renders a multi-lane text diagram in the terminal,
    # showing every block, its inputs, and the algebraic-loop breakers.
    # This is extremely useful for verifying the wiring before committing
    # a long simulation run.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("BLOCK DIAGRAM TOPOLOGY")
    print("─" * 70)
    sim.topo.print_console()

    # ─────────────────────────────────────────────────────────────────────────
    # TOPOLOGY GUI  (browser SVG — opens automatically)
    # ─────────────────────────────────────────────────────────────────────────
    # show_gui() exports an interactive SVG to a temp HTML file and opens it
    # in the default browser.  The diagram is zoomable and pan-able.
    # Comment this out for headless / CI runs.
    #
    # Equivalent call:  sim.topo.export_html("pmsm_foc_topology.html")
    # ─────────────────────────────────────────────────────────────────────────
    try:
        print("\nOpening topology GUI in browser...")
        sim.topo.show_gui()
    except Exception as exc:
        # Graceful fallback: GUI is a nice-to-have, never block the sim run
        print(f"  [topology GUI skipped: {exc}]")

    # Also export a permanent HTML copy next to the script so it can be
    # committed to the repository or shared with colleagues.
    topo_html = os.path.join(_HERE, "pmsm_foc_topology.html")
    try:
        sim.topo.export_html(topo_html)
        print(f"  Topology diagram saved → {topo_html}")
    except Exception as exc:
        print(f"  [topology export skipped: {exc}]")

    # ─────────────────────────────────────────────────────────────────────────
    # FMU INITIALISATION
    # ─────────────────────────────────────────────────────────────────────────
    # The OpenModelica-compiled PMSM FMU must be initialised before the
    # EmbedSim engine calls it.  initialize_fmu() resets all FMU states to
    # their default initial conditions (zero speed, zero angle, zero current).
    # ─────────────────────────────────────────────────────────────────────────
    print("\nInitializing FMU...")
    sim.motor.initialize_fmu(t_start=0)

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION RUN
    # ─────────────────────────────────────────────────────────────────────────
    # sim.run() steps through [0, T_END] in increments of DT using RK4.
    # Each step:
    #   1. FMU motor plant advances one step (outputs: ia, ib, ic, θm, ωm)
    #   2. MotorOutputBlock packs the FMU outputs into a flat 5-vector
    #   3. VectorDelay("motor_fb") breaks the algebraic loop by feeding the
    #      PREVIOUS step's motor state to the FOC chain
    #   4. NoisySensorBlock adds Gaussian noise to ia/ib/ic, ωm, θe
    #   5. FOC chain (SpeedPI → DQRef → Clarke → Park → SMC →
    #      InvPark → InvClarke → SVPWM) computes new duty cycles
    #   6. Duty cycles feed back into the FMU for the next step
    # ─────────────────────────────────────────────────────────────────────────
    print("\n⚙️  Running simulation...")
    sim.run()
    print("Simulation complete!")

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    t            = np.array(sim.scope.t)
    speed_data   = np.array(sim.scope.data.get("speed[0]",    [0]))
    current_data = np.array(sim.scope.data.get("currents[0]", [0]))

    final_rpm   = (speed_data[-1] * 60.0 / (2.0 * np.pi)
                   if len(speed_data) > 0 else 0.0)
    max_current = (np.max(np.abs(current_data))
                   if len(current_data) > 0 else 0.0)

    print(f"\nFinal Speed : {final_rpm:.1f} RPM")
    print(f"Target      : {SPEED_RPM} RPM")
    print(f"Error       : {abs(final_rpm - SPEED_RPM):.1f} RPM  "
          f"({abs(final_rpm - SPEED_RPM)/SPEED_RPM*100:.2f} %)")
    print(f"Max current : {max_current:.2f} A")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT
    # ─────────────────────────────────────────────────────────────────────────
    out_png = os.path.join(_HERE, "pmsm_foc_results.png")
    plot_results(sim, out_png)

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 05121967 — AURIX TRICORE C CODE GENERATION
    # ─────────────────────────────────────────────────────────────────────────
    # generate_loop() walks every block between cg_start and cg_end in
    # topological order and emits two files into
    #   <_PROJECT_ROOT>/embedsim_gen/
    #
    #   embedsim_loop.c
    #   ───────────────
    #   Contains a single function:
    #       void embedsim_loop_step(real32_T dt)
    #   which calls each block's C step function in order, passing local
    #   real32_T arrays between them.  This is the function you register as
    #   an Aurix MCAL ISR callback (or call from your OS task at 10 kHz).
    #
    #   embedsim_loop.h
    #   ───────────────
    #   Public API header:
    #     • #define EMBEDSIM_DT   — sample period derived from dt_hz
    #     • extern real32_T motor_reg[5] — shared sensor register that the
    #       Aurix integration layer writes before each step call
    #     • void embedsim_loop_init(void) — call once at startup
    #     • void embedsim_loop_step(real32_T dt) — call every 100 µs
    #
    # Aurix / TriCore specific notes:
    #   • All arithmetic uses real32_T (float32) — matches Aurix FPU width
    #   • TASKING compiler pragmas are embedded in the individual block .c
    #     files (e.g. speed_pi_controller.c) — not repeated here
    #   • MISRA C:2012 compatible: no dynamic allocation, no VLAs, no
    #     implicit function declarations
    #   • ASIL-D ready: all buffers are statically sized; no stdlib calls
    #     other than memcpy (from <string.h>)
    #
    # dt_hz parameter:
    #   Pass 1.0/DT (10 000 Hz for DT=1e-4) so the header emits the correct
    #   EMBEDSIM_DT constant.  On Aurix you would typically run the FOC ISR
    #   at 10 kHz from a GTM timer or STM compare match.
    #
    # output_dir:
    #   Files land in <_PROJECT_ROOT>/embedsim_gen/ so they sit at the repo
    #   root alongside the embedsim/ package — easy to add to your Aurix
    #   TASKING / HighTec project include path.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("FEATURE 05121967 — Aurix TriCore C Code Generation")
    print("─" * 70)

    # The output directory is <project_root>/embedsim_gen/
    # (one level above the examples/pmsm/ folder).
    gen_dir = _PROJECT_ROOT   # LoopGenerator appends /embedsim_gen automatically

    print(f"  Output directory : {os.path.join(gen_dir, "embedsim_gen")}")
    print(f"  Sample rate      : {1.0/DT:.0f} Hz  (dt = {DT*1e6:.0f} µs)")
    print(f"  Target platform  : Infineon Aurix TC3xx  (TriCore, real32_T)")
    print(f"  MISRA C:2012     : yes   ASIL-D : yes")
    print()

    try:
        # generate_loop() requires the simulation to have been run at least
        # one step so that block.output is populated and signal sizes are
        # known by the LoopGenerator / PYXInspector.
        result = sim.cg_end.generate_loop(
            cg_start   = sim.cg_start,  # CodeGenStart marker (input boundary)
            output_dir = gen_dir,        # writes into gen_dir/embedsim_gen/
            dt_hz      = 1.0 / DT,      # 10 000.0 Hz → EMBEDSIM_DT = 0.0001f
            write_files= True,           # set False to preview without writing
        )

        # Report what was generated
        c_path = os.path.join(gen_dir, "embedsim_gen", "embedsim_loop.c")
        h_path = os.path.join(gen_dir, "embedsim_gen", "embedsim_loop.h")
        # gen_dir/_PROJECT_ROOT + /embedsim_gen/ = project_root/embedsim_gen/

        if os.path.exists(c_path):
            c_lines = result["c"].count("\n")
            h_lines = result["h"].count("\n")
            print(f"  ✓  embedsim_loop.c  ({c_lines} lines)")
            print(f"  ✓  embedsim_loop.h  ({h_lines} lines)")
            print()
            print("  Aurix integration checklist:")
            print("  ─────────────────────────────────────────────────────")
            print("  1. Add embedsim_gen/ to your TASKING / HighTec include path")
            print("  2. Add the individual block .c files to your project:")
            print("     electrical_blocks/c_src/speed_pi_controller.c")
            print("     electrical_blocks/c_src/sliding_mode_controller.c")
            print("     electrical_blocks/c_src/Coordinate_Transform.c")
            print("     electrical_blocks/c_src/svpwm.c")
            print("  3. In your 10 kHz ISR or OS task:")
            print("     embedsim_loop_init();          // call once at startup")
            print("     motor_reg[0..4] = <ADC/resolver readings>;")
            print("     embedsim_loop_step(EMBEDSIM_DT); // call every 100 µs")
            print("  4. Read duty cycles from the last svpwm output buffer")
            print("     and write to your GTM / CCU6 PWM compare registers")
            print("  ─────────────────────────────────────────────────────")
        else:
            # generate_loop() may emit into a nested embedsim_gen/embedsim_gen/
            # if output_dir already ends in embedsim_gen.  Warn the user.
            print(f"  [Note] Files written to {gen_dir} "
                  f"(check for nested embedsim_gen/ subfolder)")

    except Exception as exc:
        # Code generation is non-blocking — a missing PYX_FILE or an
        # uninspected block will emit a /* Python-only */ comment in the C
        # output rather than raising, but we still catch hard errors here.
        print(f"  [CodeGen error: {exc}]")
        print("  Tip: ensure the simulation ran at least one step and all")
        print("  block .pyx files are present in electrical_blocks/c_src/")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()