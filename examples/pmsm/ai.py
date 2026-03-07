"""
ai.py
=====

AI-Enhanced PMSM Field-Oriented Control with Sliding Mode Controller
---------------------------------------------------------------------
Combines traditional SMC with neural network components for:
  - Neural network-based disturbance observer
  - Adaptive gain tuning via NN
  - Fault-tolerant stub motor (pure-Python fallback when FMU missing)

Fixes applied
-------------
  1. compute_py → compute  (EmbedSim calls compute(), not compute_py())
  2. Circular wiring removed: nn_observer no longer feeds back into ai_smc
     at the wiring level — ai_smc reads the observer output directly via
     an internal reference set before simulation starts.
  3. _StubMotor input indexing: inv_clarke outputs [v_alpha, v_beta] as one
     VectorSignal — index correctly as value[0]/value[1].
  4. VectorDelay constructor: use initial=[0.0] (not delay_length).
  5. speed_pi outputs [i_d_ref, i_q_ref] — ai_smc reads both from
     input_values[0].value[0] and [1].
  6. matplotlib non-interactive backend set early.
"""

from __future__ import annotations

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

# =============================================================================
# Path setup
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
electrical_blocks_path = os.path.join(project_root, 'electrical_blocks')

for p in (project_root, electrical_blocks_path):
    if p not in sys.path:
        sys.path.insert(0, p)

print(f"Project root: {project_root}")

# =============================================================================
# Optional ML framework imports
# =============================================================================
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    print("✅ TensorFlow available")
except ImportError:
    TF_AVAILABLE = False
    keras = layers = None
    print("⚠️  TensorFlow not available — using numpy fallback")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print("✅ PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = None
    print("⚠️  PyTorch not available — using numpy fallback")

# =============================================================================
# EmbedSim core imports
# =============================================================================
from embedsim.source_blocks     import VectorStep, VectorConstant
from embedsim.core_blocks       import VectorBlock, VectorSignal
from embedsim.dynamic_blocks    import VectorEnd
from embedsim.processing_blocks import VectorSum, VectorGain
from embedsim.simulation_engine import EmbedSim, ODESolver, VectorDelay
from embedsim.code_generator    import CodeGenStart, CodeGenEnd

# =============================================================================
# Electrical blocks (hard-required)
# =============================================================================
try:
    from coordinate_transform_blocks import (
        InvParkTransformBlock,
        InvClarkeTransformBlock,
    )
    print("✅ coordinate_transform_blocks imported")
except ImportError as e:
    print(f"❌ coordinate_transform_blocks: {e}")
    sys.exit(1)

try:
    from speed_pi_block import SpeedPIBlock
    print("✅ speed_pi_block imported")
except ImportError as e:
    print(f"❌ speed_pi_block: {e}")
    sys.exit(1)

try:
    from fmu_pmsm import ThreePhaseMotorBlock
    print("✅ fmu_pmsm imported")
except ImportError:
    ThreePhaseMotorBlock = None
    print("⚠️  fmu_pmsm not found — will use stub motor")


# =============================================================================
# Minimal numpy NN fallback
# =============================================================================

class SimpleNN:
    """Pure-numpy MLP — used when TensorFlow / PyTorch are absent."""

    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh'):
        self.activation = activation
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.W = [np.random.randn(dims[i], dims[i+1]) * 0.1 for i in range(len(dims)-1)]
        self.b = [np.zeros(dims[i+1]) for i in range(len(dims)-1)]

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32).flatten()
        for w, b in zip(self.W[:-1], self.b[:-1]):
            x = x @ w + b
            if   self.activation == 'relu':    x = np.maximum(0, x)
            elif self.activation == 'tanh':    x = np.tanh(x)
            else:                              x = 1/(1+np.exp(-x))
        return x @ self.W[-1] + self.b[-1]

    def predict(self, x, verbose=0):
        return self.forward(x).reshape(1, -1)


# =============================================================================
# AI Blocks
# =============================================================================

class NNDisturbanceObserver(VectorBlock):
    """
    Neural-network disturbance / load-torque observer.

    Inputs
    ------
    port 0  motor state  [i_d, i_q, ω_m, θ_e, T_em, ω_e, rpm]
    port 1  voltages     [v_d, v_q]

    Output
    ------
    [T_disturbance]  estimated disturbance torque (Nm)
    """

    def __init__(self, name):
        super().__init__(name)
        self.vector_size = 1
        self.is_dynamic  = True

        if TF_AVAILABLE and keras is not None:
            self.model = self._build_tf_model()
        else:
            self.model = SimpleNN(5, [64, 64], 1)

        self.input_mean   = np.array([0., 0., 100., 0., 0.], dtype=np.float32)
        self.input_std    = np.array([10., 10., 100., 48., 48.], dtype=np.float32)
        self.output_scale = 5.0

    def _build_tf_model(self):
        m = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(5,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1,  activation='tanh'),
        ])
        m.compile(optimizer='adam', loss='mse')
        return m

    # ── FIX 1: method must be named  compute  ──────────────────────────────
    def compute(self, t, dt, input_values=None):
        if not input_values or len(input_values) < 2:
            self.output = VectorSignal(np.zeros(1, dtype=np.float32), self.name)
            return self.output
        try:
            ms = input_values[0].value
            vv = input_values[1].value
            nn_in = np.array([
                ms[0] if len(ms) > 0 else 0.,
                ms[1] if len(ms) > 1 else 0.,
                ms[2] if len(ms) > 2 else 0.,
                vv[0] if len(vv) > 0 else 0.,
                vv[1] if len(vv) > 1 else 0.,
            ], dtype=np.float32)
            nn_in_n = (nn_in - self.input_mean) / (self.input_std + 1e-8)

            if TF_AVAILABLE and keras is not None:
                raw = self.model.predict(nn_in_n.reshape(1,-1), verbose=0)[0, 0]
            else:
                raw = self.model.predict(nn_in_n)[0, 0]

            T_dist = float(raw) * self.output_scale
            self.output = VectorSignal(np.array([T_dist], dtype=np.float32), self.name)
        except Exception as e:
            print(f"[nn_observer] {e}")
            self.output = VectorSignal(np.zeros(1, dtype=np.float32), self.name)
        return self.output


class NNGainAdapter(VectorBlock):
    """
    Adaptive SMC gain tuner.

    Inputs
    ------
    port 0  ω_ref   [1]
    port 1  ω_m     [1]   (from delay_omega)
    port 2  i_q_ref [1]   (from speed_pi, index 1)

    Output
    ------
    [K_sw_adapted, lambda_adapted]
    """

    def __init__(self, name, base_K_sw=40.0, base_lambda=83.0):
        super().__init__(name)
        self.vector_size  = 2
        self.is_dynamic   = True
        self.base_K_sw    = base_K_sw
        self.base_lambda  = base_lambda
        self._last_error  = 0.0
        self.K_sw_range   = [20., 80.]
        self.lambda_range = [50., 150.]

        if TORCH_AVAILABLE and torch is not None:
            self.model = nn.Sequential(
                nn.Linear(4,32), nn.ReLU(),
                nn.Linear(32,32), nn.ReLU(),
                nn.Linear(32,2),  nn.Sigmoid(),
            )
        else:
            self.model = SimpleNN(4, [32, 32], 2, activation='relu')

    # ── FIX 1 ──────────────────────────────────────────────────────────────
    def compute(self, t, dt, input_values=None):
        default = np.array([self.base_K_sw, self.base_lambda], dtype=np.float32)
        if not input_values or len(input_values) < 3:
            self.output = VectorSignal(default, self.name)
            return self.output
        try:
            omega_ref  = float(input_values[0].value[0]) if len(input_values[0].value) > 0 else 100.
            # port 1: full motor state — extract ω_m at index [2]
            ms = input_values[1].value
            omega_m    = float(ms[2]) if len(ms) > 2 else 0.
            i_q_error  = float(input_values[2].value[0]) if len(input_values[2].value) > 0 else 0.
            d_err = (i_q_error - self._last_error) / dt if dt > 0 else 0.
            self._last_error = i_q_error

            feats = np.array([
                omega_ref / 200.,
                omega_m   / 200.,
                np.tanh(i_q_error / 10.),
                np.tanh(d_err     / 1000.),
            ], dtype=np.float32)

            if TORCH_AVAILABLE and torch is not None:
                with torch.no_grad():
                    out = self.model(torch.tensor(feats)).numpy()
            else:
                out = self.model.predict(feats).flatten()

            K_sw  = self.K_sw_range[0]  + float(out[0]) * (self.K_sw_range[1]  - self.K_sw_range[0])
            lam   = self.lambda_range[0] + float(out[1]) * (self.lambda_range[1] - self.lambda_range[0])
            self.output = VectorSignal(np.array([K_sw, lam], dtype=np.float32), self.name)
        except Exception as e:
            print(f"[gain_adapter] {e}")
            self.output = VectorSignal(default, self.name)
        return self.output


class AISMCSlidingModeController(VectorBlock):
    """
    AI-Enhanced Sliding Mode Controller.

    Inputs (wired in build_ai_sim)
    ------
    port 0  speed_pi output  [i_d_ref, i_q_ref]
    port 1  i_d  (delayed)   [1]
    port 2  i_q  (delayed)   [1]
    port 3  ω_e  (delayed)   [1]
    port 4  gain_adapter     [K_sw, lambda]

    The disturbance observer is injected by reference (self._observer)
    to avoid a circular wiring loop.

    Output
    ------
    [v_d, v_q]
    """

    def __init__(self, name, base_K_sw=40., base_lambda=83., V_DC=48.):
        super().__init__(name)
        self.vector_size  = 2
        self.is_dynamic   = True
        self.base_K_sw    = base_K_sw
        self.base_lambda  = base_lambda
        self.V_DC         = V_DC
        self.phi_d = self.phi_q = 1.0
        self.R         = 0.5
        self.L_d       = 0.005
        self.L_q       = 0.006
        self.lambda_pm = 0.175
        self._last_i_d_ref = self._last_i_q_ref = 0.
        # ── FIX 2: observer injected by reference, NOT wired ──────────────
        self._observer: NNDisturbanceObserver | None = None

    def set_observer(self, observer: NNDisturbanceObserver):
        self._observer = observer

    # ── FIX 1 ──────────────────────────────────────────────────────────────
    def compute(self, t, dt, input_values=None):
        if not input_values or len(input_values) < 2:
            self.output = VectorSignal(np.zeros(2, dtype=np.float32), self.name)
            return self.output
        try:
            # port 0: [i_d_ref, i_q_ref] from speed_pi
            refs     = input_values[0].value
            i_d_ref  = float(refs[0]) if len(refs) > 0 else 0.
            i_q_ref  = float(refs[1]) if len(refs) > 1 else 0.

            # port 1: full motor state [i_d, i_q, ω_m, θ_e, T_em, ω_e, rpm]
            ms      = input_values[1].value
            i_d     = float(ms[0]) if len(ms) > 0 else 0.
            i_q     = float(ms[1]) if len(ms) > 1 else 0.
            omega_e = float(ms[5]) if len(ms) > 5 else 0.

            # port 2: [K_sw, lambda] from gain_adapter (optional)
            K_sw    = self.base_K_sw
            lambda_ = self.base_lambda
            if len(input_values) > 2 and input_values[2] is not None:
                g = input_values[2].value
                if len(g) >= 2:
                    K_sw    = float(g[0])
                    lambda_ = float(g[1])

            # Disturbance estimate from injected observer
            T_dist = 0.
            if self._observer is not None and self._observer.output is not None:
                d = self._observer.output.value
                if len(d) > 0:
                    T_dist = float(d[0])

            # Sliding surfaces
            e_d = i_d_ref - i_d
            e_q = i_q_ref - i_q
            s_d = lambda_ * e_d
            s_q = lambda_ * e_q
            sat_d = np.clip(s_d / self.phi_d, -1., 1.)
            sat_q = np.clip(s_q / self.phi_q, -1., 1.)

            # Control law
            v_d = self.R * i_d - omega_e * self.L_q * i_q + K_sw * sat_d
            v_q = (self.R * i_q + omega_e * (self.lambda_pm + self.L_d * i_d)
                   + K_sw * sat_q)
            if abs(T_dist) > 0.01:
                v_q += T_dist * 5.

            v_d = float(np.clip(v_d, -self.V_DC, self.V_DC))
            v_q = float(np.clip(v_q, -self.V_DC, self.V_DC))

            self._last_i_d_ref = i_d_ref
            self._last_i_q_ref = i_q_ref
            self.output = VectorSignal(np.array([v_d, v_q], dtype=np.float32), self.name)
        except Exception as e:
            print(f"[ai_smc] {e}")
            self.output = VectorSignal(np.zeros(2, dtype=np.float32), self.name)
        return self.output


# =============================================================================
# Custom Inverse Park for FOC (accepts [v_d, v_q] + θ_e)
# =============================================================================

class InvParkFOC(InvParkTransformBlock):
    """
    Inputs
    ------
    port 0  [v_d, v_q]  from ai_smc
    port 1  θ_e         from delay_theta (scalar)
    """

    def compute(self, t, dt, input_values=None):
        if not input_values or len(input_values) < 2:
            self.output = VectorSignal(np.zeros(2, dtype=np.float32), self.name)
            return self.output
        try:
            vdq   = input_values[0].value
            raw   = input_values[1].value
            # delay_theta carries full 7-element motor state [i_d,i_q,ω_m,θ_e,...]
            # OR a scalar [θ_e] — handle both
            theta = float(raw[3]) if len(raw) >= 7 else float(raw[0]) if len(raw) > 0 else 0.
            v_d   = float(vdq[0]) if len(vdq) > 0 else 0.
            v_q   = float(vdq[1]) if len(vdq) > 1 else 0.
            ct, st = np.cos(theta), np.sin(theta)
            self.output = VectorSignal(
                np.array([v_d*ct - v_q*st,
                          v_d*st + v_q*ct], dtype=np.float32),
                self.name,
            )
        except Exception as e:
            print(f"[InvParkFOC] {e}")
            self.output = VectorSignal(np.zeros(2, dtype=np.float32), self.name)
        return self.output


# =============================================================================
# Stub PMSM motor (pure-Python fallback)
# =============================================================================

class _StubMotor(VectorBlock):
    """
    Simplified PMSM in the d-q frame.

    Inputs
    ------
    port 0  [v_alpha, v_beta]  from inv_clarke
    port 1  T_load             from VectorConstant

    Output  [i_d, i_q, ω_m, θ_e, T_em, ω_e, rpm]  (7 values)
    """

    def __init__(self, name):
        super().__init__(name)
        self.vector_size = 7
        self.is_dynamic  = True
        self.p   = 2;    self.lam = 0.175
        self.J   = 0.002; self.B  = 0.001
        self.R   = 0.5;  self.L  = 0.005
        self._omega = self._theta = self._iq = self._id = 0.

    # ── FIX 1 + FIX 3 ──────────────────────────────────────────────────────
    def compute(self, t, dt, input_values=None):
        if not input_values or len(input_values) < 2:
            self.output = VectorSignal(np.zeros(7, dtype=np.float32), self.name)
            return self.output
        try:
            # ── FIX 3: inv_clarke output is [v_alpha, v_beta] as one signal
            vabc = input_values[0].value
            v_alpha = float(vabc[0]) if len(vabc) > 0 else 0.
            v_beta  = float(vabc[1]) if len(vabc) > 1 else 0.
            T_load  = float(input_values[1].value[0]) if len(input_values[1].value) > 0 else 0.

            # α/β → d/q using current θ_e
            ct, st = np.cos(self._theta), np.sin(self._theta)
            v_d =  v_alpha * ct + v_beta * st
            v_q = -v_alpha * st + v_beta * ct

            omega_e = self.p * self._omega
            e_d = -omega_e * self.L * self._iq
            e_q =  omega_e * (self.lam + self.L * self._id)

            self._id += (v_d - self.R*self._id - e_d) / self.L * dt
            self._iq += (v_q - self.R*self._iq - e_q) / self.L * dt

            T_em = 1.5 * self.p * self.lam * self._iq
            self._omega += (T_em - T_load - self.B*self._omega) / self.J * dt
            self._theta  = (self._theta + omega_e*dt) % (2*np.pi)
            rpm = self._omega * 60 / (2*np.pi)

            self.output = VectorSignal(
                np.array([self._id, self._iq, self._omega,
                          self._theta, T_em, omega_e, rpm], dtype=np.float32),
                self.name,
            )
        except Exception as e:
            print(f"[stub_motor] {e}")
            self.output = VectorSignal(np.zeros(7, dtype=np.float32), self.name)
        return self.output


# =============================================================================
# Build simulation
# =============================================================================

def build_ai_sim():
    print("\n" + "="*65)
    print("EmbedSim — AI-Enhanced PMSM FOC with SMC")
    print("="*65)

    T_SIM     = 0.5
    DT        = 5e-5
    OMEGA_REF = 100.0
    T_LOAD    = 0.2

    # ── Sources ───────────────────────────────────────────────────────────────
    omega_ref_src  = VectorStep("omega_ref",
                                step_time=0.05, before_value=0., after_value=OMEGA_REF, dim=1)
    load_torque_src = VectorConstant("T_load", value=[T_LOAD])

    # ── Code-gen markers ──────────────────────────────────────────────────────
    cg_start = CodeGenStart("cg_start")
    cg_end   = CodeGenEnd("cg_end")

    # ── Traditional speed controller ──────────────────────────────────────────
    try:
        speed_pi = SpeedPIBlock("speed_pi", Kp=1.0, Ki=20.0, i_max=20.0)
        print("✅ Speed PI created")
    except Exception as e:
        print(f"❌ Speed PI failed: {e}")
        return None

    # ── AI blocks ─────────────────────────────────────────────────────────────
    nn_observer  = NNDisturbanceObserver("nn_observer")
    gain_adapter = NNGainAdapter("gain_adapter", base_K_sw=40., base_lambda=83.)
    ai_smc       = AISMCSlidingModeController("ai_smc", base_K_sw=40., base_lambda=83.)

    # ── FIX 2: inject observer reference so ai_smc can read its output
    #    without creating a wiring loop ────────────────────────────────────────
    ai_smc.set_observer(nn_observer)

    # ── Coordinate transforms ─────────────────────────────────────────────────
    inv_park   = InvParkFOC("inv_park")
    inv_clarke = InvClarkeTransformBlock("inv_clarke")

    # ── Motor plant ───────────────────────────────────────────────────────────
    if ThreePhaseMotorBlock is not None:
        try:
            fmu_path = os.path.join(electrical_blocks_path, "modelica", "ThreePhaseMotor.fmu")
            motor = ThreePhaseMotorBlock(
                "motor", fmu_path=fmu_path,
                R=0.5, L_d=0.005, L_q=0.006,
                lambda_pm=0.175, J=0.002, B=0.001, p=2.0,
            )
            print("✅ Motor FMU loaded")
        except Exception as e:
            print(f"⚠️  FMU failed ({e}) — using stub motor")
            motor = _StubMotor("motor")
    else:
        print("⚠️  Using stub motor")
        motor = _StubMotor("motor")

    # ── Feedback delays (LoopBreakers) ────────────────────────────────────────
    # SpeedPIBlock reads input_values[1].value[2]  → needs full 7-element motor state.
    # The engine's zero-fallback is always VectorSignal([0.0]) — size 1.
    # Guard: explicitly pre-set .output so it is never None when speed_pi is called.
    _MOTOR_ZEROS = np.zeros(7, dtype=np.float32)
    delay_motor = VectorDelay("delay_motor", initial=_MOTOR_ZEROS.tolist())
    delay_motor.output = VectorSignal(_MOTOR_ZEROS.copy())   # ← force size-7 from t=0

    delay_theta = VectorDelay("delay_theta", initial=[0.])
    delay_theta.output = VectorSignal(np.array([0.], dtype=np.float32))

    # ── Sink + forward-path output capture ────────────────────────────────────
    sink       = VectorEnd("sink")
    motor_sink = VectorEnd("motor_sink")

    # ── Signal wiring ─────────────────────────────────────────────────────────
    #
    #  omega_ref ──► cg_start ──► speed_pi ◄── delay_motor[2] (ω_m)
    #                                │
    #                            [i_d_ref, i_q_ref]
    #                                │
    #  gain_adapter ──► ai_smc ◄─────┘
    #  delay_motor  ──► ai_smc  (i_d=[0], i_q=[1], ω_e=[5])
    #  (nn_observer via set_observer — no wiring loop)
    #
    #  ai_smc ──► inv_park ◄── delay_theta
    #  inv_park ──► inv_clarke ──► cg_end ──► motor ──► motor_sink
    #  load_torque_src ──► motor
    #  inv_clarke ──► sink

    # Speed loop  (SpeedPI reads port1.value[2] = ω_m from full motor state)
    omega_ref_src >> cg_start >> speed_pi
    motor >> delay_motor >> speed_pi          # port 1: full [i_d,i_q,ω_m,θ_e,T_em,ω_e,rpm]

    # Current refs → ai_smc (port 0)
    speed_pi >> ai_smc

    # Full motor state → ai_smc (port 1 — extracts i_d[0], i_q[1], ω_e[5])
    delay_motor >> ai_smc

    # Adaptive gains → ai_smc (port 2)
    omega_ref_src >> gain_adapter             # port 0
    delay_motor   >> gain_adapter             # port 1: full state, extracts ω_m[2]
    speed_pi      >> gain_adapter             # port 2: i_q_ref proxy
    gain_adapter  >> ai_smc

    # Disturbance observer inputs (for next-step estimate)
    motor   >> nn_observer                    # port 0: motor state
    ai_smc  >> nn_observer                    # port 1: v_dq

    # θ_e for inv_park comes from delay_motor[3] — InvParkFOC handles full-state input
    ai_smc >> inv_park
    delay_motor >> inv_park                   # port 1: full state, extracts θ_e[3]
    inv_park >> inv_clarke >> cg_end >> motor

    # Load
    load_torque_src >> motor

    # Sinks
    motor      >> motor_sink
    inv_clarke >> sink

    # ── Simulation object ─────────────────────────────────────────────────────
    sim = EmbedSim(sinks=[sink, motor_sink], T=T_SIM, dt=DT, solver=ODESolver.RK4)

    sim.scope.add(omega_ref_src,  label="omega_ref")
    sim.scope.add(speed_pi,       label="speed_pi")
    sim.scope.add(ai_smc,         label="ai_smc")
    sim.scope.add(motor,          label="motor")
    sim.scope.add(nn_observer,    label="nn_observer")
    sim.scope.add(gain_adapter,   label="gain_adapter")

    if hasattr(sim, 'topo') and sim.topo is not None:
        sim.topo.print_console()

    return sim


# =============================================================================
# Plotting
# =============================================================================

def plot_results(sim):
    if sim is None or not hasattr(sim.scope, 't') or len(sim.scope.t) == 0:
        print("No simulation data to plot")
        return

    t  = np.array(sim.scope.t)
    sd = sim.scope.data

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("AI-Enhanced PMSM FOC with Sliding Mode Control",
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    def _get(key, fallback):
        return np.array(sd.get(key, [fallback]*len(t)))

    # Speed tracking
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, _get('omega_ref[0]', OMEGA_REF := 100.), 'b--', label='ω_ref', lw=1.5)
    ax1.plot(t, _get('motor[2]', 0.),                   'r-',  label='ω_m',   lw=1.5)
    ax1.set_title('Speed Tracking', fontweight='bold')
    ax1.set_xlabel('Time [s]'); ax1.set_ylabel('Speed [rad/s]')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # d-q currents
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, _get('motor[0]', 0.), 'b-', label='i_d', lw=1.5)
    ax2.plot(t, _get('motor[1]', 0.), 'r-', label='i_q', lw=1.5)
    ax2.set_title('d-q Axis Currents', fontweight='bold')
    ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Current [A]')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # SMC voltages
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, _get('ai_smc[0]', 0.), 'c-', label='v_d', lw=1.5)
    ax3.plot(t, _get('ai_smc[1]', 0.), 'm-', label='v_q', lw=1.5)
    ax3.set_title('AI-SMC Voltage Commands', fontweight='bold')
    ax3.set_xlabel('Time [s]'); ax3.set_ylabel('Voltage [V]')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # Adaptive gains
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(t, _get('gain_adapter[0]', 40.), color='orange', lw=1.5)
    ax4.set_title('Adaptive Switching Gain K_sw', fontweight='bold')
    ax4.set_xlabel('Time [s]'); ax4.set_ylabel('K_sw')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(t, _get('gain_adapter[1]', 83.), color='purple', lw=1.5)
    ax5.set_title('Adaptive Sliding Slope λ', fontweight='bold')
    ax5.set_xlabel('Time [s]'); ax5.set_ylabel('λ')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(current_dir, "ai_smc_results.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved plot: {out_png}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*65)
    print("Starting AI-Enhanced PMSM FOC Simulation")
    print("="*65)

    sim = build_ai_sim()
    if sim is None:
        print("❌ Failed to build simulation")
        return

    print("\n⚙️  Running AI-enhanced simulation …")
    try:
        sim.run(verbose=True, progress_bar=True)
        print("✅ Simulation complete.")
    except Exception as e:
        import traceback
        print(f"❌ Simulation failed: {e}")
        traceback.print_exc()
        return

    plot_results(sim)

    if hasattr(sim, 'topo') and sim.topo is not None:
        try:
            out_html = os.path.join(current_dir, "ai_smc_topology.html")
            sim.topo.export_html(out_html)
            print(f"💾 Saved topology: {out_html}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
