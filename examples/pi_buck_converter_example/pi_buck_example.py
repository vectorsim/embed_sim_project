"""
pi_buck_example.py  —  Canonical EmbedSim Buck Converter Example
=================================================================

OVERVIEW
--------
This is the single canonical example for the buck converter in EmbedSim.
It combines two simulation modes in one script:

  Mode 1 — Fixed-gain PI demo
  ───────────────────────────
  Closed-loop voltage control with a hand-tuned PI controller.
  Produces: pi_buck_response.png, pi_buck.html, embedsim_loop.c/h

  Mode 2 — FMU-Probed Neural PI Tuner  (three-phase pipeline)
  ────────────────────────────────────
  Phase 1: FMU Prober
    Sweep (V_ref, R_load, V_in) x (Kp, Ki) grid via mini-simulations.
    Operating-point variation is achieved by patching BuckConverterBlock.DEFAULT_PARAMS
    before each instantiation — no constructor kwargs needed (PMSM pattern preserved).

  Phase 2: Neural Network Training
    Train a small MLP: [V_ref, R_load, V_in] -> [Kp_best, Ki_best].

  Phase 3: Closed-Loop Demonstration
    Run full 10 ms simulation with AI-PI controller.
    Compare against fixed-gain baseline.

SYSTEM BLOCK DIAGRAM (both modes)
──────────────────────────────────
  V_ref --> [PI / AI-PI Controller] --> [BuckConverterBlock FMU] --> V_out
                      ^                           |
                      |_____ [ScalarDelay] <_______

PLANT DEFAULT PARAMETERS (BuckConverterBlock.DEFAULT_PARAMS / BuckConverter.mo)
────────────────────────────────────────────────────────────────────────────────
  L=100µH  C=100µF  R_load=10Ω  V_in=24V  f_sw=100kHz

FIXED-GAIN TUNING (Mode 1)
──────────────────────────
  Kp=0.15  Ki=8.0  duty=[0.10, 0.90]  Ts=100µs

Dependencies: PyTorch (Mode 2 only), NumPy, Matplotlib, FMPy, EmbedSim
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import sys
import time
import warnings
import json
from pathlib import Path
from typing  import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ── Project path via _path_utils ─────────────────────────────────────────────
from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE        = get_current_parent()          # examples/pi_buck_converter_example/
project_root = get_project_root()
sys.path.insert(0, get_embedsim_import_path())

# ── EmbedSim ──────────────────────────────────────────────────────────────────
from embedsim.simulation_engine import EmbedSim, ODESolver, VectorDelay
from embedsim.source_blocks     import VectorStep
from embedsim.dynamic_blocks    import VectorEnd
from embedsim.code_generator    import CodeGenStart, CodeGenEnd, StepGenerator as LoopGenerator
from embedsim.topology_printer  import TopologyPrinter
from embedsim.core_blocks       import VectorSignal
from embedsim.plot_helper       import create_plotter

# ── Buck converter blocks ─────────────────────────────────────────────────────
sys.path.append(str(project_root / "buck_converter"))
from pi_buck_block      import PI_BuckBlock
from BuckConverterBlock import BuckConverterBlock

# ── Shared constants ──────────────────────────────────────────────────────────
FMU_PATH    = str(project_root / "buck_converter" / "modelica" / "BuckConverter.fmu")
CODEGEN_DIR = str(project_root / "embedsim_gen")

# Default parameter snapshot — used to restore after mini-sim patches
_DEFAULT_PARAMS_ORIG = dict(BuckConverterBlock.DEFAULT_PARAMS)


# ==============================================================================
# SHARED HELPER — ScalarDelay
# ==============================================================================

class ScalarDelay(VectorDelay):
    """
    One-step delay that forwards only index-0 of the upstream signal.

    buck_plant outputs [V_out, I_L, I_load] (size 3).
    Only V_out (index 0) is needed as PI feedback.
    Strips I_L and I_load so EmbedSim_Input_T gets a scalar field,
    matching the in->fb_delay reference in PI_BuckBlock.C_CUSTOM_EMIT.
    """
    def compute_py(self, t, dt, input_values=None):
        sig = super().compute_py(t, dt, input_values)
        self.output = VectorSignal(
            np.array([sig.value[0]], dtype=np.float32), self.name
        )
        self.vector_size = 1
        return self.output


# ==============================================================================
# SECTION 1 — FIXED-GAIN PI DEMO
# ==============================================================================

def run_fixed_pi_demo() -> 'EmbedSim':
    """
    Mode 1: hand-tuned PI controller driving the BuckConverter FMU.

    Produces:
      pi_buck_response.png  — 3-panel voltage/duty/current plot
      pi_buck.html          — interactive topology diagram
      embedsim_gen/         — embedsim_loop.c/.h for AURIX TC38x
    """

    # ── Blocks ────────────────────────────────────────────────────────────────
    v_ref = VectorStep(
        "vref", step_time=0.001, before_value=0.0, after_value=12.0, dim=1,
    )

    # PI tuned for L=100µH, C=100µF, R_load=10Ω, V_in=24V
    pi_controller = PI_BuckBlock(
        name="pi_buck",
        Kp=0.15, Ki=8.0,
        duty_max=0.9, duty_min=0.1,
        Ts=1e-4,
        use_c_backend=True,     # uses pi_buck_wrapper.pyd
    )

    # Plant — DEFAULT_PARAMS from BuckConverter.mo (same pattern as PMSM)
    buck_plant = BuckConverterBlock(name="buck", fmu_path=FMU_PATH)

    feedback_delay = ScalarDelay("fb_delay", initial=[0.0])
    sink           = VectorEnd("sink")
    cg_start       = CodeGenStart("pi_ctrl_start")
    cg_end         = CodeGenEnd("pi_ctrl_end")

    # ── Wiring ────────────────────────────────────────────────────────────────
    v_ref         >> cg_start
    cg_start      >> pi_controller
    pi_controller >> cg_end
    cg_end        >> buck_plant
    buck_plant    >> sink
    buck_plant    >> feedback_delay
    feedback_delay >> pi_controller
    feedback_delay >> cg_start      # exposes V_meas to StepGenerator

    print("\n✅ FMU outputs:", buck_plant.OUTPUT_VARS)

    WIRE_LABELS = {
        ("vref",          "pi_ctrl_start"): "[V_ref]",
        ("pi_ctrl_start", "pi_buck"):       "[V_ref]",
        ("pi_buck",       "pi_ctrl_end"):   "[duty]",
        ("pi_ctrl_end",   "buck"):          "[duty]",
        ("buck",          "fb_delay"):      "[V_out]",
        ("fb_delay",      "pi_buck"):       "[V_meas]",
        ("buck",          "sink"):          "[V_out, I_L, I_load]",
    }

    # ── Topology + Simulation ─────────────────────────────────────────────────
    sim = EmbedSim(sinks=[sink], T=0.01, dt=1e-6, solver=ODESolver.RK4)

    TopologyPrinter(sim, title="Buck Converter — PI Voltage Control",
                    wire_labels=WIRE_LABELS).print_console()
    sim.topo.show_gui()
    sim.topo.export_html(str(_HERE / "pi_buck.html"), wire_labels=WIRE_LABELS)

    sim.scope.add(v_ref,           label="v_ref")
    sim.scope.add(pi_controller,   label="pi_ctrl")
    sim.scope.add(buck_plant,      label="buck_out", indices=[0, 1])
    sim.scope.add(feedback_delay,  label="fb_delay")

    print("\nRunning fixed-PI simulation ...")
    sim.run(verbose=True, progress_bar=True)
    print("Simulation complete!")

    # ── CodeGen ───────────────────────────────────────────────────────────────
    gen = LoopGenerator(cg_start, cg_end, prefix="EmbedSim", dt_hz=1e6)
    gen.generate(output_dir=CODEGEN_DIR)
    print(f"  -> C files written to: {CODEGEN_DIR}/")
    print( "     embedsim_loop.c  embedsim_loop.h")

    # ── Plot ──────────────────────────────────────────────────────────────────
    create_plotter(sim).plot_grid([
        dict(signal="buck_out[0]", ylabel="Voltage (V)",
             title="Output Voltage  V_out",  color="#58a6ff",
             ylim=(-1, 20), ref_val=12.0, ref_label="V_ref = 12 V", step_time=1.0),
        dict(signal="pi_ctrl[0]",  ylabel="Duty cycle",
             title="PI Controller — Duty Cycle", color="#3fb950", ylim=(0.0, 1.0)),
        dict(signal="buck_out[1]", ylabel="Current (A)",
             title="Inductor Current  I_L", color="#d2a8ff", ylim=(-12, 12)),
    ], title="Buck Converter — PI Voltage Control (EmbedSim)",
       save_path=str(_HERE / "pi_buck_response.png"))

    return sim


# ==============================================================================
# SECTION 2 — MINI-SIMULATION HELPER  (Phase 1 support)
# ==============================================================================

def run_mini_sim(
    Kp: float, Ki: float,
    V_ref:  float = 12.0,
    R_load: float = 10.0,
    V_in:   float = 24.0,
    T_sim:  float = 0.020,   # longer window — gives weak gains time to settle
    dt:     float = 1e-6,    # same as final sim — consistent dynamics
    step_at: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one closed-loop mini-simulation against the FMU.

    Operating-point variation (R_load, V_in) is applied by temporarily
    patching BuckConverterBlock.DEFAULT_PARAMS before instantiation and
    restoring original defaults afterwards.
    This preserves the PMSM constructor pattern: __init__(self, name, fmu_path).

    Returns (t [s], V_out [V]).
    Uses Euler solver for speed — acceptable for gain-pair ranking.
    """
    # ── Patch DEFAULT_PARAMS for this operating point ─────────────────────────
    BuckConverterBlock.DEFAULT_PARAMS['R_load'] = R_load
    BuckConverterBlock.DEFAULT_PARAMS['V_in']   = V_in

    try:
        v_ref_blk = VectorStep("vref", step_time=step_at,
                                before_value=0.0, after_value=V_ref, dim=1)
        pi_blk    = PI_BuckBlock("pi", Kp=Kp, Ki=Ki,
                                  duty_max=0.9, duty_min=0.1, Ts=dt,
                                  use_c_backend=False)
        # BuckConverterBlock picks up the patched R_load and V_in from DEFAULT_PARAMS
        buck_blk  = BuckConverterBlock("buck", fmu_path=FMU_PATH)
        sink      = VectorEnd("sink")
        fb_dly    = VectorDelay("fb", initial=[0.0])

        v_ref_blk >> pi_blk >> buck_blk >> sink
        buck_blk  >> fb_dly  >> pi_blk

        sim = EmbedSim(sinks=[sink], T=T_sim, dt=dt, solver=ODESolver.RK4)
        sim.scope.add(buck_blk, label="vout", indices=[0])
        sim.run(verbose=False, progress_bar=False)

        t_arr = np.array(sim.scope.t, dtype=float)
        raw   = sim.scope.data.get("vout[0]") or sim.scope.data.get("vout") or []
        v_out = np.array(list(raw), dtype=float)
        n     = min(len(t_arr), len(v_out))
        if n < 10:
            return np.linspace(0, T_sim, 20), np.zeros(20)
        return t_arr[:n], v_out[:n]

    finally:
        # ── Always restore DEFAULT_PARAMS ─────────────────────────────────────
        BuckConverterBlock.DEFAULT_PARAMS.update(_DEFAULT_PARAMS_ORIG)


# ==============================================================================
# SECTION 3 — STEP-RESPONSE METRICS
# ==============================================================================

def compute_cost(
    t: np.ndarray, v_out: np.ndarray,
    V_ref: float, step_at: float = 0.001,
) -> float:
    """
    Scalar ITAE + overshoot + SSE cost. Lower is better. 1e6 = failed sim.

    cost = ITAE_normalised + 2.0 * overshoot_fraction + 5.0 * sse_fraction
    """
    mask = t >= step_at
    if not np.any(mask) or len(v_out[mask]) < 10:
        return 1e6

    t_post = t[mask];  v_post = v_out[mask]
    tau    = t_post - t_post[0]
    err    = np.abs(v_post - V_ref)

    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    itae   = float(_trapz(tau * err, tau))
    T_win  = float(tau[-1])
    norm   = V_ref * (T_win ** 2) / 2.0
    itae_n = itae / norm if norm > 1e-12 else 1.0

    overshoot = max(0.0, (float(np.max(v_post)) - V_ref) / V_ref)
    sse       = float(np.mean(err[int(0.75 * len(v_post)):]) / V_ref)

    return float(np.clip(itae_n + 2.0 * overshoot + 5.0 * sse, 0.0, 10.0))


# ==============================================================================
# SECTION 4 — FMU PROBER  (Phase 1)
# ==============================================================================

class FMUProber:
    """
    Probe the FMU plant to find optimal PI gains at each operating point.

    Sweeps (V_ref, R_load, V_in) x (Kp, Ki) grid via mini-simulations.
    Parameter variation uses DEFAULT_PARAMS patching — no constructor kwargs.
    """

    KP_GRID_FULL = [0.05, 0.10, 0.15, 0.25, 0.40]
    KI_GRID_FULL = [3.0,  6.0, 10.0, 15.0, 20.0]
    KP_GRID_FAST = [0.08, 0.15, 0.25, 0.38]
    KI_GRID_FAST = [3.0,  8.0, 13.0, 18.0]

    VREF_GRID_FULL  = [8.0, 10.0, 12.0, 15.0]
    RLOAD_GRID_FULL = [5.0, 10.0, 18.0]
    VIN_GRID_FULL   = [20.0, 24.0]
    VREF_GRID_FAST  = [8.0, 12.0, 15.0]
    RLOAD_GRID_FAST = [5.0, 10.0, 18.0]
    VIN_GRID_FAST   = [24.0]

    def __init__(self, fast: bool = False) -> None:
        self.fast    = fast
        self.dataset: List[Dict] = []
        self._kp_grid    = self.KP_GRID_FAST    if fast else self.KP_GRID_FULL
        self._ki_grid    = self.KI_GRID_FAST    if fast else self.KI_GRID_FULL
        self._vref_grid  = self.VREF_GRID_FAST  if fast else self.VREF_GRID_FULL
        self._rload_grid = self.RLOAD_GRID_FAST if fast else self.RLOAD_GRID_FULL
        self._vin_grid   = self.VIN_GRID_FAST   if fast else self.VIN_GRID_FULL

    def probe(self) -> List[Dict]:
        op_points = [(vr, rl, vi)
                     for vr in self._vref_grid
                     for rl in self._rload_grid
                     for vi in self._vin_grid]
        n_op    = len(op_points)
        n_gains = len(self._kp_grid) * len(self._ki_grid)
        total   = n_op * n_gains

        print(f"\n{'='*65}")
        print(f"  FMU PROBER — {n_op} operating points x {n_gains} gain pairs")
        print(f"  Total mini-simulations: {total}  "
              f"({'fast' if self.fast else 'full'} mode)")
        print(f"{'='*65}\n")

        t_start = time.time()
        done    = 0

        for idx, (V_ref, R_load, V_in) in enumerate(op_points):
            best_cost = 1e9
            best_Kp   = self._kp_grid[0]
            best_Ki   = self._ki_grid[0]

            for Kp in self._kp_grid:
                for Ki in self._ki_grid:
                    try:
                        t_s, v_s = run_mini_sim(Kp=Kp, Ki=Ki,
                                                V_ref=V_ref, R_load=R_load, V_in=V_in)
                        cost = compute_cost(t_s, v_s, V_ref)
                    except Exception as exc:
                        cost = 1e6
                    if cost < best_cost:
                        best_cost = cost;  best_Kp = Kp;  best_Ki = Ki
                    done += 1

            self.dataset.append({
                "V_ref": V_ref, "R_load": R_load, "V_in": V_in,
                "Kp_best": best_Kp, "Ki_best": best_Ki, "cost_best": best_cost,
            })

            elapsed = time.time() - t_start
            remain  = (total - done) / (done / elapsed) if done else 0
            print(f"  [{idx+1:3d}/{n_op}] "
                  f"V_ref={V_ref:5.1f}V  R={R_load:4.1f}Ω  Vin={V_in:4.1f}V  "
                  f"->  Kp={best_Kp:.3f}  Ki={best_Ki:5.1f}  "
                  f"cost={best_cost:.4f}   ETA {remain/60:.1f}min")

        print(f"\n✅ Probing complete — {len(self.dataset)} data points "
              f"in {time.time()-t_start:.1f}s\n")
        return self.dataset

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.dataset, f, indent=2)
        print(f"💾 Dataset saved -> {path}")

    @staticmethod
    def load(path: str) -> List[Dict]:
        with open(path) as f:
            data = json.load(f)
        print(f"📂 Dataset loaded — {len(data)} points from {path}")
        return data

    def plot_gain_surface(self) -> None:
        if not self.dataset:
            return
        vref  = np.array([d["V_ref"]   for d in self.dataset])
        rload = np.array([d["R_load"]  for d in self.dataset])
        kp    = np.array([d["Kp_best"] for d in self.dataset])
        ki    = np.array([d["Ki_best"] for d in self.dataset])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("FMU-Probed Optimal Gain Surface", fontsize=13, fontweight='bold')
        for ax, y, lbl in zip(axes, [kp, ki], ["Kp_best", "Ki_best"]):
            sc = ax.scatter(vref, y, c=rload, cmap='plasma', s=80, alpha=0.8)
            ax.set_xlabel("V_ref (V)");  ax.set_ylabel(lbl)
            ax.set_title(f"Optimal {lbl} vs V_ref  (colour = R_load)")
            plt.colorbar(sc, ax=ax, label="R_load (Ω)")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = _HERE / "fmu_gain_surface.png"
        plt.savefig(str(out), dpi=120, bbox_inches='tight')
        print(f"📈 Gain surface saved -> {out}")
        plt.show(block=False);  plt.pause(0.1)


# ==============================================================================
# SECTION 5 — NEURAL NETWORK  (Phase 2)
# ==============================================================================

def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        return torch, nn, optim
    except ImportError:
        raise ImportError("PyTorch required for Mode 2.  Install: pip install torch")

INPUT_DIM  = 3   # V_ref, R_load, V_in
OUTPUT_DIM = 2   # Kp, Ki


class GainNet:
    """
    MLP: [V_ref/20, R_load/20, V_in/30] -> [Kp, Ki]
    PyTorch imported lazily so Mode 1 works without it installed.
    """
    KP_SCALE = 0.5
    KI_SCALE = 20.0

    def __init__(self, hidden: int = 64):
        torch, nn, _ = _import_torch()
        self._torch = torch
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, hidden//2),nn.ReLU(),
            nn.Linear(hidden//2, OUTPUT_DIM), nn.Sigmoid(),
        )

    def forward(self, x):
        raw = self.net(x)
        torch = self._torch
        return torch.cat([raw[:, 0:1] * self.KP_SCALE,
                          raw[:, 1:2] * self.KI_SCALE], dim=1)

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self.net.parameters()

    def eval(self):
        self.net.eval()

    def encode(self, V_ref: float, R_load: float, V_in: float):
        torch = self._torch
        return torch.tensor([[V_ref/20.0, R_load/20.0, V_in/30.0]],
                             dtype=torch.float32)

    def predict(self, V_ref: float, R_load: float = 10.0,
                V_in: float = 24.0) -> Tuple[float, float]:
        torch = self._torch
        with torch.no_grad():
            out = self(self.encode(V_ref, R_load, V_in))
        return float(out[0, 0]), float(out[0, 1])

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


def train_gain_net(dataset: List[Dict], epochs: int = 300,
                   lr: float = 0.005, batch: int = 16) -> GainNet:
    torch, nn, optim = _import_torch()
    print(f"\n{'='*55}")
    print(f"  NN TRAINING  —  {len(dataset)} samples, {epochs} epochs")
    print(f"{'='*55}\n")

    net = GainNet(hidden=64)
    X   = torch.tensor([[d["V_ref"]/20.0, d["R_load"]/20.0, d["V_in"]/30.0]
                         for d in dataset], dtype=torch.float32)
    Y   = torch.tensor([[d["Kp_best"], d["Ki_best"]] for d in dataset],
                        dtype=torch.float32)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    N = len(dataset);  B = min(batch, N);  losses = []

    for epoch in range(1, epochs+1):
        perm    = torch.randperm(N)
        ep_loss = 0.0;  n_batch = 0
        for i in range(0, N, B):
            xb   = X[perm[i:i+B]];  yb = Y[perm[i:i+B]]
            pred = net(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad();  loss.backward();  optimizer.step()
            ep_loss += loss.item();  n_batch += 1
        scheduler.step()
        avg = ep_loss / n_batch;  losses.append(avg)
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs}  loss={avg:.6f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.5f}")

    print(f"\n✅ Training done — final loss = {losses[-1]:.6f}\n")

    plt.figure(figsize=(9, 3))
    plt.semilogy(losses, 'b-', lw=2)
    plt.xlabel('Epoch');  plt.ylabel('MSE Loss (log)')
    plt.title('GainNet Training Loss');  plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(_HERE / "gain_net_training.png"), dpi=120, bbox_inches='tight')
    plt.show(block=False);  plt.pause(0.1)
    return net


# ==============================================================================
# SECTION 6 — AI PI CONTROLLER BLOCK
# ==============================================================================

class AIPIController(PI_BuckBlock):
    """
    PI_BuckBlock with GainNet gain scheduling.
    Predicts (Kp, Ki) from plant operating conditions each step.
    """
    def __init__(self, name: str, gain_net: GainNet,
                 R_load: float = 10.0, V_in: float = 24.0, **kwargs) -> None:
        super().__init__(name, use_c_backend=False, **kwargs)
        self.gain_net      = gain_net
        self.R_load        = R_load
        self.V_in          = V_in
        self.kp_history: List[float] = []
        self.ki_history: List[float] = []

    def compute_py(self, t: float, dt: float,
                   input_values: Optional[list] = None) -> VectorSignal:
        inputs = input_values or []
        V_ref  = float(inputs[0].value[0]) if len(inputs) > 0 else 0.0
        V_meas = float(inputs[1].value[0]) if len(inputs) > 1 else 0.0

        Kp, Ki = self.gain_net.predict(V_ref, self.R_load, self.V_in)
        Kp = float(np.clip(Kp, 0.01, 0.5))
        Ki = float(np.clip(Ki, 0.10, 20.0))
        self._Kp = Kp;  self._Ki = Ki
        self.kp_history.append(Kp);  self.ki_history.append(Ki)

        error = V_ref - V_meas
        if not hasattr(self, '_integral'):
            self._integral = 0.0
        self._integral += error * dt
        max_int = 0.5 / max(Ki, 0.001)
        self._integral = float(np.clip(self._integral, -max_int, max_int))

        duty = float(np.clip(Kp * error + Ki * self._integral,
                              self._duty_min, self._duty_max))
        self.output = VectorSignal(np.array([duty], dtype=np.float32),
                                   self.name, dtype=self.dtype)
        return self.output


# ==============================================================================
# SECTION 7 — AI CLOSED-LOOP SIMULATION  (Phase 3)
# ==============================================================================

def run_ai_simulation(
    gain_net: GainNet,
    R_load:   float = 10.0,
    V_in:     float = 24.0,
    V_ref:    float = 12.0,
    T_sim:    float = 0.010,
    load_step_at: Optional[float] = 0.005,
    load_step_to: float = 5.0,
) -> Tuple['EmbedSim', AIPIController]:

    Kp0, Ki0 = gain_net.predict(V_ref, R_load, V_in)
    print(f"\n  NN predicted initial gains:  Kp={Kp0:.4f}  Ki={Ki0:.3f}")

    v_ref_blk = VectorStep("vref", step_time=0.001,
                            before_value=0.0, after_value=V_ref, dim=1)
    ai_ctrl   = AIPIController("ai_pi", gain_net=gain_net,
                                R_load=R_load, V_in=V_in,
                                Kp=Kp0, Ki=Ki0, duty_max=0.9, duty_min=0.1)
    buck      = BuckConverterBlock("buck", fmu_path=FMU_PATH)
    sink      = VectorEnd("sink")
    fb_dly    = VectorDelay("fb", initial=[0.0])

    v_ref_blk >> ai_ctrl >> buck >> sink
    buck >> fb_dly >> ai_ctrl

    sim = EmbedSim(sinks=[sink], T=T_sim, dt=1e-6, solver=ODESolver.RK4)
    sim.scope.add(v_ref_blk, label="v_ref")
    sim.scope.add(ai_ctrl,   label="ai_ctrl")
    sim.scope.add(buck,      label="buck_out", indices=[0, 1])

    if load_step_at is not None:
        from fmpy import read_model_description as _rmd
        _r_load_vr = next(
            v.valueReference for v in _rmd(FMU_PATH).modelVariables
            if v.name == 'R_load'
        )
        _orig = sim._compute_all_blocks;  _fired = [False]
        def _with_step(t):
            if not _fired[0] and t >= load_step_at:
                buck.fmu.setReal([_r_load_vr], [float(load_step_to)])
                print(f"  ⚡ Load step: {R_load}Ω -> {load_step_to}Ω at t={t*1000:.2f}ms")
                _fired[0] = True
            _orig(t)
        sim._compute_all_blocks = _with_step

    sim.topo.show_gui(str(_HERE / "topology_ai_pi.html"))
    print("\n  Simulating (AI-PI)...")
    sim.run(verbose=True, progress_bar=True)
    return sim, ai_ctrl


def run_fixed_pi_simulation(
    Kp:   float = 0.15, Ki: float = 8.0,
    V_ref: float = 12.0, R_load: float = 10.0, V_in: float = 24.0,
    load_step_at: Optional[float] = 0.005, load_step_to: float = 5.0,
) -> 'EmbedSim':

    v_ref_blk = VectorStep("vref", step_time=0.001,
                            before_value=0.0, after_value=V_ref, dim=1)
    pi_blk    = PI_BuckBlock("fixed_pi", Kp=Kp, Ki=Ki,
                              duty_max=0.9, duty_min=0.1, use_c_backend=False)
    buck      = BuckConverterBlock("buck", fmu_path=FMU_PATH)
    sink      = VectorEnd("sink")
    fb_dly    = ScalarDelay("fb_fixed", initial=[0.0])

    v_ref_blk >> pi_blk >> buck >> sink
    buck >> fb_dly >> pi_blk

    sim = EmbedSim(sinks=[sink], T=0.010, dt=1e-6, solver=ODESolver.RK4)
    sim.scope.add(v_ref_blk, label="v_ref")
    sim.scope.add(pi_blk,    label="pi_ctrl")
    sim.scope.add(buck,      label="buck_out", indices=[0, 1])

    if load_step_at is not None:
        from fmpy import read_model_description as _rmd
        _r_load_vr = next(
            v.valueReference for v in _rmd(FMU_PATH).modelVariables
            if v.name == 'R_load'
        )
        _orig = sim._compute_all_blocks;  _fired = [False]
        def _with_step(t):
            if not _fired[0] and t >= load_step_at:
                buck.fmu.setReal([_r_load_vr], [float(load_step_to)])
                _fired[0] = True
            _orig(t)
        sim._compute_all_blocks = _with_step

    sim.topo.show_gui(str(_HERE / "topology_fixed_pi.html"))
    sim.run(verbose=False, progress_bar=True)
    return sim


# ==============================================================================
# SECTION 8 — COMPARISON PLOTS
# ==============================================================================

def _get_metrics(v_out, t_ms, V_ref):
    mask = t_ms >= 1.0
    if not np.any(mask):
        return float('nan'), float('nan'), float('nan')
    vp = v_out[mask];  tp = t_ms[mask]
    settled   = np.abs(vp - V_ref) < 0.02 * V_ref
    st        = float(tp[np.where(settled)[0][0]] - 1.0) if np.any(settled) else float(tp[-1]-1.0)
    overshoot = max(0.0, (float(np.max(vp)) - V_ref) / V_ref * 100.0)
    sse       = float(np.mean(np.abs(vp[int(0.8*len(vp)):] - V_ref)) * 1000.0)
    return st, overshoot, sse


def plot_comparison(ai_sim, fixed_sim, ai_ctrl: AIPIController,
                    V_ref: float = 12.0) -> None:
    print("\n📊 Generating comparison plots...")

    def _get(scope, key):
        return np.array(list(scope.data.get(key, [])))

    t_ai = np.array(ai_sim.scope.t)    * 1000
    t_fx = np.array(fixed_sim.scope.t) * 1000
    v_ai = _get(ai_sim.scope,    "buck_out[0]");  v_fx = _get(fixed_sim.scope, "buck_out[0]")
    d_ai = _get(ai_sim.scope,    "ai_ctrl[0]");   d_fx = _get(fixed_sim.scope, "pi_ctrl[0]")
    i_ai = _get(ai_sim.scope,    "buck_out[1]");  i_fx = _get(fixed_sim.scope, "buck_out[1]")

    fig = plt.figure(figsize=(14, 11))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.30)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t_ai, v_ai, 'royalblue', lw=2,   label='AI-Tuned PI')
    ax0.plot(t_fx, v_fx, 'g--',       lw=1.8, label='Fixed PI (Kp=0.15, Ki=8.0)')
    ax0.axhline(V_ref,  color='r',      ls=':', lw=1.2, label=f'Target {V_ref}V')
    ax0.axvline(5.0,    color='orange', ls=':', alpha=0.6, label='Load step @ 5ms')
    ax0.set_ylabel('Voltage (V)');  ax0.set_title('Output Voltage — AI vs Fixed PI')
    ax0.legend(loc='lower right', fontsize=9);  ax0.set_ylim(-1, V_ref * 1.25)
    ax0.grid(True, alpha=0.25)

    for ax, ya, yf, lbl, ttl in [
        (fig.add_subplot(gs[1, 0]), d_ai, d_fx, 'Duty cycle',  'Controller Output'),
        (fig.add_subplot(gs[1, 1]), i_ai, i_fx, 'Current (A)', 'Inductor Current I_L'),
    ]:
        ax.plot(t_ai, ya, 'royalblue', lw=1.8, label='AI')
        ax.plot(t_fx, yf, 'g--',       lw=1.5, label='Fixed')
        ax.axvline(5.0, color='orange', ls=':', alpha=0.5)
        ax.set_ylabel(lbl);  ax.set_title(ttl)
        ax.legend(fontsize=9);  ax.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[2, 0])
    n   = min(len(t_ai), len(ai_ctrl.kp_history))
    ax3.plot(t_ai[:n], ai_ctrl.kp_history[:n], 'royalblue', lw=1.5, label='Kp')
    ax3r = ax3.twinx()
    ax3r.plot(t_ai[:n], ai_ctrl.ki_history[:n], 'orange', lw=1.5, label='Ki')
    ax3.set_ylabel('Kp', color='royalblue');  ax3r.set_ylabel('Ki', color='orange')
    ax3.set_title('AI Gain Schedule');  ax3.grid(True, alpha=0.25)

    ax4 = fig.add_subplot(gs[2, 1]);  ax4.axis('off')
    st_ai, os_ai, sse_ai = _get_metrics(v_ai, t_ai, V_ref)
    st_fx, os_fx, sse_fx = _get_metrics(v_fx, t_fx, V_ref)
    def _imp(fx, ai):
        return f"{(fx-ai)/fx*100:+.1f}%" if abs(fx) > 1e-9 else "N/A"
    rows = [
        ["Settling time",  f"{st_ai:.2f}ms",  f"{st_fx:.2f}ms",  _imp(st_fx, st_ai)],
        ["Overshoot",      f"{os_ai:.1f}%",   f"{os_fx:.1f}%",   _imp(os_fx, os_ai)],
        ["Steady-st. err", f"{sse_ai:.1f}mV", f"{sse_fx:.1f}mV", _imp(sse_fx, sse_ai)],
    ]
    tbl = ax4.table(cellText=rows,
                    colLabels=["Metric", "AI-Tuned", "Fixed PI", "Δ vs Fixed"],
                    cellLoc='center', loc='center',
                    colWidths=[0.30, 0.22, 0.22, 0.24])
    tbl.auto_set_font_size(False);  tbl.set_fontsize(10);  tbl.scale(1, 2.2)
    ax4.set_title("Performance Metrics", fontweight='bold', pad=20)

    plt.suptitle(f"AI-Tuned vs Fixed PI — Buck Converter  "
                 f"(V_ref={V_ref}V, load 10Ω->5Ω @ 5ms)",
                 fontsize=13, fontweight='bold')
    out = _HERE / "ai_vs_fixed_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    print(f"📈 Comparison saved -> {out}")
    plt.show(block=False);  plt.pause(0.1)


# ==============================================================================
# SECTION 9 — MAIN
# ==============================================================================

DATASET_CACHE = _HERE / "fmu_probe_dataset.json"
MODEL_CACHE   = _HERE / "gain_net.pt"


def main() -> None:
    print("\n" + "="*65)
    print("  EmbedSim — Buck Converter PI Control")
    print("="*65)
    print("\n  [1]  Fixed-gain PI demo  (CodeGen + topology + plot)")
    print("  [2]  AI-tuned PI demo    (FMU probe + NN + comparison)")
    print("  [3]  Both")
    choice = input("\n  Choice (1/2/3): ").strip()

    if choice in ("1", "3"):
        print("\n" + "="*65)
        print("  MODE 1 — FIXED-GAIN PI DEMO")
        print("="*65)
        run_fixed_pi_demo()

    if choice in ("2", "3"):
        print("\n" + "="*65)
        print("  MODE 2 — AI-TUNED PI DEMO")
        print("="*65)

        # ── Phase 1: Dataset ──────────────────────────────────────────────────
        if DATASET_CACHE.exists():
            use_cache = input(
                f"\n  Found cached dataset. Use it? (y/n): "
            ).strip().lower() == 'y'
        else:
            use_cache = False

        if use_cache:
            dataset = FMUProber.load(str(DATASET_CACHE))
        else:
            mode    = input("\n  Probing — [f] fast (~1 min) / [n] full (~4 min): "
                            ).strip().lower()
            prober  = FMUProber(fast=(mode != 'n'))
            dataset = prober.probe()
            if input("  Save dataset? (y/n): ").strip().lower() == 'y':
                prober.save(str(DATASET_CACHE))
            prober.plot_gain_surface()

        # ── Phase 2: Neural network ───────────────────────────────────────────
        if MODEL_CACHE.exists():
            use_model = input(
                f"\n  Found saved model. Load it? (y/n): "
            ).strip().lower() == 'y'
        else:
            use_model = False

        torch, _, _ = _import_torch()
        if use_model:
            net = GainNet(hidden=64)
            net.load_state_dict(torch.load(str(MODEL_CACHE), weights_only=True))
            net.eval()
            print("✅ Model loaded.")
        else:
            net = train_gain_net(dataset, epochs=300)
            if input("  Save model? (y/n): ").strip().lower() == 'y':
                torch.save(net.state_dict(), str(MODEL_CACHE))
                print(f"💾 Model saved -> {MODEL_CACHE}")

        # ── Phase 3: Simulation ───────────────────────────────────────────────
        ai_sim, ai_ctrl = run_ai_simulation(
            gain_net=net, R_load=10.0, V_in=24.0, V_ref=12.0,
            load_step_at=0.005, load_step_to=5.0,
        )

        create_plotter(ai_sim).plot_grid([
            dict(signal="buck_out[0]", ylabel="Voltage (V)",
                 title="Output Voltage V_out", color="#58a6ff",
                 ylim=(-1, 20), ref_val=12.0, ref_label="V_ref = 12 V", step_time=1.0),
            dict(signal="ai_ctrl[0]",  ylabel="Duty cycle",
                 title="AI PI — Duty Cycle", color="#3fb950", ylim=(0.0, 1.0)),
            dict(signal="buck_out[1]", ylabel="Current (A)",
                 title="Inductor Current I_L", color="#d2a8ff"),
        ], title="AI-Tuned Buck Converter (FMU-Probed NN)",
           save_path=str(_HERE / "pi_buck_ai_response.png"))

        Kp_f, Ki_f = net.predict(12.0, 10.0, 24.0)
        print(f"\n  NN gains at 12V / 10Ω / 24V:  Kp={Kp_f:.4f}  Ki={Ki_f:.3f}")

        if input("\n  Run fixed-PI comparison? (y/n): ").strip().lower() == 'y':
            fixed_sim = run_fixed_pi_simulation(
                Kp=0.15, Ki=8.0, load_step_at=0.005, load_step_to=5.0,
            )
            plot_comparison(ai_sim, fixed_sim, ai_ctrl, V_ref=12.0)


if __name__ == "__main__":
    main()
