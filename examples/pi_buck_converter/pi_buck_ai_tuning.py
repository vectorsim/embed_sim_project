"""
pi_buck_ai_tuning.py  —  FMU-Probed Neural PI Tuner
=====================================================
Reworked architecture (three-phase pipeline):

  Phase 1 — FMU Prober
  ─────────────────────
  Sweep a grid of operating points (V_ref, R_load, V_in) × candidate gain
  pairs (Kp, Ki).  For each combination run a short closed-loop mini-simulation
  directly against the real FMU, measure the step-response quality (settling
  time, overshoot, steady-state error) and compute a scalar cost.  The
  (Kp, Ki) pair with the lowest cost for each operating point becomes the
  ground-truth label.  No heuristic formulas — the FMU decides what is best.

  Phase 2 — Neural Network Training
  ───────────────────────────────────
  Train a small MLP to map [V_ref, R_load, V_in] → [Kp_best, Ki_best] using
  the FMU-measured dataset.  The network learns the true gain surface of the
  physical plant.

  Phase 3 — Closed-Loop Demonstration
  ─────────────────────────────────────
  Run a full 10 ms EmbedSim simulation with the trained AI-PI controller,
  then compare against a fixed-gain baseline and plot results.

SYSTEM BLOCK DIAGRAM
────────────────────
  v_ref ──► [AIPIController] ──► [BuckConverterBlock FMU] ──► sink
                   ▲                          │
                   │                          ▼
                   └────── [VectorDelay] ◄────┘

Dependencies: PyTorch, NumPy, Matplotlib, FMPy, EmbedSim
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import time
import warnings
import json
from collections import deque
from pathlib import Path
from typing  import Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy  as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ── Project path setup ────────────────────────────────────────────────────────
from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE         = get_current_parent()          # …/examples/pi_buck_converter
project_root  = get_project_root()
sys.path.insert(0, get_embedsim_import_path())

# ── EmbedSim ──────────────────────────────────────────────────────────────────
from embedsim.simulation_engine import EmbedSim, ODESolver, VectorDelay
from embedsim.topology_printer  import TopologyPrinter
from embedsim.plot_helper        import create_plotter
from embedsim.source_blocks      import VectorStep
from embedsim.dynamic_blocks     import VectorEnd
from embedsim.core_blocks        import VectorSignal

# ── Buck converter blocks ──────────────────────────────────────────────────────
sys.path.append(str(project_root / "buck_converter"))
from pi_buck_block      import PI_BuckBlock
from BuckConverterBlock import BuckConverterBlock

# ── FMU path (shared) ─────────────────────────────────────────────────────────
FMU_PATH = str(project_root / "buck_converter" / "modelica" / "BuckConverter.fmu")


# ==============================================================================
# SECTION 1 — MINI-SIMULATION HELPER
# ==============================================================================
#
# run_mini_sim() is the core of Phase 1.  It builds a complete EmbedSim graph
# (source → PI → FMU → sink) and runs it for T_sim seconds with a given set
# of PI gains and plant parameters.  It returns the V_out time series so the
# caller can measure response quality.
#
# To keep probing fast:
#   - Use Euler (not RK4) — acceptable accuracy for ranking gain pairs
#   - Short horizon T_sim=8ms covers the full step transient
#   - dt=5µs gives 1600 steps — fast but still captures dynamics
# ==============================================================================

def run_mini_sim(
    Kp:       float,
    Ki:       float,
    V_ref:    float = 12.0,
    R_load:   float = 10.0,
    V_in:     float = 24.0,
    T_sim:    float = 0.012,    # 12 ms — gives sluggish gains time to settle
    dt:       float = 5e-6,     # 5 µs step — fast but accurate enough
    step_at:  float = 0.001,    # Reference step at t = 1 ms
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one closed-loop mini-simulation against the FMU.

    Returns
    -------
    t      : np.ndarray  time axis [s]
    v_out  : np.ndarray  output voltage history [V]
    """
    v_ref_blk = VectorStep(
        "vref", step_time=step_at, before_value=0.0,
        after_value=V_ref, dim=1,
    )
    pi_blk = PI_BuckBlock(
        "pi", Kp=Kp, Ki=Ki, duty_max=0.9, duty_min=0.1,
        Ts=dt, use_c_backend=False,
    )
    buck_blk = BuckConverterBlock(
        "buck", fmu_path=FMU_PATH,
        L=100e-6, C=100e-6, R_load=R_load, V_in=V_in, f_sw=100e3,
    )
    sink    = VectorEnd("sink")
    fb_dly  = VectorDelay("fb", initial=[0.0])

    v_ref_blk >> pi_blk >> buck_blk >> sink
    buck_blk  >> fb_dly  >> pi_blk

    sim = EmbedSim(sinks=[sink], T=T_sim, dt=dt, solver=ODESolver.EULER)
    sim.scope.add(buck_blk, label="vout", indices=[0])
    sim.run(verbose=False, progress_bar=False)

    t_arr = np.array(sim.scope.t, dtype=float)
    # scope key may be "vout[0]" or "vout" depending on EmbedSim version
    raw   = (sim.scope.data.get("vout[0]")
             or sim.scope.data.get("vout")
             or [])
    v_out = np.array(list(raw), dtype=float)

    n = min(len(t_arr), len(v_out))
    if n < 10:
        # Return a flat zero trace so compute_cost returns 1e6 (bad gain)
        return np.linspace(0, T_sim, 20), np.zeros(20)
    return t_arr[:n], v_out[:n]


# ==============================================================================
# SECTION 2 — STEP-RESPONSE METRICS
# ==============================================================================

def compute_cost(
    t:       np.ndarray,
    v_out:   np.ndarray,
    V_ref:   float,
    step_at: float = 0.001,
) -> float:
    """
    Compute a scalar cost from a step-response trace using ITAE + overshoot.

    ITAE (Integral of Time × Absolute Error) naturally punishes BOTH slow
    responses (large t multiplier) AND sustained oscillation (large |e|),
    without needing arbitrary weight tuning.  A gain pair that settles fast
    with small overshoot produces a tiny ITAE; one that is sluggish or rings
    produces a large ITAE.

    Cost = ITAE_normalised + 2.0 * overshoot_fraction + 5.0 * sse_fraction

    The overshoot and SSE terms are additive penalties on top of ITAE so that
    two gain pairs with identical ITAE are separated by their peak overshoot
    and residual error.

    Returns
    -------
    float  — scalar cost (lower is better; 1e6 = failed simulation)
    """
    mask = t >= step_at
    if not np.any(mask) or len(v_out[mask]) < 10:
        return 1e6

    t_post = t[mask]
    v_post = v_out[mask]
    t0     = t_post[0]
    tau    = t_post - t0          # time elapsed since step [s]
    err    = np.abs(v_post - V_ref)

    # ── ITAE ─────────────────────────────────────────────────────────────────
    # Trapezoid integration of tau * |e(tau)|
    # Normalise by V_ref * T_window^2 / 2  so result is dimensionless ≈ [0,1]
    T_win  = float(tau[-1])
    # np.trapezoid is the NumPy 2.0+ name; np.trapz was removed in 2.0
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    itae   = float(_trapz(tau * err, tau))
    norm   = V_ref * (T_win ** 2) / 2.0
    itae_n = itae / norm if norm > 1e-12 else 1.0

    # ── Overshoot fraction ────────────────────────────────────────────────────
    peak      = float(np.max(v_post))
    overshoot = max(0.0, (peak - V_ref) / V_ref)

    # ── Steady-state error (last 25% of window) ───────────────────────────────
    last = int(0.75 * len(v_post))
    sse  = float(np.mean(err[last:]) / V_ref)

    # ── Combined cost ─────────────────────────────────────────────────────────
    cost = itae_n + 2.0 * overshoot + 5.0 * sse
    return float(np.clip(cost, 0.0, 10.0))


# ==============================================================================
# SECTION 3 — FMU PROBER  (Phase 1)
# ==============================================================================
#
# FMUProber sweeps a grid of operating points and candidate gain pairs.
# For each operating point it runs len(Kp_candidates) × len(Ki_candidates)
# mini-simulations, scores each, picks the winner, and stores the result.
#
# Total mini-simulations = N_op × N_Kp × N_Ki
#
# With the default grid:
#   N_op = 4 V_ref × 3 R_load × 2 V_in = 24 operating points
#   N_gains = 5 Kp × 5 Ki = 25 gain candidates
#   Total = 24 × 25 = 600 mini-sims
#   Time  ≈ 600 × 0.4 s ≈ 4 minutes on a laptop
#
# You can shrink the grid (fast=True) for a quick smoke test.
# ==============================================================================

class FMUProber:
    """
    Probe the FMU plant to find optimal PI gains at each operating point.

    Attributes
    ----------
    dataset : list[dict]
        Each entry: {V_ref, R_load, V_in, Kp_best, Ki_best, cost_best}
    """

    # ── Gain candidate grid ───────────────────────────────────────────────────
    # Full: 5×5 = 25 combinations per operating point
    KP_GRID_FULL = [0.05, 0.10, 0.15, 0.25, 0.40]
    KI_GRID_FULL = [3.0,  6.0,  10.0, 15.0, 20.0]

    # Fast: 4×4 = 16 combinations — still enough to find a clear winner
    KP_GRID_FAST = [0.08, 0.15, 0.25, 0.38]
    KI_GRID_FAST = [3.0,  8.0,  13.0, 18.0]

    # ── Operating-point grids ─────────────────────────────────────────────────
    VREF_GRID_FULL  = [8.0,  10.0, 12.0, 15.0]
    RLOAD_GRID_FULL = [5.0,  10.0, 18.0]
    VIN_GRID_FULL   = [20.0, 24.0]

    # Fast: 3×3×1 = 9 op-points — gives NN enough variation to generalise
    VREF_GRID_FAST  = [8.0,  12.0, 15.0]
    RLOAD_GRID_FAST = [5.0,  10.0, 18.0]
    VIN_GRID_FAST   = [24.0]

    def __init__(self, fast: bool = False) -> None:
        self.fast    = fast
        self.dataset: List[Dict] = []

        self._kp_grid    = self.KP_GRID_FAST   if fast else self.KP_GRID_FULL
        self._ki_grid    = self.KI_GRID_FAST   if fast else self.KI_GRID_FULL
        self._vref_grid  = self.VREF_GRID_FAST if fast else self.VREF_GRID_FULL
        self._rload_grid = self.RLOAD_GRID_FAST if fast else self.RLOAD_GRID_FULL
        self._vin_grid   = self.VIN_GRID_FAST  if fast else self.VIN_GRID_FULL

    # ─────────────────────────────────────────────────────────────────────────
    def probe(self) -> List[Dict]:
        """
        Run the full sweep and populate self.dataset.

        For each operating point (V_ref, R_load, V_in):
          1. Try every (Kp, Ki) candidate pair via a mini-simulation.
          2. Score each via compute_cost().
          3. Record the winner.

        Returns
        -------
        list[dict]  dataset with one entry per operating point.
        """
        op_points = [
            (vr, rl, vi)
            for vr in self._vref_grid
            for rl in self._rload_grid
            for vi in self._vin_grid
        ]
        n_op    = len(op_points)
        n_gains = len(self._kp_grid) * len(self._ki_grid)
        total   = n_op * n_gains

        print(f"\n{'='*65}")
        print(f"  FMU PROBER — {n_op} operating points × {n_gains} gain pairs")
        print(f"  Total mini-simulations: {total}")
        print(f"  Estimated time: {total * 0.25 / 60:.1f} min  "
              f"({'fast' if self.fast else 'full'} mode)")
        print(f"{'='*65}\n")

        t_start = time.time()
        done    = 0

        for idx, (V_ref, R_load, V_in) in enumerate(op_points):
            best_cost = 1e9
            best_Kp   = self._kp_grid[0]
            best_Ki   = self._ki_grid[0]
            cost_table: List[Tuple[float,float,float]] = []  # (Kp, Ki, cost)

            for Kp in self._kp_grid:
                for Ki in self._ki_grid:
                    try:
                        t_sim, v_sim = run_mini_sim(
                            Kp=Kp, Ki=Ki,
                            V_ref=V_ref, R_load=R_load, V_in=V_in,
                        )
                        cost = compute_cost(t_sim, v_sim, V_ref)
                    except Exception as exc:
                        cost = 1e6
                        print(f"    ⚠ FMU error Kp={Kp} Ki={Ki}: {exc}")

                    cost_table.append((Kp, Ki, cost))

                    if cost < best_cost:
                        best_cost = cost
                        best_Kp   = Kp
                        best_Ki   = Ki

                    done += 1

            self.dataset.append({
                "V_ref":     V_ref,
                "R_load":    R_load,
                "V_in":      V_in,
                "Kp_best":   best_Kp,
                "Ki_best":   best_Ki,
                "cost_best": best_cost,
            })

            elapsed = time.time() - t_start
            rate    = done / elapsed
            remain  = (total - done) / rate if rate > 0 else 0
            print(
                f"  [{idx+1:3d}/{n_op}] "
                f"V_ref={V_ref:5.1f}V  R={R_load:4.1f}Ω  Vin={V_in:4.1f}V  "
                f"→  Kp={best_Kp:.3f}  Ki={best_Ki:5.1f}  "
                f"cost={best_cost:.4f}   "
                f"ETA {remain/60:.1f}min"
            )
            # Print full cost grid for first op-point — lets you verify the
            # ITAE cost function is ranking gain pairs correctly.
            if idx == 0:
                print(f"       ITAE cost grid (lower=better):")
                hdr = "       Kp\\Ki  " + "  ".join(f"{ki:6.1f}" for ki in self._ki_grid)
                print(hdr)
                for kp in self._kp_grid:
                    row = f"       {kp:.3f}    "
                    for ki in self._ki_grid:
                        c = next(c for (kk,kii,c) in cost_table if kk==kp and kii==ki)
                        star = "*" if (kp==best_Kp and ki==best_Ki) else " "
                        row += f" {c:.3f}{star} "
                    print(row)
                print()

        elapsed_total = time.time() - t_start
        print(f"\n✅ Probing complete — {len(self.dataset)} data points "
              f"in {elapsed_total:.1f}s\n")
        return self.dataset

    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Save dataset to JSON so probing only needs to run once."""
        with open(path, "w") as f:
            json.dump(self.dataset, f, indent=2)
        print(f"💾 Dataset saved → {path}")

    @staticmethod
    def load(path: str) -> List[Dict]:
        """Load a previously saved dataset."""
        with open(path) as f:
            data = json.load(f)
        print(f"📂 Dataset loaded — {len(data)} points from {path}")
        return data

    # ─────────────────────────────────────────────────────────────────────────
    def plot_gain_surface(self) -> None:
        """
        Visualise the optimal-gain surface over the probed operating envelope.

        Two subplots: Kp_best and Ki_best vs V_ref, coloured by R_load.
        Gives a direct visual check that the probed surface is physically
        sensible (e.g. Ki should increase with R_load to reject load steps).
        """
        if not self.dataset:
            print("No data to plot.")
            return

        vref  = np.array([d["V_ref"]   for d in self.dataset])
        rload = np.array([d["R_load"]  for d in self.dataset])
        kp    = np.array([d["Kp_best"] for d in self.dataset])
        ki    = np.array([d["Ki_best"] for d in self.dataset])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("FMU-Probed Optimal Gain Surface", fontsize=13, fontweight='bold')

        sc0 = axes[0].scatter(vref, kp, c=rload, cmap='plasma', s=80, alpha=0.8)
        axes[0].set_xlabel("V_ref (V)")
        axes[0].set_ylabel("Kp_best")
        axes[0].set_title("Optimal Kp vs V_ref  (colour = R_load)")
        plt.colorbar(sc0, ax=axes[0], label="R_load (Ω)")
        axes[0].grid(True, alpha=0.3)

        sc1 = axes[1].scatter(vref, ki, c=rload, cmap='plasma', s=80, alpha=0.8)
        axes[1].set_xlabel("V_ref (V)")
        axes[1].set_ylabel("Ki_best")
        axes[1].set_title("Optimal Ki vs V_ref  (colour = R_load)")
        plt.colorbar(sc1, ax=axes[1], label="R_load (Ω)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = _HERE / "fmu_gain_surface.png"
        plt.savefig(str(out_path), dpi=120, bbox_inches='tight')
        print(f"📈 Gain surface saved → {out_path}")
        plt.show(block=False)
        plt.pause(0.1)


# ==============================================================================
# SECTION 4 — NEURAL NETWORK  (Phase 2)
# ==============================================================================
#
# Architecture: 3-input MLP → [Kp, Ki]
#
#   The network is intentionally wider than the old one (64 neurons) because
#   the input space is now genuinely three-dimensional (V_ref, R_load, V_in)
#   rather than seven partly-synthetic features.  Fewer features → cleaner
#   signal → faster learning.
#
#   Input features (3, all normalised to ≈[0,1]):
#     [0]  V_ref  / 20.0
#     [1]  R_load / 20.0
#     [2]  V_in   / 30.0
#
#   Output:
#     Kp ∈ (0, 0.5]  — Sigmoid × 0.5
#     Ki ∈ (0, 20.0] — Sigmoid × 20.0
# ==============================================================================

INPUT_DIM  = 3    # V_ref, R_load, V_in
OUTPUT_DIM = 2    # Kp, Ki

class GainNet(nn.Module):
    """
    MLP that maps normalised plant operating conditions to PI gains.

    Trained on FMU-measured (operating_point → best_gains) data.
    """

    KP_SCALE = 0.5
    KI_SCALE = 20.0

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, OUTPUT_DIM),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3)  →  out: (batch, 2)  [Kp, Ki]"""
        raw = self.net(x)
        return torch.cat([
            raw[:, 0:1] * self.KP_SCALE,
            raw[:, 1:2] * self.KI_SCALE,
        ], dim=1)

    @staticmethod
    def encode_features(
        V_ref:  float,
        R_load: float,
        V_in:   float,
    ) -> torch.Tensor:
        """Build a normalised (1,3) feature tensor for inference."""
        return torch.tensor([[
            V_ref  / 20.0,
            R_load / 20.0,
            V_in   / 30.0,
        ]], dtype=torch.float32)

    def predict(
        self,
        V_ref:  float,
        R_load: float = 10.0,
        V_in:   float = 24.0,
    ) -> Tuple[float, float]:
        """Convenience wrapper: returns (Kp, Ki) as Python floats."""
        with torch.no_grad():
            x   = self.encode_features(V_ref, R_load, V_in)
            out = self(x)
        return float(out[0, 0]), float(out[0, 1])


# ==============================================================================
# SECTION 5 — TRAINING  (Phase 2)
# ==============================================================================

def train_gain_net(
    dataset: List[Dict],
    epochs:  int  = 300,
    lr:      float = 0.005,
    batch:   int  = 16,
) -> GainNet:
    """
    Train GainNet on FMU-probed (operating_point → best_gains) data.

    Parameters
    ----------
    dataset : list[dict]   output of FMUProber.probe() or FMUProber.load()
    epochs  : int          training epochs (300 is plenty for <100 samples)
    lr      : float        initial Adam learning rate
    batch   : int          mini-batch size (use dataset size if small)

    Returns
    -------
    GainNet  trained model
    """
    print(f"\n{'='*55}")
    print(f"  NN TRAINING  —  {len(dataset)} samples, {epochs} epochs")
    print(f"{'='*55}\n")

    # ── Build tensors ─────────────────────────────────────────────────────────
    X = torch.tensor([
        [d["V_ref"] / 20.0, d["R_load"] / 20.0, d["V_in"] / 30.0]
        for d in dataset
    ], dtype=torch.float32)

    Y = torch.tensor([
        [d["Kp_best"], d["Ki_best"]]
        for d in dataset
    ], dtype=torch.float32)

    # ── Model, optimiser, scheduler ───────────────────────────────────────────
    net       = GainNet(hidden=64)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    losses = []
    N      = len(dataset)
    B      = min(batch, N)

    for epoch in range(1, epochs + 1):
        perm   = torch.randperm(N)
        ep_loss = 0.0
        n_batch = 0

        for i in range(0, N, B):
            idx  = perm[i : i + B]
            xb   = X[idx]
            yb   = Y[idx]

            pred = net(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            n_batch += 1

        scheduler.step()
        avg = ep_loss / n_batch
        losses.append(avg)

        if epoch % 50 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"loss={avg:.6f}  lr={lr_now:.5f}")

    print(f"\n✅ Training done — final loss = {losses[-1]:.6f}\n")

    # ── Sanity check on training set ─────────────────────────────────────────
    print("  Prediction check (training set sample):")
    print(f"  {'V_ref':>6} {'R_load':>7} {'V_in':>6}  "
          f"{'Kp_true':>8} {'Ki_true':>8}  "
          f"{'Kp_pred':>8} {'Ki_pred':>8}")
    print("  " + "-"*62)

    net.eval()
    with torch.no_grad():
        preds = net(X).numpy()
    trues = Y.numpy()

    for i in range(min(8, N)):
        d = dataset[i]
        print(
            f"  {d['V_ref']:6.1f} {d['R_load']:7.1f} {d['V_in']:6.1f}  "
            f"{trues[i,0]:8.4f} {trues[i,1]:8.3f}  "
            f"{preds[i,0]:8.4f} {preds[i,1]:8.3f}"
        )

    # ── Loss curve ────────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 3))
    plt.semilogy(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log)')
    plt.title('GainNet Training Loss — FMU-Probed Dataset')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = _HERE / "gain_net_training.png"
    plt.savefig(str(loss_path), dpi=120, bbox_inches='tight')
    print(f"📈 Training loss saved → {loss_path}")
    plt.show(block=False)
    plt.pause(0.1)

    return net


# ==============================================================================
# SECTION 6 — AI PI CONTROLLER BLOCK
# ==============================================================================
#
# AIPIController wraps PI_BuckBlock.  At each time step it calls
# GainNet.predict() to get context-aware Kp/Ki, pushes them into the
# parent's PI state, then lets the parent's compute_py() run the PI law.
#
# Unlike the old version, the feature vector is only 3 numbers (the plant
# conditions), not a mix of live error signals.  The NN is a plant model,
# not a real-time signal processor — which is the correct abstraction.
# ==============================================================================

class AIPIController(PI_BuckBlock):
    """
    PI_BuckBlock with GainNet-predicted gain scheduling.

    Signal ports (same as PI_BuckBlock):
      input[0] — V_ref  (VectorStep setpoint)
      input[1] — V_meas (VectorDelay feedback from FMU)

    Parameters
    ----------
    name     : str          EmbedSim block name
    gain_net : GainNet      trained network (required)
    R_load   : float        nominal load resistance used for NN feature [Ω]
    V_in     : float        nominal supply voltage used for NN feature [V]
    **kwargs                forwarded to PI_BuckBlock
    """

    def __init__(
        self,
        name:     str,
        gain_net: GainNet,
        R_load:   float = 10.0,
        V_in:     float = 24.0,
        **kwargs,
    ) -> None:
        super().__init__(name, use_c_backend=False, **kwargs)
        self.gain_net = gain_net
        self.R_load   = R_load
        self.V_in     = V_in

        # Log gains for post-run inspection
        self.kp_history: List[float] = []
        self.ki_history: List[float] = []

    def compute_py(
        self,
        t:            float,
        dt:           float,
        input_values: Optional[list] = None,
    ) -> VectorSignal:
        """
        Override: predict gains → push into PI state → run PI law.
        """
        inputs = input_values or []
        V_ref  = float(inputs[0].value[0]) if len(inputs) > 0 else 0.0
        V_meas = float(inputs[1].value[0]) if len(inputs) > 1 else 0.0

        # ── Predict gains from operating conditions ──────────────────────────
        Kp, Ki = self.gain_net.predict(V_ref, self.R_load, self.V_in)

        # Safety clamp (Sigmoid already limits range, but belt + braces)
        Kp = float(np.clip(Kp, 0.01, 0.5))
        Ki = float(np.clip(Ki, 0.10, 20.0))

        # Push into parent PI block attributes so PI law uses fresh gains
        self._Kp = Kp
        self._Ki = Ki

        self.kp_history.append(Kp)
        self.ki_history.append(Ki)

        # ── Run standard PI law (inherited from PI_BuckBlock) ────────────────
        error = V_ref - V_meas

        # Anti-windup integral
        if not hasattr(self, '_integral'):
            self._integral = 0.0
        self._integral += error * dt
        max_int = 0.5 / max(Ki, 0.001)
        self._integral = float(np.clip(self._integral, -max_int, max_int))

        duty = Kp * error + Ki * self._integral
        duty = float(np.clip(duty, self._duty_min, self._duty_max))

        self.output = VectorSignal(
            np.array([duty], dtype=np.float32),
            self.name,
            dtype=self.dtype,
        )
        return self.output


# ==============================================================================
# SECTION 7 — FULL CLOSED-LOOP SIMULATION  (Phase 3)
# ==============================================================================

def run_ai_simulation(
    gain_net: GainNet,
    R_load:   float = 10.0,
    V_in:     float = 24.0,
    V_ref:    float = 12.0,
    T_sim:    float = 0.010,
    load_step_at: Optional[float] = 0.005,   # None = no load step
    load_step_to: float = 5.0,
) -> Tuple['EmbedSim', AIPIController]:
    """
    Run a full 10 ms EmbedSim simulation with the AI-PI controller.

    Parameters
    ----------
    gain_net      : trained GainNet
    R_load        : nominal load [Ω]
    V_in          : supply voltage [V]
    V_ref         : reference setpoint [V]
    T_sim         : simulation horizon [s]
    load_step_at  : time of load disturbance [s]; None = no disturbance
    load_step_to  : load after disturbance [Ω]

    Returns
    -------
    sim            : completed EmbedSim
    ai_controller  : AIPIController block (has kp_history, ki_history)
    """
    print(f"\n{'='*55}")
    print(f"  AI-PI SIMULATION  "
          f"V_ref={V_ref}V  R={R_load}Ω  Vin={V_in}V")
    print(f"{'='*55}\n")

    # Predict initial gains and show them
    Kp0, Ki0 = gain_net.predict(V_ref, R_load, V_in)
    print(f"  NN predicted initial gains:  Kp={Kp0:.4f}  Ki={Ki0:.3f}")

    # ── Blocks ────────────────────────────────────────────────────────────────
    v_ref_blk = VectorStep(
        "vref", step_time=0.001, before_value=0.0,
        after_value=V_ref, dim=1,
    )
    ai_ctrl = AIPIController(
        "ai_pi", gain_net=gain_net,
        R_load=R_load, V_in=V_in,
        Kp=Kp0, Ki=Ki0,
        duty_max=0.9, duty_min=0.1,
    )
    buck = BuckConverterBlock(
        "buck", fmu_path=FMU_PATH,
        L=100e-6, C=100e-6, R_load=R_load, V_in=V_in, f_sw=100e3,
    )
    sink   = VectorEnd("sink")
    fb_dly = VectorDelay("fb", initial=[0.0])

    # ── Wiring ────────────────────────────────────────────────────────────────
    v_ref_blk >> ai_ctrl >> buck >> sink
    buck >> fb_dly >> ai_ctrl

    # ── Simulation ────────────────────────────────────────────────────────────
    sim = EmbedSim(sinks=[sink], T=T_sim, dt=1e-6, solver=ODESolver.RK4)
    sim.scope.add(v_ref_blk, label="v_ref")
    sim.scope.add(ai_ctrl,   label="ai_ctrl")
    sim.scope.add(buck,      label="buck_out", indices=[0, 1])

    # ── Optional load-step disturbance ────────────────────────────────────────
    if load_step_at is not None:
        _orig  = sim._compute_all_blocks
        _fired = [False]

        def _with_disturbance(t: float) -> None:
            if not _fired[0] and t >= load_step_at:
                buck.set_R_load(load_step_to)
                print(f"  ⚡ Load step: {R_load}Ω → {load_step_to}Ω "
                      f"at t={t*1000:.2f}ms")
                _fired[0] = True
            _orig(t)

        sim._compute_all_blocks = _with_disturbance

    # ── Topology ──────────────────────────────────────────────────────────────
    sim.topo.print_console()
    _topo_html = _HERE / "topology_ai_pi.html"
    sim.topo.show_gui(str(_topo_html))
    print(f"🗺  Topology saved → {_topo_html}")

    print("\n⚙  Simulating…")
    sim.run(verbose=True, progress_bar=True)
    return sim, ai_ctrl


# ==============================================================================
# SECTION 8 — FIXED-GAIN BASELINE
# ==============================================================================

def run_fixed_pi_simulation(
    Kp:       float = 0.15,
    Ki:       float = 8.0,
    V_ref:    float = 12.0,
    R_load:   float = 10.0,
    V_in:     float = 24.0,
    load_step_at: Optional[float] = 0.005,
    load_step_to: float = 5.0,
) -> 'EmbedSim':
    """Fixed-gain PI baseline for comparison."""
    print(f"\n{'='*55}")
    print(f"  FIXED-PI  Kp={Kp}  Ki={Ki}")
    print(f"{'='*55}\n")

    v_ref_blk = VectorStep(
        "vref", step_time=0.001, before_value=0.0,
        after_value=V_ref, dim=1,
    )
    pi_blk = PI_BuckBlock(
        "fixed_pi", Kp=Kp, Ki=Ki, duty_max=0.9, duty_min=0.1,
        use_c_backend=False,
    )
    buck = BuckConverterBlock(
        "buck", fmu_path=FMU_PATH,
        L=100e-6, C=100e-6, R_load=R_load, V_in=V_in, f_sw=100e3,
    )
    sink   = VectorEnd("sink")
    fb_dly = VectorDelay("fb_fixed", initial=[0.0])

    v_ref_blk >> pi_blk >> buck >> sink
    buck >> fb_dly >> pi_blk

    sim = EmbedSim(sinks=[sink], T=0.010, dt=1e-6, solver=ODESolver.RK4)
    sim.scope.add(v_ref_blk, label="v_ref")
    sim.scope.add(pi_blk,    label="pi_ctrl")
    sim.scope.add(buck,      label="buck_out", indices=[0, 1])

    if load_step_at is not None:
        _orig  = sim._compute_all_blocks
        _fired = [False]

        def _with_disturbance(t: float) -> None:
            if not _fired[0] and t >= load_step_at:
                buck.set_R_load(load_step_to)
                _fired[0] = True
            _orig(t)

        sim._compute_all_blocks = _with_disturbance

    # ── Topology ──────────────────────────────────────────────────────────────
    sim.topo.print_console()
    _topo_html = _HERE / "topology_fixed_pi.html"
    sim.topo.show_gui(str(_topo_html))
    print(f"🗺  Topology saved → {_topo_html}")

    sim.run(verbose=False, progress_bar=True)
    return sim


# ==============================================================================
# SECTION 9 — COMPARISON PLOTS
# ==============================================================================

def _get_metrics(
    v_out: np.ndarray, t_ms: np.ndarray, V_ref: float,
) -> Tuple[float, float, float]:
    """Return (settling_ms, overshoot_pct, sse_mV) for a step response."""
    step_ms = 1.0
    mask    = t_ms >= step_ms
    if not np.any(mask):
        return float('nan'), float('nan'), float('nan')

    vp = v_out[mask]
    tp = t_ms[mask]

    band    = 0.02 * V_ref
    settled = np.abs(vp - V_ref) < band
    if np.any(settled):
        st = float(tp[np.where(settled)[0][0]] - step_ms)
    else:
        st = float(tp[-1] - step_ms)

    overshoot = max(0.0, (float(np.max(vp)) - V_ref) / V_ref * 100.0)

    last20 = int(0.8 * len(vp))
    sse    = float(np.mean(np.abs(vp[last20:] - V_ref)) * 1000.0)

    return st, overshoot, sse


def plot_comparison(
    ai_sim:    'EmbedSim',
    fixed_sim: 'EmbedSim',
    ai_ctrl:   AIPIController,
    V_ref:     float = 12.0,
) -> None:
    """
    Six-panel comparison figure:
      Row 0  (full width) — Output voltage
      Row 1  left         — Duty cycle
      Row 1  right        — Inductor current
      Row 2  left         — AI Kp/Ki history
      Row 2  right        — Performance metrics table
    """
    print("\n📊 Generating comparison plots…")

    t_ai    = np.array(ai_sim.scope.t)    * 1000   # → ms
    t_fx    = np.array(fixed_sim.scope.t) * 1000

    def _get(scope, key):
        return np.array(list(scope.data.get(key, [])))

    v_ai   = _get(ai_sim.scope,    "buck_out[0]")
    v_fx   = _get(fixed_sim.scope, "buck_out[0]")
    d_ai   = _get(ai_sim.scope,    "ai_ctrl[0]")
    d_fx   = _get(fixed_sim.scope, "pi_ctrl[0]")
    i_ai   = _get(ai_sim.scope,    "buck_out[1]")
    i_fx   = _get(fixed_sim.scope, "buck_out[1]")

    fig = plt.figure(figsize=(14, 11))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.30)

    # ── Row 0: Voltage ────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t_ai, v_ai, 'royalblue',      lw=2,   label='AI-Tuned PI')
    ax0.plot(t_fx, v_fx, 'g--',            lw=1.8, label=f'Fixed PI (Kp=0.15, Ki=8.0)')
    ax0.axhline(V_ref, color='r', ls=':',  lw=1.2, label=f'Target {V_ref}V')
    ax0.axvline(1.0,   color='grey',  ls=':', alpha=0.5)
    ax0.axvline(5.0,   color='orange',ls=':', alpha=0.6, label='Load step @ 5ms')
    ax0.set_ylabel('Voltage (V)')
    ax0.set_title('Output Voltage — AI-Tuned vs Fixed PI')
    ax0.legend(loc='lower right', fontsize=9)
    ax0.set_ylim(-1, V_ref * 1.25)
    ax0.grid(True, alpha=0.25)

    # ── Row 1: Duty / Current ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t_ai, d_ai, 'royalblue', lw=1.8, label='AI')
    ax1.plot(t_fx, d_fx, 'g--',       lw=1.5, label='Fixed')
    ax1.axvline(1.0, color='grey',   ls=':', alpha=0.5)
    ax1.axvline(5.0, color='orange', ls=':', alpha=0.5)
    ax1.set_ylabel('Duty cycle')
    ax1.set_title('Controller Output')
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(t_ai, i_ai, 'royalblue', lw=1.8, label='AI')
    ax2.plot(t_fx, i_fx, 'g--',       lw=1.5, label='Fixed')
    ax2.axvline(5.0, color='orange', ls=':', alpha=0.5)
    ax2.set_ylabel('Current (A)')
    ax2.set_title('Inductor Current I_L')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    # ── Row 2 left: Gain history ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    n   = min(len(t_ai), len(ai_ctrl.kp_history))
    ax3.plot(t_ai[:n], ai_ctrl.kp_history[:n], 'royalblue',
             lw=1.5, label='Kp (left)')
    ax3_r = ax3.twinx()
    ax3_r.plot(t_ai[:n], ai_ctrl.ki_history[:n], 'orange',
               lw=1.5, label='Ki (right)')
    ax3.set_ylabel('Kp', color='royalblue')
    ax3_r.set_ylabel('Ki', color='orange')
    ax3.set_title('AI Gain Schedule')
    ax3.set_xlabel('Time (ms)')
    ax3.grid(True, alpha=0.25)
    # combined legend
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3_r.get_legend_handles_labels()
    ax3.legend(h1+h2, l1+l2, fontsize=8, loc='upper right')

    # ── Row 2 right: Metrics table ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')

    st_ai, os_ai, sse_ai    = _get_metrics(v_ai, t_ai, V_ref)
    st_fx, os_fx, sse_fx    = _get_metrics(v_fx, t_fx, V_ref)

    def _imp(fixed, ai):
        if abs(fixed) < 1e-9:
            return "N/A"
        v = (fixed - ai) / fixed * 100
        return f"{v:+.1f}%"

    rows = [
        ["Settling time",  f"{st_ai:.2f}ms",   f"{st_fx:.2f}ms",  _imp(st_fx, st_ai)],
        ["Overshoot",      f"{os_ai:.1f}%",    f"{os_fx:.1f}%",   _imp(os_fx, os_ai)],
        ["Steady-st. err", f"{sse_ai:.1f}mV",  f"{sse_fx:.1f}mV", _imp(sse_fx, sse_ai)],
    ]
    tbl = ax4.table(
        cellText=rows,
        colLabels=["Metric", "AI-Tuned", "Fixed PI", "Δ vs Fixed"],
        cellLoc='center', loc='center',
        colWidths=[0.30, 0.22, 0.22, 0.24],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    # Colour the improvement column
    for i, row in enumerate(rows):
        try:
            val  = float(row[3].replace('+',''))
            col  = '#90EE90' if val > 0 else '#FFB6C1'
        except ValueError:
            col = '#FFFFFF'
        tbl[(i+1, 3)].set_facecolor(col)

    ax4.set_title("Performance Metrics", fontweight='bold', pad=20)

    plt.suptitle(
        f"AI-Tuned vs Fixed PI — Buck Converter  "
        f"(V_ref={V_ref}V, load 10Ω→5Ω @ 5ms)",
        fontsize=13, fontweight='bold',
    )
    out_path = _HERE / "ai_vs_fixed_comparison.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"📈 Comparison saved → {out_path}")
    plt.show(block=False)
    plt.pause(0.1)


# ==============================================================================
# SECTION 10 — MAIN
# ==============================================================================

DATASET_CACHE = _HERE / "fmu_probe_dataset.json"
MODEL_CACHE   = _HERE / "gain_net.pt"


def main() -> None:
    """
    Interactive main entry point.

    Workflow:
      1. Probe the FMU (or load cached dataset)
      2. Train / load GainNet
      3. Plot the FMU-discovered gain surface
      4. Run AI-PI closed-loop simulation
      5. Optionally run fixed-PI baseline and compare
    """
    print("\n" + "="*65)
    print("  EmbedSim — FMU-Probed Neural PI Tuner")
    print("="*65)

    # ── Phase 1: Dataset ──────────────────────────────────────────────────────
    if DATASET_CACHE.exists():
        use_cache = input(
            f"\nFound cached dataset ({DATASET_CACHE.name}).  "
            "Use it? (y/n): "
        ).strip().lower() == 'y'
    else:
        use_cache = False

    if use_cache:
        dataset = FMUProber.load(str(DATASET_CACHE))
    else:
        mode = input(
            "\nProbing mode:\n"
            "  [f] fast  — ~18 mini-sims,  ~1 min\n"
            "  [n] full  — ~600 mini-sims, ~4 min\n"
            "Choice (f/n): "
        ).strip().lower()
        fast   = (mode != 'n')
        prober = FMUProber(fast=fast)
        dataset = prober.probe()

        # Save prompt BEFORE plot — on Windows plt.show() is blocking
        save = input("\nSave dataset for next run? (y/n): ").strip().lower()
        if save == 'y':
            prober.save(str(DATASET_CACHE))

        prober.plot_gain_surface()  # show after prompt so it doesn't block input

    # ── Phase 2: Training ─────────────────────────────────────────────────────
    if MODEL_CACHE.exists():
        use_model = input(
            f"\nFound saved model ({MODEL_CACHE.name}).  "
            "Load it? (y/n): "
        ).strip().lower() == 'y'
    else:
        use_model = False

    if use_model:
        net = GainNet(hidden=64)
        net.load_state_dict(torch.load(str(MODEL_CACHE), weights_only=True))
        net.eval()
        print("✅ Model loaded.")
    else:
        net = train_gain_net(dataset, epochs=300)
        save_m = input("\nSave trained model? (y/n): ").strip().lower()
        if save_m == 'y':
            torch.save(net.state_dict(), str(MODEL_CACHE))
            print(f"💾 Model saved → {MODEL_CACHE}")

    # ── Phase 3: Simulation ───────────────────────────────────────────────────
    ai_sim, ai_ctrl = run_ai_simulation(
        gain_net     = net,
        R_load       = 10.0,
        V_in         = 24.0,
        V_ref        = 12.0,
        load_step_at = 0.005,
        load_step_to = 5.0,
    )

    # Save individual AI result plots
    ai_png = str(_HERE / "pi_buck_ai_response.png")
    create_plotter(ai_sim).plot_grid([
        dict(signal="buck_out[0]", ylabel="Voltage (V)",
             title="Output Voltage V_out", color="#58a6ff",
             ylim=(-1, 20), ref_val=12.0, ref_label="V_ref = 12 V",
             step_time=1.0),
        dict(signal="ai_ctrl[0]",  ylabel="Duty cycle",
             title="AI PI Controller — Duty Cycle", color="#3fb950",
             ylim=(0.0, 1.0)),
        dict(signal="buck_out[1]", ylabel="Current (A)",
             title="Inductor Current I_L", color="#d2a8ff"),
    ], title="AI-Tuned Buck Converter (FMU-Probed NN)", save_path=ai_png)

    # Final gain report
    Kp_final, Ki_final = net.predict(12.0, 10.0, 24.0)
    print(f"\n{'='*50}")
    print(f"  NN gains at nominal op-point (12V, 10Ω, 24V):")
    print(f"  Kp = {Kp_final:.4f}")
    print(f"  Ki = {Ki_final:.3f}")
    print(f"{'='*50}")

    # ── Optional comparison ───────────────────────────────────────────────────
    do_cmp = input("\nRun fixed-PI comparison? (y/n): ").strip().lower()
    if do_cmp == 'y':
        fixed_sim = run_fixed_pi_simulation(
            Kp=0.15, Ki=8.0, load_step_at=0.005, load_step_to=5.0,
        )
        plot_comparison(ai_sim, fixed_sim, ai_ctrl, V_ref=12.0)


if __name__ == "__main__":
    main()

  