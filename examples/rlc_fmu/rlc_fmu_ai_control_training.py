"""
===============================================================================
RLC DIGITAL TWIN — DIFFERENTIABLE AI CONTROLLER (FMU + PYTORCH)
===============================================================================

Author: Paul Abraham / VectorSim
Version: 2.0
Date: 2026-02-24

===============================================================================
OVERVIEW
===============================================================================

This script implements a differentiable neural-network controller for an
RLC circuit Digital Twin exported as an FMU (Functional Mock-up Unit).

The controller is trained in two phases:

    Phase 1 — Supervised PI Warm-Up (No FMU)
    Phase 2 — FMU-in-the-Loop Surrogate Gradient Training

The plant model is a Modelica-based RLC system compiled into an FMU
and executed through the VectorSim FMUBlock interface.

The controller is implemented in PyTorch and trained to track
a sinusoidal voltage reference signal.

-------------------------------------------------------------------------------
DIGITAL TWIN ARCHITECTURE
-------------------------------------------------------------------------------

Reference Signal  →  Neural Network Controller  →  FMU (RLC Plant)  →  Output
                             ↑
                             │
                        Feedback (V_out)

The neural network receives:
    • Instantaneous tracking error:   e(k) = V_ref - V_out
    • Integral of error:              ∫ e(k) dt

It outputs:
    • Control voltage u(k) applied to the FMU plant

-------------------------------------------------------------------------------
WHY TWO-PHASE TRAINING?
-------------------------------------------------------------------------------

FMUs are black-box dynamic systems.
They do NOT provide gradients for backpropagation.

To train a neural controller, we use:

PHASE 1 — Supervised Pre-Training
----------------------------------
The NN is trained to imitate a reasonable PI controller:

    u_target = Kp * error + Ki * integral

This stabilizes learning and avoids random FMU excitation.

PHASE 2 — Surrogate Gradient Learning
--------------------------------------
Since the FMU is non-differentiable, we:

1. Run a full episode through the FMU (forward simulation only)
2. Estimate plant gain using finite differences:
       G ≈ dVout/du
3. Build a differentiable surrogate model:
       Vout_surrogate = G * u
4. Backpropagate loss through this surrogate

This approximates policy-gradient / model-based RL training.

-------------------------------------------------------------------------------
SURROGATE GRADIENT MATHEMATICS
-------------------------------------------------------------------------------

True system:
    V_out = FMU(u)

We approximate locally:
    V_out ≈ G * u

Loss:
    L = MSE(V_out_surrogate, V_ref)

Backprop uses:

    dL/dθ ≈ (dL/dV_out) · G · (d u / dθ)

where:
    θ = neural network parameters
    G = finite-difference plant gain

-------------------------------------------------------------------------------
KEY FEATURES
-------------------------------------------------------------------------------

✓ FMU-in-the-loop closed-loop training
✓ Finite-difference plant gain estimation
✓ Gradient clipping for stability
✓ Vectorized loss computation
✓ Best-weight checkpointing
✓ Final evaluation and visualization
✓ Control saturation handling

-------------------------------------------------------------------------------
SIMULATION PARAMETERS
-------------------------------------------------------------------------------

Reference amplitude:  10 V
Frequency:            50 Hz
Simulation length:    50 ms
Time step:            10 µs
Total steps:          5000
Control limit:        ±50 V

-------------------------------------------------------------------------------
OUTPUT
-------------------------------------------------------------------------------

• Training progress (RMSE per epoch)
• Final RMSE after training
• Plot showing:
      - Reference voltage
      - Plant output voltage
      - Control signal
• Saved PNG:
      rlc_fmu_ai_model_training.png

-------------------------------------------------------------------------------
ENGINEERING SIGNIFICANCE
-------------------------------------------------------------------------------

This implementation demonstrates:

• How to train AI controllers on black-box FMU models
• Differentiable control with non-differentiable plants
• Hybrid control (classical + neural)
• Practical surrogate-gradient learning
• AI-enabled Digital Twin optimization

It represents a production-ready pattern for:
    - Smart grid control
    - Power electronics
    - Industrial drives
    - Hardware-in-the-loop AI training
    - Physics-informed reinforcement learning

===============================================================================
END HEADER
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os, sys, time

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

from embedsim import FMUBlock
from embedsim.core_blocks import VectorSignal

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
VREF_AMPL = 10.0
FREQ      = 50.0
dt        = 1e-5
T_sim     = 0.05           # 50 ms
t_steps   = np.arange(0, T_sim, dt)
N_steps   = len(t_steps)   # 5000

U_LIMIT   = 50.0           # Control saturation [V]
HIDDEN    = 32             # Slightly wider than v1 for better expressivity

# ---------------------------------------------------------------------------
# Load FMU plant
# ---------------------------------------------------------------------------
FMU_PATH = Path(_HERE) / "modelica" / "RLC_Sine_DigitalTwin_OM.fmu"
if not FMU_PATH.exists():
    raise FileNotFoundError(f"FMU not found at: {FMU_PATH}")

plant_fmu = FMUBlock(
    name="RLC_plant",
    fmu_path=FMU_PATH,
    input_names=["Vcontrol_python"],
    output_names=["Vout"],
    parameters={
        "R": 10.0,
        "L": 10e-3,
        "C": 100e-6,
        "Vref_ampl": VREF_AMPL,
        "freq": FREQ,
        "usePythonControl": 1.0,
    },
)

# ---------------------------------------------------------------------------
# Reference signal
# ---------------------------------------------------------------------------
ref_signal = VREF_AMPL * np.sin(2 * np.pi * FREQ * t_steps)

# ---------------------------------------------------------------------------
# Neural Network Controller
# ---------------------------------------------------------------------------
class NNController(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., 2]  →  [..., 1]  (un-clipped control voltage)"""
        return self.net(x)

controller = NNController()

# ---------------------------------------------------------------------------
# Helper: run one closed-loop episode through the FMU (numpy, no grad)
# Returns y_out [N], u_out [N], integrals [N]
# ---------------------------------------------------------------------------
def run_episode_numpy(ctrl_net: NNController) -> tuple:
    plant_fmu.reset()
    y_out    = np.zeros(N_steps)
    u_out    = np.zeros(N_steps)
    integrals = np.zeros(N_steps)
    integ    = 0.0

    with torch.no_grad():
        for k in range(N_steps):
            fb  = y_out[k-1] if k > 0 else 0.0
            err = ref_signal[k] - fb
            integ += err * dt
            integrals[k] = integ

            nn_in = torch.tensor([[err, integ]], dtype=torch.float32)
            u = float(ctrl_net(nn_in).item())
            u = float(np.clip(u, -U_LIMIT, U_LIMIT))

            sig_in   = VectorSignal([u], "ctrl")
            y_out[k] = float(plant_fmu.compute(k * dt, dt, [sig_in]).value[0])
            u_out[k] = u

    return y_out, u_out, integrals

# ---------------------------------------------------------------------------
# Estimate linearised plant gain G = dVout/du using finite difference
# Evaluated at the mid-point of the episode once per FD call
# ---------------------------------------------------------------------------
def estimate_plant_gain(u_nominal: float, step_size: float = 0.5) -> float:
    """Single-step FD around u_nominal.  Quick: just 2 FMU steps."""
    # Warm up plant to steady state with u_nominal for a few steps
    plant_fmu.reset()
    for _ in range(50):
        plant_fmu.compute(0.0, dt, [VectorSignal([u_nominal], "ctrl")])

    sig_pos = VectorSignal([u_nominal + step_size], "ctrl")
    sig_neg = VectorSignal([u_nominal - step_size], "ctrl")

    plant_fmu.reset()
    for _ in range(50):
        plant_fmu.compute(0.0, dt, [VectorSignal([u_nominal], "ctrl")])
    y_pos = float(plant_fmu.compute(50 * dt, dt, [sig_pos]).value[0])

    plant_fmu.reset()
    for _ in range(50):
        plant_fmu.compute(0.0, dt, [VectorSignal([u_nominal], "ctrl")])
    y_neg = float(plant_fmu.compute(50 * dt, dt, [sig_neg]).value[0])

    G = (y_pos - y_neg) / (2 * step_size)
    return float(G) if abs(G) > 1e-6 else 1.0   # fallback to 1 if degenerate

# ---------------------------------------------------------------------------
# PHASE 1 — Supervised pre-training (no FMU, fast)
#
# We teach the NN to behave like a proportional-integral controller:
#   u_target = Kp * err + Ki * integ
# with Kp = 2.0, Ki = 50.0 (reasonable starting point for this RLC).
# This avoids early random thrashing through the FMU.
# ---------------------------------------------------------------------------
print("=" * 60)
print("PHASE 1 — Supervised PI warm-up (no FMU)")
print("=" * 60)

Kp_init, Ki_init = 2.0, 50.0
PRETRAIN_EPOCHS  = 200
PRETRAIN_SAMPLES = 2000

optimizer_pre = optim.Adam(controller.parameters(), lr=5e-3)
loss_fn = nn.MSELoss()

t_pre = time.time()
for ep in range(PRETRAIN_EPOCHS):
    # Random (error, integral) pairs spanning realistic operating range
    errs   = (torch.rand(PRETRAIN_SAMPLES, 1) * 2 - 1) * VREF_AMPL * 1.5
    integs = (torch.rand(PRETRAIN_SAMPLES, 1) * 2 - 1) * VREF_AMPL * 0.5 / FREQ

    nn_in  = torch.cat([errs, integs], dim=1)
    u_pred = controller(nn_in)                   # [N, 1]
    u_tgt  = torch.clamp(
        Kp_init * errs + Ki_init * integs,
        -U_LIMIT, U_LIMIT
    )

    loss = loss_fn(u_pred, u_tgt)
    optimizer_pre.zero_grad()
    loss.backward()
    optimizer_pre.step()

    if (ep + 1) % 50 == 0:
        print(f"  Pre-train epoch {ep+1}/{PRETRAIN_EPOCHS}, Loss: {loss.item():.5f}")

print(f"  Pre-training done in {time.time()-t_pre:.1f}s\n")

# ---------------------------------------------------------------------------
# PHASE 2 — FMU-in-the-loop training with surrogate gradients
# ---------------------------------------------------------------------------
print("=" * 60)
print("PHASE 2 — FMU-in-the-loop surrogate gradient training")
print("=" * 60)

FMU_EPOCHS    = 100
LR_FMU        = 1e-3
GRAD_CLIP     = 1.0
GD_INTERVAL   = 10    # Re-estimate plant gain every N epochs

optimizer_fmu = optim.Adam(controller.parameters(), lr=LR_FMU)

G = estimate_plant_gain(u_nominal=5.0)   # Initial gain estimate
print(f"  Initial plant gain estimate G = {G:.4f}")

best_rmse = np.inf
best_state = None
t_phase2 = time.time()

for epoch in range(FMU_EPOCHS):

    # --- Re-estimate plant gain periodically ---
    if epoch % GD_INTERVAL == 0 and epoch > 0:
        u_mid = float(np.mean(np.abs(u_out)))   # use last episode mean |u|
        G = estimate_plant_gain(u_nominal=u_mid if u_mid > 0.1 else 5.0)

    # --- Run episode (numpy, no grad) to get actual plant response ---
    y_out, u_out, integrals = run_episode_numpy(controller)
    tracking_errors = ref_signal - y_out
    rmse = float(np.sqrt(np.mean(tracking_errors ** 2)))

    # Track best
    if rmse < best_rmse:
        best_rmse  = rmse
        best_state = {k: v.clone() for k, v in controller.state_dict().items()}

    # --- Surrogate gradient pass ---
    # Build (error, integral) tensors from the episode we just ran
    errs_np   = tracking_errors                        # e(k) = ref - y_out
    integ_np  = integrals                              # cumulative integral
    nn_inputs = torch.tensor(
        np.stack([errs_np, integ_np], axis=1),
        dtype=torch.float32
    )  # [N, 2]

    u_pred  = controller(nn_inputs)                    # [N, 1]  — with grad
    u_clip  = torch.clamp(u_pred, -U_LIMIT, U_LIMIT)  # [N, 1]

    # Surrogate output: Vout_surrogate = G * u  (linear approximation)
    # Loss = MSE(Vout_surrogate, Vref)
    vref_t  = torch.tensor(ref_signal, dtype=torch.float32).unsqueeze(1)  # [N,1]
    vout_surr = G * u_clip
    loss    = loss_fn(vout_surr, vref_t)

    optimizer_fmu.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(controller.parameters(), GRAD_CLIP)
    optimizer_fmu.step()

    if (epoch + 1) % 10 == 0:
        elapsed = time.time() - t_phase2
        print(f"  Epoch {epoch+1:3d}/{FMU_EPOCHS} | "
              f"RMSE: {rmse:.4f} V | "
              f"G: {G:.4f} | "
              f"Elapsed: {elapsed:.1f}s")

print(f"\nBest RMSE achieved: {best_rmse:.4f} V")

# Restore best weights
if best_state is not None:
    controller.load_state_dict(best_state)

# ---------------------------------------------------------------------------
# Final evaluation run
# ---------------------------------------------------------------------------
print("\nRunning final evaluation episode...")
y_out_final, u_out_final, _ = run_episode_numpy(controller)
rmse_final = float(np.sqrt(np.mean((ref_signal - y_out_final) ** 2)))
print(f"Final evaluation RMSE: {rmse_final:.4f} V")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax1 = axes[0]
ax1.plot(t_steps * 1e3, ref_signal,    label="V_ref",              color="tab:blue",   lw=1.5)
ax1.plot(t_steps * 1e3, y_out_final,   label="V_out (NN ctrl)",    color="tab:orange", lw=1.5)
ax1.set_ylabel("Voltage (V)")
ax1.set_title(f"RLC Digital Twin — NN Controller (FMU)  |  RMSE = {rmse_final:.4f} V")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(t_steps * 1e3, u_out_final,   label="Control u(t)", color="tab:green")
ax2.axhline( U_LIMIT, color="red",   ls="--", lw=0.8, label=f"+{U_LIMIT} V limit")
ax2.axhline(-U_LIMIT, color="red",   ls="--", lw=0.8, label=f"-{U_LIMIT} V limit")
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Control Voltage (V)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_png = os.path.join(_HERE, "rlc_fmu_ai_control_training.png")
plt.savefig(out_png, dpi=150)
plt.show()
print(f"Plot saved → {out_png}")