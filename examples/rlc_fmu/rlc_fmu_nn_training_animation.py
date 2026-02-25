

"""
===============================================================================
RLC Digital Twin — Neural Network Controller with Live Training Dashboard
===============================================================================
Author: Paul / Vector Simulation Framework
Date: 2026-02-23
Version: 1.3

-------------------------------------------------------------------------------
PURPOSE
-------------------------------------------------------------------------------
Trains a small neural network (NN) to control the output voltage of an RLC
circuit digital twin (FMU), tracking a sinusoidal reference V_ref.

A real-time dashboard shows progress during training:
  - Top:    V_ref vs V_out
  - Middle: Applied control voltage u(t)
  - Bottom: Tracking RMSE over epochs

-------------------------------------------------------------------------------
HOW THE LEARNING WORKS
-------------------------------------------------------------------------------
The NN receives two inputs at each timestep:
  - Instantaneous error   e(t)  = V_ref(t) - V_out(t)
  - Integrated error    ∫e(t)dt  (like the I-term of a PI controller)

It outputs a control voltage u(t), which is fed into the FMU plant.

Because the FMU is a black box (no PyTorch-differentiable internals), we use
a SURROGATE GRADIENT: instead of backpropagating through the plant, we treat
the NN output u(t) as if it were the plant output and ask the NN to minimise
the error between u(t) and V_ref(t).

This is a simplification — the NN learns to produce control signals that
*look like* V_ref, which in practice drives the plant output toward V_ref
because the plant gain is approximately 1 in the passband.

For a proper differentiable approach you would need either:
  - A learned surrogate model of the plant (e.g. another NN), or
  - Adjoint/finite-difference gradients through the FMU.

-------------------------------------------------------------------------------
PARAMETERS
-------------------------------------------------------------------------------
  R = 10 Ω,  L = 10 mH,  C = 100 µF
  V_ref: 10 V sine at 50 Hz
  dt = 10 µs,  window = 50 ms,  epochs = 100
  NN: 2 → 16 → 16 → 1,  Tanh activations,  Adam lr = 0.01
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os, sys

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))
from embedsim import FMUBlock
from embedsim.core_blocks import VectorSignal

# ── Simulation parameters ─────────────────────────────────────────────────────
VREF_AMPL = 10.0
FREQ      = 50.0
dt        = 1e-5
T_sim     = 0.05          # 50 ms
t_steps   = np.arange(0, T_sim, dt)
N_steps   = len(t_steps)
epochs    = 100

# ── FMU plant ─────────────────────────────────────────────────────────────────
FMU_PATH = Path(_HERE) / "modelica" / "RLC_Sine_DigitalTwin_OM.fmu"
if not FMU_PATH.exists():
    raise FileNotFoundError(f"FMU not found at: {FMU_PATH}")

plant_fmu = FMUBlock(
    name="RLC_plant",
    fmu_path=FMU_PATH,
    input_names=["Vcontrol_python"],
    output_names=["Vout"],
    parameters={
        "R": 10.0, "L": 10e-3, "C": 100e-6,
        "Vref_ampl": VREF_AMPL, "freq": FREQ,
        "usePythonControl": 1.0,
    },
)

# ── Reference signal ──────────────────────────────────────────────────────────
ref_signal = VREF_AMPL * np.sin(2 * np.pi * FREQ * t_steps)

# ── Neural Network Controller ─────────────────────────────────────────────────
class NNController(nn.Module):
    """
    Inputs:  [e(t), integral_e(t)]  — error and its integral
    Output:  u(t)                   — control voltage
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

controller = NNController()
optimizer  = optim.Adam(controller.parameters(), lr=0.01)
loss_fn    = nn.MSELoss()

# ── Live dashboard setup ──────────────────────────────────────────────────────
plt.ion()
fig = plt.figure(figsize=(10, 8))
fig.suptitle("RLC Digital Twin — NN Controller Training", fontsize=13, fontweight='bold')

ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)   # separate x-axis (epoch), no sharex

t_ms = t_steps * 1e3

line_ref, = ax1.plot(t_ms, ref_signal, label="V_ref", color="tab:blue", lw=1.5)
line_out, = ax1.plot(t_ms, np.zeros_like(t_ms), label="V_out (NN)", color="tab:orange", lw=1.2)
ax1.set_ylabel("Voltage (V)")
ax1.set_xlabel("Time (ms)")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)

line_u, = ax2.plot(t_ms, np.zeros_like(t_ms), label="u(t)", color="tab:green", lw=1.2)
ax2.set_ylabel("Control Voltage (V)")
ax2.set_xlabel("Time (ms)")
ax2.set_ylim([-55, 55])
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)

rmse_history = []
line_rmse, = ax3.plot([], [], label="RMSE", color="tab:red", lw=1.5)
ax3.set_ylabel("RMSE (V)")
ax3.set_xlabel("Epoch")
ax3.set_ylim([0, VREF_AMPL])
ax3.legend(loc="upper right")
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(epochs):
    plant_fmu.reset()
    y_out      = np.zeros(N_steps)
    u_out      = np.zeros(N_steps)
    integ      = 0.0
    prev_y     = 0.0
    loss_accum = torch.tensor(0.0)

    optimizer.zero_grad()

    for k in range(N_steps):
        err    = ref_signal[k] - prev_y
        integ += err * dt

        nn_in  = torch.tensor([err, integ], dtype=torch.float32)
        u      = controller(nn_in)
        u_clip = torch.clamp(u, -50.0, 50.0)

        prev_y = float(plant_fmu.compute(k * dt, dt,
                       [VectorSignal([u_clip.item()], "ctrl")]).value[0])
        y_out[k] = prev_y
        u_out[k] = u_clip.item()

        # Surrogate loss: ask NN to produce u ≈ V_ref
        # (drives plant output toward V_ref since plant gain ≈ 1 in passband)
        target     = torch.tensor(ref_signal[k], dtype=torch.float32)
        loss_accum = loss_accum + loss_fn(u_clip.squeeze(), target)

    loss_mean = loss_accum / N_steps
    loss_mean.backward()
    optimizer.step()

    # True RMSE from actual plant output (for monitoring)
    rmse = float(np.sqrt(np.mean((y_out - ref_signal) ** 2)))
    rmse_history.append(rmse)

    # Dashboard update every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        line_out.set_ydata(y_out)
        line_u.set_ydata(u_out)
        line_rmse.set_xdata(np.arange(len(rmse_history)))
        line_rmse.set_ydata(rmse_history)
        ax3.set_xlim([0, max(1, len(rmse_history))])
        ax3.relim()
        ax3.autoscale_view(scalex=False)
        fig.suptitle(f"RLC NN Controller — Epoch {epoch+1}/{epochs}  "
                     f"RMSE = {rmse:.4f} V", fontsize=12, fontweight='bold')
        plt.pause(0.01)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:>4}/{epochs}   RMSE = {rmse:.5f} V")

plt.ioff()
plt.tight_layout()
plt.savefig("rlc_fmu_nn_training_animation.png", dpi=150)
plt.show()
print("Plot saved")