"""
===============================================================================
RLC DIGITAL TWIN — DIFFERENTIABLE AI CONTROLLER (FMU + PYTORCH)
===============================================================================

Author: Paul Abraham / VectorSim
Version: 2.0
Date: 2026-02-24

===============================================================================
WHAT THIS SCRIPT DOES
===============================================================================

This script trains a neural-network controller to regulate an RLC circuit
Digital Twin exported as an FMU (Functional Mock-up Unit).

The objective is to make the plant output voltage track a 50 Hz sinusoidal
reference signal as accurately as possible.

Because FMUs are black-box dynamic systems and do NOT provide gradients,
the controller is trained using a two-stage strategy:

-------------------------------------------------------------------------------
1) PHASE 1 — Supervised PI Warm-Up (No FMU)
-------------------------------------------------------------------------------
The neural network is first trained to imitate a classical PI controller:

    u = Kp * error + Ki * integral(error)

This stabilizes the controller before interacting with the FMU and prevents
unstable or random control behavior.

-------------------------------------------------------------------------------
2) PHASE 2 — FMU-in-the-Loop Training (Surrogate Gradients)
-------------------------------------------------------------------------------
The trained controller is then connected in closed loop with the FMU.

Since the FMU is non-differentiable, the script:

    • Runs a full closed-loop simulation episode
    • Estimates the plant gain using finite differences
    • Builds a differentiable linear surrogate model
    • Backpropagates tracking loss through this surrogate

This enables gradient-based optimization of the neural controller
despite the black-box nature of the FMU.

-------------------------------------------------------------------------------
RESULT
-------------------------------------------------------------------------------

The script produces:

    • A trained neural controller
    • RMSE tracking performance metrics
    • A plot comparing:
          - Reference voltage
          - Plant output voltage
          - Control signal
    • A saved PNG visualization of the final closed-loop response

-------------------------------------------------------------------------------
ENGINEERING PURPOSE
-------------------------------------------------------------------------------

This implementation demonstrates how to:

    ✓ Train AI controllers on black-box FMU models
    ✓ Apply surrogate-gradient learning to physical systems
    ✓ Combine classical control with neural networks
    ✓ Perform AI optimization on Digital Twins

It provides a reusable pattern for AI-based control of
power electronics, smart grids, industrial drives,
and hardware-in-the-loop digital twin systems.

===============================================================================
END HEADER
===============================================================================
"""
# -----------------------------------------------------------------------------
# Standard Libraries
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os, sys, time

# -----------------------------------------------------------------------------
# Path Utilities (project-specific helpers)
# -----------------------------------------------------------------------------
from _path_utils import (
    get_embedsim_import_path,
    get_modelica_path,
    get_current_parent,
)
from examples.rlc_fmu.rlc_fmu_nn_training_animation import epoch

# Ensure EmbedSim can be imported
sys.path.insert(0, get_embedsim_import_path())

# -----------------------------------------------------------------------------
# FMU Interface
# -----------------------------------------------------------------------------
from embedsim import FMUBlock
from embedsim.core_blocks import VectorSignal


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

VREF_AMPL = 10.0      # Reference sine amplitude [V]
FREQ      = 50.0      # Reference frequency [Hz]

dt        = 1e-5      # Simulation timestep [s]
T_sim     = 0.05      # Total simulation time [s] (50 ms)

t_steps   = np.arange(0, T_sim, dt)
N_steps   = len(t_steps)

U_LIMIT   = 50.0      # Control saturation limit [V]
HIDDEN    = 32        # Hidden layer width


# =============================================================================
# LOAD FMU DIGITAL TWIN
# =============================================================================

# Path to compiled Modelica FMU
FMU_PATH = get_modelica_path("RLC_Sine_DigitalTwin_OM.fmu")

# Create FMU block
plant_fmu = FMUBlock(
    name="RLC_plant",
    fmu_path=FMU_PATH,
    input_names=["Vcontrol_python"],     # Control input
    output_names=["Vout"],               # Measured output
    parameters={
        "R": 10.0,
        "L": 10e-3,
        "C": 100e-6,
        "Vref_ampl": VREF_AMPL,
        "freq": FREQ,
        "usePythonControl": 1.0,         # Enable external control
    },
)


# =============================================================================
# REFERENCE SIGNAL
# =============================================================================

# Target voltage waveform to track
ref_signal = VREF_AMPL * np.sin(2 * np.pi * FREQ * t_steps)


# =============================================================================
# NEURAL NETWORK CONTROLLER
# =============================================================================

class NNController(nn.Module):
    """
    Neural controller that mimics and extends a PI controller.

    Inputs:
        [ error, integral_of_error ]

    Output:
        Control voltage (unclipped)
    """

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
        return self.net(x)


controller = NNController()


# =============================================================================
# CLOSED-LOOP FMU SIMULATION (NO GRADIENTS)
# =============================================================================

def run_episode_numpy(ctrl_net: NNController):
    """
    Runs a full closed-loop simulation through the FMU.

    Gradients are disabled because FMU is non-differentiable.

    Returns:
        y_out     : plant output voltage trajectory
        u_out     : control signal trajectory
        integrals : integral state trajectory
    """

    plant_fmu.reset()

    y_out    = np.zeros(N_steps)
    u_out    = np.zeros(N_steps)
    integrals = np.zeros(N_steps)

    integ = 0.0

    with torch.no_grad():
        for k in range(N_steps):

            # Feedback signal
            fb = y_out[k-1] if k > 0 else 0.0

            # Tracking error
            err = ref_signal[k] - fb

            # Integral of error
            integ += err * dt
            integrals[k] = integ

            # Neural network input
            nn_in = torch.tensor([[err, integ]], dtype=torch.float32)

            # Compute control signal
            u = float(ctrl_net(nn_in).item())

            # Apply actuator saturation
            u = float(np.clip(u, -U_LIMIT, U_LIMIT))

            # Send control to FMU
            sig_in = VectorSignal([u], "ctrl")
            y_out[k] = float(
                plant_fmu.compute(k * dt, dt, [sig_in]).value[0]
            )

            u_out[k] = u

    return y_out, u_out, integrals


# =============================================================================
# PLANT GAIN ESTIMATION (FINITE DIFFERENCE)
# =============================================================================

def estimate_plant_gain(u_nominal: float, step_size: float = 0.5):
    """
    Estimates local plant gain:

        G ≈ dVout/du

    Uses symmetric finite difference around u_nominal.
    """

    plant_fmu.reset()

    # Warm-up at nominal control
    for _ in range(50):
        plant_fmu.compute(0.0, dt, [VectorSignal([u_nominal], "ctrl")])

    # Evaluate positive perturbation
    sig_pos = VectorSignal([u_nominal + step_size], "ctrl")
    sig_neg = VectorSignal([u_nominal - step_size], "ctrl")

    plant_fmu.reset()
    for _ in range(50):
        plant_fmu.compute(0.0, dt, [VectorSignal([u_nominal], "ctrl")])
    y_pos = float(
        plant_fmu.compute(50 * dt, dt, [sig_pos]).value[0]
    )

    # Evaluate negative perturbation
    plant_fmu.reset()
    for _ in range(50):
        plant_fmu.compute(0.0, dt, [VectorSignal([u_nominal], "ctrl")])
    y_neg = float(
        plant_fmu.compute(50 * dt, dt, [sig_neg]).value[0]
    )

    # Central difference approximation
    G = (y_pos - y_neg) / (2 * step_size)

    # Prevent degenerate zero gain
    return float(G) if abs(G) > 1e-6 else 1.0


# =============================================================================
# PHASE 1 — SUPERVISED PI WARM-UP
# =============================================================================

"""
The neural network is first trained to imitate a classical PI controller:

    u = Kp * error + Ki * integral

This prevents unstable behavior when starting FMU training.
"""

Kp_init, Ki_init = 2.0, 50.0
PRETRAIN_EPOCHS  = 200
PRETRAIN_SAMPLES = 2000

optimizer_pre = optim.Adam(controller.parameters(), lr=5e-3)
loss_fn = nn.MSELoss()

for ep in range(PRETRAIN_EPOCHS):

    print("Phase 1-   EPOCH: ", epoch)

    # Random samples across realistic error ranges
    errs   = (torch.rand(PRETRAIN_SAMPLES, 1) * 2 - 1) * VREF_AMPL * 1.5
    integs = (torch.rand(PRETRAIN_SAMPLES, 1) * 2 - 1) * VREF_AMPL * 0.5 / FREQ

    nn_in  = torch.cat([errs, integs], dim=1)

    # Neural output
    u_pred = controller(nn_in)

    # PI target output
    u_tgt = torch.clamp(
        Kp_init * errs + Ki_init * integs,
        -U_LIMIT,
        U_LIMIT
    )

    loss = loss_fn(u_pred, u_tgt)

    optimizer_pre.zero_grad()
    loss.backward()
    optimizer_pre.step()


# =============================================================================
# PHASE 2 — FMU-IN-THE-LOOP SURROGATE TRAINING
# =============================================================================

"""
Because the FMU does not provide gradients, we approximate:

    V_out ≈ G * u

where G is an estimated local plant gain.

This allows gradient backpropagation through the surrogate model.
"""

FMU_EPOCHS  = 100
LR_FMU      = 1e-3
GRAD_CLIP   = 1.0
GD_INTERVAL = 10

optimizer_fmu = optim.Adam(controller.parameters(), lr=LR_FMU)

# Initial gain estimate
G = estimate_plant_gain(u_nominal=5.0)

best_rmse = np.inf
best_state = None

for epoch in range(FMU_EPOCHS):

    print("Phase 2-   EPOCH: ", epoch)

    # Periodically re-estimate plant gain
    if epoch % GD_INTERVAL == 0 and epoch > 0:
        u_mid = float(np.mean(np.abs(u_out)))
        G = estimate_plant_gain(u_nominal=u_mid if u_mid > 0.1 else 5.0)

    # Run real FMU episode
    y_out, u_out, integrals = run_episode_numpy(controller)

    tracking_errors = ref_signal - y_out
    rmse = float(np.sqrt(np.mean(tracking_errors ** 2)))

    # Save best performing model
    if rmse < best_rmse:
        best_rmse = rmse
        best_state = {
            k: v.clone() for k, v in controller.state_dict().items()
        }

    # Build differentiable surrogate training batch
    nn_inputs = torch.tensor(
        np.stack([tracking_errors, integrals], axis=1),
        dtype=torch.float32
    )

    u_pred = controller(nn_inputs)
    u_clip = torch.clamp(u_pred, -U_LIMIT, U_LIMIT)

    vref_t = torch.tensor(ref_signal, dtype=torch.float32).unsqueeze(1)

    # Surrogate plant output
    vout_surr = G * u_clip

    loss = loss_fn(vout_surr, vref_t)

    optimizer_fmu.zero_grad()
    loss.backward()

    # Gradient clipping for stability
    nn.utils.clip_grad_norm_(controller.parameters(), GRAD_CLIP)

    optimizer_fmu.step()


# =============================================================================
# FINAL EVALUATION
# =============================================================================

if best_state is not None:
    controller.load_state_dict(best_state)

y_out_final, u_out_final, _ = run_episode_numpy(controller)

rmse_final = float(
    np.sqrt(np.mean((ref_signal - y_out_final) ** 2))
)

print(f"Final evaluation RMSE: {rmse_final:.4f} V")


# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(t_steps * 1e3, ref_signal, label="V_ref")
axes[0].plot(t_steps * 1e3, y_out_final, label="V_out (NN)")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t_steps * 1e3, u_out_final, label="Control")
axes[1].axhline(U_LIMIT,  ls="--")
axes[1].axhline(-U_LIMIT, ls="--")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

out_png = os.path.join(
    get_current_parent(),
    "rlc_fmu_nn_pi_control_training.png"
)

plt.savefig(out_png, dpi=150)
plt.show()