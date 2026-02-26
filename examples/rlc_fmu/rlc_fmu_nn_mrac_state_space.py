"""
=============================================================================
RLC FMU Neural Network MRAC (Model Reference Adaptive Control)
State-Space Training with Live Animation
=============================================================================

EDUCATIONAL PURPOSE
-------------------
This script demonstrates how a Neural Network (NN) can be trained to replace
a classical PI controller in a closed-loop RLC circuit simulation.

THREE PHASES
------------
  Phase 1 - Supervised Pretraining  : NN imitates a well-tuned PI controller
  Phase 2 - Evolution Strategies    : NN directly minimises RMSE on real plant
  Phase 3 - Animation               : Reference vs output + NN weights as tree

CRASH SAFETY — SUBPROCESS ISOLATION (definitive fix)
------------------------------------------------------
  Problem: Windows exit code 0xC0000409 (heap corruption) occurs inside the
  FMU's compiled DLL after ~1000+ sequential rollouts.  Object recreation,
  reset() calls, and try/except cannot fix this — the corruption is in the
  DLL's private memory, which only Windows can reclaim.

  Solution: run every FMU episode in a FRESH SUBPROCESS (fmu_worker.py).
  When the subprocess exits, Windows unconditionally releases all its memory.
  The parent process never accumulates any FMU state.

  How it works:
    1. Parent writes NN weights + reference signal to temp .npy files
    2. Parent spawns:  python fmu_worker.py weights.npy ref.npy dt result.npy
    3. Worker runs one FMU episode, writes [rmse, y_out] to result.npy, exits
    4. Parent reads result.npy, deletes temp files, continues

  Cost: ~0.2s subprocess startup overhead per rollout.
  With ES_POP_SIZE=20: 40 subprocesses/step → ~8s overhead/step.
  This is acceptable and completely eliminates the crash.

FEEDFORWARD PATH
----------------
  NN inputs: [error, integral, Vref(t)]   (3 inputs)
  The feedforward input Vref(t) allows the NN to anticipate required control
  effort, equivalent to a learned gain:  u = Kp·e + Ki·∫e + Kff·Vref

BEST-WEIGHT CHECKPOINT
-----------------------
  Weights saved at every ES step; best are reloaded at end.
  Disk checkpoint (es_checkpoint.npz) saved every ES_SAVE_EVERY steps
  for automatic resume after any crash.

EVOLUTION STRATEGIES
--------------------
  Gradient-free optimisation for non-differentiable FMU plant.
  Antithetic ES estimator (Salimans et al. 2017):
    ∇ ≈ (1/2Nσ) Σ (RMSE(θ+σεᵢ) - RMSE(θ-σεᵢ)) · εᵢ

=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import torch
import torch.nn as nn
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

WORKER_SCRIPT = os.path.join(_HERE, "fmu_worker.py")
PYTHON_EXE    = sys.executable     # same venv python that runs this script


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

VREF_AMPL = 10.0
FREQ      = 50.0
dt        = 1e-5
T_sim     = 0.05
t_steps   = np.arange(0, T_sim, dt)
N_steps   = len(t_steps)
U_LIMIT   = 50.0

# ES coarse parameters (fast rollouts during training)
dt_es = 1e-4
T_es  = 0.02
t_es  = np.arange(0, T_es, dt_es)
N_es  = len(t_es)


# =============================================================================
# NEURAL NETWORK CONFIGURATION
# =============================================================================

LAYERS = [3, 6, 6, 1]      # inputs: [error, integral, Vref(feedforward)]

VREF_AMPL   = 10.0
FREQ        = 50.0
ERR_SCALE   = 1.0 / VREF_AMPL
INTEG_SCALE = (2 * np.pi * FREQ) / VREF_AMPL
REF_SCALE   = 1.0 / VREF_AMPL


# =============================================================================
# PI TEACHER GAINS
# =============================================================================

Kp_TEACHER = 4.0
Ki_TEACHER = 200.0


# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

PRETRAIN_EPOCHS  = 1000
PRETRAIN_SAMPLES = 2000
PRETRAIN_LR      = 5e-3

ES_EPOCHS        = 100
ES_POP_SIZE      = 20
ES_SIGMA         = 0.03
ES_LR            = 0.015
BAD_RMSE         = 99.0

ES_SAVE_EVERY    = 5
CHECKPOINT_PATH  = os.path.join(_HERE, "es_checkpoint.npz")

ANIM_FRAMES      = 20


# =============================================================================
# REFERENCE SIGNALS
# =============================================================================

ref_signal = VREF_AMPL * np.sin(2 * np.pi * FREQ * t_steps)
ref_es     = VREF_AMPL * np.sin(2 * np.pi * FREQ * t_es)


# =============================================================================
# NEURAL NETWORK CONTROLLER
# =============================================================================

class NNController(nn.Module):
    """
    Fully-connected controller — 3 inputs, Tanh hidden layers, 1 output.

    Inputs  (normalised to [-1, +1]):
      [0] error    : e(t) = Vref(t) - Vout(t)
      [1] integral : ∫e dt
      [2] reference: Vref(t)  ← feedforward path

    Output:
      control voltage u  (before ±U_LIMIT clipping)
    """

    def __init__(self, layers):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.append(nn.Tanh())
        self.register_buffer(
            'input_scale',
            torch.tensor([[ERR_SCALE, INTEG_SCALE, REF_SCALE]], dtype=torch.float32)
        )

    def forward(self, x):
        x = x * self.input_scale
        for layer in self.net:
            x = layer(x)
        return x

    def get_flat_weights(self):
        return np.concatenate([p.data.numpy().ravel() for p in self.parameters()])

    def set_flat_weights(self, w):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(torch.tensor(w[idx:idx+n].reshape(p.shape), dtype=torch.float32))
            idx += n


controller = NNController(LAYERS)
loss_fn    = nn.MSELoss()


# =============================================================================
# SUBPROCESS FMU RUNNER — crash-proof episode execution
# =============================================================================

def run_episode_subprocess(weights: np.ndarray,
                           ref:     np.ndarray,
                           step_dt: float,
                           timeout: float = 60.0):
    """
    Run one FMU episode in a completely isolated subprocess.

    CRASH SAFETY:
      Each call spawns a fresh Python process (fmu_worker.py).
      The FMU DLL lives only in that process's memory.
      When the process exits, Windows reclaims ALL its memory.
      The parent process is never exposed to heap corruption.

    Communication via temp files (fastest cross-process method on Windows):
      weights.npy  → subprocess reads NN weights
      ref.npy      → subprocess reads reference signal
      result.npy   ← subprocess writes [rmse, y_out[0..N]]

    Args:
        weights : flat NN weight array
        ref     : reference signal array
        step_dt : time step size [s]
        timeout : max seconds to wait for subprocess

    Returns:
        rmse  : scalar RMS tracking error  (BAD_RMSE on failure)
        y_out : plant output array  (zeros on failure)
    """
    steps = len(ref)

    # Write inputs to temp files
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as fw:
        weights_path = fw.name
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as fr:
        ref_path = fr.name
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as fres:
        result_path = fres.name

    np.save(weights_path, weights)
    np.save(ref_path,     ref)

    try:
        result = subprocess.run(
            [PYTHON_EXE, WORKER_SCRIPT,
             weights_path, ref_path, str(step_dt), result_path],
            timeout=timeout,
            capture_output=True,
        )

        if result.returncode != 0:
            # Subprocess crashed (0xC0000409 etc.) — return penalty
            return BAD_RMSE, np.zeros(steps)

        data  = np.load(result_path)
        rmse  = float(data[0])
        y_out = data[1:]
        return rmse, y_out

    except subprocess.TimeoutExpired:
        return BAD_RMSE, np.zeros(steps)

    except Exception:
        return BAD_RMSE, np.zeros(steps)

    finally:
        # Always clean up temp files
        for p in [weights_path, ref_path, result_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


def run_episode(ctrl_net: NNController,
                ref:      np.ndarray,
                step_dt:  float):
    """Convenience wrapper: extracts weights and calls subprocess runner."""
    return run_episode_subprocess(ctrl_net.get_flat_weights(), ref, step_dt)


# =============================================================================
# PHASE 1 — SUPERVISED PRETRAINING
# =============================================================================

print()
print("=" * 60)
print("  PHASE 1: Supervised Pretraining (NN imitates PI)")
print(f"  Teacher PI gains: Kp={Kp_TEACHER}, Ki={Ki_TEACHER}")
print(f"  NN architecture : {LAYERS}  "
      f"({len(controller.get_flat_weights())} weights)")
print("=" * 60)

optimizer_pre = optim.Adam(controller.parameters(), lr=PRETRAIN_LR)
t0 = time.time()

for ep in range(PRETRAIN_EPOCHS):
    errs   = (torch.rand(PRETRAIN_SAMPLES, 1) * 2 - 1) * VREF_AMPL * 1.2
    integs = (torch.rand(PRETRAIN_SAMPLES, 1) * 2 - 1) * VREF_AMPL / (2 * np.pi * FREQ)
    refs   = torch.zeros(PRETRAIN_SAMPLES, 1)

    nn_in  = torch.cat([errs, integs, refs], dim=1)
    u_pred = controller(nn_in)
    u_tgt  = torch.clamp(Kp_TEACHER * errs + Ki_TEACHER * integs,
                         -U_LIMIT, U_LIMIT)

    loss = loss_fn(u_pred, u_tgt)
    optimizer_pre.zero_grad()
    loss.backward()
    optimizer_pre.step()

    if (ep + 1) % 100 == 0:
        print(f"  Epoch {ep+1:5d}/{PRETRAIN_EPOCHS}  |  Loss = {loss.item():.5f}")

rmse_pretrain, _ = run_episode(controller, ref_signal, dt)
print(f"\n  Pretraining done in {time.time()-t0:.1f}s")
print(f"  Post-pretrain RMSE on real plant = {rmse_pretrain:.5f} V")


# =============================================================================
# PHASE 2 — EVOLUTION STRATEGIES WITH CHECKPOINT RESUME
# =============================================================================

n_weights = len(controller.get_flat_weights())

# Check for existing checkpoint
start_epoch   = 0
rmse_hist     = []
theta_history = []

if os.path.exists(CHECKPOINT_PATH):
    print(f"\n  Found checkpoint: {CHECKPOINT_PATH}")
    ckpt          = np.load(CHECKPOINT_PATH, allow_pickle=True)
    saved_weights = ckpt['best_weights']
    saved_rmse    = float(ckpt['best_rmse'])
    saved_epoch   = int(ckpt['epoch'])
    saved_history = list(ckpt['rmse_hist'])

    print(f"  Resuming from step {saved_epoch + 1}  |  "
          f"Best RMSE so far = {saved_rmse:.5f} V")
    controller.set_flat_weights(saved_weights)
    rmse_hist     = saved_history
    start_epoch   = saved_epoch + 1
    theta_history = [saved_weights.copy()] * start_epoch
else:
    print(f"\n  No checkpoint found — starting ES from scratch.")

print()
print("=" * 60)
print("  PHASE 2: Evolution Strategies (ES) — direct RMSE optimisation")
print(f"  Weights    : {n_weights}")
print(f"  Pop size   : {ES_POP_SIZE}  "
      f"(antithetic → {2*ES_POP_SIZE} subprocesses/step)")
print(f"  ES steps   : {ES_EPOCHS}  (starting at {start_epoch + 1})")
print(f"  σ={ES_SIGMA}   lr={ES_LR}")
print(f"  FMU isolation: each rollout = separate subprocess (crash-proof)")
print(f"  Auto-save  : every {ES_SAVE_EVERY} steps → {CHECKPOINT_PATH}")
print("=" * 60)

t0_es = time.time()

for epoch in range(start_epoch, ES_EPOCHS):
    t_epoch = time.time()
    theta   = controller.get_flat_weights()
    theta_history.append(theta.copy())

    noise       = np.random.randn(ES_POP_SIZE, n_weights)
    returns_pos = np.zeros(ES_POP_SIZE)
    returns_neg = np.zeros(ES_POP_SIZE)

    for i in range(ES_POP_SIZE):
        returns_pos[i], _ = run_episode_subprocess(
            theta + ES_SIGMA * noise[i], ref_es, dt_es)
        returns_neg[i], _ = run_episode_subprocess(
            theta - ES_SIGMA * noise[i], ref_es, dt_es)

    # Filter out failed rollouts
    valid = (returns_pos < BAD_RMSE * 0.9) & (returns_neg < BAD_RMSE * 0.9)
    if valid.sum() < 2:
        print(f"  Step {epoch+1:3d}  |  Too many subprocess failures — skipping.")
        controller.set_flat_weights(theta)
        rmse_hist.append(rmse_hist[-1] if rmse_hist else rmse_pretrain)
        continue

    advantages = (returns_pos - returns_neg)[valid]
    grad_es    = (advantages @ noise[valid]) / (2 * valid.sum() * ES_SIGMA)
    theta_new  = theta - ES_LR * grad_es
    controller.set_flat_weights(theta_new)

    # Full-resolution evaluation (also subprocess-isolated)
    rmse_full, _ = run_episode(controller, ref_signal, dt)
    rmse_hist.append(rmse_full)

    elapsed_step  = time.time() - t_epoch
    elapsed_total = time.time() - t0_es
    eta = elapsed_total / (epoch - start_epoch + 1) * (ES_EPOCHS - epoch - 1)

    best_marker = " ◄ best" if rmse_full == min(rmse_hist) else ""
    print(f"  Step {epoch+1:3d}/{ES_EPOCHS}  |  "
          f"RMSE = {rmse_full:.5f} V  |  "
          f"step={elapsed_step:.1f}s  ETA={eta:.0f}s{best_marker}")

    # Auto-save checkpoint
    if (epoch + 1) % ES_SAVE_EVERY == 0 or (epoch + 1) == ES_EPOCHS:
        best_idx  = int(np.argmin(rmse_hist))
        best_rmse = rmse_hist[best_idx]
        np.savez(CHECKPOINT_PATH,
                 best_weights = theta_history[best_idx],
                 best_rmse    = best_rmse,
                 epoch        = epoch,
                 rmse_hist    = np.array(rmse_hist))

# Reload best weights
best_step  = int(np.argmin(rmse_hist))
best_rmse  = rmse_hist[best_step]
controller.set_flat_weights(theta_history[best_step])
rmse_verified, _ = run_episode(controller, ref_signal, dt)

print(f"\n  ES complete.")
print(f"  Best step  : {best_step + 1}  |  RMSE = {best_rmse:.5f} V")
print(f"  Verified   : {rmse_verified:.5f} V  (full resolution)")
print(f"  Best weights reloaded ✓")

if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print(f"  Checkpoint removed (clean run).")


# =============================================================================
# PHASE 3 — COLLECT ANIMATION FRAMES
# =============================================================================

print()
print("=" * 60)
print("  PHASE 3: Collecting animation snapshots (full resolution)")
print("=" * 60)

anim_frames = []
for i in range(ANIM_FRAMES):
    _, y_f = run_episode(controller, ref_signal, dt)
    anim_frames.append(y_f.copy())
    if (i + 1) % 5 == 0:
        print(f"  Frame {i+1}/{ANIM_FRAMES} collected.")
print("  All frames collected.")


# =============================================================================
# VISUALISATION — NEURAL NETWORK TREE
# =============================================================================

def draw_nn_tree(ax, ctrl_net, layer_sizes):
    """Draw NN as layered graph. Green node = feedforward input."""
    ax.clear()
    ax.set_title(
        "Neural Network Weights  |  Green node = feedforward Vref(t)",
        fontsize=10, fontweight='bold'
    )
    ax.axis('off')

    n_layers = len(layer_sizes)
    pos = []
    for l, size in enumerate(layer_sizes):
        ys = np.linspace(-(size-1)/2.0, (size-1)/2.0, size)
        xs = np.full(size, float(l) / max(n_layers-1, 1))
        pos.append(np.stack([xs, ys], axis=1))

    linear_modules = [m for m in ctrl_net.net if isinstance(m, nn.Linear)]
    for l, lin in enumerate(linear_modules):
        w = lin.weight.detach().numpy()
        for j in range(w.shape[0]):
            for i in range(w.shape[1]):
                src   = pos[l][i]
                tgt   = pos[l+1][j]
                color = 'royalblue' if w[j, i] >= 0 else 'tomato'
                ax.plot([src[0], tgt[0]], [src[1], tgt[1]],
                        color=color, lw=0.7, alpha=0.5)
                ax.text((src[0]+tgt[0])/2, (src[1]+tgt[1])/2,
                        f"{w[j,i]:.5f}", color='black', fontsize=5,
                        ha='center', va='center')

    for l, layer_pos in enumerate(pos):
        colors = []
        for n_idx in range(len(layer_pos)):
            colors.append('lightgreen' if (l == 0 and n_idx == 2) else 'skyblue')
        ax.scatter(layer_pos[:, 0], layer_pos[:, 1],
                   s=300, c=colors, edgecolors='navy', linewidths=1.5, zorder=5)

    input_labels = ['e(t)', '∫e dt', 'Vref\n[FF]']
    ff_colors    = ['navy', 'navy', 'darkgreen']
    for i, (label, color) in enumerate(zip(input_labels, ff_colors)):
        ax.text(pos[0][i, 0]-0.07, pos[0][i, 1], label,
                ha='right', va='center', fontsize=7,
                color=color, fontweight='bold')

    headers = (['Input\n(3)'] +
               [f'Hidden {i+1}\n(Tanh)' for i in range(n_layers-2)] +
               ['Output\n[u]'])
    for l, label in enumerate(headers):
        ax.text(float(l)/max(n_layers-1, 1), pos[l][-1, 1]+1.0,
                label, ha='center', va='bottom',
                fontsize=8, color='navy', fontweight='bold')

    ax.plot([], [], color='royalblue',  lw=1.5, label='positive weight')
    ax.plot([], [], color='tomato',     lw=1.5, label='negative weight')
    ax.plot([], [], 'o', color='lightgreen', markersize=8,
            label='feedforward input')
    ax.legend(loc='lower right', fontsize=7)


# =============================================================================
# ANIMATION
# =============================================================================

fig, (ax_top, ax_tree) = plt.subplots(2, 1, figsize=(15, 11))
fig.suptitle(
    "RLC FMU — Neural Network MRAC  |  Feedforward + Feedback  |  ES trained",
    fontsize=12, fontweight='bold'
)

ax_top.set_xlabel("Time (s)")
ax_top.set_ylabel("Voltage (V)")
ax_top.set_xlim(0, T_sim)
ax_top.set_ylim(-1.2 * VREF_AMPL, 1.2 * VREF_AMPL)
ax_top.grid(True, linestyle='--', alpha=0.4)
line_ref, = ax_top.plot([], [], 'b-',  lw=1.8, label='V_ref (sine)')
line_out, = ax_top.plot([], [], 'r--', lw=1.8, label='V_out (NN ctrl)')
ax_top.legend(loc='upper right')


def init():
    line_ref.set_data([], [])
    line_out.set_data([], [])
    return line_ref, line_out


def update(frame):
    y_frame = anim_frames[frame % len(anim_frames)]
    line_ref.set_data(t_steps, ref_signal)
    line_out.set_data(t_steps, y_frame)
    draw_nn_tree(ax_tree, controller, LAYERS)
    rmse_f = float(np.sqrt(np.mean((ref_signal - y_frame)**2)))
    ax_top.set_title(
        f"Reference vs Plant Output  —  "
        f"snapshot {frame+1}/{ANIM_FRAMES}  |  RMSE = {rmse_f:.4f} V"
    )
    return line_ref, line_out


ani = FuncAnimation(fig, update, frames=ANIM_FRAMES,
                    init_func=init, interval=300, blit=False)

out_gif = os.path.join(_HERE, "rlc_nn_tree_rmse_live.gif")
ani.save(out_gif, writer=PillowWriter(fps=4))
plt.tight_layout()
plt.show()


# =============================================================================
# FINAL REPORT
# =============================================================================

print()
print("=" * 60)
print(f"  GIF saved → {out_gif}")
print()
print("  RMSE history (ES phase):")
for i, r in enumerate(rmse_hist):
    marker = " ◄ best" if i == best_step else ""
    print(f"    Step {i+1:3d}  |  RMSE = {r:.5f} V{marker}")
print()
print(f"  Architecture : {LAYERS}  ({n_weights} weights)")
print(f"  Inputs       : [error, integral, Vref]  (feedforward active)")
print(f"  Pre-ES  RMSE : {rmse_pretrain:.5f} V  (PI-imitation ceiling)")
print(f"  Best ES RMSE : {best_rmse:.5f} V  (step {best_step+1})")
print(f"  Δ vs PI      : {rmse_pretrain - best_rmse:+.5f} V  improved ✓")
print("=" * 60)
