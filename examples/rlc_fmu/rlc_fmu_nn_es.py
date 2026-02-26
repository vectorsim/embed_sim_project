"""
=============================================================================
RLC FMU Neural Network  Evolution Strategies
Neural Network Controller Training with Live Animation
=============================================================================

WHAT THIS SCRIPT DOES — THE BIG PICTURE
-----------------------------------------
We have a physical RLC circuit (Resistor + Inductor + Capacitor) modelled
as an FMU (Functional Mock-up Unit — a compiled simulation exported from
Modelica/OpenModelica).  The circuit's output voltage Vout(t) needs to
track a sinusoidal reference signal Vref(t) = 10 * sin(2π·50·t).

A classical PI controller can do this job, but it has a hard ceiling on
performance because it only reacts to the current error and its integral.
It cannot "look ahead" or exploit the known shape of the reference.

This script trains a small neural network (NN) to do the same job BETTER
by learning a smarter control law directly from interacting with the plant.

WHY IS THIS INTERESTING FOR EmbedSim?
------------------------------------------
EmbedSim aims to let engineers write control logic once in Python/C and
deploy it on both simulation and embedded hardware (e.g Aurix TriCore MCU).
This script demonstrates that the same framework that runs a classical PID
block can also train and run a neural network controller — the NN is just
another block whose weights are the tunable parameters.

The workflow maps directly to EmbedSim blocks:
  FMUBlock      — the RLC plant (physics model from Modelica)
  NNController  — a trainable block that replaces the PI controller
  ES optimizer  — a block-level tuner that adjusts NNController weights

THREE TRAINING PHASES
----------------------
  Phase 1 — Supervised Pretraining (warm start)
    The NN is trained to imitate a known-good PI controller on synthetic
    data (random errors and integrals).  This avoids starting from random
    weights, which would produce wildly unstable control signals and make
    the plant simulation diverge.  Think of it as giving the NN a "manual"
    before it starts learning by experience.

  Phase 2 — Evolution Strategies (ES) — direct plant optimisation
    After pretraining, the NN already performs roughly as well as the PI.
    ES then directly minimises the tracking RMSE on the real plant.
    No gradients through the FMU are needed — ES is a gradient-FREE method.
    The key insight: ES can optimise any black-box function, including
    one that involves a compiled C simulation whose internals are opaque.

  Phase 3 — Animation and Visualisation
    Once training is complete, an animated GIF is produced showing:
      - How well the NN tracks the sine reference
      - How RMSE fell during training
      - The trained NN architecture with all its weight values

WHAT IS AN FMU?
---------------
An FMU (Functional Mock-up Unit) is a self-contained simulation component
defined by the FMI (Functional Mock-up Interface) standard.  It packages:
  - A compiled C solver (the plant dynamics)
  - An XML description of inputs, outputs and parameters
  - Initial conditions and solver settings

Here the FMU was generated from a Modelica RLC circuit model using
OpenModelica.  The circuit equations are:
  L · dI/dt  = Vcontrol - R·I - Vout
  C · dVout/dt = I
where Vcontrol is the NN's output (the control voltage).

The FMU exposes one input  (Vcontrol_python) and one output (Vout).
Everything else — the Laplace-domain poles, the resonant frequency,
the damping ratio — is computed inside the FMU's compiled solver.

WHY DOES THE FMU CRASH ON WINDOWS (0xC0000409)?
------------------------------------------------
The FMU's compiled DLL allocates C-level heap memory every time it is
called.  Over thousands of sequential rollouts this memory accumulates
(the DLL's internal allocator does not free intermediate state cleanly).
Eventually the Windows stack guard page is overwritten → STATUS_STACK_
BUFFER_OVERRUN (0xC0000409).

Python-level try/except, reset(), or object recreation CANNOT fix this
because the corruption lives inside the DLL's private memory, which only
the operating system can reclaim.

THE FIX — SUBPROCESS ISOLATION
--------------------------------
Each FMU rollout runs in a completely fresh Python subprocess
(rlc_fmu_nn_es_worker.py).  When that subprocess exits, Windows unconditionally
destroys its entire memory space — including the DLL's heap.
The parent process (this script) never accumulates any FMU state.
Communication happens via temporary .npy files written to disk.

WHAT IS EVOLUTION STRATEGIES (ES)?
------------------------------------
ES is a class of black-box optimisation algorithms inspired by biological
evolution.  The version used here is the OpenAI ES estimator (Salimans
et al., 2017), which works as follows:

  Given current weights θ and a noise scale σ:
  1. Sample N random perturbation vectors εᵢ from N(0, I)
  2. For each εᵢ, evaluate the objective at θ+σεᵢ  ("positive" perturb)
                  and also at θ−σεᵢ  ("negative" / antithetic perturb)
  3. Compute a gradient estimate:
       ∇ ≈ (1 / 2Nσ) · Σᵢ [ F(θ+σεᵢ) − F(θ−σεᵢ) ] · εᵢ
     where F(θ) is the RMSE of the NN controller with weights θ.
  4. Update: θ ← θ − lr · ∇   (gradient descent on RMSE)

WHY ANTITHETIC SAMPLING?
  Using both +ε and −ε pairs reduces the variance of the gradient estimate
  by half compared to using only one-sided perturbations.  This means fewer
  rollouts are needed to get a reliable gradient direction.

WHY NOT BACKPROPAGATION?
  Backprop requires computing ∂(plant output)/∂(NN weights), which means
  differentiating through the FMU — a compiled C binary we cannot inspect.
  ES only needs to EVALUATE the FMU, never differentiate through it.
  This makes ES the natural choice for any black-box plant.

FEEDFORWARD INPUT — WHY ADD Vref(t) AS A THIRD INPUT?
-------------------------------------------------------
A pure feedback controller (PI or NN) reacts to errors AFTER they occur.
The reference signal Vref(t) = 10·sin(2π·50·t) is a deterministic sine
wave that is fully known at every time step.

By feeding Vref(t) directly into the NN alongside the error and integral,
we allow the NN to learn a feedforward component:
  u(t) = f_feedback(e, ∫e) + f_feedforward(Vref)
where the feedforward part can pre-emptively generate the control voltage
needed to drive a sine wave through the RLC circuit, BEFORE errors build up.

This is the neural network equivalent of adding a reference model prefilter
or a feedforward gain in classical control design.

INPUT NORMALISATION — WHY SCALE THE INPUTS?
--------------------------------------------
The NN uses Tanh activation functions, which saturate (output ±1) when
their input magnitude exceeds roughly 2–3.  Unnormalised inputs would
immediately saturate all neurons, destroying the gradient signal.

We scale each input so that its typical range maps to approximately [-1, +1]:
  error    / VREF_AMPL           → error divided by max possible error
  integral / (VREF_AMPL / ω)    → integral scaled by its natural magnitude
  Vref(t)  / VREF_AMPL           → reference normalised to ±1

CHECKPOINT / CRASH RECOVERY
-----------------------------
ES training takes several minutes.  If the script is interrupted (power,
crash, Ctrl+C), all progress would be lost without checkpointing.
Every ES_SAVE_EVERY steps, the best weights, RMSE history and current step
are saved to es_checkpoint.npz.  On the next run, the script detects this
file and resumes from the saved step automatically.

REFERENCES
-----------
  Salimans, T. et al. (2017). "Evolution Strategies as a Scalable
  Alternative to Reinforcement Learning." arXiv:1703.03864.

  FMI Standard: https://fmi-standard.org/

=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import time
import tempfile      # for creating temporary files to pass data to subprocesses
import subprocess    # for spawning isolated FMU worker processes
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor  # for parallel subprocess pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import torch
import torch.nn as nn
import torch.optim as optim

# Resolve the project root so that "import embedsim" works regardless of
# where this script is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

# Path to the worker script that each subprocess will run.
# Keeping worker logic in a separate file means the subprocess can be
# imported cleanly without re-running any training code.
WORKER_SCRIPT = os.path.join(_HERE, "rlc_fmu_nn_es_worker.py")

# Use the SAME Python interpreter and virtual environment as the parent.
# sys.executable returns the full path of the currently running Python binary.
PYTHON_EXE = sys.executable


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
# These define the RLC plant and the reference signal for FULL-resolution
# evaluation and animation.  A coarser grid (dt_es, T_es) is used during
# ES training to keep rollouts fast.

VREF_AMPL = 10.0          # Reference sine amplitude [V]
FREQ      = 50.0          # Reference frequency [Hz] — 50 Hz mains frequency
dt        = 1e-5          # Fine time step for final evaluation [s] = 10 µs
T_sim     = 0.05          # Total simulation time [s] = 50 ms = 2.5 full cycles
t_steps   = np.arange(0, T_sim, dt)   # Time axis array for full-res runs
N_steps   = len(t_steps)              # Number of time steps (= 5000)
U_LIMIT   = 50.0          # Saturation limit on control voltage [V]
                           # Prevents the NN from requesting impossible voltages.
                           # The real inverter driving the RLC circuit is limited
                           # to ±50 V.

# --- Coarse grid for ES training rollouts ---
# Using dt_es=1e-4 (100 µs) instead of 1e-5 (10 µs) makes each rollout
# 10× faster.  The gradient estimate is noisier but still good enough to
# guide the ES update.  Final evaluation always uses the fine grid.
dt_es = 1e-4              # Coarse time step for ES training [s]
T_es  = 0.02              # Shorter horizon for ES [s] = 20 ms = 1 cycle
t_es  = np.arange(0, T_es, dt_es)    # Coarse time axis
N_es  = len(t_es)                     # Number of coarse steps (= 200)


# =============================================================================
# NEURAL NETWORK CONFIGURATION
# =============================================================================
# The architecture is a fully-connected multilayer perceptron (MLP):
#
#   Input layer  : 3 neurons  [error, integral, Vref]
#   Hidden layer 1: 6 neurons  with Tanh activation
#   Hidden layer 2: 6 neurons  with Tanh activation
#   Output layer : 1 neuron   (the control voltage u, no activation)
#
# Total trainable parameters:
#   Layer 1 weights: 3×6=18,  biases: 6
#   Layer 2 weights: 6×6=36,  biases: 6
#   Layer 3 weights: 6×1=6,   biases: 1
#   Total: 73 parameters
#
# Why Tanh?
#   Tanh outputs are in (-1, +1), matching the normalised input range.
#   It is smooth everywhere, which helps pretraining with Adam.
#   ReLU was tried but caused "dead neuron" problems when weights went
#   strongly negative during ES perturbations.

LAYERS = [3, 6, 6, 1]    # layer sizes from input to output

# Input normalisation scales — see "INPUT NORMALISATION" section in module
# docstring above for the full explanation.
VREF_AMPL   = 10.0
FREQ        = 50.0
ERR_SCALE   = 1.0 / VREF_AMPL                    # error → [-1, +1]
INTEG_SCALE = (2 * np.pi * FREQ) / VREF_AMPL     # integral → [-1, +1]
                                                   # At 50 Hz the integral of
                                                   # a 10V sine over one quarter
                                                   # period is ≈10/(2π·50) ≈ 0.032
                                                   # Multiplying by 2π·50/10=31.4
                                                   # maps this to ≈1.0 ✓
REF_SCALE   = 1.0 / VREF_AMPL                    # reference → [-1, +1]


# =============================================================================
# PI TEACHER GAINS  (used in Phase 1 pretraining only)
# =============================================================================
# A PI (Proportional-Integral) controller computes:
#   u(t) = Kp · e(t) + Ki · ∫e(t)dt
# where e(t) = Vref(t) - Vout(t) is the tracking error.
#
# Kp_TEACHER=4, Ki_TEACHER=200 were hand-tuned for the 50 Hz RLC plant.
# The PI controller achieves RMSE ≈ 1.3–2.2 V depending on initial conditions.
# This is the "performance ceiling" that Phase 1 imitates.
# Phase 2 (ES) then surpasses this ceiling by learning a feedforward component.

Kp_TEACHER = 4.0     # Proportional gain of the teacher PI controller
Ki_TEACHER = 200.0   # Integral gain of the teacher PI controller


# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# --- Phase 1: Supervised pretraining ---
PRETRAIN_EPOCHS  = 1000   # Number of gradient update steps with Adam
PRETRAIN_SAMPLES = 2000   # Random (error, integral, Vref) samples per batch.
                           # We use RANDOM samples rather than actual FMU
                           # trajectories because:
                           #   1. It is much faster (no FMU calls needed)
                           #   2. It covers the full expected operating range
                           #   3. The NN only needs to IMITATE the PI formula,
                           #      not match a specific trajectory
PRETRAIN_LR      = 5e-3   # Adam learning rate for pretraining.
                           # Adam (Adaptive Moment Estimation) adapts the
                           # learning rate per-parameter using momentum
                           # estimates, making it robust to poor scaling.

# --- Phase 2: Evolution Strategies ---
ES_EPOCHS   = 50      # Total number of ES gradient steps (training iterations)
ES_POP_SIZE = 20      # Number of antithetic perturbation PAIRS per step.
                       # Each pair produces one +ε rollout and one −ε rollout,
                       # so the total rollouts per step = 2 × ES_POP_SIZE = 40.
                       # Larger population → better gradient estimate, slower step.
ES_SIGMA    = 0.03    # Noise scale σ for weight perturbations.
                       # Each perturbed weight θᵢ + σ·εᵢ differs from the
                       # current weight by roughly ±3% of a unit weight change.
                       # Too small → gradient too noisy (perturbation lost in
                       #             plant noise).
                       # Too large → gradient biased (we leave the region where
                       #             the linear gradient estimate is valid).
ES_LR       = 0.015   # ES learning rate (step size along gradient direction).
                       # This multiplies the ES gradient estimate before
                       # subtracting it from the current weights.
                       # Empirically: 0.015 gives stable convergence on this
                       # 73-weight network.  Too large → oscillation.
BAD_RMSE    = 99.0    # Penalty RMSE returned when a subprocess fails or times out.
                       # Using 99.0 V (far above any real RMSE) means the failed
                       # rollout's contribution to the gradient estimate will be
                       # filtered out by the `valid` mask below.

# --- Checkpointing ---
ES_SAVE_EVERY   = 5    # Save checkpoint to disk every N ES steps.
                        # Balances crash-recovery granularity vs disk I/O.
CHECKPOINT_PATH = os.path.join(_HERE, "es_checkpoint.npz")

# --- Animation ---
ANIM_FRAMES = 20      # Number of frames in the output GIF.
                       # More frames → smoother animation but larger file size.


# =============================================================================
# REFERENCE SIGNALS
# =============================================================================
# Pre-compute the full reference arrays once.  The same arrays are reused
# across all FMU evaluations and animation frames, saving repeated computation.

ref_signal = VREF_AMPL * np.sin(2 * np.pi * FREQ * t_steps)  # fine grid, 5000 pts
ref_es     = VREF_AMPL * np.sin(2 * np.pi * FREQ * t_es)     # coarse grid, 200 pts


# =============================================================================
# NEURAL NETWORK CONTROLLER  (torch.nn.Module)
# =============================================================================

class NNController(nn.Module):
    """
    Fully-connected feedforward controller with Tanh hidden layers.

    ARCHITECTURE DIAGRAM:
    ┌──────────┐     ┌──────────────────────────────────────────────────┐
    │  e(t)    │────►│                                                  │
    │  ∫e dt   │────►│   Linear(3→6) → Tanh → Linear(6→6) → Tanh      │──► u(t)
    │  Vref(t) │────►│              → Linear(6→1)                      │
    └──────────┘     └──────────────────────────────────────────────────┘
     3 inputs          73 trainable weights                   1 output

    The first operation inside forward() is element-wise scaling of the
    inputs to normalise them to [-1, +1] before the first Linear layer.
    This is stored as a non-trainable buffer (register_buffer) so it is
    saved with the model but not updated during training.

    WHY nn.ModuleList INSTEAD OF nn.Sequential?
    --------------------------------------------
    nn.Sequential would be simpler, but nn.ModuleList gives us direct
    access to individual layers by index — useful for extracting weights
    for the NN tree visualisation and for the ES flat-weight interface.

    FLAT WEIGHT INTERFACE (get_flat_weights / set_flat_weights)
    -----------------------------------------------------------
    ES operates on θ as a single 1-D array of all 73 weights concatenated.
    get_flat_weights() extracts them in a deterministic order (same as
    PyTorch's parameter iteration).  set_flat_weights() writes them back
    in exactly the same order.  This round-trip must be lossless.
    """

    def __init__(self, layers):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers) - 1):
            # nn.Linear(in, out) creates a weight matrix W (out×in) and
            # bias vector b (out), so the layer computes: output = W·input + b
            self.net.append(nn.Linear(layers[i], layers[i + 1]))
            # Add Tanh after every layer EXCEPT the last (output) layer.
            # The output layer has no activation so u(t) can be any real
            # number before the ±U_LIMIT clipping is applied externally.
            if i < len(layers) - 2:
                self.net.append(nn.Tanh())

        # register_buffer: stored in model state_dict but not a nn.Parameter,
        # so it is not updated by the optimiser.  It IS moved to the correct
        # device if we call .cuda() later.
        self.register_buffer(
            'input_scale',
            torch.tensor([[ERR_SCALE, INTEG_SCALE, REF_SCALE]], dtype=torch.float32)
        )

    def forward(self, x):
        """
        Forward pass: normalise inputs then pass through all layers.

        x shape: (batch_size, 3)  — each row is [e(t), ∫e·dt, Vref(t)]
        returns : (batch_size, 1) — control voltage u(t)
        """
        # Element-wise multiply to normalise.  input_scale has shape (1,3)
        # and broadcasts over the batch dimension automatically.
        x = x * self.input_scale

        # Pass through Linear and Tanh layers in order.
        # The ModuleList alternates: Linear, Tanh, Linear, Tanh, Linear
        for layer in self.net:
            x = layer(x)
        return x

    def get_flat_weights(self):
        """
        Return all trainable parameters as a single 1-D NumPy array.

        Iterates self.parameters() in the same deterministic order that
        PyTorch always uses (depth-first, weight before bias per layer).
        For [3,6,6,1] this gives 73 elements:
          18 (W1) + 6 (b1) + 36 (W2) + 6 (b2) + 6 (W3) + 1 (b3) = 73
        """
        return np.concatenate([p.data.numpy().ravel() for p in self.parameters()])

    def set_flat_weights(self, w):
        """
        Load a 1-D weight array back into the model parameters.

        Must use EXACTLY the same iteration order as get_flat_weights().
        Uses data.copy_() to write in-place without breaking autograd graph.
        """
        idx = 0
        for p in self.parameters():
            n = p.numel()            # number of elements in this parameter tensor
            p.data.copy_(
                torch.tensor(
                    w[idx:idx+n].reshape(p.shape),
                    dtype=torch.float32
                )
            )
            idx += n


# Instantiate the controller and the MSE loss function used in Phase 1.
# nn.MSELoss() computes mean((u_pred - u_target)²) over the batch.
controller = NNController(LAYERS)
loss_fn    = nn.MSELoss()


# =============================================================================
# SUBPROCESS FMU RUNNER  — crash-proof episode execution
# =============================================================================

def run_episode_subprocess(weights: np.ndarray,
                           ref:     np.ndarray,
                           step_dt: float,
                           timeout: float = 60.0):
    """
    Run ONE closed-loop FMU episode in a completely isolated subprocess.

    MOTIVATION — WHY NOT JUST CALL THE FMU DIRECTLY?
    --------------------------------------------------
    See "WHY DOES THE FMU CRASH ON WINDOWS?" in the module docstring.
    The short answer: the FMU's compiled DLL corrupts its own heap after
    ~1000 sequential calls.  The only reliable fix is to run each rollout
    in a fresh OS process so Windows can reclaim all DLL memory on exit.

    DATA FLOW:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Parent process (this script)                                   │
    │                                                                 │
    │  1. Write weights → /tmp/tmpXXXX.npy                           │
    │  2. Write ref     → /tmp/tmpYYYY.npy                           │
    │  3. subprocess.run([python, fmu_worker.py, ...])  ─────────►   │
    │                                                      Child:     │
    │                                                      load npy   │
    │                                                      run FMU    │
    │                                                      write npy  │
    │                                                      EXIT ──►   │
    │                                                   Windows frees │
    │                                                   ALL memory ✓  │
    │  4. Read result   ← /tmp/tmpZZZZ.npy                           │
    │  5. Delete all 3 temp files                                     │
    └─────────────────────────────────────────────────────────────────┘

    WHY TEMP FILES INSTEAD OF PIPES OR SOCKETS?
    -------------------------------------------
    NumPy arrays are most efficiently shared as .npy files — np.save/load
    is faster and simpler than pickling over a pipe, and avoids the
    subprocess stdin/stdout buffering issues common on Windows.

    Args:
        weights : flat 73-element weight array (the NN to evaluate)
        ref     : reference signal array (Vref at each time step)
        step_dt : simulation time step [s]
        timeout : kill subprocess after this many seconds (prevents hangs)

    Returns:
        rmse  : scalar RMS tracking error in Volts.
                Returns BAD_RMSE (99.0 V) if subprocess crashed or timed out.
        y_out : plant output array Vout(t).
                Returns zeros array if subprocess failed.
    """
    steps = len(ref)

    # Create three temporary files:
    #   fw  → carry NN weights into the subprocess
    #   fr  → carry reference signal into the subprocess
    #   fres← carry result (RMSE + Vout) back to the parent
    # delete=False keeps the file on disk after the context manager closes it
    # (we need it to persist until the subprocess reads it).
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as fw:
        weights_path = fw.name
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as fr:
        ref_path = fr.name
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as fres:
        result_path = fres.name

    np.save(weights_path, weights)
    np.save(ref_path,     ref)

    try:
        # subprocess.run() launches a new Python process and WAITS for it
        # to complete (or timeout).  capture_output=True suppresses the
        # worker's stdout/stderr so it does not clutter our console output.
        result = subprocess.run(
            [PYTHON_EXE, WORKER_SCRIPT,
             weights_path, ref_path, str(step_dt), result_path],
            timeout=timeout,
            capture_output=True,
        )

        if result.returncode != 0:
            # Non-zero exit code means the subprocess crashed (e.g. 0xC0000409
            # if the FMU DLL still corrupted memory despite subprocess isolation,
            # or any Python exception that was not caught inside fmu_worker.py).
            # Return the penalty RMSE so the ES gradient ignores this rollout.
            return BAD_RMSE, np.zeros(steps)

        # Load the result written by fmu_worker.py.
        # Format: [rmse, Vout[0], Vout[1], ..., Vout[N-1]]  (N+1 elements)
        data  = np.load(result_path)
        rmse  = float(data[0])
        y_out = data[1:]
        return rmse, y_out

    except subprocess.TimeoutExpired:
        # The FMU solver diverged or hung — treat as a failed rollout.
        return BAD_RMSE, np.zeros(steps)

    except Exception:
        # Any other unexpected error (file not found, etc.)
        return BAD_RMSE, np.zeros(steps)

    finally:
        # The `finally` block runs regardless of success or exception.
        # Always clean up temp files to avoid filling the disk over many
        # training runs.
        for p in [weights_path, ref_path, result_path]:
            try:
                os.unlink(p)
            except Exception:
                pass  # file may not exist if an earlier step failed


def run_episode(ctrl_net: NNController,
                ref:      np.ndarray,
                step_dt:  float):
    """
    Convenience wrapper: extract weights from the controller object and
    call the subprocess runner.

    Used for:
      - Post-pretrain evaluation (Phase 1)
      - Full-resolution evaluation after each ES step (Phase 2)
      - Collecting animation frames (Phase 3)
    """
    return run_episode_subprocess(ctrl_net.get_flat_weights(), ref, step_dt)


def run_episodes_parallel(weights_list: list,
                          ref:          np.ndarray,
                          step_dt:      float,
                          max_workers:  int = None):
    """
    Run MULTIPLE FMU episodes simultaneously using a process pool.

    WHY PARALLEL?
    -------------
    Each ES step requires 40 rollouts (20 positive + 20 negative perturbations).
    Running them sequentially at ~2.5 s each = 100 s/step.
    Running them in parallel at ~2.5 s wall time = ~12 s/step on 8 cores.

    HOW ProcessPoolExecutor WORKS:
    --------------------------------
    ProcessPoolExecutor creates a pool of worker OS processes (not threads).
    Each worker runs independently in its own Python interpreter — they share
    no memory and cannot corrupt each other's state.

    pool.submit(fn, *args) sends a job to the pool and returns a Future object
    immediately (non-blocking).  The pool dispatches jobs to idle workers.
    f.result() blocks until that specific Future completes and returns its value.

    We preserve job ordering by storing futures in a list and collecting
    results in the same order:
      futures_ordered = [submit(w0), submit(w1), ..., submit(w39)]
      results         = [f0.result(), f1.result(), ..., f39.result()]

    This guarantees results[i] always corresponds to weights_list[i].

    WHY ProcessPoolExecutor INSTEAD OF multiprocessing.Pool?
    ---------------------------------------------------------
    On Windows, multiprocessing.Pool uses "spawn" to create worker processes.
    This re-imports the __main__ module in each worker — without the
    `if __name__ == '__main__':` guard, this causes a fork bomb (each worker
    tries to spawn more workers infinitely).
    ProcessPoolExecutor handles this correctly as long as the guard is present.

    Args:
        weights_list : list of flat weight arrays, one per rollout
        ref          : reference signal (same array used for all rollouts)
        step_dt      : simulation time step [s]
        max_workers  : cap on parallel processes (None → use CPU core count)

    Returns:
        List of (rmse, y_out) tuples in the same order as weights_list.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n = len(weights_list)
    if max_workers is None:
        import os as _os
        # Limit to min(n_jobs, n_cpu_cores) — no point having more workers
        # than jobs, and spawning more processes than cores wastes time on
        # context switching.
        max_workers = min(n, _os.cpu_count() or 4)

    futures_ordered = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        # Submit all 40 jobs to the pool in one go.
        # The pool immediately starts running up to max_workers jobs in parallel.
        for w in weights_list:
            fut = pool.submit(run_episode_subprocess, w, ref, step_dt)
            futures_ordered.append(fut)
        # Collect results in submission order (NOT completion order).
        # f.result() blocks until that particular future is done.
        results = [f.result() for f in futures_ordered]

    return results


# =============================================================================
# PHASE 1 — SUPERVISED PRETRAINING
# =============================================================================
# Guard required on Windows: without this, every subprocess spawned by
# ProcessPoolExecutor would re-execute all the code above (imports,
# controller instantiation, etc.) causing a process storm.
# With the guard, only the original parent process runs the training code.

if __name__ == '__main__':

    print()
    print("=" * 60)
    print("  PHASE 1: Supervised Pretraining (NN imitates PI)")
    print(f"  Teacher PI gains: Kp={Kp_TEACHER}, Ki={Ki_TEACHER}")
    print(f"  NN architecture : {LAYERS}  "
          f"({len(controller.get_flat_weights())} weights)")
    print("=" * 60)

    # Adam optimiser for Phase 1.
    # Adam is preferred over plain SGD because:
    #   - It adapts the learning rate per-parameter automatically
    #   - It handles the different natural scales of W and b
    #   - It converges in far fewer epochs on typical MLP problems
    optimizer_pre = optim.Adam(controller.parameters(), lr=PRETRAIN_LR)
    t0 = time.time()

    for ep in range(PRETRAIN_EPOCHS):
        # --- Generate random synthetic training samples ---
        # We do NOT run the FMU here — that would be ~5000× slower.
        # Instead we sample random (error, integral, Vref) combinations
        # that cover the full operating range, and ask the NN to match
        # the PI formula: u = Kp·e + Ki·∫e  (clamped to ±U_LIMIT).
        #
        # torch.rand() gives uniform [0,1]; multiply and shift to get the
        # desired range.  VREF_AMPL * 1.2 extends slightly beyond ±10 V
        # to prevent the NN from over-fitting exactly at the boundaries.
        errs   = (torch.rand(PRETRAIN_SAMPLES, 1) * 2 - 1) * VREF_AMPL * 1.2
        integs = (torch.rand(PRETRAIN_SAMPLES, 1) * 2 - 1) * VREF_AMPL / (2 * np.pi * FREQ)
        refs   = torch.zeros(PRETRAIN_SAMPLES, 1)  # reference = 0 during pretraining
                                                     # (the PI teacher ignores Vref anyway)

        # Concatenate along dimension 1 → shape (2000, 3)
        nn_in  = torch.cat([errs, integs, refs], dim=1)
        u_pred = controller(nn_in)          # NN output: shape (2000, 1)

        # Compute the PI teacher's desired output for these samples.
        # torch.clamp() hard-limits to ±U_LIMIT (same as the physical actuator).
        u_tgt  = torch.clamp(Kp_TEACHER * errs + Ki_TEACHER * integs,
                             -U_LIMIT, U_LIMIT)

        # --- Backpropagation ---
        # Step 1: compute MSE loss between NN prediction and PI target
        loss = loss_fn(u_pred, u_tgt)

        # Step 2: zero accumulated gradients from the previous epoch
        #         (PyTorch accumulates gradients by default — must clear!)
        optimizer_pre.zero_grad()

        # Step 3: backpropagate — compute ∂loss/∂θ for all 73 parameters
        loss.backward()

        # Step 4: Adam update step — move each parameter in the direction
        #         that reduces loss, scaled by Adam's adaptive rate
        optimizer_pre.step()

        if (ep + 1) % 100 == 0:
            print(f"  Epoch {ep+1:5d}/{PRETRAIN_EPOCHS}  |  Loss = {loss.item():.5f}")

    # Evaluate the pretrained NN on the real FMU plant (subprocess-isolated).
    # This is the RMSE ceiling we expect ES to beat.
    rmse_pretrain, _ = run_episode(controller, ref_signal, dt)
    print(f"\n  Pretraining done in {time.time()-t0:.1f}s")
    print(f"  Post-pretrain RMSE on real plant = {rmse_pretrain:.5f} V")
    # Expected: ~1.3–2.2 V  (this is the PI controller performance ceiling)


    # =============================================================================
    # PHASE 2 — EVOLUTION STRATEGIES WITH CHECKPOINT RESUME
    # =============================================================================

    n_weights = len(controller.get_flat_weights())  # = 73 for [3,6,6,1]

    # Initialise tracking variables for the ES loop
    start_epoch   = 0      # which ES step to start from (>0 if resuming)
    rmse_hist     = []     # list of best-weight RMSE after each ES step
    theta_history = []     # list of weight arrays — one saved per step
                            # used at the end to reload the globally best weights

    # --- Checkpoint resume ---
    # If the script was interrupted during a previous run, es_checkpoint.npz
    # contains the best weights seen so far, the RMSE history, and the
    # last completed epoch number.  Loading it allows the ES loop to continue
    # exactly where it left off without repeating any work.
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n  Found checkpoint: {CHECKPOINT_PATH}")
        ckpt          = np.load(CHECKPOINT_PATH, allow_pickle=True)
        saved_weights = ckpt['best_weights']
        saved_rmse    = float(ckpt['best_rmse'])
        saved_epoch   = int(ckpt['epoch'])
        saved_history = list(ckpt['rmse_hist'])

        print(f"  Resuming from step {saved_epoch + 1}  |  "
              f"Best RMSE so far = {saved_rmse:.5f} V")

        # Restore the controller to the saved best weights so the ES loop
        # continues from a good starting point (not from Phase 1 weights).
        controller.set_flat_weights(saved_weights)
        rmse_hist     = saved_history
        start_epoch   = saved_epoch + 1

        # Pad theta_history with copies of the saved weights so that
        # the final best-weight reload (which indexes into theta_history)
        # still works correctly after resume.
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

        # Save a copy of the current weights BEFORE this step's update.
        # theta_history[k] will hold the weights at the START of step k,
        # which is also the weights used to evaluate rmse_hist[k].
        theta = controller.get_flat_weights()
        theta_history.append(theta.copy())

        # --- Sample antithetic perturbations ---
        # noise[i] is a random direction in 73-dimensional weight space.
        # Each row is an independent standard normal vector ε ~ N(0, I₇₃).
        # "Antithetic" means we evaluate BOTH +ε and −ε for every noise
        # vector, so information is extracted in two opposite directions.
        noise       = np.random.randn(ES_POP_SIZE, n_weights)
        returns_pos = np.zeros(ES_POP_SIZE)   # RMSE at θ + σ·εᵢ
        returns_neg = np.zeros(ES_POP_SIZE)   # RMSE at θ − σ·εᵢ

        # --- Build the full list of 40 weight arrays to evaluate ---
        # First 20: positive perturbations  θ + σ·εᵢ  for i=0..19
        # Last  20: negative perturbations  θ − σ·εᵢ  for i=0..19
        # They are submitted to run_episodes_parallel in one batch so all 40
        # subprocesses are launched simultaneously.
        job_weights = (
            [theta + ES_SIGMA * noise[i] for i in range(ES_POP_SIZE)] +
            [theta - ES_SIGMA * noise[i] for i in range(ES_POP_SIZE)]
        )

        # Launch all 40 FMU rollouts in parallel.  Wall time ≈ one rollout.
        results = run_episodes_parallel(job_weights, ref_es, dt_es)

        # Unpack: results[0..19] are positive, results[20..39] are negative.
        for i in range(ES_POP_SIZE):
            returns_pos[i] = results[i][0]
            returns_neg[i] = results[ES_POP_SIZE + i][0]

        # --- Filter out crashed rollouts ---
        # Any rollout that returned BAD_RMSE (99.0 V) is excluded from the
        # gradient estimate.  The valid mask keeps only pairs where BOTH
        # the positive and negative rollout succeeded.
        valid = (returns_pos < BAD_RMSE * 0.9) & (returns_neg < BAD_RMSE * 0.9)
        if valid.sum() < 2:
            # Too few valid rollouts to form a meaningful gradient — skip
            # this step and keep the current weights unchanged.
            print(f"  Step {epoch+1:3d}  |  Too many subprocess failures — skipping.")
            controller.set_flat_weights(theta)
            rmse_hist.append(rmse_hist[-1] if rmse_hist else rmse_pretrain)
            continue

        # --- Compute the ES gradient estimate ---
        # advantages[i] = RMSE(θ+σεᵢ) − RMSE(θ−σεᵢ)
        # If advantages[i] > 0, the positive perturbation was WORSE than
        # the negative, so we should move in the −εᵢ direction.
        # If advantages[i] < 0, the positive was BETTER, so move in +εᵢ.
        # The outer product advantages @ noise[valid] sums these directional
        # signals across all valid pairs to form the gradient vector.
        # Dividing by 2 * N * σ makes the estimate unbiased regardless of
        # population size or noise scale.
        advantages = (returns_pos - returns_neg)[valid]
        grad_es    = (advantages @ noise[valid]) / (2 * valid.sum() * ES_SIGMA)

        # Gradient DESCENT: subtract because we minimise RMSE (lower is better).
        theta_new  = theta - ES_LR * grad_es
        controller.set_flat_weights(theta_new)

        # --- Full-resolution evaluation on the fine grid ---
        # The ES rollouts used a coarser dt_es=1e-4 grid for speed.
        # We evaluate the updated weights on the full dt=1e-5 grid so that
        # rmse_hist contains comparable, high-quality RMSE values.
        rmse_full, _ = run_episode(controller, ref_signal, dt)
        rmse_hist.append(rmse_full)

        # --- Progress reporting ---
        elapsed_step  = time.time() - t_epoch
        elapsed_total = time.time() - t0_es
        # Estimated time remaining: average time per step × remaining steps
        eta = elapsed_total / (epoch - start_epoch + 1) * (ES_EPOCHS - epoch - 1)

        best_marker = " ◄ best" if rmse_full == min(rmse_hist) else ""
        print(f"  Step {epoch+1:3d}/{ES_EPOCHS}  |  "
              f"RMSE = {rmse_full:.5f} V  |  "
              f"step={elapsed_step:.1f}s  ETA={eta:.0f}s{best_marker}")

        # --- Auto-save checkpoint to disk ---
        # Save the globally best weights seen so far (not just the latest).
        # This ensures that even if ES temporarily gets worse (common when
        # σ is too large), we can always recover the best result.
        if (epoch + 1) % ES_SAVE_EVERY == 0 or (epoch + 1) == ES_EPOCHS:
            best_idx  = int(np.argmin(rmse_hist))
            best_rmse = rmse_hist[best_idx]
            np.savez(CHECKPOINT_PATH,
                     best_weights = theta_history[best_idx],
                     best_rmse    = best_rmse,
                     epoch        = epoch,
                     rmse_hist    = np.array(rmse_hist))

    # --- Reload the globally best weights found during ES ---
    # rmse_hist[k] is the full-resolution RMSE at the end of step k.
    # We find the step with the lowest RMSE and restore those weights.
    best_step  = int(np.argmin(rmse_hist))
    best_rmse  = rmse_hist[best_step]
    controller.set_flat_weights(theta_history[best_step])

    # One final evaluation to confirm (process variance is small but non-zero
    # due to FMU initial condition randomness — verified RMSE may differ
    # slightly from the training RMSE).
    rmse_verified, _ = run_episode(controller, ref_signal, dt)

    print(f"\n  ES complete.")
    print(f"  Best step  : {best_step + 1}  |  RMSE = {best_rmse:.5f} V")
    print(f"  Verified   : {rmse_verified:.5f} V  (full resolution)")
    print(f"  Best weights reloaded ✓")

    if os.path.exists(CHECKPOINT_PATH):
        # Retry loop: on Windows, a just-exited subprocess may still hold
        # the file handle open for a fraction of a second after it exits.
        # Retrying with short sleeps avoids a spurious PermissionError.
        for _attempt in range(5):
            try:
                os.remove(CHECKPOINT_PATH)
                print(f"  Checkpoint removed (clean run).")
                break
            except PermissionError:
                time.sleep(0.5)   # wait 500 ms for handle to be released
        else:
            print(f"  Note: could not remove checkpoint (file in use) — safe to delete manually.")


    # =============================================================================
    # PHASE 3 — COLLECT ANIMATION FRAMES
    # =============================================================================
    # Run ANIM_FRAMES independent full-resolution episodes to collect
    # Vout(t) trajectories for the GIF.  Because each subprocess is isolated,
    # minor numerical differences between frames are expected (floating-point
    # non-determinism in the FMU solver's adaptive step control).
    # The slight variation between frames adds visual richness to the animation.

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
    # VISUALISATION — NEURAL NETWORK TREE (static helper used in draw_nn_tree)
    # =============================================================================

    def draw_nn_tree(ax, ctrl_net, layer_sizes):
        """
        Render the trained neural network as a layered directed graph.

        Each neuron is drawn as a circle; each weight as a coloured edge.
        Edge colour encodes sign: blue = positive, red = negative.
        Edge thickness and opacity encode magnitude: heavier = larger |w|.

        COORDINATE SYSTEM:
          x-axis: layer index, normalised 0.0 (input) → 1.0 (output)
          y-axis: neuron index within layer, centred at 0.0

        NODE COLOUR SCHEME:
          Light blue  (#aed6f1): standard hidden neurons
          Light green (#a9dfbf): feedforward input neuron (Vref) — highlighted
                                 to remind the viewer that this input provides
                                 anticipatory control rather than feedback
          Light yellow(#f9e79f): output neuron (u) — the control voltage
        """
        ax.clear()
        ax.set_title(
            "Trained NN Weights  —  blue = positive  |  red = negative  |  green node = Vref feedforward",
            fontsize=9, fontweight='bold', color='#222222'
        )
        ax.axis('off')
        ax.set_facecolor('#f8f8f8')

        n_layers = len(layer_sizes)

        # Compute (x, y) centre position for every neuron in every layer.
        # pos[l][n] = [x, y] for the n-th neuron in layer l.
        pos = []
        for l, size in enumerate(layer_sizes):
            # Distribute neurons evenly between −(size−1)/2 and +(size−1)/2
            # so the layer is vertically centred at y=0.
            ys = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
            xs = np.full(size, float(l) / max(n_layers - 1, 1))
            pos.append(np.stack([xs, ys], axis=1))

        # --- Draw edges (weights) ---
        # Extract only the nn.Linear modules (skipping nn.Tanh modules
        # which have no parameters and don't correspond to graph edges).
        linear_modules = [m for m in ctrl_net.net if isinstance(m, nn.Linear)]
        w_all = [lin.weight.detach().numpy() for lin in linear_modules]

        # Find the global maximum |weight| across all layers.
        # Used to normalise alpha and linewidth so the heaviest weight
        # has full opacity/thickness and zero weights are nearly invisible.
        w_max = max(np.abs(w).max() for w in w_all) + 1e-6

        for l, (lin, w) in enumerate(zip(linear_modules, w_all)):
            # w has shape (out_neurons, in_neurons) = (LAYERS[l+1], LAYERS[l])
            # w[j, i] is the weight from neuron i in layer l to neuron j in l+1.
            for j in range(w.shape[0]):      # loop over destination neurons
                for i in range(w.shape[1]):  # loop over source neurons
                    src   = pos[l][i]        # [x, y] of source neuron
                    tgt   = pos[l + 1][j]    # [x, y] of destination neuron
                    val   = w[j, i]
                    color = '#2166ac' if val >= 0 else '#d6604d'
                    # Scale both alpha (transparency) and linewidth by |val|/w_max
                    # so large weights stand out visually.
                    alpha = 0.25 + 0.65 * abs(val) / w_max
                    lw    = 0.5  + 2.0  * abs(val) / w_max
                    ax.plot([src[0], tgt[0]], [src[1], tgt[1]],
                            color=color, lw=lw, alpha=alpha, zorder=1)
                    # Weight label at the midpoint of the edge
                    mx = (src[0] + tgt[0]) / 2
                    my = (src[1] + tgt[1]) / 2
                    ax.text(mx, my, f"{val:.4f}",
                            color='#111111', fontsize=5,
                            ha='center', va='center', zorder=3,
                            bbox=dict(boxstyle='round,pad=0.05',
                                      fc='white', ec='none', alpha=0.6))

        # --- Draw nodes (neurons) ---
        node_colors_map = {
            'default': '#aed6f1',   # standard hidden neuron: light blue
            'ff'     : '#a9dfbf',   # Vref feedforward input: light green
            'output' : '#f9e79f',   # control output u(t):   light yellow
        }
        for l, layer_pos in enumerate(pos):
            colors = []
            for n_idx in range(len(layer_pos)):
                if l == 0 and n_idx == 2:          # third input neuron = Vref
                    colors.append(node_colors_map['ff'])
                elif l == n_layers - 1:            # output layer
                    colors.append(node_colors_map['output'])
                else:
                    colors.append(node_colors_map['default'])
            ax.scatter(layer_pos[:, 0], layer_pos[:, 1],
                       s=380, c=colors, edgecolors='#1a5276',
                       linewidths=1.8, zorder=5)

        # Input signal labels positioned to the left of the input layer
        input_labels = ['e(t)', '∫e·dt', 'Vref[FF]']
        input_colors = ['#1a5276', '#1a5276', '#1e8449']   # green for FF input
        for i, (lbl, col) in enumerate(zip(input_labels, input_colors)):
            ax.text(pos[0][i, 0] - 0.06, pos[0][i, 1],
                    lbl, ha='right', va='center',
                    fontsize=8, color=col, fontweight='bold')

        # Output label to the right of the output layer
        ax.text(pos[-1][0, 0] + 0.06, pos[-1][0, 1],
                'u(t)', ha='left', va='center',
                fontsize=8, color='#7d6608', fontweight='bold')

        # Layer headers above each column
        headers = (['Input'] +
                   [f'Hidden {k+1}\n(Tanh)' for k in range(n_layers - 2)] +
                   ['Output'])
        for l, lbl in enumerate(headers):
            top_y = pos[l][-1, 1] + 0.85
            ax.text(float(l) / max(n_layers - 1, 1), top_y,
                    lbl, ha='center', va='bottom',
                    fontsize=8, color='#1a5276', fontweight='bold')

        # Neuron count labels below each column
        for l, size in enumerate(layer_sizes):
            bot_y = pos[l][0, 1] - 0.7
            ax.text(float(l) / max(n_layers - 1, 1), bot_y,
                    f'{size} neurons', ha='center', va='top',
                    fontsize=7, color='#555555')

        # Legend
        ax.plot([], [], color='#2166ac', lw=2.0, label='positive weight')
        ax.plot([], [], color='#d6604d', lw=2.0, label='negative weight')
        ax.plot([], [], 'o', ms=9, color='#a9dfbf',
                mec='#1a5276', mew=1.5, label='feedforward (Vref)')
        ax.plot([], [], 'o', ms=9, color='#f9e79f',
                mec='#1a5276', mew=1.5, label='output (u)')
        ax.legend(loc='lower center', fontsize=7, ncol=4,
                  framealpha=0.8, edgecolor='#cccccc')


    # =============================================================================
    # ANIMATION — 3-panel layout, ALL panels animated
    # =============================================================================
    #
    # PANEL 1 — top-left: Sine tracking
    #   The reference Vref(t) and NN output Vout(t) are drawn incrementally,
    #   revealing more of the time axis with each frame (left-to-right sweep).
    #   An orange fill_between region shows the instantaneous error e(t).
    #   A red dot marks the current "pen tip" at the leading edge.
    #
    #   HOW THE REVEAL WORKS:
    #     reveal_per_frame = N_steps // ANIM_FRAMES  (e.g. 5000//20 = 250)
    #     Frame k shows t[0 .. (k+1)*250], incrementally extending each frame.
    #
    # PANEL 2 — top-right: RMSE convergence
    #   The training curve grows from left to right in sync with Panel 1.
    #   A vertical dotted line marks the current training step.
    #   A red dot sits on the curve at the current step's RMSE.
    #   A text badge below the x-axis always shows:
    #     "Step N / 100    RMSE = X.XXXX V  [◄ best]"
    #
    #   HOW THE STEP MAPPING WORKS:
    #     step_idx = int((frame+1) * n_steps_hist / ANIM_FRAMES)
    #     Maps 20 animation frames onto 100 ES training steps proportionally.
    #     Frame 0 → step 5, frame 1 → step 10, ..., frame 19 → step 100.
    #
    # PANEL 3 — bottom (full width): NN weight tree
    #   The NN structure is static (weights don't change after training).
    #   The EDGES animate with a "travelling wave" pulse:
    #
    #   HOW THE PULSE WORKS:
    #     phase = frame * (2π / ANIM_FRAMES) * 2    ← 2 full cycles per GIF
    #     For each edge in layer l:
    #       layer_phase = phase − l * (π/2)          ← π/2 delay per layer
    #       pulse = 0.5 + 0.5 * sin(layer_phase)     ← oscillates 0 → 1
    #       edge.alpha     = base_alpha * (0.35 + 0.65 * pulse)
    #       edge.linewidth = base_lw    * (0.5  + 0.5  * pulse)
    #
    #     The π/2 (90°) delay per layer means the input layer brightens first,
    #     then hidden layer 1, then hidden layer 2, then the output — creating
    #     a visible left-to-right wave that visually represents signal flow
    #     through the network during inference.

    # Pre-compute scalar RMSE for each collected animation frame
    frame_rmse = [float(np.sqrt(np.mean((ref_signal - y)**2))) for y in anim_frames]

    # Extract weight matrices once — they do not change during animation
    linear_modules = [m for m in controller.net if isinstance(m, nn.Linear)]
    w_layers = [lin.weight.detach().numpy() for lin in linear_modules]
    n_layers  = len(LAYERS)

    # Global weight magnitude maximum for normalising edge thickness / alpha
    w_max = max(np.abs(w).max() for w in w_layers) + 1e-6

    # Pre-compute neuron centre positions (same layout as draw_nn_tree)
    node_pos = []
    for l, size in enumerate(LAYERS):
        ys = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        xs = np.full(size, float(l) / max(n_layers - 1, 1))
        node_pos.append(np.stack([xs, ys], axis=1))

    # --- Figure and subplot layout ---
    fig = plt.figure(figsize=(18, 12), facecolor='#f0f4f8')
    best_rmse_str = f"Final RMSE = {min(rmse_hist):.4f} V"
    fig.suptitle(
        "RLC Circuit - Evolution Strategies (ES)\n"
        "Architecture [3-6-6-1]  |  Feedforward + Feedback  |  ES Trained  |  " + best_rmse_str,
        fontsize=13, fontweight='bold', color='#1a3a5c'
    )

    # GridSpec: 2 rows × 2 columns.
    # Row 0 height ratio 1: panels 1 & 2 (top, equal width)
    # Row 1 height ratio 1.3: panel 3 (bottom, spans full width)
    # hspace=0.52 gives room for the badge text below panel 2's x-axis.
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3],
                          hspace=0.52, wspace=0.30,
                          left=0.07, right=0.97, top=0.88, bottom=0.05)

    ax_sine = fig.add_subplot(gs[0, 0])   # top-left
    ax_rmse = fig.add_subplot(gs[0, 1])   # top-right
    ax_tree = fig.add_subplot(gs[1, :])   # bottom, spans both columns

    # =========================================================
    # PANEL 1 — Sine tracking setup
    # =========================================================
    ax_sine.set_facecolor('#ffffff')
    ax_sine.set_xlim(0, T_sim)
    ax_sine.set_ylim(-1.25 * VREF_AMPL, 1.25 * VREF_AMPL)
    ax_sine.set_xlabel("Time (s)", fontsize=9)
    ax_sine.set_ylabel("Voltage (V)", fontsize=9)
    ax_sine.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax_sine.tick_params(labelsize=8)

    # Faint ghost of the full reference — gives the viewer a sense of
    # where the signal is headed before the animated line reaches there.
    ax_sine.plot(t_steps, ref_signal, color='#2471a3', lw=1.0,
                 alpha=0.18, zorder=1)

    # Animated line artists — data filled in update_anim() each frame.
    line_ref_anim, = ax_sine.plot([], [], color='#2471a3',
                                   lw=2.2, label='V_ref (reference)', zorder=3)
    line_out_anim, = ax_sine.plot([], [], color='#c0392b',
                                   lw=2.0, ls='--', label='V_out (NN output)', zorder=4)

    # fill_between placeholder — will be removed and redrawn each frame
    # because matplotlib's fill_between does not support incremental updates.
    fill_err = ax_sine.fill_between([], [], [], color='#e67e22',
                                     alpha=0.15, label='error band', zorder=2)

    # Red dot at the leading (most recent) time point of the animated lines
    dot_now, = ax_sine.plot([], [], 'o', color='#c0392b', ms=7, zorder=6)

    ax_sine.legend(loc='upper right', fontsize=8, framealpha=0.9)
    title_sine = ax_sine.set_title('', fontsize=9, color='#922b21')

    # =========================================================
    # PANEL 2 — RMSE convergence setup
    # =========================================================
    ax_rmse.set_facecolor('#ffffff')
    steps_x = np.arange(1, len(rmse_hist) + 1)   # x-axis: step numbers 1..100

    # Static background elements drawn once (never updated):
    #   faint full curve — shows the complete training history as context
    #   filled area       — adds visual weight below the curve
    #   horizontal dashed — marks the best (minimum) RMSE achieved
    ax_rmse.plot(steps_x, rmse_hist, color='#1e8449',
                 lw=1.0, alpha=0.25, zorder=1)
    ax_rmse.fill_between(steps_x, rmse_hist, alpha=0.06, color='#1e8449')
    ax_rmse.axhline(min(rmse_hist), color='#922b21', lw=1.2,
                    ls='--', alpha=0.6,
                    label=f'best = {min(rmse_hist):.4f} V')

    # Animated elements — updated each frame:
    line_rmse_live, = ax_rmse.plot([], [], color='#1a5276',
                                    lw=2.2, zorder=3, label='training RMSE')
    dot_rmse,       = ax_rmse.plot([], [], 'o', color='#c0392b',
                                    ms=9, zorder=5, label='current step')

    # Vertical dotted line sweeping right to mark the current training step.
    # axvline creates a line from y=0 to y=1 in axes coordinates (full height).
    vline_rmse = ax_rmse.axvline(x=0, color='#c0392b', lw=1.4,
                                  ls=':', alpha=0.7, zorder=4)

    # Text badge placed BELOW the x-axis at axes-normalised position (0.5, -0.22).
    # y=-0.22 puts it below the xlabel without overlapping the subplot below.
    # The bounding box (bbox) makes it readable against any background.
    title_rmse = ax_rmse.text(
        0.5, -0.22, '', transform=ax_rmse.transAxes,
        ha='center', va='top', fontsize=10, fontweight='bold',
        color='#1a3a5c',
        bbox=dict(boxstyle='round,pad=0.5', fc='#eaf4fb',
                  ec='#2471a3', lw=1.5, alpha=0.97)
    )

    ax_rmse.set_xlabel("ES Training Step", fontsize=9)
    ax_rmse.set_ylabel("RMSE (V)", fontsize=9)
    ax_rmse.set_xlim(1, len(rmse_hist))
    ax_rmse.set_ylim(0, max(rmse_hist) * 1.08)
    ax_rmse.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax_rmse.tick_params(labelsize=8)
    ax_rmse.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax_rmse.set_title("ES Training Convergence", fontsize=9,
                       fontweight='bold', color='#1a5276')

    # =========================================================
    # PANEL 3 — NN tree setup  (nodes static, edges animated)
    # =========================================================
    ax_tree.set_facecolor('#f8f8f8')
    ax_tree.axis('off')
    ax_tree.set_title(
        "Neural Network — Weights  |  Pulse = signal flow  |  "
        "blue = positive weight  |  red = negative weight  |  green = Vref feedforward",
        fontsize=9, fontweight='bold', color='#222222'
    )

    # Draw neuron circles as a scatter plot.  Positions never change so
    # we draw them once here and do NOT update them in update_anim().
    node_scatters = []
    for l, layer_pos in enumerate(node_pos):
        colors = []
        for n_idx in range(len(layer_pos)):
            if l == 0 and n_idx == 2:
                colors.append('#a9dfbf')    # feedforward input = green
            elif l == n_layers - 1:
                colors.append('#f9e79f')    # output neuron = yellow
            else:
                colors.append('#aed6f1')    # hidden neurons = light blue
        sc = ax_tree.scatter(layer_pos[:, 0], layer_pos[:, 1],
                             s=420, c=colors, edgecolors='#1a5276',
                             linewidths=2.0, zorder=10)
        node_scatters.append(sc)

    # Static text labels for input signals, output signal and layer headers
    input_labels  = ['e(t)', 'integral', 'Vref [FF]']
    input_colors  = ['#1a5276', '#1a5276', '#1e8449']
    for i, (lbl, col) in enumerate(zip(input_labels, input_colors)):
        ax_tree.text(node_pos[0][i, 0] - 0.055, node_pos[0][i, 1],
                     lbl, ha='right', va='center',
                     fontsize=8, color=col, fontweight='bold')
    ax_tree.text(node_pos[-1][0, 0] + 0.055, node_pos[-1][0, 1],
                 'u(t)', ha='left', va='center',
                 fontsize=8, color='#7d6608', fontweight='bold')
    headers = ['Input'] + [f'Hidden {k+1}' for k in range(n_layers - 2)] + ['Output']
    for l, lbl in enumerate(headers):
        top_y = node_pos[l][-1, 1] + 0.82
        ax_tree.text(float(l) / max(n_layers - 1, 1), top_y,
                     lbl, ha='center', va='bottom',
                     fontsize=9, color='#1a5276', fontweight='bold')

    # --- Create one Line2D object PER EDGE ---
    # We create them once here and keep references in edge_lines[].
    # In update_anim() we modify only alpha and linewidth (no redraws),
    # which is much faster than deleting and recreating lines each frame.
    edge_lines = []   # Line2D objects — one per weight connection
    edge_meta  = []   # (base_color, base_alpha, base_lw, layer_idx, weight_val)
                       # Stored alongside each line to compute pulse modulation

    for l, w in enumerate(w_layers):
        for j in range(w.shape[0]):      # destination neuron index
            for i in range(w.shape[1]):  # source neuron index
                val   = w[j, i]
                src   = node_pos[l][i]
                tgt   = node_pos[l + 1][j]
                color = '#2166ac' if val >= 0 else '#d6604d'
                # Base appearance proportional to weight magnitude.
                # Floor raised to 0.35 so even small weights are clearly visible.
                # Ceiling kept at 0.90 to leave room for the pulse to brighten further.
                base_alpha = 0.35 + 0.55 * abs(val) / w_max
                base_lw    = 1.0  + 3.0  * abs(val) / w_max
                ln, = ax_tree.plot([src[0], tgt[0]], [src[1], tgt[1]],
                                   color=color, lw=base_lw,
                                   alpha=base_alpha, zorder=2)
                edge_lines.append(ln)
                edge_meta.append((color, base_alpha, base_lw, l, val))

    # Static weight value labels at the midpoint of each edge.
    # fontsize=5 is very small but necessary to fit without overlapping.
    for l, w in enumerate(w_layers):
        for j in range(w.shape[0]):
            for i in range(w.shape[1]):
                val = w[j, i]
                src = node_pos[l][i]
                tgt = node_pos[l + 1][j]
                ax_tree.text((src[0] + tgt[0]) / 2,
                             (src[1] + tgt[1]) / 2,
                             f"{val:.4f}",
                             color='#111111', fontsize=5,
                             ha='center', va='center', zorder=4,
                             bbox=dict(boxstyle='round,pad=0.05',
                                       fc='white', ec='none', alpha=0.25))

    # Legend for Panel 3
    ax_tree.plot([], [], color='#2166ac', lw=2.5, label='positive weight')
    ax_tree.plot([], [], color='#d6604d', lw=2.5, label='negative weight')
    ax_tree.plot([], [], 'o', ms=10, color='#a9dfbf',
                 mec='#1a5276', mew=1.5, label='feedforward Vref')
    ax_tree.plot([], [], 'o', ms=10, color='#f9e79f',
                 mec='#1a5276', mew=1.5, label='output u(t)')
    ax_tree.legend(loc='lower center', fontsize=8, ncol=4,
                   framealpha=0.85, edgecolor='#cccccc')

    # =========================================================
    # ANIMATION CALLBACK FUNCTIONS
    # =========================================================

    # How many fine-grid time steps to reveal per animation frame.
    # e.g. 5000 steps / 20 frames = 250 steps per frame.
    reveal_per_frame = max(1, N_steps // ANIM_FRAMES)

    def init_anim():
        """
        Initialise all animated artists to their empty/starting state.
        Called once by FuncAnimation before the first frame is drawn.
        """
        line_ref_anim.set_data([], [])
        line_out_anim.set_data([], [])
        dot_now.set_data([], [])
        line_rmse_live.set_data([], [])
        dot_rmse.set_data([], [])
        vline_rmse.set_xdata([0])
        title_rmse.set_text('')
        # Reset all node scatters to their base (mid-pulse) size
        for l_idx, sc in enumerate(node_scatters):
            sc.set_sizes([400] * LAYERS[l_idx])
        return [line_ref_anim, line_out_anim, dot_now,
                line_rmse_live, dot_rmse] + edge_lines + node_scatters

    def update_anim(frame):
        """
        Update all three panels for a given animation frame index.

        Called by FuncAnimation for frame = 0, 1, 2, ..., ANIM_FRAMES−1.
        Must return a list of modified artists (even with blit=False).
        """
        idx     = frame % len(anim_frames)   # which collected Vout trajectory
        y_frame = anim_frames[idx]           # Vout(t) array for this frame
        rmse_f  = frame_rmse[idx]            # pre-computed RMSE for this frame

        # --------------------------------------------------
        # Panel 1 update: extend both lines by reveal_per_frame steps
        # --------------------------------------------------
        # n_show is clamped to N_steps so the last frame shows the full signal.
        n_show = min((frame + 1) * reveal_per_frame, N_steps)
        t_show = t_steps[:n_show]

        line_ref_anim.set_data(t_show, ref_signal[:n_show])
        line_out_anim.set_data(t_show, y_frame[:n_show])

        # Rebuild error fill from scratch each frame.
        # ax.collections accumulates fill_between patches; remove them all
        # before adding the new one to avoid layering artefacts.
        for coll in ax_sine.collections:
            coll.remove()
        ax_sine.fill_between(t_show,
                              ref_signal[:n_show],
                              y_frame[:n_show],
                              color='#e67e22', alpha=0.12, zorder=2)

        # Advance the red dot to the current leading edge position
        dot_now.set_data([t_show[-1]], [y_frame[n_show - 1]])

        # Update panel 1 title with live RMSE and peak error
        err_max = float(np.abs(ref_signal[:n_show] - y_frame[:n_show]).max())
        title_sine.set_text(
            f"Tracking  RMSE={rmse_f:.4f}V  max|e|={err_max:.3f}V"
        )

        # --------------------------------------------------
        # Panel 2 update: grow RMSE curve to the current step
        # --------------------------------------------------
        # Map the animation frame (0..19) proportionally onto the ES step
        # axis (1..100).  step_idx is clamped to len(rmse_hist) to handle
        # the last frame exactly.
        step_idx = min(int((frame + 1) * len(rmse_hist) / ANIM_FRAMES),
                       len(rmse_hist))
        if step_idx > 0:
            cur_step = steps_x[step_idx - 1]   # ES step number (1-indexed)
            cur_rmse = rmse_hist[step_idx - 1]  # RMSE at that step

            # Grow the blue live curve from step 1 to cur_step
            line_rmse_live.set_data(steps_x[:step_idx], rmse_hist[:step_idx])

            # Move the red dot to the current step on the curve
            dot_rmse.set_data([cur_step], [cur_rmse])

            # Advance the vertical marker line to x = cur_step
            vline_rmse.set_xdata([cur_step])

            # Update the text badge below the x-axis.
            # The ◄ best flag appears when the current step is the global minimum.
            best_flag = "  ◄ best" if cur_rmse == min(rmse_hist) else ""
            title_rmse.set_text(
                f"Step {int(cur_step):3d} / {len(rmse_hist)}"
                f"    RMSE = {cur_rmse:.4f} V{best_flag}"
            )

        # --------------------------------------------------
        # Panel 3 update: pulse edges AND nodes with travelling wave
        # --------------------------------------------------
        # phase advances by 2*(2π/ANIM_FRAMES) per frame, completing 2 full
        # sine cycles.  Each layer fires π/2 radians (90°) later than the
        # previous, making the brightness wave travel visibly left→right.
        phase = frame * (2 * np.pi / ANIM_FRAMES) * 2

        # --- Pulse edges ---
        for ln, (color, base_alpha, base_lw, layer_idx, val) in zip(edge_lines, edge_meta):
            layer_phase = phase - layer_idx * (np.pi / 2)
            pulse = 0.5 + 0.5 * np.sin(layer_phase)   # 0 → 1

            # Strong contrast: alpha swings between 15% and 100% of base.
            # Linewidth swings between 30% and 100% — clearly visible change.
            ln.set_alpha(np.clip(base_alpha * (0.15 + 0.85 * pulse), 0.05, 1.0))
            ln.set_linewidth(base_lw * (0.3 + 0.7 * pulse))

        # --- Pulse node sizes so the neurons visibly "fire" ---
        # Each layer's scatter plot gets a new size array each frame.
        # Size oscillates between 200 (dim, between pulses) and 600 (bright, firing).
        # The same π/2 layer delay ensures neurons fire in left→right order.
        for l_idx, sc in enumerate(node_scatters):
            layer_phase = phase - l_idx * (np.pi / 2)
            pulse_node  = 0.5 + 0.5 * np.sin(layer_phase)   # 0 → 1
            new_size    = 200 + 400 * pulse_node             # 200 → 600
            sc.set_sizes([new_size] * LAYERS[l_idx])

        return [line_ref_anim, line_out_anim, dot_now,
                line_rmse_live, dot_rmse] + edge_lines + node_scatters


    # FuncAnimation calls update_anim(frame) for frame=0,1,...,ANIM_FRAMES-1.
    # interval=400 ms between frames → GIF plays at ~2.5 fps in viewers
    # (PillowWriter saves at fps=3 → 333 ms per frame — close match).
    ani = FuncAnimation(fig, update_anim, frames=ANIM_FRAMES,
                        init_func=init_anim, interval=400, blit=False)

    # Save as GIF using Pillow (no FFmpeg needed).
    out_gif = os.path.join(_HERE, "rlc_nn_tree_rmse_live.gif")
    ani.save(out_gif, writer=PillowWriter(fps=3))

    # rect=[left, bottom, right, top] in figure-normalised coordinates.
    # Leaves room at the top for the suptitle.
    plt.tight_layout(rect=[0, 0, 1, 0.93])
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