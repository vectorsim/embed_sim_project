"""
dfc_nn_tuner.py
===============
Neural-network surrogate tuner for DFControllerBlock gains.

Strategy
--------
Phase 1 — Exploration  (Latin-Hypercube sampling)
    Run N_EXPLORE simulations with random gains sampled across physics-derived
    bounds.  Each simulation produces a scalar cost from _cost_metrics().
    This builds a (gains → cost) dataset.

Phase 2 — NN surrogate training
    Train a small MLP on the dataset:
        Input  : [Kp_speed, Kp_id, Kp_iq]   (3 features, normalised)
        Output : predicted cost              (1 scalar)
    Loss: MSE.  Trained with Adam, ~500 epochs.

Phase 3 — Surrogate optimisation
    Run gradient descent on the trained NN to find the input that minimises
    predicted cost.  Start from the best observed point.
    Multiple restarts from random initialisations guard against local minima.

Phase 4 — Verification
    Run one real simulation at the NN-recommended gains and report true cost.

Why NN instead of GP (Bayesian)?
---------------------------------
GP scales as O(N³) with dataset size — expensive beyond ~200 points.
NN scales as O(N) and the gradient is free (backprop), so Phase 3 is fast.
For 3 gain parameters and ~100 samples the NN is more than expressive enough.

Cost function  (same weights as SMC tuner for direct comparison)
-----------------------------------------------------------------
    cost = W_SS * ss_err [RPM]  +  W_ID * id_rms [A]  +  W_CHAT * iq_chat [A]

    W_SS   = 2.0   steady-state speed error
    W_ID   = 50.0  d-axis RMS  (MTPA penalty — id should be 0)
    W_CHAT = 4.0   iq chattering  (std of iq_ref in steady state)

Bounds  (physics-derived)
--------------------------
    Kp_speed : [0.02, 0.40]  A/(rad/s)   iq_ref = I_MAX at [9, 178] rad/s error
    Kp_id    : [0.50, 8.00]  V/A         current P gain d-axis
    Kp_iq    : [0.50, 8.00]  V/A         current P gain q-axis
"""

from __future__ import annotations

import sys
import math
import time
import numpy as np
import importlib
from pathlib import Path

# ── path setup (mirrors db42s02_closed_loop_dfc_20k.py) ──────────────────────
from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE    = get_current_parent()
_ROOT    = get_project_root()
_FS_ELEC = _ROOT / "fs_electrical_machines"

for _p in (get_embedsim_import_path(), str(_FS_ELEC), str(_FS_ELEC / "c_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import sim components
from smc_controller_block import _DB42S02
import db42s02_closed_loop_dfc_20k as _dfc_sim

# ── Tuner hyper-parameters ────────────────────────────────────────────────────

N_EXPLORE   = 40     # LHS exploration simulations
N_EPOCHS    = 600    # NN training epochs
N_RESTARTS  = 12     # gradient-descent restarts from random initialisations
LR_SURR     = 3e-3   # Adam lr for surrogate training
LR_OPT      = 5e-2   # Adam lr for surrogate optimisation
N_OPT_STEPS = 400    # steps per optimisation restart

# Cost weights
W_SS   = 2.0
W_ID   = 50.0
W_CHAT = 4.0

T_SIM  = _dfc_sim.T_SIM
TARGET_RPM = _dfc_sim.TARGET_RPM

# ── Physics-derived bounds ────────────────────────────────────────────────────

I_MAX = _DB42S02.SMC_I_MAX

BOUNDS = np.array([
    [0.02,  0.40],   # Kp_speed  [A/(rad/s)]
    [0.50,  8.00],   # Kp_id     [V/A]
    [0.50,  8.00],   # Kp_iq     [V/A]
], dtype=np.float64)

PARAM_NAMES = ["Kp_speed", "Kp_id   ", "Kp_iq   "]


# =============================================================================
# Cost metrics
# =============================================================================

def _cost_metrics(d: dict | None) -> dict | None:
    """Compute scalar cost from a _run_sim() result dict."""
    if d is None:
        return None

    t   = d["t"]
    rpm = d["speed_rpm"]
    idd = d["id"]
    iqr = d["iq_ref"]

    if len(t) < 200:
        return None

    # Hard guard: diverged
    if float(np.max(np.abs(rpm))) > TARGET_RPM * 3.0:
        return None

    # Steady-state mask: last 15% of sim
    ss = t > 0.85 * T_SIM
    if not np.any(ss):
        return None

    ref_ss  = float(np.mean(d["omega_ref_rpm"][ss])) if "omega_ref_rpm" in d else TARGET_RPM
    ss_err  = float(np.mean(np.abs(rpm[ss] - ref_ss)))
    id_rms  = float(np.sqrt(np.mean(idd[ss] ** 2)))
    iq_chat = float(np.std(iqr[ss]))

    # Hard guard: SS error > 800 RPM → controller failed
    if ss_err > 800.0:
        return None

    cost = W_SS * ss_err + W_ID * id_rms + W_CHAT * iq_chat
    return {"cost": cost, "ss_err": ss_err, "id_rms": id_rms, "iq_chat": iq_chat}


def _run_with_gains(kp_speed: float, kp_id: float, kp_iq: float) -> dict | None:
    """Run one DFC simulation with given gains and return cost metrics."""
    # Patch the DFControllerBlock defaults on the fly via _run_sim
    # by monkey-patching the module-level sim function.
    import db42s02_closed_loop_dfc_20k as sim_mod
    from diff_flatness_controller_block import DFControllerBlock

    # Temporarily override default gains — cleanest without re-architecting sim
    _orig = sim_mod._run_sim

    def _patched_run():
        import math
        from embedsim import EmbedSim, ODESolver, VectorEnd
        from embedsim.core_blocks import VectorSignal, DEFAULT_DTYPE
        from embedsim.source_blocks import VectorStep, VectorConstant
        from embedsim.simulation_engine import VectorDelay
        from embedsim.code_generator import CodeGenStart, CodeGenEnd
        from motor_utility_blocks import SVPWMPackBlock
        from svpwm_block import SVPWMBlock
        from ctrl_packer import CtrlPacker
        from machine_feedback import db42s02_feedback_profile

        V_DC   = sim_mod.V_DC
        DT     = sim_mod.DT
        T_SIM  = sim_mod.T_SIM
        T_RAMP = sim_mod._RAMP_TIME
        TRADS  = sim_mod.TARGET_RADS_MECH
        MSIZ   = sim_mod._MOTOR_OUT_SIZE

        try:
            cg_start = CodeGenStart("cg_start")
            dfc = DFControllerBlock(
                "dfc",
                P_POLES   = int(_DB42S02.SMC_P_POLES),
                R_S       = _DB42S02.SMC_R_S,
                L_D       = _DB42S02.SMC_L_D,
                L_Q       = _DB42S02.SMC_L_Q,
                LAMBDA_PM = _DB42S02.SMC_LAMBDA_PM,
                V_DC      = V_DC,
                I_MAX     = I_MAX,
                dt_s      = DT,
                Kp_id     = kp_id,
                Kp_iq     = kp_iq,
                Kp_speed  = kp_speed,
                smo_k     = _DB42S02.SMC_SMO_K,
                smo_tau   = 1.0 / (2.0 * math.pi * _DB42S02.SMC_SMO_FC),
                fusion_omega_lo = 50.0,
                fusion_omega_hi = 250.0,
                fusion_gamma    = 2.0,
                fusion_iir_lo   = 0.05,
                fusion_iir_hi   = 0.30,
            )
            svpwm_pack  = SVPWMPackBlock("svpwm_pack", v_dc=V_DC)
            svpwm       = SVPWMBlock("svpwm", use_c_backend=False)
            cg_end      = CodeGenEnd("cg_end")
            speed_ref   = VectorStep("speed_ref", step_time=0.0,
                                     before_value=TRADS, after_value=TRADS)
            load_torque = VectorConstant("load_torque", value=sim_mod.T_LOAD_ZERO)
            motor       = sim_mod.DB42S02PlantBlock("motor")
            motor_delay = VectorDelay("motor_delay", initial=[0.0] * MSIZ)
            ctrl        = CtrlPacker("ctrl_packer",
                                     target_rads_mech=TRADS, ramp_time=T_RAMP,
                                     feedback=db42s02_feedback_profile(
                                         enc_glitch=False, adc_noise=False, adc_sat=False))
            sink        = VectorEnd("sink")
            sink_cg     = VectorEnd("sink_cg")

            cg_start >> dfc >> svpwm_pack >> svpwm >> cg_end
            motor >> motor_delay >> ctrl
            speed_ref   >> ctrl
            ctrl        >> cg_start
            cg_end      >> motor
            load_torque >> motor
            motor       >> sink
            cg_end      >> sink_cg

            sim = EmbedSim(sinks=[sink, sink_cg], T=T_SIM, dt=DT,
                           solver=ODESolver.EULER)
            sim.scope.add(motor, indices=[0, 6, 7], label="Motor")
            sim.run()
        except Exception:
            return None

        sc = sim.scope
        t  = np.array(sc.t, dtype=np.float32)
        ld = dfc.log_data

        def _m(pos):
            s = sc.get_signal("Motor", pos)
            return s if s is not None else np.zeros(len(t), np.float32)

        def _i(key):
            if len(ld["t"]) > 1:
                return np.interp(t, ld["t"], ld[key]).astype(np.float32)
            return np.zeros(len(t), np.float32)

        return {
            "t":             t,
            "speed_rpm":     _m(0),
            "id":            _m(1),
            "iq_ref":        _i("iq_ref"),
            "omega_ref_rpm": _i("speed_ref"),
        }

    return _cost_metrics(_patched_run())


# =============================================================================
# Latin-Hypercube sampling
# =============================================================================

def _lhs(n: int, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Latin-Hypercube sample — n points in d dimensions."""
    d    = bounds.shape[0]
    cuts = np.linspace(0.0, 1.0, n + 1)
    X    = np.zeros((n, d))
    for j in range(d):
        u = rng.uniform(cuts[:-1], cuts[1:])
        rng.shuffle(u)
        X[:, j] = bounds[j, 0] + u * (bounds[j, 1] - bounds[j, 0])
    return X


# =============================================================================
# Minimal MLP surrogate  (pure numpy — no PyTorch/TF dependency)
# =============================================================================

class MLP:
    """
    2-hidden-layer MLP, trained with Adam, pure NumPy.

    Architecture:  3 → 32 → 32 → 1   (tanh activations, linear output)
    """

    def __init__(self, n_in: int = 3, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        def _w(r, c): return rng.standard_normal((r, c)) * np.sqrt(2.0 / r)
        self.W1 = _w(hidden, n_in);   self.b1 = np.zeros(hidden)
        self.W2 = _w(hidden, hidden); self.b2 = np.zeros(hidden)
        self.W3 = _w(1, hidden);      self.b3 = np.zeros(1)
        # Adam state
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]
        self._t = 0

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (N, 3) → (N,)"""
        h1 = np.tanh(X @ self.W1.T + self.b1)
        h2 = np.tanh(h1 @ self.W2.T + self.b2)
        return (h2 @ self.W3.T + self.b3).squeeze(-1)

    def _forward_with_cache(self, X):
        z1 = X @ self.W1.T + self.b1;  h1 = np.tanh(z1)
        z2 = h1 @ self.W2.T + self.b2; h2 = np.tanh(z2)
        y  = (h2 @ self.W3.T + self.b3).squeeze(-1)
        return y, h1, h2, z1, z2

    def loss_and_grad(self, X: np.ndarray, y_true: np.ndarray):
        """MSE loss + gradients via backprop."""
        N   = X.shape[0]
        y, h1, h2, z1, z2 = self._forward_with_cache(X)
        err = y - y_true                       # (N,)
        L   = float(np.mean(err ** 2))

        # Output layer
        dL_dy  = 2.0 * err / N                 # (N,)
        dW3    = dL_dy[:, None] * h2           # (N, hidden)
        db3    = dL_dy
        dh2    = dL_dy[:, None] * self.W3      # (N, hidden)

        # Hidden 2
        dz2    = dh2 * (1.0 - h2 ** 2)
        dW2    = dz2.T @ h1
        db2    = dz2.sum(0)
        dh1    = dz2 @ self.W2

        # Hidden 1
        dz1    = dh1 * (1.0 - h1 ** 2)
        dW1    = dz1.T @ X
        db1    = dz1.sum(0)

        grads = [dW1, db1, dW2, db2, dW3.mean(0, keepdims=True), db3]
        return L, grads

    def adam_step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        t = self._t
        params = self._params()
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            m_hat = self._m[i] / (1 - beta1 ** t)
            v_hat = self._v[i] / (1 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 600, lr: float = 3e-3,
              verbose: bool = True) -> list:
        losses = []
        for ep in range(epochs):
            L, grads = self.loss_and_grad(X, y)
            self.adam_step(grads, lr=lr)
            losses.append(L)
            if verbose and (ep % 100 == 0 or ep == epochs - 1):
                print(f"    epoch {ep:4d}  MSE={L:.6f}")
        return losses

    def predict_scalar_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Forward + gradient of output w.r.t. input x (shape: (3,)).
        Used for gradient-descent optimisation of gains.
        """
        X  = x[None, :]   # (1, 3)
        y, h1, h2, z1, z2 = self._forward_with_cache(X)

        # Backprop to input
        dL_dy = np.ones(1)
        dh2   = dL_dy[:, None] * self.W3         # (1, hidden)
        dz2   = dh2 * (1.0 - h2 ** 2)
        dh1   = dz2 @ self.W2
        dz1   = dh1 * (1.0 - h1 ** 2)
        dx    = (dz1 @ self.W1).squeeze(0)       # (3,)
        return float(y.squeeze()), dx


# =============================================================================
# Surrogate optimisation  (gradient descent on MLP input)
# =============================================================================

def _optimise_surrogate(mlp: MLP,
                        X_norm: np.ndarray,
                        y: np.ndarray,
                        bounds_norm: np.ndarray,
                        n_restarts: int,
                        n_steps: int,
                        lr: float,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Find x ∈ [0,1]^3 that minimises mlp.forward(x).
    Returns best normalised gain vector found.
    """
    best_x    = X_norm[np.argmin(y)].copy()
    best_cost = float(mlp.forward(best_x[None, :])[0])

    for restart in range(n_restarts):
        # Start from best observed or random
        if restart == 0:
            x = best_x.copy()
        else:
            x = rng.uniform(0.0, 1.0, size=3)

        # Adam state for input optimisation
        m = np.zeros(3); v = np.zeros(3); t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for _ in range(n_steps):
            cost, grad = mlp.predict_scalar_grad(x)
            t += 1
            m  = beta1 * m + (1 - beta1) * grad
            v  = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
            x = np.clip(x, 0.0, 1.0)   # stay in bounds

        cost_final = float(mlp.forward(x[None, :])[0])
        if cost_final < best_cost:
            best_cost = cost_final
            best_x    = x.copy()

    return best_x


# =============================================================================
# Main tuner
# =============================================================================

def run_dfc_tuner() -> tuple[float, float, float]:
    """
    Run the full NN surrogate tuner.

    Returns
    -------
    (Kp_speed_best, Kp_id_best, Kp_iq_best)
    """
    rng = np.random.default_rng(seed=42)

    print("\n" + "=" * 70)
    print("  DFC Gain Tuner  —  NN Surrogate + Gradient Optimisation")
    print("=" * 70)
    print(f"  Phase 1: {N_EXPLORE} LHS exploration simulations")
    print(f"  Phase 2: MLP training ({N_EPOCHS} epochs)")
    print(f"  Phase 3: {N_RESTARTS} gradient-descent restarts on surrogate")
    print(f"  Phase 4: verification simulation")
    print()
    print(f"  Bounds:")
    for name, (lo, hi) in zip(PARAM_NAMES, BOUNDS):
        print(f"    {name}  [{lo:.3f}, {hi:.3f}]")
    print(f"\n  Cost: {W_SS}*ss_err_RPM + {W_ID}*id_rms_A + {W_CHAT}*iq_chat_A")
    print("=" * 70)

    # ── Phase 1: LHS exploration ──────────────────────────────────────────────
    print("\n[Phase 1] LHS exploration ...")
    X_raw   = _lhs(N_EXPLORE, BOUNDS, rng)   # (N_EXPLORE, 3) physical gains
    costs   = []
    results = []

    t0 = time.perf_counter()
    for i, gains in enumerate(X_raw):
        kp_s, kp_d, kp_q = gains
        print(f"  [{i+1:2d}/{N_EXPLORE}]  Kp_speed={kp_s:.4f}  "
              f"Kp_id={kp_d:.3f}  Kp_iq={kp_q:.3f}", end="  ", flush=True)
        try:
            met = _run_with_gains(kp_s, kp_d, kp_q)
        except KeyboardInterrupt:
            print("\n  Interrupted — using data collected so far.")
            X_raw = X_raw[:i]
            break

        if met is None:
            print("→ UNSTABLE")
            costs.append(1e6)
        else:
            print(f"→ cost={met['cost']:.1f}  "
                  f"ss={met['ss_err']:.0f}RPM  "
                  f"id={met['id_rms']:.3f}A  "
                  f"chat={met['iq_chat']:.3f}A")
            costs.append(met["cost"])
            results.append((gains, met))

    elapsed = time.perf_counter() - t0
    print(f"\n  Exploration done  ({elapsed:.0f}s)")

    costs_arr = np.array(costs, dtype=np.float64)
    valid     = costs_arr < 1e5
    if valid.sum() < 4:
        print("  ERROR: fewer than 4 valid simulations — widen bounds or check model.")
        return 0.119, 2.0, 2.0

    X_valid = X_raw[valid]
    y_valid = costs_arr[valid]

    # Best observed so far
    best_idx     = int(np.argmin(y_valid))
    best_obs     = X_valid[best_idx]
    best_cost_obs = y_valid[best_idx]
    print(f"\n  Best observed:  cost={best_cost_obs:.2f}")
    for name, val in zip(PARAM_NAMES, best_obs):
        print(f"    {name} = {val:.4f}")

    # ── Phase 2: NN surrogate training ────────────────────────────────────────
    print("\n[Phase 2] Training MLP surrogate ...")

    # Normalise inputs to [0,1] and outputs to zero-mean unit-variance
    X_norm = (X_valid - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    y_mean = y_valid.mean(); y_std = max(y_valid.std(), 1e-8)
    y_norm = (y_valid - y_mean) / y_std

    mlp = MLP(n_in=3, hidden=32, seed=0)
    mlp.train(X_norm, y_norm, epochs=N_EPOCHS, lr=LR_SURR, verbose=True)

    # Surrogate quality check on training set
    y_pred = mlp.forward(X_norm)
    r2     = 1.0 - np.var(y_norm - y_pred) / np.var(y_norm)
    print(f"\n  Surrogate R² on training set: {r2:.4f}  (>0.85 is good)")

    # ── Phase 3: surrogate optimisation ──────────────────────────────────────
    print(f"\n[Phase 3] Gradient descent on surrogate ({N_RESTARTS} restarts) ...")
    best_norm = _optimise_surrogate(
        mlp, X_norm, y_norm,
        bounds_norm=np.array([[0, 1], [0, 1], [0, 1]], dtype=np.float64),
        n_restarts=N_RESTARTS,
        n_steps=N_OPT_STEPS,
        lr=LR_OPT,
        rng=rng)

    # Denormalise
    gains_pred = BOUNDS[:, 0] + best_norm * (BOUNDS[:, 1] - BOUNDS[:, 0])
    kp_s_pred, kp_d_pred, kp_q_pred = gains_pred
    cost_surr  = float(mlp.forward(best_norm[None, :])[0]) * y_std + y_mean
    print(f"  Surrogate minimum:  predicted cost = {cost_surr:.2f}")
    for name, val in zip(PARAM_NAMES, gains_pred):
        print(f"    {name} = {val:.4f}")

    # ── Phase 4: verification ─────────────────────────────────────────────────
    print("\n[Phase 4] Verification simulation with NN-recommended gains ...")
    met_verify = _run_with_gains(kp_s_pred, kp_d_pred, kp_q_pred)
    if met_verify is None:
        print("  Verification UNSTABLE — returning best observed gains.")
        return float(best_obs[0]), float(best_obs[1]), float(best_obs[2])

    cost_verify = met_verify["cost"]

    # Use the better of verified NN gain and best observed
    if cost_verify < best_cost_obs:
        final_gains = gains_pred
        print(f"  NN gains BETTER than best observed:  "
              f"cost={cost_verify:.2f} < {best_cost_obs:.2f}")
    else:
        final_gains = best_obs
        print(f"  Best observed still better:  "
              f"cost={best_cost_obs:.2f} < {cost_verify:.2f}")
        print(f"  (NN surrogate may need more exploration data — increase N_EXPLORE)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TUNING COMPLETE")
    print("=" * 70)
    print(f"\n  {'Parameter':<14} {'Before':>10}  {'After':>10}  {'Delta':>8}")
    print(f"  {'-'*46}")
    defaults = [0.119, 2.0, 2.0]
    for name, default, tuned in zip(PARAM_NAMES, defaults, final_gains):
        delta = (tuned - default) / (abs(default) + 1e-12) * 100.0
        sign  = "UP" if delta > 0.0 else "DN"
        print(f"  {name}  {default:>10.4f}  {tuned:>10.4f}  "
              f"{sign} {abs(delta):5.1f}%")

    best_met = met_verify if cost_verify < best_cost_obs else results[best_idx][1]
    print(f"\n  Best cost        : {min(cost_verify, best_cost_obs):.2f}")
    print(f"  SS speed error   : {best_met['ss_err']:.1f} RPM")
    print(f"  id RMS  (MTPA)   : {best_met['id_rms']:.4f} A")
    print(f"  iq chattering    : {best_met['iq_chat']:.4f} A")
    print("=" * 70)

    return float(final_gains[0]), float(final_gains[1]), float(final_gains[2])


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    kp_speed, kp_id, kp_iq = run_dfc_tuner()

    print(f"\n[Done] Use these gains in db42s02_closed_loop_dfc_20k.py:")
    print(f"    Kp_speed = {kp_speed:.4f},")
    print(f"    Kp_id    = {kp_id:.4f},")
    print(f"    Kp_iq    = {kp_iq:.4f},")
