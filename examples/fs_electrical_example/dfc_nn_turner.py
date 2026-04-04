"""
dfc_nn_tuner.py
===============
Neural-network surrogate tuner for DFControllerBlock gains.
Uses the working simulation from db42s02_closed_loop_dfc_20k.py
"""

from __future__ import annotations

import sys
import math
import time
import json
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import contextlib
import io

# ── path setup ──────────────────────────────────────────────────────────────
from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE = get_current_parent()
_ROOT = get_project_root()
_FS_ELEC = _ROOT / "fs_electrical_machines"

for _p in (get_embedsim_import_path(), str(_FS_ELEC), str(_FS_ELEC / "c_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the working simulation
import db42s02_closed_loop_dfc_20k as sim_mod
from diff_flatness_controller_block import DFControllerBlock
from smc_controller_block import _DB42S02


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TunerConfig:
    """Tuner hyperparameters"""
    n_explore: int = 10
    n_verification: int = 2
    n_epochs: int = 100
    hidden_size: int = 32
    lr_train: float = 3e-3
    early_stopping_patience: int = 50
    n_restarts: int = 5
    n_opt_steps: int = 200
    lr_opt: float = 5e-2

    # Cost weights (from working simulation)
    w_ss: float = 2.0
    w_id: float = 50.0
    w_chat: float = 4.0

    target_rpm: float = 2000.0
    t_sim: float = 5.0

    # Tuning bounds
    bounds: np.ndarray = None

    def __post_init__(self):
        if self.bounds is None:
            self.bounds = np.array([
                [0.02,  0.40],   # Kp_speed
                [0.50,  8.00],   # Kp_id
                [0.50,  8.00],   # Kp_iq
            ], dtype=np.float64)

    @property
    def param_names(self) -> List[str]:
        return ["Kp_speed", "Kp_id", "Kp_iq"]


@dataclass
class TuningResult:
    gains: np.ndarray
    cost: float
    ss_error: float
    id_rms: float
    iq_chatter: float
    convergence: bool
    n_simulations: int
    timestamp: str


# =============================================================================
# Simulation Runner - Uses the working simulation
# =============================================================================

class DFCSimulationRunner:
    """Run DFC simulation with custom gains"""

    def __init__(self, config: TunerConfig):
        self.config = config

    def run(self, kp_speed: float, kp_id: float, kp_iq: float,
            verbose: bool = False) -> Optional[Dict[str, Any]]:
        """Run simulation with given gains"""

        # Suppress all output during simulation
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                return self._run_simulation(kp_speed, kp_id, kp_iq, verbose)
            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                return None

    def _run_simulation(self, kp_speed: float, kp_id: float, kp_iq: float,
                        verbose: bool) -> Dict[str, Any]:
        """Internal simulation runner"""

        from embedsim import EmbedSim, ODESolver, VectorEnd
        from embedsim.core_blocks import VectorSignal, DEFAULT_DTYPE
        from embedsim.source_blocks import VectorStep, VectorConstant
        from embedsim.simulation_engine import VectorDelay
        from embedsim.code_generator import CodeGenStart, CodeGenEnd
        from motor_utility_blocks import SVPWMPackBlock
        from svpwm_block import SVPWMBlock
        from ctrl_packer import CtrlPacker
        from machine_feedback import db42s02_feedback_profile

        V_DC = sim_mod.V_DC
        DT = sim_mod.DT
        T_SIM = sim_mod.T_SIM
        T_RAMP = sim_mod._RAMP_TIME
        TRADS = sim_mod.TARGET_RADS_MECH
        MSIZ = sim_mod._MOTOR_OUT_SIZE

        cg_start = CodeGenStart("cg_start")

        # Create DFC with custom gains
        dfc = DFControllerBlock(
            "dfc",
            P_POLES=int(_DB42S02.SMC_P_POLES),
            R_S=_DB42S02.SMC_R_S,
            L_D=_DB42S02.SMC_L_D,
            L_Q=_DB42S02.SMC_L_Q,
            LAMBDA_PM=_DB42S02.SMC_LAMBDA_PM,
            V_DC=V_DC,
            I_MAX=_DB42S02.SMC_I_MAX,
            dt_s=DT,
            Kp_id=kp_id,
            Kp_iq=kp_iq,
            Kp_speed=kp_speed,
            smo_k=_DB42S02.SMC_SMO_K,
            smo_tau=1.0 / (2.0 * math.pi * _DB42S02.SMC_SMO_FC),
            fusion_omega_lo=50.0,
            fusion_omega_hi=250.0,
            fusion_gamma=2.0,
            fusion_iir_lo=0.05,
            fusion_iir_hi=0.30,
        )

        svpwm_pack = SVPWMPackBlock("svpwm_pack", v_dc=V_DC)
        svpwm = SVPWMBlock("svpwm", use_c_backend=False)
        cg_end = CodeGenEnd("cg_end")
        speed_ref = VectorStep("speed_ref", step_time=0.0,
                              before_value=TRADS, after_value=TRADS)
        load_torque = VectorConstant("load_torque", value=sim_mod.T_LOAD_ZERO)
        motor = sim_mod.DB42S02PlantBlock("motor")
        motor_delay = VectorDelay("motor_delay", initial=[0.0] * MSIZ)
        ctrl = CtrlPacker("ctrl_packer",
                         target_rads_mech=TRADS, ramp_time=T_RAMP,
                         feedback=db42s02_feedback_profile(
                             enc_glitch=False, adc_noise=False, adc_sat=False))
        sink = VectorEnd("sink")
        sink_cg = VectorEnd("sink_cg")

        cg_start >> dfc >> svpwm_pack >> svpwm >> cg_end
        motor >> motor_delay >> ctrl
        speed_ref >> ctrl
        ctrl >> cg_start
        cg_end >> motor
        load_torque >> motor
        motor >> sink
        cg_end >> sink_cg

        # Run simulation (without verbose parameter)
        sim = EmbedSim(sinks=[sink, sink_cg], T=T_SIM, dt=DT, solver=ODESolver.EULER)

        # Add signals to scope
        sim.scope.add(motor, indices=[0, 6, 7], label="Motor")

        sim.run()

        # Extract results
        sc = sim.scope
        t = np.array(sc.t, dtype=np.float32)
        ld = dfc.log_data

        def get_motor(idx):
            s = sc.get_signal("Motor", idx)
            return s if s is not None else np.zeros(len(t), np.float32)

        def get_log(key):
            if len(ld["t"]) > 1:
                return np.interp(t, ld["t"], ld[key]).astype(np.float32)
            return np.zeros(len(t), np.float32)

        return {
            "t": t,
            "speed_rpm": get_motor(0),
            "id": get_motor(6),
            "iq": get_motor(7),
            "iq_ref": get_log("iq_ref"),
            "omega_ref_rpm": get_log("speed_ref"),
        }


# =============================================================================
# Cost Function
# =============================================================================

class CostFunction:
    def __init__(self, config: TunerConfig):
        self.config = config

    def compute(self, results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        if results is None:
            return None

        t = results["t"]
        rpm = results["speed_rpm"]
        id_meas = results["id"]
        iq_ref = results["iq_ref"]

        if len(t) < 200:
            return None

        # Check for divergence
        if float(np.max(np.abs(rpm))) > self.config.target_rpm * 3.0:
            return None

        # Steady-state mask (last 15%)
        ss_mask = t > 0.85 * self.config.t_sim
        if not np.any(ss_mask):
            return None

        # Reference speed
        ref_speed = self.config.target_rpm

        # Compute metrics
        ss_error = float(np.mean(np.abs(rpm[ss_mask] - ref_speed)))
        id_rms = float(np.sqrt(np.mean(id_meas[ss_mask] ** 2)))
        iq_chatter = float(np.std(iq_ref[ss_mask]))

        # Hard failure
        if ss_error > 800.0:
            return None

        # Weighted cost
        cost = (self.config.w_ss * ss_error +
                self.config.w_id * id_rms +
                self.config.w_chat * iq_chatter)

        return {
            "cost": cost,
            "ss_error": ss_error,
            "id_rms": id_rms,
            "iq_chatter": iq_chatter
        }


# =============================================================================
# Latin Hypercube Sampling
# =============================================================================

class LatinHypercubeSampler:
    def __init__(self, bounds: np.ndarray, rng: np.random.Generator = None):
        self.bounds = bounds
        self.rng = rng or np.random.default_rng()
        self.n_dims = bounds.shape[0]

    def sample(self, n_points: int) -> np.ndarray:
        samples = np.zeros((n_points, self.n_dims))
        for j in range(self.n_dims):
            cuts = np.linspace(0.0, 1.0, n_points + 1)
            u = self.rng.uniform(cuts[:-1], cuts[1:])
            self.rng.shuffle(u)
            lo, hi = self.bounds[j, 0], self.bounds[j, 1]
            samples[:, j] = lo + u * (hi - lo)
        return samples


# =============================================================================
# MLP Surrogate Model
# =============================================================================

class MLPSurrogate:
    def __init__(self, n_inputs: int = 3, hidden_size: int = 32, seed: int = 0):
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)

        def xavier_init(rows, cols):
            return rng.standard_normal((rows, cols)) * np.sqrt(2.0 / (rows + cols))

        self.W1 = xavier_init(hidden_size, n_inputs)
        self.b1 = np.zeros(hidden_size)
        self.W2 = xavier_init(hidden_size, hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W3 = xavier_init(1, hidden_size)
        self.b3 = np.zeros(1)

        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]
        self._t = 0

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, X: np.ndarray) -> np.ndarray:
        h1 = np.tanh(X @ self.W1.T + self.b1)
        h2 = np.tanh(h1 @ self.W2.T + self.b2)
        return (h2 @ self.W3.T + self.b3).squeeze(-1)

    def train(self, X, y, epochs=100, lr=3e-3, verbose=True):
        for epoch in range(epochs):
            # Forward
            h1 = np.tanh(X @ self.W1.T + self.b1)
            h2 = np.tanh(h1 @ self.W2.T + self.b2)
            y_pred = (h2 @ self.W3.T + self.b3).squeeze(-1)

            # Loss
            loss = np.mean((y_pred - y) ** 2)

            # Backward
            grad_y = 2 * (y_pred - y) / len(X)

            grad_W3 = (grad_y[:, None] * h2).mean(axis=0, keepdims=True)
            grad_b3 = grad_y.mean(axis=0)
            grad_h2 = grad_y[:, None] * self.W3

            grad_z2 = grad_h2 * (1 - h2 ** 2)
            grad_W2 = grad_z2.T @ h1 / len(X)
            grad_b2 = grad_z2.mean(axis=0)
            grad_h1 = grad_z2 @ self.W2

            grad_z1 = grad_h1 * (1 - h1 ** 2)
            grad_W1 = grad_z1.T @ X / len(X)
            grad_b1 = grad_z1.mean(axis=0)

            # Adam update
            self._t += 1
            for param, grad, m, v in zip(self._params(),
                                        [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3],
                                        self._m, self._v):
                m[:] = 0.9 * m + 0.1 * grad
                v[:] = 0.999 * v + 0.001 * (grad ** 2)
                m_hat = m / (1 - 0.9 ** self._t)
                v_hat = v / (1 - 0.999 ** self._t)
                param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print(f"    epoch {epoch:4d}  loss={loss:.6f}")


# =============================================================================
# Main Tuner
# =============================================================================

class DFCGainTuner:
    def __init__(self, config: TunerConfig = None, output_dir: Path = None):
        self.config = config or TunerConfig()
        self.output_dir = output_dir or _HERE / "tuning_results"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.runner = DFCSimulationRunner(self.config)
        self.cost_fn = CostFunction(self.config)
        self.sampler = LatinHypercubeSampler(self.config.bounds)
        self.rng = np.random.default_rng(42)

        self.exploration_results = []
        self.best_result = None

    def run(self) -> TuningResult:
        print("\n" + "=" * 70)
        print("  DFC Gain Tuner")
        print("=" * 70)
        self._print_config()

        # First, test default gains to ensure simulation works
        print("\n[Test] Testing default gains...")
        default_results = self.runner.run(0.119, 2.0, 2.0, verbose=False)
        default_metrics = self.cost_fn.compute(default_results)

        if default_metrics is None:
            print("  ERROR: Default gains simulation failed!")
            print("  Please ensure db42s02_closed_loop_dfc_20k.py runs successfully first.")
            sys.exit(1)

        print(f"  ✓ Default gains work!")
        print(f"    cost={default_metrics['cost']:.2f}, "
              f"ss_error={default_metrics['ss_error']:.1f}RPM, "
              f"id_rms={default_metrics['id_rms']:.3f}A")

        self.best_result = TuningResult(
            gains=np.array([0.119, 2.0, 2.0]),
            cost=default_metrics['cost'],
            ss_error=default_metrics['ss_error'],
            id_rms=default_metrics['id_rms'],
            iq_chatter=default_metrics['iq_chatter'],
            convergence=False,
            n_simulations=1,
            timestamp=datetime.now().isoformat()
        )

        # Phase 1: Exploration
        self._phase1_exploration()

        # Phase 2: Train surrogate (if we have enough data)
        if len(self.exploration_results) >= 4:
            self._phase2_train_surrogate()
            best_norm = self._phase3_optimize_surrogate()
            self._phase4_verify(best_norm)
        else:
            print(f"\n[Note] Only {len(self.exploration_results)} valid simulations found.")
            print("  Using best observed gains from exploration.")

        self._save_results()
        return self.best_result

    def _print_config(self):
        print(f"\n  Configuration:")
        print(f"    Exploration: {self.config.n_explore} simulations")
        print(f"    NN: {self.config.hidden_size} hidden units, {self.config.n_epochs} epochs")
        print(f"    Bounds:")
        for name, (lo, hi) in zip(self.config.param_names, self.config.bounds):
            print(f"      {name}: [{lo:.3f}, {hi:.3f}]")

    def _phase1_exploration(self):
        print(f"\n[Phase 1] Exploring {self.config.n_explore} gain combinations...")

        X_raw = self.sampler.sample(self.config.n_explore)

        for i, gains in enumerate(X_raw):
            kp_s, kp_d, kp_q = gains
            print(f"  [{i+1:2d}/{self.config.n_explore}]  "
                  f"Kp_speed={kp_s:.4f}  Kp_id={kp_d:.3f}  Kp_iq={kp_q:.3f}",
                  end="  ", flush=True)

            results = self.runner.run(kp_s, kp_d, kp_q, verbose=False)
            metrics = self.cost_fn.compute(results)

            if metrics is None:
                print("→ UNSTABLE")
            else:
                print(f"→ cost={metrics['cost']:.2f}  "
                      f"ss={metrics['ss_error']:.0f}RPM  "
                      f"id={metrics['id_rms']:.3f}A")
                self.exploration_results.append((gains.copy(), metrics))

                if metrics['cost'] < self.best_result.cost:
                    self.best_result = TuningResult(
                        gains=gains.copy(),
                        cost=metrics['cost'],
                        ss_error=metrics['ss_error'],
                        id_rms=metrics['id_rms'],
                        iq_chatter=metrics['iq_chatter'],
                        convergence=False,
                        n_simulations=self.best_result.n_simulations + 1,
                        timestamp=datetime.now().isoformat()
                    )
                    print(f"    ★ NEW BEST! cost={metrics['cost']:.2f}")

        print(f"\n  Exploration complete.")
        print(f"  Valid simulations: {len(self.exploration_results)}/{self.config.n_explore}")
        print(f"  Best cost: {self.best_result.cost:.2f}")
        print(f"  Best gains: Kp_speed={self.best_result.gains[0]:.4f}, "
              f"Kp_id={self.best_result.gains[1]:.3f}, Kp_iq={self.best_result.gains[2]:.3f}")

    def _phase2_train_surrogate(self):
        print("\n[Phase 2] Training surrogate model...")

        # Prepare data
        X = np.array([g for g, _ in self.exploration_results])
        y = np.array([m['cost'] for _, m in self.exploration_results])

        # Normalize
        bounds = self.config.bounds
        X_norm = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        y_mean, y_std = y.mean(), y.std()
        y_norm = (y - y_mean) / (y_std + 1e-8)

        # Train
        self.surrogate = MLPSurrogate(n_inputs=3, hidden_size=self.config.hidden_size)
        self.surrogate.train(X_norm, y_norm, epochs=self.config.n_epochs, verbose=True)

        # Save surrogate data
        self.X_norm = X_norm
        self.y_mean = y_mean
        self.y_std = y_std

    def _phase3_optimize_surrogate(self) -> np.ndarray:
        print(f"\n[Phase 3] Optimizing surrogate...")

        # Start from best gains
        best_norm = (self.best_result.gains - self.config.bounds[:, 0]) / (self.config.bounds[:, 1] - self.config.bounds[:, 0])

        # Local search with random restarts
        for restart in range(self.config.n_restarts):
            if restart == 0:
                x = best_norm.copy()
            else:
                x = best_norm + self.rng.normal(0, 0.1, size=3)
                x = np.clip(x, 0, 1)

            # Gradient descent
            for step in range(self.config.n_opt_steps):
                # Finite difference gradient
                eps = 0.01
                grad = np.zeros(3)
                for j in range(3):
                    x_plus = x.copy()
                    x_plus[j] = min(1, x[j] + eps)
                    x_minus = x.copy()
                    x_minus[j] = max(0, x[j] - eps)
                    y_plus = self.surrogate.forward(x_plus[None, :])[0]
                    y_minus = self.surrogate.forward(x_minus[None, :])[0]
                    grad[j] = (y_plus - y_minus) / (2 * eps)

                # Update
                x = x - self.config.lr_opt * grad
                x = np.clip(x, 0, 1)

            # Evaluate
            cost_pred = self.surrogate.forward(x[None, :])[0] * self.y_std + self.y_mean
            if cost_pred < self.best_result.cost:
                best_norm = x

        return best_norm

    def _phase4_verify(self, best_norm: np.ndarray):
        print(f"\n[Phase 4] Verifying best gains...")

        gains = self.config.bounds[:, 0] + best_norm * (self.config.bounds[:, 1] - self.config.bounds[:, 0])

        for i in range(self.config.n_verification):
            results = self.runner.run(gains[0], gains[1], gains[2], verbose=False)
            metrics = self.cost_fn.compute(results)

            if metrics and metrics['cost'] < self.best_result.cost:
                self.best_result = TuningResult(
                    gains=gains.copy(),
                    cost=metrics['cost'],
                    ss_error=metrics['ss_error'],
                    id_rms=metrics['id_rms'],
                    iq_chatter=metrics['iq_chatter'],
                    convergence=True,
                    n_simulations=self.best_result.n_simulations + 1,
                    timestamp=datetime.now().isoformat()
                )
                print(f"  ✓ Verified: cost={metrics['cost']:.2f} (improved!)")
                return

        print(f"  Verification complete. Best cost remains: {self.best_result.cost:.2f}")

    def _save_results(self):
        results_path = self.output_dir / "tuning_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'best_gains': self.best_result.gains.tolist(),
                'best_cost': self.best_result.cost,
                'ss_error': self.best_result.ss_error,
                'id_rms': self.best_result.id_rms,
                'iq_chatter': self.best_result.iq_chatter,
                'n_simulations': self.best_result.n_simulations,
                'timestamp': self.best_result.timestamp
            }, f, indent=2)
        print(f"\n  Results saved to {results_path}")

    def print_summary(self):
        print("\n" + "=" * 70)
        print("  TUNING COMPLETE")
        print("=" * 70)

        defaults = [0.119, 2.0, 2.0]
        print(f"\n  {'Parameter':<14} {'Default':>10}  {'Tuned':>10}  {'Change':>12}")
        print(f"  {'-'*50}")

        for name, default, tuned in zip(self.config.param_names, defaults, self.best_result.gains):
            delta = (tuned - default) / (abs(default) + 1e-12) * 100
            print(f"  {name:<14} {default:>10.4f}  {tuned:>10.4f}  {delta:>+11.1f}%")

        print(f"\n  Best cost        : {self.best_result.cost:.2f}")
        print(f"  SS speed error   : {self.best_result.ss_error:.1f} RPM")
        print(f"  id RMS           : {self.best_result.id_rms:.4f} A")
        print(f"  iq chattering    : {self.best_result.iq_chatter:.4f} A")
        print("=" * 70)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DFC Gain Tuner")
    parser.add_argument("--explore", type=int, default=10, help="Exploration simulations")
    parser.add_argument("--epochs", type=int, default=100, help="NN training epochs")
    parser.add_argument("--quick", action="store_true", help="Quick test")

    args = parser.parse_args()

    if args.quick:
        config = TunerConfig(n_explore=5, n_epochs=50, n_restarts=3)
    else:
        config = TunerConfig(n_explore=args.explore, n_epochs=args.epochs)

    tuner = DFCGainTuner(config)
    result = tuner.run()
    tuner.print_summary()

    print(f"\n[Done] Use these gains in db42s02_closed_loop_dfc_20k.py:")
    print(f"    Kp_speed = {result.gains[0]:.6f},")
    print(f"    Kp_id    = {result.gains[1]:.6f},")
    print(f"    Kp_iq    = {result.gains[2]:.6f},")