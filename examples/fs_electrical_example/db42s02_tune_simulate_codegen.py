"""
db42s02_tune_simulate_codegen.py
=================================
NANOTEC DB42S02 — Complete workflow in three steps:

  Step 1 — TUNE
    DE + GP Bayesian optimiser (smc_fmu_tuner.py) finds the best SMC gains
    for the target operating point using PMSM_Python_Plant (RK4, no FMU).
    Saves  smc_best_gains.json  and patches  smc_gains_config.h.

  Step 2 — SIMULATE
    Full EmbedSim closed-loop at 20 kHz with the tuned gains.
    Uses the same PMSM_Python_Plant + coordinate_transform_blocks pipeline.
    Produces  db42s02_smc_foc_results.png.

  Step 3 — CODEGEN
    StepGenerator emits  embedsim_gen/embedsim_step.c/.h  for the AURIX TC3xx.
    The generated ISR calls SMC_Controller_Step() → SVPWM in a single flat
    function with no dynamic allocation.

Usage
-----
  python db42s02_tune_simulate_codegen.py            # full workflow
  python db42s02_tune_simulate_codegen.py --no-tune  # skip tuning, use defaults
  python db42s02_tune_simulate_codegen.py --rpm 1000 --t_sim 1.5

Paul Abraham / EmbedSim 2025
"""

from __future__ import annotations

import sys, os, math, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE   = get_current_parent()
_ROOT   = get_project_root()
_FS_ELEC = _ROOT / "fs_electrical_machines"

for _p in (get_embedsim_import_path(),
           str(_FS_ELEC),
           str(_FS_ELEC / "c_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── EmbedSim imports ──────────────────────────────────────────────────────────
from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep, VectorConstant
from embedsim.simulation_engine import VectorDelay
from embedsim.code_generator import CodeGenStart, CodeGenEnd

from motor_utility_blocks import SVPWMPackBlock
from svpwm_block import SVPWMBlock
from pmsm_python_plant import PMSM_Python_Plant
from smc_controller_block import SMCControllerBlock, _DB42S02
from smc_fmu_tuner import SMCGains, SMCTuner, patch_header

# ── Constants ─────────────────────────────────────────────────────────────────
V_DC            = _DB42S02.SMC_V_DC        # 17.0 V
_MOTOR_OUT_SIZE = 8    # [rpm, ia, ib, ic, theta_m, T_em, id, iq]

# Load schedule — kept for reference but not applied at 400 RPM no-load.
# Re-enable T_LOAD_LIGHT / T_LOAD_HEAVY when testing at higher speeds
# or with an external load on the AURIX kit.
T_LOAD_ZERO  = 0.000   # N·m — active at 400 RPM smoke test
T_LOAD_LIGHT = 0.005   # N·m  5 mN·m  (reserved)
T_LOAD_HEAVY = 0.020   # N·m  20 mN·m (reserved)


# =============================================================================
#  Plant block
# =============================================================================
class DB42S02PlantBlock(PMSM_Python_Plant):
    """DB42S02 plant — wraps PMSM_Python_Plant with constant no-load torque.

    Load schedule removed: the AURIX motor kit runs at 400 RPM no-load.
    T_load = 0 throughout the simulation so the tuner and verification
    sim are consistent with the real hardware condition.
    """

    TOPO_CATEGORY     = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[rpm,ia,ib,ic,theta_m,Tem,id,iq]"
    _P = _DB42S02.SMC_P_POLES

    def __init__(self, name: str, **kwargs):
        super().__init__(
            name      = name,
            R         = _DB42S02.SMC_R_S,
            L_d       = _DB42S02.SMC_L_D,
            L_q       = _DB42S02.SMC_L_Q,
            lambda_pm = _DB42S02.SMC_LAMBDA_PM,
            J         = _DB42S02.SMC_J_ROTOR,
            B_fric    = _DB42S02.SMC_B_FRICTION,
            p         = float(_DB42S02.SMC_P_POLES),
            v_dc      = V_DC,
        )

    def compute_py(self, t, dt, input_values=None):
        ta = tb = tc = 0.5
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                ta, tb, tc = float(v[0]), float(v[1]), float(v[2])

        augmented = [VectorSignal(
            np.array([ta, tb, tc, V_DC, T_LOAD_ZERO], dtype=DEFAULT_DTYPE))]
        return super().compute_py(t, dt, augmented)

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
#  CtrlPacker
# =============================================================================
class CtrlPacker(VectorBlock):
    """Packs motor feedback bus into SMC input format."""
    INPUT_NAMES       = ["omega_ref_mech", "theta_m", "ia", "ib", "ic"]
    INPUT_KEEP        = [0, 1, 2, 3, 4]
    C_CODEGEN_EXCLUDE = True

    _RAMP_RATE = (_DB42S02.SMC_P_POLES * 2 * math.pi * 2000 / 60) / 0.5  # reach 2000 RPM in 0.5s

    def __init__(self, name="ctrl_packer", **kw):
        super().__init__(name, **kw)
        self.output_label    = "[omega_ref,theta_m,ia,ib,ic]"
        self._omega_ref_filt = 0.0

    def reset(self):
        super().reset()
        self._omega_ref_filt = 0.0

    def compute_py(self, t, dt, input_values=None):
        m = (input_values[0].value if input_values and len(input_values) > 0
             else np.zeros(_MOTOR_OUT_SIZE, dtype=DEFAULT_DTYPE))
        r = (input_values[1].value if input_values and len(input_values) > 1
             else np.zeros(1, dtype=DEFAULT_DTYPE))

        omega_target = float(r[0]) if len(r) > 0 else 0.0
        max_step = self._RAMP_RATE * dt
        self._omega_ref_filt += max(-max_step,
                                    min(max_step,
                                        omega_target - self._omega_ref_filt))

        self.output = VectorSignal(np.array([
            self._omega_ref_filt,
            float(m[4]) if len(m) > 4 else 0.0,   # theta_m
            float(m[1]) if len(m) > 1 else 0.0,   # ia
            float(m[2]) if len(m) > 2 else 0.0,   # ib
            float(m[3]) if len(m) > 3 else 0.0,   # ic
        ], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
#  Step 1 — TUNE
# =============================================================================
def step1_tune(target_rpm: float,
               t_sim: float,
               dt: float,
               de_iters: int,
               gp_iters: int,
               out_json: str,
               header_path: str) -> SMCGains:
    """
    Run DE + GP Bayesian optimiser to find best SMC gains.
    Returns the best SMCGains found.
    """
    print("\n" + "=" * 70)
    print("  STEP 1 — TUNE  (DE + GP Bayesian optimisation)")
    print(f"  Target: {target_rpm:.0f} RPM   t_sim={t_sim}s   dt={dt*1e6:.0f}µs")
    print(f"  DE iters={de_iters}   GP iters={gp_iters}")
    print("=" * 70)

    tuner = SMCTuner(
        omega_cmd_rpm = target_rpm,
        t_sim         = t_sim,
        dt            = dt,
        de_iters      = de_iters,
        gp_iters      = gp_iters,
        workers       = 1,
        out_file      = out_json,
    )
    best = tuner.run()

    print(f"\n  Best gains found:")
    print(f"    SMC_KS_W  = {best.SMC_KS_W:.6f}  N·m")
    print(f"    SMC_ETA_W = {best.SMC_ETA_W:.6f}")
    print(f"    SMC_PHI_W = {best.SMC_PHI_W:.4f}  rad/s")
    print(f"    SMC_KS_I  = {best.SMC_KS_I:.6f}  V")
    print(f"    SMC_PHI_I = {best.SMC_PHI_I:.6f}  A")
    print(f"\n  Saved to: {out_json}")

    if header_path and Path(header_path).exists():
        patch_header(best, header_path)
        print(f"  Header patched: {header_path}")

    return best


# =============================================================================
#  Step 2 — SIMULATE
# =============================================================================
def step2_simulate(gains: SMCGains,
                   target_rpm: float,
                   t_sim: float,
                   dt: float) -> dict:
    """
    Full EmbedSim closed-loop simulation at 20 kHz with tuned gains.
    Returns history dict for plotting.
    """
    print("\n" + "=" * 70)
    print("  STEP 2 — SIMULATE  (EmbedSim closed-loop @ 20 kHz)")
    print(f"  Target: {target_rpm:.0f} RPM   t_sim={t_sim}s")
    print(f"  Gains: KS_W={gains.SMC_KS_W:.4f}  ETA_W={gains.SMC_ETA_W:.4f}"
          f"  PHI_W={gains.SMC_PHI_W:.3f}  KS_I={gains.SMC_KS_I:.4f}"
          f"  PHI_I={gains.SMC_PHI_I:.4f}")
    print("=" * 70)

    target_rads = target_rpm * 2.0 * math.pi / 60.0

    # ── Build blocks ─────────────────────────────────────────────────────────
    cg_start = CodeGenStart("cg_start")
    smc = SMCControllerBlock(
        "smc",
        SMC_V_DC   = V_DC,
        SMC_KS_W   = gains.SMC_KS_W,
        SMC_ETA_W  = gains.SMC_ETA_W,
        SMC_PHI_W  = gains.SMC_PHI_W,
        SMC_KS_I   = gains.SMC_KS_I,
        SMC_PHI_I  = gains.SMC_PHI_I,
        dt_s       = dt,
        use_c_backend = False,
        integrator    = "tustin",
    )
    svpwm_pack = SVPWMPackBlock("svpwm_pack", v_dc=V_DC)
    svpwm      = SVPWMBlock("svpwm", use_c_backend=False)
    cg_end     = CodeGenEnd("cg_end")

    speed_ref  = VectorStep("speed_ref", step_time=0.0,
                            before_value=target_rads,
                            after_value=target_rads)
    motor      = DB42S02PlantBlock("motor")
    motor_delay= VectorDelay("motor_delay",
                             initial=[0.0] * _MOTOR_OUT_SIZE)
    ctrl       = CtrlPacker("ctrl_packer")
    sink       = VectorEnd("sink")
    sink_cg    = VectorEnd("sink_cg")

    # ── Wire ─────────────────────────────────────────────────────────────────
    cg_start >> smc >> svpwm_pack >> svpwm >> cg_end
    motor    >> motor_delay
    motor_delay >> ctrl
    speed_ref   >> ctrl
    ctrl        >> cg_start
    cg_end      >> motor
    motor       >> sink
    cg_end      >> sink_cg

    # ── Scope ─────────────────────────────────────────────────────────────────
    sim = EmbedSim(sinks=[sink, sink_cg], T=t_sim, dt=dt,
                   solver=ODESolver.EULER)

    print("\n[Topology]")
    sim.topo.print_console()

    sim.scope.add(speed_ref,  indices=[0],          label="SpeedRef")
    sim.scope.add(smc,        indices=[0, 1],        label="Vab")
    sim.scope.add(svpwm_pack, indices=[0],           label="Vref")
    sim.scope.add(svpwm,      indices=[0, 1, 2, 3],  label="Duties")
    sim.scope.add(motor, indices=[0, 1, 2, 3, 5, 6, 7], label="Motor")

    print(f"\nRunning simulation ({t_sim}s @ {1/dt:.0f} Hz)...")
    sim.run()
    print(f"  Completed: {len(sim.scope.t)} steps")

    # ── CodeGen ───────────────────────────────────────────────────────────────
    print("\n[CodeGen] Generating embedsim_step.c / .h ...")
    result = cg_end.generate_step(
        cg_start   = cg_start,
        output_dir = _ROOT,
        dt_hz      = 1.0 / dt,
        prefix     = "EmbedSim",
        write_files= True,
    )
    if result:
        gen_dir = _ROOT / "embedsim_gen"
        print(f"  → {gen_dir}/embedsim_step.h")
        print(f"  → {gen_dir}/embedsim_step.c")
        print(f"  Input_T  : omega_ref_mech, theta_m, ia, ib, ic")
        print(f"  Output_T : ta, tb, tc, sector")

    # ── Extract data ──────────────────────────────────────────────────────────
    sc = sim.scope
    t  = np.array(sc.t, dtype=np.float32)
    ld = smc.log_data

    def _motor(pos):
        s = sc.get_signal("Motor", pos)
        return s if s is not None else np.zeros(len(t), dtype=np.float32)

    def _scope(label, pos):
        s = sc.get_signal(label, pos)
        return s if s is not None else np.zeros(len(t), dtype=np.float32)

    def interp(key):
        if len(ld["t"]) > 1:
            return np.interp(t, ld["t"], ld[key]).astype(np.float32)
        return np.zeros(len(t), dtype=np.float32)

    return {
        "t":            t,
        "speed_rpm":    _motor(0),
        "omega_ref_rpm": interp("speed_ref"),
        "iq_ref":       interp("iq_ref"),
        "iq":           interp("iq"),
        "id":           interp("id"),
        "id_plant":     _motor(5),
        "iq_plant":     _motor(6),
        "torque":       _motor(4),
        "v_alpha":      _scope("Vab", 0),
        "v_beta":       _scope("Vab", 1),
        "vref":         _scope("Vref", 0),
        "ta":           _scope("Duties", 0),
        "tb":           _scope("Duties", 1),
        "tc":           _scope("Duties", 2),
    }


# =============================================================================
#  Step 3 — PLOT (summary of Step 2 results)
# =============================================================================
def step3_plot(d: dict, target_rpm: float,
               path: str = "db42s02_smc_foc_results.png"):
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle(
        f"DB42S02 — SMC FOC  |  {target_rpm:.0f} RPM  |  20 kHz  |  "
        f"Tuned gains",
        fontsize=12, fontweight="bold")
    t = d["t"]

    def _axl(ax, *plots, ylabel="", title="", hlines=()):
        for y, kw in plots:
            ax.plot(t, y, **kw)
        for yv, color, ls in hlines:
            ax.axhline(yv, color=color, ls=ls, lw=0.8, alpha=0.6)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(alpha=0.3); ax.set_xlabel("t [s]")

    _axl(axes[0,0],
         (d["omega_ref_rpm"], dict(color="k", ls="--", lw=1.5, label="ω_ref")),
         (d["speed_rpm"],     dict(color="C0",          lw=1.5, label="ω_actual")),
         ylabel="Speed [RPM]", title="Speed tracking")
    axes[0,0].legend(fontsize=8)

    _axl(axes[0,1],
         (d["speed_rpm"] - d["omega_ref_rpm"], dict(color="C1", lw=0.8)),
         ylabel="Error [RPM]", title="Speed error",
         hlines=[(0, "k", "--")])

    _axl(axes[1,0],
         (d["iq_ref"],    dict(color="k",  ls="--", lw=1.2, label="iq_ref")),
         (d["iq_plant"],  dict(color="C0",          lw=1.0, label="iq (plant)")),
         (d["id_plant"],  dict(color="C5",          lw=1.0, label="id (plant)")),
         ylabel="Current [A]", title="dq currents (MTPA — id=0)",
         hlines=[(0, "gray", "--"),
                 (_DB42S02.SMC_I_MAX,  "gray", "--"),
                 (-_DB42S02.SMC_I_MAX, "gray", "--")])
    axes[1,0].legend(fontsize=8)

    _axl(axes[1,1],
         (d["id_plant"], dict(color="C5", lw=0.8)),
         ylabel="id [A]", title="id (should ≈ 0 — MTPA)",
         hlines=[(0, "k", "--")])

    _axl(axes[2,0],
         (d["v_alpha"], dict(color="C0", lw=0.8, label="v_α")),
         (d["v_beta"],  dict(color="C1", lw=0.8, label="v_β")),
         ylabel="Voltage [V]", title="Stator voltage commands")
    axes[2,0].legend(fontsize=8)

    _axl(axes[2,1],
         (d["vref"], dict(color="C5", lw=0.8)),
         ylabel="Vref [norm]", title="SVPWM modulation index",
         hlines=[(0.95, "red", "--")])

    _axl(axes[3,0],
         (d["ta"], dict(color="C3", lw=0.7, label="ta")),
         (d["tb"], dict(color="C2", lw=0.7, label="tb")),
         (d["tc"], dict(color="C1", lw=0.7, label="tc")),
         ylabel="Duty", title="SVPWM duties")
    axes[3,0].set_ylim(-0.05, 1.05)
    axes[3,0].legend(fontsize=8)

    _axl(axes[3,1],
         (d["torque"] * 1000, dict(color="C4", lw=0.8, label="T_em")),
         ylabel="Torque [mN·m]", title="Electromagnetic torque (no-load)")
    axes[3,1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] {path}")


def print_summary(d: dict, target_rpm: float):
    n  = len(d["t"])
    ss = int(0.8 * n)
    err     = float(np.mean(np.abs(d["speed_rpm"][ss:] - d["omega_ref_rpm"][ss:])))
    id_ss   = float(np.mean(np.abs(d["id_plant"][ss:])))
    iq_ss   = float(np.mean(np.abs(d["iq_plant"][ss:])))
    spd_fin = float(d["speed_rpm"][-1])

    print("\n" + "=" * 60)
    print("  SMC FOC — Performance Summary")
    print("=" * 60)
    print(f"  Final speed      : {spd_fin:+.1f} RPM  (target {target_rpm:.0f})")
    print(f"  SS speed error   : {err:.2f} RPM  (last 20%)")
    print(f"  SS id            : {id_ss:.4f} A   (target 0 — MTPA)")
    print(f"  SS iq            : {iq_ss:.4f} A")
    print(f"  Load             : no-load (T_load = 0)")
    print("=" * 60)


# =============================================================================
#  CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="DB42S02 SMC FOC — Tune → Simulate → CodeGen")
    p.add_argument("--no-tune",  action="store_true",
                   help="Skip tuning, use default gains")
    p.add_argument("--rpm",      type=float, default=2000.0,
                   help="Target speed [RPM]  (default 2000)")
    p.add_argument("--t_sim",    type=float, default=2.0,
                   help="Simulation duration [s]  (default 2.0)")
    p.add_argument("--dt",       type=float, default=50e-6,
                   help="Time step [s]  (default 50µs = 20 kHz)")
    p.add_argument("--de_iters", type=int,   default=50,
                   help="DE iterations for tuner  (default 50)")
    p.add_argument("--gp_iters", type=int,   default=30,
                   help="GP iterations for tuner  (default 30)")
    p.add_argument("--gains_json", default="smc_best_gains.json",
                   help="JSON file to save/load gains")
    p.add_argument("--header",   default="",
                   help="Path to smc_gains_config.h to patch (NOT embed_sim_smc_controller.h)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    header_path = args.header
    if not header_path:
        _cand = (_ROOT / "fs_electrical_machines" / "c_src" /
                 "smc_gains_config.h")
        if _cand.exists():
            header_path = str(_cand)

    print("=" * 70)
    print("  NANOTEC DB42S02 — SMC FOC  |  Tune → Simulate → CodeGen")
    print(f"  Target: {args.rpm:.0f} RPM   dt={args.dt*1e6:.0f}µs   "
          f"t_sim={args.t_sim}s")
    print("=" * 70)

    # ── Step 1: Tune ──────────────────────────────────────────────────────────
    if args.no_tune:
        # Use defaults or load from JSON if it exists
        gains_path = Path(args.gains_json)
        if gains_path.exists():
            with open(gains_path) as f:
                d = json.load(f)
            gains = SMCGains(
                SMC_KS_W  = d["SMC_KS_W"],
                SMC_ETA_W = d["SMC_ETA_W"],
                SMC_PHI_W = d["SMC_PHI_W"],
                SMC_KS_I  = d["SMC_KS_I"],
                SMC_PHI_I = d["SMC_PHI_I"],
            )
            print(f"\n[Tune] Loaded gains from {gains_path}")
        else:
            gains = SMCGains()
            print(f"\n[Tune] Using default gains (no JSON found)")
        print(f"  {gains}")
    else:
        # Tune at a lower RPM/shorter sim for speed, then simulate at full target
        tune_rpm  = min(args.rpm, 400.0)   # tune at 400 RPM for fast convergence
        tune_tsim = 1.0                     # 1s sufficient to evaluate transient
        gains = step1_tune(
            target_rpm  = tune_rpm,
            t_sim       = tune_tsim,
            dt          = args.dt,
            de_iters    = args.de_iters,
            gp_iters    = args.gp_iters,
            out_json    = args.gains_json,
            header_path = header_path,
        )

    # ── Step 2: Simulate + CodeGen ────────────────────────────────────────────
    data = step2_simulate(
        gains      = gains,
        target_rpm = args.rpm,
        t_sim      = args.t_sim,
        dt         = args.dt,
    )

    # ── Step 3: Plot ──────────────────────────────────────────────────────────
    step3_plot(data, args.rpm)
    print_summary(data, args.rpm)

    print("\n[Done]")
    print("  db42s02_smc_foc_results.png")
    print("  embedsim_gen/embedsim_step.c")
    print("  embedsim_gen/embedsim_step.h")
    if not args.no_tune:
        print(f"  {args.gains_json}")
