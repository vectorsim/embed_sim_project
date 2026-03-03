"""
example_pmsm_foc.py
===================

PMSM Field-Oriented Control — pmsm_blocks library demo.

Control topology
----------------

  ω_ref ──►[Σ]──►[speed PI]──────────────────► iq*
            ▲                                    │
      [omega_delay]◄──────────────────────── [motor ω]
                                                 │
                    id* = 0 ──►[Σ]──►[id PI]──►[vd]─┐
                                ▲                     ├──►[InvPark(θ)]──►[vα,vβ]──►[MOTOR]
                                │    iq* ──►[Σ]──►[iq PI]──►[vq]─┘
                                │          ▲
                         [id_delay]    [iq_delay]
                                │          │
                          [Park(θ)]◄──────[dq]
                               ▲
                 [Clarke]◄──[abc_delay]◄──[motor ia,ib,ic]
                    ▲
             [theta_delay]◄──[motor θ]     (also feeds InvPark)

Loop breakers (ExtractDelay):
  omega_delay  — channels [3]   (ω from motor 5-vector)
  theta_delay  — channels [4]   (θ from motor 5-vector)
  abc_delay    — channels [0,1,2] (ia,ib,ic from motor 5-vector)
  id_delay     — channels [0]   (id from park 2-vector)
  iq_delay     — channels [1]   (iq from park 2-vector)

Set USE_C_BACKEND = True once Cython extensions are compiled.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ── Project path ──────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / '.project_root_marker').exists():
            return parent
    return current.parent

root = get_project_root()
sys.path.insert(0, str(root / "embedsim"))


# ── Imports ───────────────────────────────────────────────────────────────────

from embedsim import (
    EmbedSim, VectorConstant, VectorStep, VectorSum, ODESolver,
)

from pmsm_blocks import (
    PMSMMotorBlock,
    ClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
    PIControllerBlock,
    VectorCombineBlock,
    RecordingSinkBlock,
    ExtractDelay,          # ← fused extract + loop-breaking delay
)

# ── Backend flag ──────────────────────────────────────────────────────────────
USE_C_BACKEND = True    # set True after: cd pmsm_blocks/c_src && ./build_all.sh


# ==============================================================================
# Simulation
# ==============================================================================

def run_pmsm_foc():
    be = "C" if USE_C_BACKEND else "Python"
    print("\n" + "=" * 70)
    print(f"PMSM FOC SIMULATION  —  backend: {be}")
    print("=" * 70)

    # ── References ────────────────────────────────────────────────────────────
    speed_ref   = VectorStep("speed_ref",
                              step_time=0.1, before_value=0.0,
                              after_value=100.0, dim=1)
    id_ref      = VectorConstant("id_ref",  [0.0])   # MTPA: id* = 0
    load_torque = VectorStep("load",
                              step_time=0.5, before_value=0.0,
                              after_value=5.0,   dim=1)

    # ── Summing junctions ─────────────────────────────────────────────────────
    speed_error = VectorSum("speed_error", signs=[1, -1])  # ω*  − ω
    id_error    = VectorSum("id_error",    signs=[1, -1])  # id* − id
    iq_error    = VectorSum("iq_error",    signs=[1, -1])  # iq* − iq

    # ── PI controllers ────────────────────────────────────────────────────────
    speed_pi = PIControllerBlock("speed_pi", Kp=1.0,  Ki=20.0,   limit=100.0,
                                  use_c_backend=USE_C_BACKEND)
    id_pi    = PIControllerBlock("id_pi",    Kp=50.0, Ki=1000.0, limit=200.0,
                                  use_c_backend=USE_C_BACKEND)
    iq_pi    = PIControllerBlock("iq_pi",    Kp=50.0, Ki=1000.0, limit=200.0,
                                  use_c_backend=USE_C_BACKEND)

    # ── Transforms ────────────────────────────────────────────────────────────
    dq_combine = VectorCombineBlock("dq_combine", n_inputs=2)
    inv_park   = InvParkTransformBlock("inv_park", use_c_backend=USE_C_BACKEND)
    clarke     = ClarkeTransformBlock("clarke",    use_c_backend=USE_C_BACKEND)
    park       = ParkTransformBlock("park",        use_c_backend=USE_C_BACKEND)

    # ── Motor ─────────────────────────────────────────────────────────────────
    motor = PMSMMotorBlock(
        "motor",
        Rs=0.5, Ld=0.002, Lq=0.002, psi_pm=0.3, J=0.001, B=0.001,
        use_c_backend=USE_C_BACKEND,
    )

    # ── ExtractDelay loop breakers ────────────────────────────────────────────
    # Motor output is a 5-vector: [ia, ib, ic, ω, θ].
    # Park output is a 2-vector:  [id, iq].
    #
    # ExtractDelay is both a channel extractor AND a LoopBreaker:
    #   • pre-initialised by the engine before the main compute pass
    #   • extracts and stores channels from its upstream block each step
    #   • outputs the PREVIOUS step's extracted value to downstream blocks
    #
    # This avoids the ordering problem where a plain VectorDelay placed after
    # a SignalExtractBlock would try to run before the extractor.

    omega_delay = ExtractDelay("omega_delay", channels=[3],      initial=[0.0])
    theta_delay = ExtractDelay("theta_delay", channels=[4],      initial=[0.0])
    abc_delay   = ExtractDelay("abc_delay",   channels=[0,1,2],  initial=[0.0, 0.0, 0.0])
    id_delay    = ExtractDelay("id_delay",    channels=[0],      initial=[0.0])
    iq_delay    = ExtractDelay("iq_delay",    channels=[1],      initial=[0.0])

    # ── Recording sinks ───────────────────────────────────────────────────────
    sink_motor = RecordingSinkBlock("motor_out")   # records [ia, ib, ic, ω, θ]
    sink_dq    = RecordingSinkBlock("dq_out")      # records [id, iq] from park

    # ── Connections ───────────────────────────────────────────────────────────
    print("\nConnecting blocks...")

    # Speed outer loop
    speed_ref   >> speed_error      # ω* reference
    omega_delay >> speed_error      # ω  feedback (one-step delayed)
    speed_error >> speed_pi         # → iq*

    # Current inner loops
    id_ref    >> id_error           # id* = 0
    speed_pi  >> iq_error           # iq* from speed PI

    id_delay  >> id_error           # id feedback (one-step delayed)
    iq_delay  >> iq_error           # iq feedback (one-step delayed)

    id_error  >> id_pi
    iq_error  >> iq_pi

    # Voltage vector → InvPark → [vα, vβ]
    id_pi       >> dq_combine       # vd
    iq_pi       >> dq_combine       # vq
    dq_combine  >> inv_park         # port 0: [vd, vq]
    theta_delay >> inv_park         # port 1: θ (delayed)

    # Motor
    inv_park    >> motor            # port 0: [vα, vβ]
    load_torque >> motor            # port 1: T_load

    # Motor → recording sink
    motor >> sink_motor

    # Motor → ExtractDelay loop breakers (all read the 5-vector)
    motor >> omega_delay            # extracts [ω] → speed loop
    motor >> theta_delay            # extracts [θ] → inv_park, park
    motor >> abc_delay              # extracts [ia,ib,ic] → Clarke

    # Current feedback chain
    abc_delay >> clarke             # [ia,ib,ic] → [α,β]
    clarke    >> park               # [α,β] → [id,iq]
    theta_delay >> park             # θ (shared delayed angle)

    # dq → recording + ExtractDelay loop breakers
    park >> sink_dq                 # record [id, iq] directly from Park
    park >> id_delay                # extracts [id] → id_error
    park >> iq_delay                # extracts [iq] → iq_error

    print("✓  Connections complete")

    # ── Run ───────────────────────────────────────────────────────────────────
    sim = EmbedSim(
        sinks  = [sink_motor, sink_dq],
        T      = 1.0,
        dt     = 0.0001,
        solver = ODESolver.RK4,
    )

    print("\nRunning simulation  (T=1.0 s,  dt=0.1 ms,  RK4)...")
    sim.run(verbose=False, progress_bar=True)

    motor_data, t = sink_motor.get_data()  # (N, 5): [ia, ib, ic, ω, θ]
    dq_data,    _ = sink_dq.get_data()     # (N, 2): [id, iq]

    print(f"\nData shapes:")
    print(f"  motor_data : {motor_data.shape}   (expected (10001, 5))")
    print(f"  dq_data    : {dq_data.shape}")

    _print_performance(motor_data, dq_data, t)
    _plot_results(motor_data, dq_data, t, be)

    print("\n" + "=" * 70)
    print("SIMULATION STATISTICS")
    print("=" * 70)
    print(f"  Backend      : {be}")
    print(f"  Steps        : {sim.stats.total_steps}")
    print(f"  Compute time : {sim.stats.compute_time:.3f} s")
    print(f"  Avg step     : {sim.stats.avg_step_time * 1e6:.1f} µs")
    print(f"  Loop breakers: {sim.stats.loop_breakers_count}")

    return sim, motor_data, dq_data, t


# ==============================================================================
# Performance report
# ==============================================================================

def _print_performance(motor_data, dq_data, t):
    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)

    omega  = motor_data[:, 3]
    ia, ib, ic = motor_data[:, 0], motor_data[:, 1], motor_data[:, 2]
    target = 100.0

    ss_mask = t > 0.85
    if np.any(ss_mask):
        ss_w = float(np.mean(omega[ss_mask]))
        err  = target - ss_w
        print(f"\nSpeed control:")
        print(f"  Steady-state ω : {ss_w:.2f} rad/s   (ref {target:.1f})")
        print(f"  Steady error   : {err:+.3f} rad/s  ({abs(err/target)*100:.2f} %)")

    band    = 0.02 * target
    in_band = np.abs(omega - target) < band
    if np.any(in_band):
        first = t[np.where(in_band)[0][0]]
        print(f"  First enters ±2% band at t = {first:.4f} s")

    for lbl, msk in [("Before load (0.25–0.45 s)", (t>0.25)&(t<0.45)),
                     ("After  load (0.65–0.85 s)", (t>0.65)&(t<0.85))]:
        if np.any(msk):
            ra = float(np.sqrt(np.mean(ia[msk]**2)))
            rb = float(np.sqrt(np.mean(ib[msk]**2)))
            rc = float(np.sqrt(np.mean(ic[msk]**2)))
            print(f"\nPhase current RMS  {lbl}:")
            print(f"  ia={ra:.3f} A    ib={rb:.3f} A    ic={rc:.3f} A")

    if dq_data.ndim == 2 and dq_data.shape[1] >= 2 and np.any(ss_mask):
        id_ss = float(np.mean(dq_data[ss_mask, 0]))
        iq_ss = float(np.mean(dq_data[ss_mask, 1]))
        ok    = "✓  Good" if abs(id_ss) < 0.5 else "⚠  Needs improvement"
        print(f"\ndq steady-state (> 0.85 s):")
        print(f"  id = {id_ss:+.4f} A   (target 0)   {ok}")
        print(f"  iq = {iq_ss:+.4f} A   (torque component)")


# ==============================================================================
# Plots
# ==============================================================================

def _plot_results(motor_data, dq_data, t, be="Python"):
    ia    = motor_data[:, 0]
    ib    = motor_data[:, 1]
    ic    = motor_data[:, 2]
    omega = motor_data[:, 3]
    i_rms = np.sqrt((ia**2 + ib**2 + ic**2) / 3.0)

    ds    = max(1, len(t) // 8000)   # downsample index step for plotting

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"PMSM Field-Oriented Control  [{be} backend]\n"
        f"Speed step: 0→100 rad/s @ t=0.1 s  |  Load step: 5 N·m @ t=0.5 s",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.44, wspace=0.30)

    kw_s = dict(x=0.1, color="green",  lw=1.8, ls=":", label="Speed step  t=0.1 s")
    kw_l = dict(x=0.5, color="orange", lw=1.8, ls=":", label="Load  step  t=0.5 s")

    # ── 1. Rotor speed ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t[::ds], omega[::ds], "b-", lw=2.5, label="ω  measured")
    ax.axhline(100.0, color="r",  ls="--", lw=2,  label="ω* = 100 rad/s")
    ax.axvline(**kw_s); ax.axvline(**kw_l)
    ax.fill_between(t[::ds], 98, 102, alpha=0.13, color="gray", label="±2 % band")
    ax.set(ylabel="Speed  [rad/s]", xlabel="Time  [s]",
           title="Rotor Speed  ω  [rad/s]", ylim=[-5, 115])
    ax.legend(fontsize=9, ncol=4); ax.grid(alpha=0.3)

    # ── 2. Three-phase currents (full run) ────────────────────────────────────
    ax = fig.add_subplot(gs[1, :])
    ax.plot(t[::ds], ia[::ds], "r-", lw=1.2, alpha=0.85, label="ia")
    ax.plot(t[::ds], ib[::ds], "g-", lw=1.2, alpha=0.85, label="ib")
    ax.plot(t[::ds], ic[::ds], "b-", lw=1.2, alpha=0.85, label="ic")
    ax.axvline(**kw_s); ax.axvline(**kw_l)
    ax.set(ylabel="Current  [A]", xlabel="Time  [s]",
           title="Three-Phase Stator Currents  ia / ib / ic  [A]")
    ax.legend(fontsize=10, ncol=3); ax.grid(alpha=0.3)

    # ── 3. Startup zoom (0–0.25 s) ────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    msk = t <= 0.25
    ax.plot(t[msk], ia[msk], "r-", lw=2, label="ia")
    ax.plot(t[msk], ib[msk], "g-", lw=2, label="ib")
    ax.plot(t[msk], ic[msk], "b-", lw=2, label="ic")
    ax.axvline(x=0.1, color="green", ls=":", lw=1.8)
    ax.set(ylabel="[A]", xlabel="Time  [s]",
           title="Startup Transient  (0–0.25 s)", xlim=[0, 0.25])
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── 4. Load transient zoom (0.45–0.65 s) ─────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    msk = (t >= 0.45) & (t <= 0.65)
    ax.plot(t[msk], ia[msk], "r-", lw=2, label="ia")
    ax.plot(t[msk], ib[msk], "g-", lw=2, label="ib")
    ax.plot(t[msk], ic[msk], "b-", lw=2, label="ic")
    ax.axvline(x=0.5, color="orange", ls=":", lw=1.8)
    ax.set(ylabel="[A]", xlabel="Time  [s]",
           title="Load Transient  (0.45–0.65 s)", xlim=[0.45, 0.65])
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── 5. dq-axis currents ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    if dq_data.ndim == 2 and dq_data.shape[1] >= 2:
        ax.plot(t[::ds], dq_data[::ds, 0], "c-", lw=2, label="id")
        ax.plot(t[::ds], dq_data[::ds, 1], "m-", lw=2, label="iq")
        ax.axhline(0, color="k", alpha=0.25, lw=1)
    ax.axvline(**kw_s); ax.axvline(**kw_l)
    ax.set(ylabel="[A]", xlabel="Time  [s]",
           title="dq-axis Currents  id / iq  [A]")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # ── 6. RMS current ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    ax.plot(t[::ds], i_rms[::ds], color="purple", lw=2.5, label="I_rms")
    for lbl, msk_r, clr in [
        ("Before load", (t > 0.25) & (t < 0.45), "green"),
        ("After  load", (t > 0.65) & (t < 0.85), "orange"),
    ]:
        if np.any(msk_r):
            lvl = float(np.mean(i_rms[msk_r]))
            ax.axhline(lvl, color=clr, ls="--", lw=1.5,
                       label=f"{lbl}  {lvl:.2f} A")
    ax.axvline(**kw_s); ax.axvline(**kw_l)
    ax.set(ylabel="[A RMS]", xlabel="Time  [s]", title="RMS Current Magnitude")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.savefig("pmsm_foc_library_demo.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n✓  Plot saved → pmsm_foc_library_demo.png")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    sim, motor_data, dq_data, t = run_pmsm_foc()
    print("\n" + "=" * 70)
    print("✓  PMSM FOC simulation complete")
    print("=" * 70)
    print("\nTo enable C backend:")
    print("  1.  cd pmsm_blocks/c_src")
    print("  2.  ./build_all.sh  (Linux/macOS)  or  build_all.bat  (Windows)")
    print("  3.  Set USE_C_BACKEND = True at the top of this file")
