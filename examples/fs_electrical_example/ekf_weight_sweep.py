# ekf_weight_sweep.py  v3
"""
ekf_weight_sweep.py  v3
=======================
EKF weight sweep — steady-state speed accuracy under AURIX noise.

The v1/v2 sweeps measured only P[2,2] convergence speed, which is
purely algebraic (P0_omega/r_i ratio) and independent of signal content.
The real question is: once P has converged, how accurately does the EKF
track true omega_m under AURIX noise (ADC 12-bit, PWM spikes)?

Approach
--------
Self-contained PMSM Euler integrator — no EmbedSim, no SMC, no imports
beyond numpy.  Runs a 200 ms trajectory:
  0–30 ms   : ramp 0 → 2000 RPM using simple PI FOC
  30–100 ms : hold 2000 RPM, no load
  100–200 ms: hold 2000 RPM, load step 20 mN·m

AURIX noise applied to ia, ib, ic before feeding EKF:
  - ADC 12-bit quantisation + Gaussian thermal noise (σ = 1.5 LSB)
  - PWM switching spike on ia (5% probability, ±0.5 A)

For each (q_omega, r_i) pair measures:
  - Steps to P[2,2] convergence  (cold-start robustness)
  - Steady-state RMS speed error  (accuracy under noise)
  - Peak speed error at load step (disturbance rejection)

Outputs
-------
  ekf_weight_sweep.png         — 2-panel heatmap: conv steps | SS error
  ekf_weight_sweep_detail.png  — omega_m traces for 5 best candidates
"""

from __future__ import annotations
import sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_FS   = _ROOT / "fs_electrical_machines"
for _p in (str(_ROOT / "embedsim"), str(_FS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diff_flatness_controller_block import ExtendedKalmanFilter

# ── DB42S02 motor constants ────────────────────────────────────────────────────
R_S       = 0.285
L_D = L_Q = 0.3675e-3
LAMBDA_PM = 0.0014
P_POLES   = 4
I_MAX     = 3.57
V_DC      = 17.0
J         = 2.4e-6
B_FRIC    = 1e-6
KT        = 1.5 * P_POLES * LAMBDA_PM   # 0.0084 N·m/A
V_MAX     = V_DC / math.sqrt(3.0)
DT        = 50e-6   # 20 kHz

# ── Simulation profile ────────────────────────────────────────────────────────
TARGET_RPM   = 2000.0
TARGET_RADS  = TARGET_RPM * 2.0 * math.pi / 60.0
N_SIM        = 4000          # 200 ms total
RAMP_END     = 600           # step where ramp finishes (30 ms)
LOAD_STEP    = 2000          # load applied at step 2000 (100 ms)
T_LOAD       = 0.020         # N·m heavy load
SS_START     = 3000          # steady-state analysis window start (150 ms)

# Simple PI current controller gains (fast inner loop for signal generation)
KP_ID = 0.4;  KI_ID = 50.0
KP_IQ = 8.0;  KI_IQ = 500.0
KP_SPEED = 0.4

# ── Sweep grid ────────────────────────────────────────────────────────────────
Q_OMEGA_VALUES = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 10.0]
R_I_VALUES     = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
P_CONV_THRESHOLD = 800.0  # elec speed variance (rad/s_e)^2 = 50*(4^2)
P0_OMEGA         = 1e6

DETAIL_PAIRS = [
    (1e-2, 1e-5),
    (1e-2, 1e-4),
    (1e-2, 1e-3),
    (1e-1, 1e-4),
    (1.0,  1e-4),
]

_DARK  = "#111111"
_PANEL = "#1a1a1a"

# ── AURIX noise ───────────────────────────────────────────────────────────────
_ADC_LSB_A       = (I_MAX * 2.0) / (2**12)
_ADC_NOISE_SIGMA = 1.5 * _ADC_LSB_A
_ADC_SAT         = I_MAX * 1.26
_SPIKE_PROB      = 0.05
_SPIKE_AMP       = 0.50
_RNG             = np.random.default_rng(seed=20240101)

def _add_noise(ia: float, ib: float, ic: float):
    """Apply AURIX ADC noise + phase-A PWM spike."""
    def _adc(i):
        n = i + _RNG.normal(0.0, _ADC_NOISE_SIGMA)
        q = round(n / _ADC_LSB_A) * _ADC_LSB_A
        return float(np.clip(q, -_ADC_SAT, _ADC_SAT))
    ia_n = _adc(ia)
    ib_n = _adc(ib)
    ic_n = _adc(ic)
    if _RNG.random() < _SPIKE_PROB:
        ia_n += float(_RNG.choice([-1.0, 1.0])) * _SPIKE_AMP
    return ia_n, ib_n, ic_n


# =============================================================================
# Self-contained PMSM + PI FOC signal generator
# =============================================================================

def _generate_signals() -> dict:
    """
    Euler-integrate PMSM with PI FOC for N_SIM steps.

    Returns per-step arrays: ia_n, ib_n, ic_n (noisy), v_alpha, v_beta,
    omega_true_rads, t_s.
    All float32 except t_s.
    """
    print("  Generating PMSM signals (Euler, PI FOC, AURIX noise) ...")

    # State: id, iq, omega_m, theta_e  (all in SI)
    id_ = iq_ = om_ = th_e = 0.0

    # PI integrators
    int_id = int_iq = 0.0

    ia_n_arr = np.zeros(N_SIM, np.float32)
    ib_n_arr = np.zeros(N_SIM, np.float32)
    ic_n_arr = np.zeros(N_SIM, np.float32)
    va_arr   = np.zeros(N_SIM, np.float32)
    vb_arr   = np.zeros(N_SIM, np.float32)
    om_arr   = np.zeros(N_SIM, np.float32)   # true omega_m

    for k in range(N_SIM):
        omega_e = P_POLES * om_
        t_load  = T_LOAD if k >= LOAD_STEP else 0.0

        # Speed P-loop → iq_ref
        ramp_ref = TARGET_RADS * min(1.0, k / max(1, RAMP_END))
        speed_err = ramp_ref - om_
        iq_ref = float(np.clip(KP_SPEED * speed_err, -I_MAX, I_MAX))
        id_ref = 0.0  # MTPA

        # Current PI
        err_id = id_ref - id_;  err_iq = iq_ref - iq_
        int_id += KI_ID * err_id * DT
        int_iq += KI_IQ * err_iq * DT

        vd = KP_ID * err_id + int_id - omega_e * L_Q * iq_
        vq = KP_IQ * err_iq + int_iq + omega_e * (L_D * id_ + LAMBDA_PM)

        # Hexagon voltage clamp
        mag = math.sqrt(vd * vd + vq * vq)
        if mag > V_MAX:
            vd *= V_MAX / mag
            vq *= V_MAX / mag

        # Inverse Park → αβ
        cos_t = math.cos(th_e);  sin_t = math.sin(th_e)
        v_alpha = vd * cos_t - vq * sin_t
        v_beta  = vd * sin_t + vq * cos_t

        # Euler: electrical dynamics
        did = (vd - R_S * id_ + omega_e * L_Q * iq_) / L_D
        diq = (vq - R_S * iq_ - omega_e * (L_D * id_ + LAMBDA_PM)) / L_Q

        id_ += did * DT
        iq_ += diq * DT

        # Mechanical dynamics
        T_em = 1.5 * P_POLES * (LAMBDA_PM * iq_ + (L_D - L_Q) * id_ * iq_)
        dom  = (T_em - B_FRIC * om_ - t_load) / J
        om_ += dom * DT
        th_e += P_POLES * om_ * DT
        # Wrap theta_e to [-pi, pi]
        while th_e >  math.pi: th_e -= 2.0 * math.pi
        while th_e < -math.pi: th_e += 2.0 * math.pi

        # abc currents (inverse Park + Clarke)
        i_alpha = id_ * cos_t - iq_ * sin_t
        i_beta  = id_ * sin_t + iq_ * cos_t
        ia = i_alpha
        ib = -0.5 * i_alpha + (math.sqrt(3.0) / 2.0) * i_beta
        ic = -ia - ib

        # Apply AURIX noise
        ia_n, ib_n, ic_n = _add_noise(ia, ib, ic)

        ia_n_arr[k] = ia_n
        ib_n_arr[k] = ib_n
        ic_n_arr[k] = ic_n
        va_arr[k]   = v_alpha
        vb_arr[k]   = v_beta
        om_arr[k]   = om_

    rpm_ss = float(np.mean(om_arr[SS_START:])) * 60.0 / (2.0 * math.pi)
    ia_rms = float(np.sqrt(np.mean(ia_n_arr[SS_START:] ** 2)))
    print(f"  Generated {N_SIM} steps — "
          f"SS speed={rpm_ss:.0f} RPM  ia_rms={ia_rms:.3f} A")

    return {"ia": ia_n_arr, "ib": ib_n_arr, "ic": ic_n_arr,
            "va": va_arr, "vb": vb_arr, "omega_true": om_arr}


# =============================================================================
# EKF evaluation for one (q_omega, r_i) pair
# =============================================================================

def _evaluate(q_i, q_omega, r_i, sig: dict) -> dict:
    """
    Run EKF on signals, return convergence + accuracy metrics.
    """
    ekf = ExtendedKalmanFilter(
        R_s=R_S, L_d=L_D, L_q=L_Q, lambda_pm=LAMBDA_PM,
        p_poles=P_POLES,
        q_i=q_i, q_omega=q_omega, r_i=r_i,
        p0_i=1.0, p0_omega=P0_OMEGA,
        i_max=I_MAX * 10.0,
        omega_max=float(P_POLES) * 700.0,  # electrical rad/s
        warmup_steps=0,
    )
    ia = sig["ia"]; ib = sig["ib"]; ic = sig["ic"]
    va = sig["va"]; vb = sig["vb"]
    omega_true = sig["omega_true"]

    omega_ekf = np.zeros(N_SIM, np.float32)
    p_trace   = np.zeros(N_SIM, np.float32)
    conv_step = N_SIM + 1

    for k in range(N_SIM):
        ekf.step(float(ia[k]), float(ib[k]), float(ic[k]),
                 float(va[k]), float(vb[k]), DT)
        omega_ekf[k] = float(ekf.omega_m)
        p_trace[k]   = float(ekf.P[2, 2])
        if conv_step > N_SIM and float(ekf.P[2, 2]) < P_CONV_THRESHOLD:
            conv_step = k + 1

    # Steady-state error [RPM]
    err_rpm = (omega_ekf[SS_START:] - omega_true[SS_START:]) * 60.0 / (2.0 * math.pi)
    ss_rms  = float(np.sqrt(np.mean(err_rpm ** 2)))

    # Load step peak error: window around LOAD_STEP
    w0 = max(0, LOAD_STEP - 20);  w1 = min(N_SIM, LOAD_STEP + 200)
    peak_err = float(np.max(np.abs(
        (omega_ekf[w0:w1] - omega_true[w0:w1]) * 60.0 / (2.0 * math.pi))))

    return {
        "conv_step": conv_step,
        "ss_rms_rpm": ss_rms,
        "peak_err_rpm": peak_err,
        "omega_ekf": omega_ekf,
        "p_trace": p_trace,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  EKF weight sweep v3 — steady-state accuracy under AURIX noise")
    print(f"  {N_SIM} steps ({N_SIM*DT*1e3:.0f} ms)  target={TARGET_RPM:.0f} RPM")
    print(f"  Load step at {LOAD_STEP*DT*1e3:.0f} ms  SS window {SS_START*DT*1e3:.0f}–{N_SIM*DT*1e3:.0f} ms")
    print("=" * 65)

    sig = _generate_signals()
    t_ms = np.arange(N_SIM) * DT * 1e3

    # ── Full sweep ────────────────────────────────────────────────────────
    conv_grid   = np.zeros((len(Q_OMEGA_VALUES), len(R_I_VALUES)), np.float32)
    ss_err_grid = np.zeros_like(conv_grid)

    print(f"\n  {'q_omega':>10}  {'r_i':>10}  {'conv':>8}  {'ss_rms':>9}  {'peak':>9}")
    for i, q_omega in enumerate(Q_OMEGA_VALUES):
        for j, r_i in enumerate(R_I_VALUES):
            res = _evaluate(1e-4, q_omega, r_i, sig)
            conv_grid[i, j]   = res["conv_step"]
            ss_err_grid[i, j] = res["ss_rms_rpm"]
            ok = "OK" if res["conv_step"] <= N_SIM else "--"
            print(f"  {q_omega:>10.0e}  {r_i:>10.0e}  "
                  f"{res['conv_step']:>5} {ok}  "
                  f"{res['ss_rms_rpm']:>8.2f} RPM  "
                  f"{res['peak_err_rpm']:>8.2f} RPM")

    # ── Plot 1: dual heatmap ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=_DARK)
    fig.suptitle(
        f"DB42S02 EKF weights — p0_omega={P0_OMEGA:.0e}  "
        f"P_conv={P_CONV_THRESHOLD}  AURIX noise ON",
        color="white", fontsize=12)

    def _hmap(ax, data, title, fmt, cmap, vmin, vmax):
        from matplotlib.patches import Rectangle
        im = ax.imshow(data, aspect="auto", origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(R_I_VALUES)))
        ax.set_xticklabels([f"{v:.0e}" for v in R_I_VALUES],
                           rotation=45, ha="right", color="#ccc", fontsize=8)
        ax.set_yticks(range(len(Q_OMEGA_VALUES)))
        ax.set_yticklabels([f"{v:.0e}" for v in Q_OMEGA_VALUES],
                           color="#ccc", fontsize=8)
        ax.set_xlabel("r_i  (measurement noise)", color="#aaa", fontsize=10)
        ax.set_ylabel("q_omega  (process noise)", color="#aaa", fontsize=10)
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.set_facecolor(_PANEL)
        for i in range(len(Q_OMEGA_VALUES)):
            for j in range(len(R_I_VALUES)):
                val = data[i, j]
                txt = fmt(val)
                bg  = ax.images[0].cmap(ax.images[0].norm(val))
                lum = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]
                col = "black" if lum > 0.5 else "white"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=col, fontweight="bold")
        return im

    # Left: convergence steps (green = fast)
    conv_disp = np.where(conv_grid > N_SIM, N_SIM + 1, conv_grid)
    im0 = _hmap(axes[0], conv_disp,
                f"Cold-start convergence steps\nP[2,2] < {P_CONV_THRESHOLD}  (green=fast)",
                lambda v: f"{int(v)}" if v <= N_SIM else "X",
                "RdYlGn_r", 1, N_SIM + 1)
    cb0 = fig.colorbar(im0, ax=axes[0])
    cb0.set_label("Steps to converge", color="#aaa")
    cb0.ax.yaxis.label.set_color("#aaa")

    # Right: steady-state RMS error RPM (green = accurate)
    ss_max = float(np.percentile(ss_err_grid, 90))
    im1 = _hmap(axes[1], ss_err_grid,
                f"SS speed error RMS [RPM]\n(last {(N_SIM-SS_START)*DT*1e3:.0f} ms, AURIX noise)",
                lambda v: f"{v:.1f}",
                "RdYlGn_r", 0, max(ss_max, 1.0))
    cb1 = fig.colorbar(im1, ax=axes[1])
    cb1.set_label("SS RMS error [RPM]", color="#aaa")
    cb1.ax.yaxis.label.set_color("#aaa")

    # Mark best cell (lowest SS error + converged)
    mask = conv_grid <= N_SIM
    if np.any(mask):
        masked = np.where(mask, ss_err_grid, np.inf)
        bi, bj = np.unravel_index(np.argmin(masked), masked.shape)
        from matplotlib.patches import Rectangle
        for ax in axes:
            ax.add_patch(Rectangle((bj-0.5, bi-0.5), 1, 1,
                         lw=2.5, edgecolor="cyan", facecolor="none"))
        print(f"\n  Best cell (cyan): "
              f"q_omega={Q_OMEGA_VALUES[bi]:.0e}  "
              f"r_i={R_I_VALUES[bj]:.0e}  "
              f"conv={int(conv_grid[bi,bj])} steps  "
              f"ss_err={ss_err_grid[bi,bj]:.2f} RPM")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out1 = _HERE / "ekf_weight_sweep.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close(fig)
    print(f"  Saved {out1}")

    # ── Plot 2: omega_m traces for best candidates ────────────────────────
    omega_true_rpm = sig["omega_true"] * 60.0 / (2.0 * math.pi)

    fig2, axes2 = plt.subplots(len(DETAIL_PAIRS), 1,
                               figsize=(14, 3 * len(DETAIL_PAIRS)),
                               facecolor=_DARK)
    fig2.suptitle("EKF omega_m traces — DB42S02  AURIX noise ON",
                  color="white", fontsize=11)

    for row, (q_omega, r_i) in enumerate(DETAIL_PAIRS):
        res = _evaluate(1e-4, q_omega, r_i, sig)
        ekf_rpm = res["omega_ekf"] * 60.0 / (2.0 * math.pi)
        ax = axes2[row]
        ax.set_facecolor(_PANEL)
        ax.spines[:].set_color("#333")
        ax.tick_params(colors="#888", labelsize=8)

        ax.plot(t_ms, omega_true_rpm, color="#44bbff", lw=1.2,
                label="true omega_m", alpha=0.8)
        ax.plot(t_ms, ekf_rpm, color="#ff9944", lw=1.0,
                label="EKF estimate", alpha=0.9)
        ax.axvline(LOAD_STEP * DT * 1e3, color="orange", lw=0.9,
                   ls="--", alpha=0.7, label="load step")
        ax.axvline(SS_START * DT * 1e3, color="#888", lw=0.8,
                   ls=":", alpha=0.6, label="SS window")

        conv_ms = res["conv_step"] * DT * 1e3
        if res["conv_step"] <= N_SIM:
            ax.axvline(conv_ms, color="lime", lw=1.0, ls="--",
                       label=f"conv @ {conv_ms:.2f} ms")

        ax.set_ylabel("Speed [RPM]", color="#aaa", fontsize=8)
        ax.set_xlabel("t [ms]", color="#888", fontsize=8)
        ax.set_title(
            f"q_ω={q_omega:.0e}  r_i={r_i:.0e}  |  "
            f"conv={res['conv_step']} steps  "
            f"SS_rms={res['ss_rms_rpm']:.2f} RPM  "
            f"peak={res['peak_err_rpm']:.2f} RPM",
            color="#cccccc", fontsize=9)
        ax.legend(fontsize=7, facecolor="#222",
                  labelcolor="white", edgecolor="#444", loc="lower right")

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = _HERE / "ekf_weight_sweep_detail.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close(fig2)
    print(f"  Saved {out2}")

    # ── Recommendation ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Recommended AURIX weights (converged + lowest SS error):")
    if np.any(mask):
        print(f"    q_omega  = {Q_OMEGA_VALUES[bi]:.0e}")
        print(f"    r_i      = {R_I_VALUES[bj]:.0e}")
        print(f"    p0_omega = {P0_OMEGA:.0e}  (cold-start)")
        print(f"    conv     = {int(conv_grid[bi,bj])} steps  "
              f"({conv_grid[bi,bj]*DT*1e3:.2f} ms)")
        print(f"    SS error = {ss_err_grid[bi,bj]:.2f} RPM RMS")
    print(f"{'='*65}")
