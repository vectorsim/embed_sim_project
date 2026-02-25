"""
ControlForge — ScriptBlock Example
====================================

Scenario: Closed-loop speed control of a simple first-order plant
  - Plant:       dv/dt = (u - v) / tau   (first-order lag, tau = 0.5 s)
  - Controller:  PI controller written as a ScriptBlock
  - Setpoint:    step from 0 to 10 m/s at t = 0
  - Feedback:    loop broken by VectorDelay

Block diagram:

  [setpoint] ──► [error_sum] ──► [pi_ctrl] ──┬──► [plant_deriv] ──► [plant_int] ──► [speed_sink]
                     ▲                        │    (in0: u_ctrl)          │
                     │ (sign -1)              └──► [ctrl_sink]            │
                     │                             (in1: v_prev)          │
                     └──────────────── [fb_delay] ◄────────────────────────┘

  fb_delay — single block, two output connections:
    → error_sum   (sign -1, closes speed feedback)
    → plant_deriv (input 1, provides v_prev for dv/dt computation)

Both the PI controller and the plant derivative are ScriptBlocks, showing that
any computation can be expressed as user Python code and dropped into the diagram.

Run:
    python script_block_example.py
"""

import sys
import os
import numpy as np
import matplotlib

from embedsim import EmbedSim

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),  '..')))
# ─────────────────────────────────────────────────────────────────────────────

from embedsim.core_blocks                import VectorSignal
from embedsim.source_blocks              import VectorConstant
from embedsim.processing_blocks          import VectorSum
from embedsim.dynamic_blocks             import VectorIntegrator, VectorEnd
from embedsim.simulation_engine import (
    EmbedSim, VectorDelay, ODESolver
)
from embedsim.script_blocks              import ScriptBlock


# =============================================================================
# Parameters
# =============================================================================
V_REF  = 10.0    # m/s   setpoint speed
V_INIT =  0.0    # m/s   initial speed
TAU    =  0.5    # s     plant time constant
T_SIM  = 10.0    # s     simulation duration
DT     =  0.02   # s     time step

# PI gains — closed-loop bandwidth ≈ Kp/tau = 16 rad/s, fast settling
KP = 8.0
KI = 4.0


# =============================================================================
# ScriptBlock scripts
# =============================================================================

# PI controller
# u[0] is the numpy array from VectorSum (dim=1).
# 'integral' is declared as a parameter (= 0.0) so it is picked up on the
# first call; after that the framework keeps it alive via script_locals.
PI_SCRIPT = """
error     = u[0][0]
integral  = integral + error * dt
u_control = Kp * error + Ki * integral
u_control = float(np.clip(u_control, -100.0, 100.0))
output    = np.array([u_control])
"""

# First-order plant derivative  dv/dt = (u - v) / tau
PLANT_SCRIPT = """
u_ctrl = u[0][0]    # PI output (input 0)
v_prev = u[1][0]    # previous speed from VectorDelayEnhanced (input 1)
dv_dt  = (u_ctrl - v_prev) / tau
output = np.array([dv_dt])
"""


# =============================================================================
# Part 1 — Sanity check
# =============================================================================
print("=" * 60)
print("ControlForge  ScriptBlock Example")
print("=" * 60)
print("\n[1] ScriptBlock sanity check")

pi_check = ScriptBlock(
    name       = "pi_check",
    script     = PI_SCRIPT,
    parameters = {"Kp": KP, "Ki": KI, "integral": 0.0},
    output_dim = 1,
    mode       = "python",
)

# Two manual steps, error = 5.0 m/s, dt = 0.01 s
# integral starts at 0.0 (from parameters)
# Step 1:  integral → 0 + 5*0.01 = 0.05    u = Kp*5 + Ki*0.05
# Step 2:  integral → 0.05 + 5*0.01 = 0.10 u = Kp*5 + Ki*0.10
sig5 = VectorSignal([5.0])
out1 = pi_check.compute(t=0.00, dt=0.01, input_values=[sig5])
out2 = pi_check.compute(t=0.01, dt=0.01, input_values=[sig5])

exp1 = KP * 5.0 + KI * 0.05
exp2 = KP * 5.0 + KI * 0.10

print(f"  Step 1 — output={out1.value[0]:.6f}  expected={exp1:.6f}  "
      f"{'OK' if abs(out1.value[0] - exp1) < 1e-9 else 'MISMATCH'}")
print(f"  Step 2 — output={out2.value[0]:.6f}  expected={exp2:.6f}  "
      f"{'OK' if abs(out2.value[0] - exp2) < 1e-9 else 'MISMATCH'}")


# =============================================================================
# Part 2 — Build closed-loop block diagram
# =============================================================================
print("\n[2] Building closed-loop block diagram")

setpoint  = VectorConstant("setpoint", [V_REF])
fb_delay  = VectorDelay("fb_delay", initial=[V_INIT])

error_sum = VectorSum("error_sum", signs=[1, -1])
setpoint >> error_sum
fb_delay >> error_sum

pi_ctrl = ScriptBlock(
    name       = "pi_ctrl",
    script     = PI_SCRIPT,
    parameters = {"Kp": KP, "Ki": KI, "integral": 0.0},
    output_dim = 1,
    mode       = "python",
)
error_sum >> pi_ctrl

plant_deriv = ScriptBlock(
    name       = "plant_deriv",
    script     = PLANT_SCRIPT,
    parameters = {"tau": TAU},
    output_dim = 1,
    mode       = "python",
)
pi_ctrl  >> plant_deriv    # input 0: control signal
fb_delay >> plant_deriv    # input 1: previous speed

plant_int = VectorIntegrator("plant_int", initial_state=[V_INIT], dim=1)
plant_deriv >> plant_int

plant_int >> fb_delay      # close feedback loop through the delay

speed_sink = VectorEnd("speed")
ctrl_sink  = VectorEnd("control")
plant_int >> speed_sink
pi_ctrl   >> ctrl_sink


# =============================================================================
# Part 3 — Run simulation
# =============================================================================
print("\n[3] Running simulation")

sim = EmbedSim(
    sinks  = [speed_sink, ctrl_sink],
    T      = T_SIM,
    dt     = DT,
    solver = ODESolver.EULER,
)
sim.scope.add(plant_int,   label="Speed")
sim.scope.add(pi_ctrl,     label="Control")
sim.scope.add(plant_deriv, label="Derivative")
sim.scope.add(setpoint,    label="Setpoint")

sim.print_topology()
sim.print_topology_tree()
sim.print_topology_sources2sink()
sim.run(verbose=True, progress_bar=True)


# =============================================================================
# Metrics
# =============================================================================
t_arr  = np.array(sim.scope.t)
v_arr  = sim.scope.get_signal("Speed",      index=0)
u_arr  = sim.scope.get_signal("Control",    index=0)
dv_arr = sim.scope.get_signal("Derivative", index=0)
sp_arr = sim.scope.get_signal("Setpoint",   index=0)

final_v   = v_arr[-1]
err_arr   = sp_arr - v_arr
iae       = np.cumsum(np.abs(err_arr)) * DT
overshoot = max(0.0, (v_arr.max() - V_REF) / V_REF * 100.0)

y10 = V_INIT + 0.10 * (V_REF - V_INIT)
y90 = V_INIT + 0.90 * (V_REF - V_INIT)
t10 = next((t_arr[i] for i, v in enumerate(v_arr) if v >= y10), None)
t90 = next((t_arr[i] for i, v in enumerate(v_arr) if v >= y90), None)
rise_time = (t90 - t10) if (t10 is not None and t90 is not None) else None

print(f"\n  Final speed        : {final_v:.4f} m/s  (setpoint {V_REF} m/s)")
print(f"  Steady-state error : {abs(V_REF - final_v):.4f} m/s")
print(f"  Overshoot          : {overshoot:.2f} %")
if rise_time:
    print(f"  Rise time (10-90%) : {rise_time:.2f} s")
print(f"  Final IAE          : {iae[-1]:.3f} m/s·s")


# =============================================================================
# Part 4 — 4-panel dashboard
# =============================================================================
print("\n[4] Generating plot")

BG   = "#F8FAFC"
DARK = "#1E293B"
BLUE = "#2563EB"
RED  = "#DC2626"
GRN  = "#16A34A"
ORNG = "#EA580C"
GRAY = "#9CA3AF"
PURP = "#7C3AED"

fig = plt.figure(figsize=(14, 10), facecolor=BG)
gs  = gridspec.GridSpec(
    3, 2, figure=fig,
    hspace=0.52, wspace=0.34,
    left=0.08, right=0.97,
    top=0.93,  bottom=0.07,
)

# ── A: Speed response (full width) ───────────────────────────────────────────
ax_sp = fig.add_subplot(gs[0, :])
ax_sp.set_facecolor("white")
ax_sp.plot(t_arr, sp_arr, color=GRAY, lw=1.5, ls="--", label="Setpoint", zorder=2)
ax_sp.plot(t_arr, v_arr,  color=BLUE, lw=2.2,           label="Speed",    zorder=3)

for ymark, lbl, col in [(y10, "10 %", GRN), (y90, "90 %", ORNG)]:
    ax_sp.axhline(ymark, color=col, lw=0.9, ls=":", alpha=0.8)
    ax_sp.text(t_arr[-1] * 1.003, ymark, lbl, va="center", fontsize=8, color=col)

above = v_arr > V_REF
if above.any():
    ax_sp.fill_between(t_arr, V_REF, v_arr, where=above,
                       color=RED, alpha=0.15,
                       label=f"Overshoot  {overshoot:.1f} %")

if rise_time is not None:
    ax_sp.annotate(
        f"Rise time\n{rise_time:.2f} s",
        xy=(t90, y90),
        xytext=(t90 + 0.5, y90 - 2.5),
        arrowprops=dict(arrowstyle="->", color=ORNG, lw=1.2),
        fontsize=8.5, color=ORNG,
    )

ax_sp.set_ylabel("Speed  [m/s]", fontsize=10)
ax_sp.set_title(
    f"ControlForge — Closed-Loop Speed Control  "
    f"(PI ScriptBlock   Kp={KP}  Ki={KI}  tau={TAU} s)",
    fontsize=11, fontweight="bold",
)
ax_sp.legend(fontsize=9, loc="lower right")
ax_sp.grid(True, alpha=0.2)
ax_sp.set_xlim(t_arr[0], t_arr[-1])
ax_sp.set_ylim(-0.5, V_REF * 1.20)

# ── B: Control signal ─────────────────────────────────────────────────────────
ax_u = fig.add_subplot(gs[1, 0])
ax_u.set_facecolor("white")
ax_u.plot(t_arr, u_arr, color=RED, lw=1.8, label="PI output")
ax_u.fill_between(t_arr, 0, u_arr, color=RED, alpha=0.10)
ax_u.axhline(0, color=GRAY, lw=0.8, ls="--", alpha=0.6)
ax_u.set_xlabel("Time  [s]", fontsize=9)
ax_u.set_ylabel("Control  u  [m/s²]", fontsize=9)
ax_u.set_title("B — PI Control Signal  (ScriptBlock)", fontsize=9, fontweight="bold")
ax_u.legend(fontsize=8)
ax_u.grid(True, alpha=0.2)
ax_u.set_xlim(t_arr[0], t_arr[-1])

# ── C: Plant derivative dv/dt ─────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_facecolor("white")
if dv_arr is not None:
    ax_d.plot(t_arr, dv_arr, color=PURP, lw=1.8, label="dv/dt")
    ax_d.fill_between(t_arr, 0, dv_arr, where=(dv_arr >= 0), color=PURP, alpha=0.10)
    ax_d.fill_between(t_arr, 0, dv_arr, where=(dv_arr  < 0), color=RED,  alpha=0.10)
    ax_d.axhline(0, color=GRAY, lw=0.8, ls="--", alpha=0.6)
ax_d.set_xlabel("Time  [s]", fontsize=9)
ax_d.set_ylabel("dv/dt  [m/s²]", fontsize=9)
ax_d.set_title("C — Plant Derivative  (ScriptBlock)", fontsize=9, fontweight="bold")
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.2)
ax_d.set_xlim(t_arr[0], t_arr[-1])

# ── D: IAE ────────────────────────────────────────────────────────────────────
ax_iae = fig.add_subplot(gs[2, 0])
ax_iae.set_facecolor("white")
ax_iae.plot(t_arr, iae, color=GRN, lw=1.8, label="IAE")
ax_iae.fill_between(t_arr, 0, iae, color=GRN, alpha=0.12)
ax_iae.set_xlabel("Time  [s]", fontsize=9)
ax_iae.set_ylabel("IAE  [m/s · s]", fontsize=9)
ax_iae.set_title("D — Integral Absolute Error", fontsize=9, fontweight="bold")
ax_iae.legend(fontsize=8)
ax_iae.grid(True, alpha=0.2)
ax_iae.set_xlim(t_arr[0], t_arr[-1])
ax_iae.text(t_arr[-1] * 0.97, iae[-1],
            f"Final IAE\n{iae[-1]:.2f} m/s·s",
            ha="right", va="top", fontsize=8, color=GRN)

# ── E: Performance card ───────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[2, 1])
ax_c.set_facecolor(DARK)
ax_c.set_axis_off()

rise_str = f"{rise_time:.2f} s" if rise_time is not None else "n/a"
over_str = f"{overshoot:.2f} %" if overshoot > 0.01 else "none"

card = [
    ("ControlForge  Performance",               0.95, 10.5, "#94A3B8", "bold"),
    ("=" * 34,                                  0.89,  8.0, "#334155", "normal"),
    ("PLANT  (ScriptBlock)",                    0.83,  9.5, "#60A5FA", "bold"),
    ("  dv/dt = (u - v) / tau",                 0.77,  9.0, "#E2E8F0", "normal"),
    (f"  tau = {TAU} s",                        0.71,  9.0, "#E2E8F0", "normal"),
    ("=" * 34,                                  0.65,  8.0, "#334155", "normal"),
    ("PI CONTROLLER  (ScriptBlock)",            0.59,  9.5, "#34D399", "bold"),
    (f"  Kp = {KP}   Ki = {KI}",               0.53,  9.0, "#E2E8F0", "normal"),
    ("  anti-windup clip  [-100, 100]",         0.47,  9.0, "#E2E8F0", "normal"),
    ("=" * 34,                                  0.41,  8.0, "#334155", "normal"),
    ("PERFORMANCE",                             0.35,  9.5, "#FBBF24", "bold"),
    (f"  Setpoint      {V_REF:.1f} m/s",        0.29,  9.0, "#E2E8F0", "normal"),
    (f"  Final speed   {final_v:.4f} m/s",      0.23,  9.0, "#E2E8F0", "normal"),
    (f"  Steady error  {abs(V_REF-final_v):.4f} m/s", 0.17, 9.0, "#E2E8F0", "normal"),
    (f"  Rise time     {rise_str}",             0.11,  9.0, "#E2E8F0", "normal"),
    (f"  Overshoot     {over_str}",             0.05,  9.0, "#E2E8F0", "normal"),
]

for text, y, fsize, col, weight in card:
    ax_c.text(0.04, y, text,
              transform=ax_c.transAxes,
              fontsize=fsize, color=col,
              fontweight=weight, fontfamily="monospace", va="top")

# ── save ──────────────────────────────────────────────────────────────────────
# Save in current directory
plot_path = os.path.join(os.getcwd(), "scriptblock_example.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

# Show the plot
plt.show()  # <-- this will display the plot in a window or notebook

# Close the figure
plt.close(fig)

print(f"Plot saved → {plot_path}")
print("\n" + "=" * 60 + "\nDone.\n")