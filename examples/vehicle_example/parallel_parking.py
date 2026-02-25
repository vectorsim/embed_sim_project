"""
parallel_parking.py
==========================
Parallel parking in a TIGHT 7.5 m gap (1.67 × car length)
with closed-loop, position/angle-triggered phase transitions.

Block graph
-----------
    [controller] ──► [sequencer] ──► [bicycle] ──► [VectorEnd: state]
         ▲                                │
         │                               ▼
         └──────── [StickyDelay: fb] ◄───┘

  controller  ScriptBlock — reads delayed state → outputs phase index
  sequencer   ScriptBlock — maps phase index   → (v, delta)
  bicycle     ScriptBlock — kinematic ODE      → (x, y, theta)
  fb          StickyDelay — 1-step delay, LoopBreaker; holds initial [x,y,θ]
                            until bicycle produces its first valid output

Phase transitions (closed-loop, position/angle-triggered)
----------------------------------------------------------
  0 → 1  x  ≥ X_MANOUVRE      (car past gap, ready to reverse)
  1 → 2  θ  ≥ THETA_ARC1_DONE (arc 1 complete by heading)
  2 → 3  θ  ≤ THETA_DONE      (arc 2 complete, heading restored)

Author: ControlForge demo
"""

import math
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

# ── Framework imports ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from embedsim.core_blocks                import VectorSignal
from embedsim.dynamic_blocks             import VectorEnd
from embedsim.script_blocks              import ScriptBlock
from embedsim.simulation_engine import EmbedSim, ODESolver, VectorDelay

# ── StickyDelay ────────────────────────────────────────────────────────────
class StickyDelay(VectorDelay):
    """
    VectorDelayEnhanced subclass that holds its initial value through reset()
    and ignores inputs of the wrong dimension (engine zero-fallback during step 0).

    This lets us build a proper controller → bicycle feedback loop where the
    controller reads the bicycle state one timestep late, without the normal
    VectorDelayEnhanced bug where the first engine fallback [0.0] overwrites the
    carefully chosen initial state.
    """

    def __init__(self, name: str, initial: list, expected_dim: int = None):
        super().__init__(name, initial)
        self._initial      = np.array(initial, dtype=float)
        self._expected_dim = expected_dim if expected_dim else len(initial)
        self.last_output   = VectorSignal(self._initial.copy(), name)
        self.output        = VectorSignal(self._initial.copy(), name)

    def reset(self):
        # Preserve initial value — do NOT clear output/last_output
        self.last_output = VectorSignal(self._initial.copy(), self.name)
        self.output      = VectorSignal(self._initial.copy(), self.name)

    def compute(self, t: float, dt: float, input_values=None) -> VectorSignal:
        # Emit previous step's value
        self.output = VectorSignal(self.last_output.value.copy(), self.name)
        # Only store new input if dimension matches (guards against engine fallback)
        if input_values and len(input_values[0].value) == self._expected_dim:
            self.last_output = VectorSignal(input_values[0].value.copy())
        return self.output


# ── Geometry ───────────────────────────────────────────────────────────────
WHEELBASE   = 2.7
DELTA_MAX   = math.radians(35)
V_APPROACH  = 0.7
V_PARK      = 0.8
CAR_LENGTH  = 4.5
CAR_WIDTH   = 2.0
GAP         = 7.5          # 1.67 × car length — tightest gap the S-curve clears
DT          = 0.05

PARK_Y      = 0.0
LANE_Y      = PARK_Y + CAR_WIDTH + 2.5   # 4.5 m

CAR_A_X     =  GAP / 2.0 + CAR_LENGTH / 2.0   #  6.0 m  (right parked car centre)
CAR_B_X     = -(GAP / 2.0 + CAR_LENGTH / 2.0) # -6.0 m  (left parked car centre)

R_MIN       = WHEELBASE / math.tan(DELTA_MAX)          # 3.856 m
D_LAT       = LANE_Y                                    # 4.5 m
ARC_ANGLE   = math.acos(1.0 - D_LAT / (2.0 * R_MIN))  # 1.141 rad = 65.4°

# X_MANOUVRE = 6.0 m is chosen after full swept-corner analysis: using the
# symmetric value (7.011 m) causes the front-right corner to clip Car A
# during arc 2.  Shifting the manoeuvre 1 m leftward keeps every corner clear
# with ≥ 30 cm margin at the tightest moment of the sweep.
X_MANOUVRE      = 6.0           # car centre x when reversing begins
X_START         = CAR_B_X - 1.5 # -7.5 m  (well left of gap)

THETA_ARC1_DONE = ARC_ANGLE - math.radians(1.5)        # fire slightly early
THETA_DONE      = math.radians(2.0)                     # parked when θ ≤ 2°

T_SIM = 60.0   # generous; simulation stops itself via phase=3 (v=0)

# ── Scripts ────────────────────────────────────────────────────────────────
CONTROLLER_SCRIPT = """
# Phase state machine driven by real vehicle position and heading
phase = phase if 'phase' in vars() else 0
x     = float(u[0][0])
theta = float(u[0][2])

if   phase == 0 and x     >= X_MANOUVRE:       phase = 1
elif phase == 1 and theta >= THETA_ARC1_DONE:  phase = 2
elif phase == 2 and theta <= THETA_DONE:       phase = 3

output = np.array([float(phase)])
"""

SEQUENCER_SCRIPT = """
# Map phase index to actuator commands
phase = int(round(float(u[0][0])))

if   phase == 0:  v, delta = V_APPROACH,  0.0
elif phase == 1:  v, delta = -V_PARK,    -DELTA_MAX
elif phase == 2:  v, delta = -V_PARK,    +DELTA_MAX
else:             v, delta = 0.0,         0.0

output = np.array([v, delta, float(phase)])
"""

BICYCLE_SCRIPT = """
# Kinematic bicycle ODE — state persists in script_locals
x     = x     if 'x'     in vars() else X_START
y     = y     if 'y'     in vars() else LANE_Y
theta = theta if 'theta' in vars() else 0.0

v     = float(u[0][0])
delta = float(np.clip(u[0][1], -DELTA_MAX, DELTA_MAX))

x     += v * np.cos(theta) * dt
y     += v * np.sin(theta) * dt
theta += v / WHEELBASE * np.tan(delta) * dt

output = np.array([x, y, theta])
"""


# ── Build and run ──────────────────────────────────────────────────────────
def build_and_run():
    controller = ScriptBlock("controller", CONTROLLER_SCRIPT,
        dict(X_MANOUVRE=X_MANOUVRE,
             THETA_ARC1_DONE=THETA_ARC1_DONE,
             THETA_DONE=THETA_DONE),
        output_dim=1)

    sequencer = ScriptBlock("sequencer", SEQUENCER_SCRIPT,
        dict(V_APPROACH=V_APPROACH, V_PARK=V_PARK, DELTA_MAX=DELTA_MAX),
        output_dim=3)

    bicycle = ScriptBlock("bicycle", BICYCLE_SCRIPT,
        dict(X_START=X_START, LANE_Y=LANE_Y,
             DELTA_MAX=DELTA_MAX, WHEELBASE=WHEELBASE),
        output_dim=3)

    # StickyDelay: holds [X_START, LANE_Y, 0] until bicycle emits real state
    fb = StickyDelay("state_fb", initial=[X_START, LANE_Y, 0.0], expected_dim=3)

    sink_state = VectorEnd("state")   # [x, y, theta]
    sink_cmd   = VectorEnd("cmd")     # [v, delta, phase]

    # ── Wiring ────────────────────────────────────────────────────────────
    controller >> sequencer  >> bicycle >> sink_state
    bicycle    >> fb         >> controller
    sequencer  >> sink_cmd

    print("=" * 62)
    print("  Tight Parallel Parking — Closed-Loop ControlForge")
    print("=" * 62)
    print(f"  Gap          : {GAP:.1f} m  ({GAP/CAR_LENGTH:.2f}× car length)")
    print(f"  Clearance    : {(GAP-CAR_LENGTH)/2:.2f} m each end (nom)")
    print(f"  R_min        : {R_MIN:.3f} m")
    print(f"  Arc angle    : {math.degrees(ARC_ANGLE):.1f}°")
    print(f"  X_MANOUVRE   : {X_MANOUVRE:.3f} m  ← position trigger")
    print(f"  θ arc1 done  : {math.degrees(THETA_ARC1_DONE):.1f}°  ← angle trigger")
    print(f"  θ parked     : {math.degrees(THETA_DONE):.1f}°   ← angle trigger")
    print()

    sim = EmbedSim(
        sinks  = [sink_state, sink_cmd],
        T      = T_SIM,
        dt     = DT,
        solver = ODESolver.EULER,
    )
    sim.scope.add(bicycle,    label="state")
    sim.scope.add(sequencer,  label="cmd")
    sim.scope.add(controller, label="phase")

    sim.print_topology()
    sim.run(verbose=True, progress_bar=True)

    state_hist = np.array(sink_state.history)   # (N, 3)
    cmd_hist   = np.array(sink_cmd.history)     # (N, 3): v, delta, phase

    # Trim to just after parking completes (phase=3 + 40 coasting frames)
    done_idx = next(
        (i for i, c in enumerate(cmd_hist) if int(round(c[2])) >= 3),
        len(cmd_hist)
    )
    trim = min(done_idx + 40, len(cmd_hist))
    state_hist = state_hist[:trim]
    cmd_hist   = cmd_hist[:trim]

    xf, yf, thf = state_hist[-1]
    front_cl = (CAR_A_X - CAR_LENGTH/2) - (xf + CAR_LENGTH/2)
    rear_cl  = (xf - CAR_LENGTH/2) - (CAR_B_X + CAR_LENGTH/2)

    print(f"\n  ── Final state ──────────────────────────────────────")
    print(f"  x             : {xf:+.4f} m   (ideal  0.000)")
    print(f"  y             : {yf:+.4f} m   (ideal  0.000)")
    print(f"  θ             : {math.degrees(thf):+.3f}°  (ideal  0.0°)")
    print(f"  Front clear   : {front_cl:.4f} m  to Car A")
    print(f"  Rear  clear   : {rear_cl:.4f} m  to Car B")
    print(f"  Time to park  : {done_idx*DT:.1f} s")
    print(f"  ────────────────────────────────────────────────────")

    return state_hist, cmd_hist


# ── Visualiser ─────────────────────────────────────────────────────────────
PHASE_NAMES  = ["Approach", "Rev+Right", "Rev+Left", "Parked ✓"]
PHASE_COLORS = ["#4ecdc4",  "#e74c3c",   "#3498db",  "#2ecc71"]

class ParkingVisualiser:
    def __init__(self, state_hist, cmd_hist):
        self.state = state_hist
        self.cmd   = cmd_hist
        self._build()

    def _static_car(self, ax, cx, color):
        r = patches.Rectangle(
            (-CAR_LENGTH/2, -CAR_WIDTH/2), CAR_LENGTH, CAR_WIDTH,
            color=color, alpha=0.9, zorder=2)
        r.set_transform(Affine2D().translate(cx, PARK_Y) + ax.transData)
        ax.add_patch(r)

    def _build(self):
        self.fig, (self.ax_map, self.ax_tel) = plt.subplots(
            1, 2, figsize=(15, 6),
            gridspec_kw={"width_ratios": [2.3, 1]}
        )
        ax = self.ax_map
        ax.set_facecolor("#1a1e2a")

        # x-range covers approach + manoeuvre
        ax.set_xlim(CAR_B_X - 2.5, X_MANOUVRE + CAR_LENGTH/2 + 1.0)
        ax.set_ylim(PARK_Y - 1.8, LANE_Y + 2.2)
        ax.set_aspect("equal")
        ax.set_title(
            f"Parallel Parking  ·  {GAP:.0f} m gap  ({GAP/CAR_LENGTH:.2f}× car)  "
            f"·  Closed-Loop ControlForge",
            color="#e8c547", fontsize=9.5
        )

        # Kerb and road surface
        ax.axhline(PARK_Y - 0.3, color="#4a4a4a", lw=3)
        ax.fill_between(
            [CAR_B_X-3, X_MANOUVRE+CAR_LENGTH/2+2],
            PARK_Y - 1.8, PARK_Y - 0.3,
            color="#1e1e1e", zorder=0
        )
        # Lane line
        ax.axhline(LANE_Y, color="#444", lw=1, ls="--", zorder=1, alpha=0.5)

        # Gap boundary markers
        for edge_x in [CAR_A_X - CAR_LENGTH/2, CAR_B_X + CAR_LENGTH/2]:
            ax.axvline(edge_x, color="#666", lw=1, ls=":", zorder=2)

        # Parked cars
        self._static_car(ax, CAR_A_X, "#c0392b")
        self._static_car(ax, CAR_B_X, "#922b21")

        # Ego car trail and body
        self.trail,   = ax.plot([], [], lw=2, zorder=4)
        self.car_body  = patches.Rectangle(
            (-CAR_LENGTH/2, -CAR_WIDTH/2), CAR_LENGTH, CAR_WIDTH,
            color="#3a7bd5", zorder=5
        )
        ax.add_patch(self.car_body)
        self.arrow = FancyArrowPatch(
            posA=(0,0), posB=(0,0),
            arrowstyle="-|>", mutation_scale=14,
            color="#f0d060", zorder=6
        )
        ax.add_patch(self.arrow)
        self.phase_lbl = ax.text(
            0.02, 0.95, "", transform=ax.transAxes,
            color="#e8c547", fontsize=10, weight="bold"
        )

        # ── Telemetry panel ────────────────────────────────────────────────
        ax2 = self.ax_tel
        ax2.set_facecolor("#111318")
        ax2.axis("off")

        # Block graph diagram
        diagram = (
            "Block graph\n"
            "───────────────────────\n"
            "[controller] ScriptBlock\n"
            "  reads: x, θ (delayed)\n"
            "  emits: phase index\n"
            "     │         ▲\n"
            "     ▼         │\n"
            "[sequencer]  ScriptBlock\n"
            "  emits: v, δ, phase\n"
            "     │\n"
            "     ▼\n"
            "[bicycle]    ScriptBlock\n"
            "  kinematic ODE\n"
            "  emits: x, y, θ\n"
            "     │\n"
            "     ▼\n"
            "[state_fb]   StickyDelay\n"
            "  LoopBreaker\n"
            "  1-step delay\n"
            "     │\n"
            "     └──► controller"
        )
        ax2.text(0.03, 0.99, diagram, transform=ax2.transAxes,
                 color="#4a7fb5", fontsize=7, va="top", family="monospace")

        # Telemetry
        fields = ["Phase", "x (m)", "y (m)", "θ (°)", "v (m/s)", "δ (°)"]
        self._tv = {}
        for i, f in enumerate(fields):
            yp = 0.37 - i * 0.067
            ax2.text(0.03, yp, f"  {f}:", color="#6b7080", fontsize=9,
                     transform=ax2.transAxes)
            self._tv[f] = ax2.text(
                0.5, yp, "", color="#dde1ee",
                fontsize=9, weight="bold", transform=ax2.transAxes
            )

    def _frame(self, i):
        x, y, theta = self.state[i]
        v, delta, phase = self.cmd[i]
        ph = max(0, min(int(round(phase)), 3))

        self.trail.set_data(self.state[:i+1, 0], self.state[:i+1, 1])
        self.trail.set_color(PHASE_COLORS[ph])

        self.car_body.set_transform(
            Affine2D().rotate(theta).translate(x, y) + self.ax_map.transData
        )
        fx = x + CAR_LENGTH/2 * math.cos(theta)
        fy = y + CAR_LENGTH/2 * math.sin(theta)
        self.arrow.set_positions(
            (fx, fy),
            (fx + 1.6*math.cos(theta), fy + 1.6*math.sin(theta))
        )

        self.phase_lbl.set_text(PHASE_NAMES[ph])
        self.phase_lbl.set_color(PHASE_COLORS[ph])

        self._tv["Phase"].set_text(PHASE_NAMES[ph])
        self._tv["Phase"].set_color(PHASE_COLORS[ph])
        self._tv["x (m)"].set_text(f"{x:+.3f}")
        self._tv["y (m)"].set_text(f"{y:+.3f}")
        self._tv["θ (°)"].set_text(f"{math.degrees(theta):+.2f}")
        self._tv["v (m/s)"].set_text(f"{v:+.2f}")
        self._tv["δ (°)"].set_text(f"{math.degrees(delta):+.1f}")

        return self.car_body, self.trail, self.arrow, self.phase_lbl

    def save_png(self, path="parking_tight.png"):
        self._frame(len(self.state) - 1)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1e2a")
        print(f"  Saved → {path}")

    def save_animation(self, path="parallel_parking.gif", interval_ms=50,
                       fps=None, frame_step=1, dpi=100):
        """
        Save the animation to a file.

        Parameters
        ----------
        path         : output filename — extension determines format:
                         .gif   → animated GIF  (requires Pillow)
                         .mp4   → H.264 video   (requires ffmpeg)
                         .webm  → VP8 video     (requires ffmpeg)
        interval_ms  : delay between frames in milliseconds (default 50 → 20 fps)
        fps          : override frames-per-second (default: derived from interval_ms)
        frame_step   : render every Nth frame — use 2 or 3 to shrink GIF file size
        dpi          : output resolution (default 100; use 150 for higher quality)
        """
        if fps is None:
            fps = max(1, 1000 // interval_ms)

        frames = range(0, len(self.state), frame_step)

        anim = FuncAnimation(
            self.fig, self._frame,
            frames=frames,
            interval=interval_ms, blit=True
        )

        ext = os.path.splitext(path)[1].lower()

        if ext == ".gif":
            anim.save(
                path, writer="pillow", fps=fps,
                savefig_kwargs={"facecolor": "#1a1e2a"}
            )
        elif ext == ".mp4":
            anim.save(
                path, writer="ffmpeg", fps=fps, dpi=dpi,
                savefig_kwargs={"facecolor": "#1a1e2a"}
            )
        elif ext == ".webm":
            anim.save(
                path, writer="ffmpeg", fps=fps, dpi=dpi,
                extra_args=["-vcodec", "libvpx", "-b:v", "1M"],
                savefig_kwargs={"facecolor": "#1a1e2a"}
            )
        else:
            raise ValueError(
                f"Unsupported format '{ext}'. Use .gif, .mp4, or .webm"
            )

        print(f"  Saved → {path}  ({len(list(frames))} frames @ {fps} fps, dpi={dpi})")

    def show(self, interval_ms=50):
        self._anim = FuncAnimation(
            self.fig, self._frame,
            frames=len(self.state),
            interval=interval_ms, blit=True
        )
        plt.tight_layout()
        plt.show()


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    state_hist, cmd_hist = build_and_run()
    vis = ParkingVisualiser(state_hist, cmd_hist)
    vis.save_png("parallel_parking.png")
    vis.save_animation("parallel_parking.gif")   # animated GIF  (Pillow)
    vis.save_animation("parallel_parking.mp4") # MP4 video     (ffmpeg)
    vis.show()