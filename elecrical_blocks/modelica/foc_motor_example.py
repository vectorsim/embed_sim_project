"""
foc_motor_example.py
====================
Field-Oriented Control (FOC) example using the ThreePhaseMotor FMU
and the MotorFMUClient wrapper.

CONTROL STRATEGY: id = 0 (Maximum Torque Per Ampere for SPM)
--------------------------------------------------------------
  - d-axis current reference  : id_ref = 0  (no flux weakening)
  - q-axis current reference  : iq_ref = T_ref / (1.5 * p * lambda_pm)
  - Two independent PI controllers close the current loops
  - Outer speed PI loop generates the torque (iq) reference
  - Clarke/Park transforms omitted here — we operate directly in dq frame
    since the FMU already accepts v_d / v_q commands

BLOCK DIAGRAM
-------------

  speed_ref ──►[Speed PI]──► iq_ref
                                │
  id_ref = 0 ──────────────────►[d-PI]──► v_d ──►┐
  iq_ref ──────────────────────►[q-PI]──► v_q ──►├──► PMSM FMU
                                                   │
  omega_m, i_d, i_q, theta_e ◄────────────────────┘

USAGE
-----
  python foc_motor_example.py

  Produces a matplotlib plot of:
    - Speed tracking (reference vs actual)
    - d/q axis currents
    - Electromagnetic torque
    - Phase voltage commands
"""

import numpy as np
import matplotlib.pyplot as plt
from motor_fmu_client import MotorFMUClient


# =============================================================================
# PI Controller (anti-windup via clamping)
# =============================================================================

class PIController:
    """Simple PI controller with anti-windup output clamping."""

    def __init__(
        self,
        Kp: float,
        Ki: float,
        output_min: float = -np.inf,
        output_max: float = np.inf,
        name: str = "PI",
    ):
        self.Kp  = Kp
        self.Ki  = Ki
        self.out_min = output_min
        self.out_max = output_max
        self.name    = name
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def update(self, error: float, dt: float) -> float:
        # Proportional
        p_term = self.Kp * error

        # Integral with clamping anti-windup
        self._integral += self.Ki * error * dt

        raw = p_term + self._integral

        # Clamp output
        clamped = np.clip(raw, self.out_min, self.out_max)

        # Back-calculation anti-windup: desaturate integral if clamped
        if raw != clamped:
            self._integral -= (raw - clamped)

        return clamped


# =============================================================================
# FOC Simulation
# =============================================================================

def run_foc_simulation(fmu_path: str = "ThreePhaseMotor.fmu"):
    """
    Run a complete FOC simulation using the ThreePhaseMotor FMU.

    Profile:
      0.0–0.5 s  : Ramp to 1000 RPM, no load
      0.5–1.0 s  : Hold 1000 RPM, apply 3 N.m load step at t=0.6 s
      1.0–1.5 s  : Ramp to 1500 RPM
      1.5–2.0 s  : Hold 1500 RPM, load removed at t=1.8 s
    """

    # ------------------------------------------------------------------
    # Motor parameters (must match the .mo model)
    # ------------------------------------------------------------------
    R         = 0.5
    L_d       = 0.005
    L_q       = 0.006
    lambda_pm = 0.175
    J         = 0.002
    B         = 0.001
    p         = 2          # pole pairs

    # ------------------------------------------------------------------
    # Simulation settings
    # ------------------------------------------------------------------
    dt      = 1e-4         # 100 µs step (matches PWM carrier)
    t_stop  = 2.0
    N_steps = int(t_stop / dt)

    # ------------------------------------------------------------------
    # FOC Controllers
    # ------------------------------------------------------------------
    V_DC    = 48.0         # DC bus voltage
    V_LIMIT = V_DC / np.sqrt(3.0)   # max phase voltage

    # Current controllers – fast inner loop
    pi_id = PIController(Kp=10.0,  Ki=500.0,  output_min=-V_LIMIT, output_max=V_LIMIT, name="id-PI")
    pi_iq = PIController(Kp=10.0,  Ki=500.0,  output_min=-V_LIMIT, output_max=V_LIMIT, name="iq-PI")

    # Speed controller – outer loop (slower)
    I_MAX  = 15.0          # peak current limit [A]
    pi_spd = PIController(Kp=0.05, Ki=1.0,    output_min=-I_MAX,   output_max=I_MAX,   name="speed-PI")

    # Torque constant
    Kt = 1.5 * p * lambda_pm  # [N.m / A]

    # ------------------------------------------------------------------
    # Speed reference profile
    # ------------------------------------------------------------------
    def speed_reference(t: float) -> float:
        """Trapezoidal speed profile [rad/s]."""
        rpm1 = 1000.0
        rpm2 = 1500.0
        w1   = rpm1 * 2 * np.pi / 60.0
        w2   = rpm2 * 2 * np.pi / 60.0

        if t < 0.5:
            return w1 * (t / 0.5)           # ramp 0→1000 RPM
        elif t < 1.0:
            return w1                        # hold 1000 RPM
        elif t < 1.5:
            return w1 + (w2 - w1) * ((t - 1.0) / 0.5)  # ramp to 1500
        else:
            return w2                        # hold 1500 RPM

    def load_torque(t: float) -> float:
        """Step load profile [N.m]."""
        if 0.6 <= t < 1.8:
            return 3.0
        return 0.0

    # ------------------------------------------------------------------
    # Data logging
    # ------------------------------------------------------------------
    log_t       = np.zeros(N_steps)
    log_omega   = np.zeros(N_steps)
    log_omega_r = np.zeros(N_steps)
    log_id      = np.zeros(N_steps)
    log_iq      = np.zeros(N_steps)
    log_vd      = np.zeros(N_steps)
    log_vq      = np.zeros(N_steps)
    log_Tem     = np.zeros(N_steps)
    log_Tload   = np.zeros(N_steps)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  FOC Motor Simulation | FMU: {fmu_path}")
    print(f"  dt={dt*1e3:.2f} ms | t_stop={t_stop} s | {N_steps} steps")
    print(f"{'='*60}\n")

    with MotorFMUClient(
        fmu_path=fmu_path,
        dt=dt,
        t_start=0.0,
        t_stop=t_stop,
    ) as motor:

        motor.initialize(parameters={
            "R": R, "L_d": L_d, "L_q": L_q,
            "lambda_pm": lambda_pm, "J": J, "B": B, "p": float(p)
        })

        for k in range(N_steps):
            t = motor.time

            # Current state from last step output
            omega_m = motor.read_output("omega_m") if k > 0 else 0.0
            i_d     = motor.read_output("i_d")     if k > 0 else 0.0
            i_q     = motor.read_output("i_q")     if k > 0 else 0.0

            # ── Outer speed loop ──────────────────────────────────────
            omega_ref = speed_reference(t)
            speed_err = omega_ref - omega_m
            iq_ref    = pi_spd.update(speed_err, dt)
            id_ref    = 0.0                  # id=0 MTPA strategy

            # ── Inner d-axis current loop ─────────────────────────────
            id_err = id_ref - i_d
            v_d    = pi_id.update(id_err, dt)

            # ── Inner q-axis current loop ─────────────────────────────
            iq_err = iq_ref - i_q
            v_q    = pi_iq.update(iq_err, dt)

            # ── Feed-forward decoupling (optional but improves response)
            omega_e = p * omega_m
            v_d    -= omega_e * L_q * i_q          # d-axis decoupling
            v_q    += omega_e * (L_d * i_d + lambda_pm)  # q-axis decoupling

            # Clamp final voltages to inverter limit
            v_d = np.clip(v_d, -V_LIMIT, V_LIMIT)
            v_q = np.clip(v_q, -V_LIMIT, V_LIMIT)

            # ── Step FMU ─────────────────────────────────────────────
            T_load  = load_torque(t)
            outputs = motor.step(v_d=v_d, v_q=v_q, T_load=T_load)

            # ── Log ──────────────────────────────────────────────────
            log_t[k]       = t
            log_omega[k]   = outputs["omega_m"]
            log_omega_r[k] = omega_ref
            log_id[k]      = outputs["i_d"]
            log_iq[k]      = outputs["i_q"]
            log_vd[k]      = v_d
            log_vq[k]      = v_q
            log_Tem[k]     = outputs["T_em"]
            log_Tload[k]   = T_load

            if k % 5000 == 0:
                rpm = outputs["speed_rpm"]
                print(f"  t={t:.3f}s | {rpm:7.1f} RPM | "
                      f"id={outputs['i_d']:6.2f} A | iq={outputs['i_q']:6.2f} A | "
                      f"Tem={outputs['T_em']:5.2f} N.m | Tload={T_load:.1f} N.m")

    print(f"\n  Simulation complete.\n{'='*60}\n")

    # ------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("FOC PMSM Motor Simulation — EmbedSim / ThreePhaseMotor FMU",
                 fontsize=13, fontweight="bold")

    # 1. Speed
    ax = axes[0]
    ax.plot(log_t, log_omega * 60 / (2 * np.pi), "b-",  lw=1.5, label="Actual [RPM]")
    ax.plot(log_t, log_omega_r * 60 / (2 * np.pi), "r--", lw=1.2, label="Reference [RPM]")
    ax.set_ylabel("Speed [RPM]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Shaft Speed Tracking")

    # 2. dq Currents
    ax = axes[1]
    ax.plot(log_t, log_id, "g-",  lw=1.2, label="i_d [A]")
    ax.plot(log_t, log_iq, "m-",  lw=1.2, label="i_q [A]")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Current [A]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("d-q Axis Currents (FOC id=0)")

    # 3. Torque
    ax = axes[2]
    ax.plot(log_t, log_Tem,   "b-",  lw=1.2, label="T_em [N.m]")
    ax.plot(log_t, log_Tload, "r--", lw=1.2, label="T_load [N.m]")
    ax.set_ylabel("Torque [N.m]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Electromagnetic vs Load Torque")

    # 4. Voltage commands
    ax = axes[3]
    ax.plot(log_t, log_vd, "c-",  lw=1.2, label="v_d [V]")
    ax.plot(log_t, log_vq, "y-",  lw=1.2, label="v_q [V]")
    ax.axhline( V_LIMIT, color="gray", lw=0.8, ls=":", label=f"±{V_LIMIT:.1f} V limit")
    ax.axhline(-V_LIMIT, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Voltage [V]")
    ax.set_xlabel("Time [s]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("dq Voltage Commands")

    plt.tight_layout()
    plt.savefig("foc_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Plot saved → foc_results.png")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import sys

    fmu_path = sys.argv[1] if len(sys.argv) > 1 else "ThreePhaseMotor.fmu"

    if not __import__("os").path.exists(fmu_path):
        print(f"""
[ERROR] FMU not found: '{fmu_path}'

To generate the FMU from OpenModelica:
  1. Open OMEdit and load ThreePhaseMotor.mo
  2. Right-click model → Export → FMU (Co-Simulation, FMI 2.0)
     OR run in omc:
       loadFile("ThreePhaseMotor.mo");
       buildModelFMU(ThreePhaseMotor, version="2.0", fmuType="cs");
  3. Copy ThreePhaseMotor.fmu to this directory and re-run.
""")
        sys.exit(1)

    run_foc_simulation(fmu_path)
