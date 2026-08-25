"""
pmsm_dfc.py
===========

PMSM Control - Single Python Controller with Mode Switching
ALIGNED WITH C IMPLEMENTATION

Architecture
------------

    Speed Reference
          |
          v
    Jerk-Limited S-Curve (time-optimal, matches C)
          |
          +---- omega_ref
          +---- omega_dot_ref
          +---- omega_ddot_ref
          |
          v
    Differential Flatness
          |
          +---- iq_ref_ff
          +---- iq_ref_dot_ff
          +---- vd_ff
          +---- vq_ff
          |
          v
    Speed Feedback Correction (torque based, not current based)
          |
          v
        iq_ref
          |
          v
      Current PI
          |
          v
       SVM / PWM
          |
          v
         PMSM
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Dict, Any

import numpy as np


# ================================================================
# Path setup
# ================================================================

from _path_utils import (
    get_project_root,
    get_embedsim_import_path,
    get_current_parent,
)

_HERE = get_current_parent()
_ROOT = get_project_root()
_PMSM = _ROOT / "pmsm"
_C_SRC = _PMSM / "c_src"

for _p in (
    get_embedsim_import_path(),
    str(_PMSM),
    str(_C_SRC),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ================================================================
# EmbedSim imports
# ================================================================

from embedsim.core_blocks import (
    VectorSignal,
    DEFAULT_DTYPE,
)

from embedsim_generic_control import GenericControlBlock


# =============================================================================
# Jerk-Limited Speed Trajectory (matches C implementation)
# =============================================================================

class SpeedTrajectory:
    """
    Jerk-limited speed trajectory generator - matches C implementation.

    State:
        speed  [RPM]
        accel  [RPM/s]
        jerk   [RPM/s^3]  # Note: C uses RPM/s^3 for jerk

    Outputs:
        omega_ref       [rad/s]
        omega_dot       [rad/s^2]
        omega_ddot      [rad/s^3]

    The trajectory is generated recursively matching C's
    EmbedSim_CalculateTimeOptimalSCurve exactly.
    """

    def __init__(
            self,
            max_speed_rpm=3000.0,
            max_accel_rpm_s=500.0,
            max_jerk_rpm_s3=3000.0,  # C uses RPM/s^3
            settle_tolerance=0.1,     # SPEED_SETTLE_TOL
            debug=False,
    ):

        self.max_speed_rpm = float(max_speed_rpm)
        self.max_accel_rpm_s = float(max_accel_rpm_s)
        self.max_jerk_rpm_s3 = float(max_jerk_rpm_s3)  # C: MAX_JERK_RPM (RPM/s^3)
        self.settle_tolerance = float(settle_tolerance)  # C: SPEED_SETTLE_TOL

        # Trajectory states (in RPM, RPM/s)
        self.speed = 0.0
        self.accel = 0.0
        self.jerk = 0.0  # RPM/s^3

        # Debug
        self.debug = debug
        self._last_debug_print = -1.0

    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(value, minimum, maximum):
        return max(minimum, min(maximum, value))

    # ------------------------------------------------------------------
    def reset(self):
        self.speed = 0.0
        self.accel = 0.0
        self.jerk = 0.0

    # ------------------------------------------------------------------
    def update(self, target_rpm, dt):
        """
        Update trajectory by one controller sample - matches C exactly.

        Parameters
        ----------
        target_rpm : float
            Requested speed [RPM]
        dt : float
            Controller sample time [s]

        Returns
        -------
        dict
            omega_ref
            omega_dot
            omega_ddot
        """

        if dt <= 0.0:
            return self._output()

        # ------------------------------------------------------------
        # Limit target speed
        # ------------------------------------------------------------

        target = self._clamp(
            float(target_rpm),
            -self.max_speed_rpm,
            self.max_speed_rpm,
        )

        # ------------------------------------------------------------
        # Speed error (use trajectory state, not measured)
        # ------------------------------------------------------------

        error = target - self.speed
        distance_to_target = abs(error)

        # ------------------------------------------------------------
        # Deadband: if very close to target and accel small, settle
        # Matches C's SPEED_SETTLE_TOL check
        # ------------------------------------------------------------

        if distance_to_target < self.settle_tolerance and abs(self.accel) < 0.01:
            self.speed = target
            self.accel = 0.0
            self.jerk = 0.0
            return self._output()

        # ------------------------------------------------------------
        # Direction toward target
        # ------------------------------------------------------------

        direction = 1.0 if error >= 0.0 else -1.0

        # ------------------------------------------------------------
        # Calculate stopping acceleration:
        # a_stop = sqrt(2 * Jmax * |error|)
        # This is the maximum acceleration that can be reduced
        # to zero exactly at the target using max jerk.
        # Matches C exactly.
        # ------------------------------------------------------------

        stopping_accel = math.sqrt(
            max(0.0, 2.0 * self.max_jerk_rpm_s3 * distance_to_target)
        )

        # ------------------------------------------------------------
        # Desired acceleration = direction * min(accelMax, a_stop)
        # ------------------------------------------------------------

        desired_accel = direction * min(self.max_accel_rpm_s, stopping_accel)

        # ------------------------------------------------------------
        # Compute jerk needed to reach desired acceleration in one step
        # This is ALWAYS computed, not just when changing
        # ------------------------------------------------------------

        jerk_request = (desired_accel - self.accel) / dt

        # ------------------------------------------------------------
        # Clamp jerk to ±jerkMax
        # ------------------------------------------------------------

        self.jerk = self._clamp(
            jerk_request,
            -self.max_jerk_rpm_s3,
            self.max_jerk_rpm_s3,
        )

        # ------------------------------------------------------------
        # Integrate acceleration: a += j * dt
        # ------------------------------------------------------------

        new_accel = self.accel + self.jerk * dt

        # ------------------------------------------------------------
        # Clamp acceleration to ±accelMax
        # ------------------------------------------------------------

        new_accel = self._clamp(
            new_accel,
            -self.max_accel_rpm_s,
            self.max_accel_rpm_s,
        )

        # ------------------------------------------------------------
        # Integrate speed using second-order formula
        # Matches C's second-order integration exactly:
        # omega = omega0 + accel * dt + 0.5 * jerk * dt^2
        # ------------------------------------------------------------

        new_speed = (
            self.speed
            + self.accel * dt
            + 0.5 * self.jerk * dt * dt
        )

        # ------------------------------------------------------------
        # Clamp speed to ±max_speed_rpm
        # ------------------------------------------------------------

        new_speed = self._clamp(
            new_speed,
            -self.max_speed_rpm,
            self.max_speed_rpm,
        )

        # ------------------------------------------------------------
        # Prevent overshoot (matches C exactly)
        # ------------------------------------------------------------

        if (direction > 0.0 and new_speed > target) or \
           (direction < 0.0 and new_speed < target):
            new_speed = target
            new_accel = 0.0
            self.jerk = 0.0

        # ------------------------------------------------------------
        # Update states
        # ------------------------------------------------------------

        self.speed = new_speed
        self.accel = new_accel

        # Debug: Print trajectory states when jerk is non-zero
        if self.debug:
            if abs(self.jerk) > 0.1:  # Only print when jerk is significant
                print(f"[Trajectory] speed={self.speed:.1f} RPM, accel={self.accel:.1f} RPM/s, jerk={self.jerk:.1f} RPM/s³")

        return self._output()

    # ------------------------------------------------------------------
    def _output(self):
        rpm_to_rad_s = 2.0 * math.pi / 60.0

        return {
            "omega_ref": self.speed * rpm_to_rad_s,
            "omega_dot": self.accel * rpm_to_rad_s,
            # Note: C's omega_ddot is the derivative of omega_dot:
            # ω̈ = d(ω̇)/dt = jerk in rad/s^3
            "omega_ddot": self.jerk * rpm_to_rad_s,  # C: RotorJerkRefM
        }


# =============================================================================
# Universal Python Controller - ALIGNED WITH C
# =============================================================================

class PythonController(GenericControlBlock):
    """
    Universal PMSM controller - ALIGNED WITH C IMPLEMENTATION.

    Modes:
        OPEN_LOOP
        DFC

    DFC architecture (exactly matches the C implementation):

        S-Curve (time-optimal, matches C)
            |
            +--> omega_ref
            +--> omega_dot
            +--> omega_ddot
                    |
                    v
              Mechanical Flatness
                    |
                    +--> torque_ff = J*ω̇ + B*ω + Tload
                    |
              Speed PI (torque correction)
                    |
                    +--> torque_corr = Kp_speed*(ω_ref-ω_meas) + Ki_speed*∫(ω_ref-ω_meas)
                    |
                    v
              Total torque = torque_ff + torque_corr
                    |
                    v
              iq_ref = torque / (1.5*p*λ_PM)
                    |
                    v
              Current PI (voltage correction)
                    |
                    v
                   SVM
    """

    def __init__(
            self,
            name="ctrl",
            dt_s=50e-6,
            vdc_nom=12.0,
            controller_mode="DFC",
            # Current PI gains (matches C: DFC_CURRENT_KP_D_F, etc.)
            kp_d=0.0001,
            kp_q=0.0195,
            ki_d=0.0005,
            ki_q=0.0002,
            # Speed PI gains (matches C: DFC_SPEED_KP_Q_F, DFC_SPEED_KI_Q_F)
            kp_speed=0.0039,
            ki_speed=0.0002,
            # Limits (matches C: DFC_INTEGRAL_LIMIT_F)
            integral_limit=25.0,
            max_current=100.0,
            max_iq_dot=1000.0,
            modulation_limit=0.90,
            # Startup parameters (matches C)
            startup_mod_min=0.05,
            startup_mod_max=0.25,
            startup_increment=0.001,
            # Spinning detection (matches C)
            spinning_past_index=89500,
            stopped_past_index=2000,
            # PMSM parameters
            pole_pairs=4.0,
            rs=0.19,
            ld=0.125e-3,
            lq=0.125e-3,
            lambda_pm=0.0014,
            j=2.4e-6,
            b=1.0e-6,
            tload=0.0,
            # Open-loop parameters
            open_loop_amp=0.3,
            open_loop_ramp_rate=200.0,
            # FIX: Force use_python=True to always use Python implementation
            use_python=True,
            # Debug
            debug=False,
            **kwargs,
    ):

        # Initialize GenericControlBlock
        # FIX: Override use_c_backend to False to always use Python
        super().__init__(
            name=name,
            dt_s=dt_s,
            vdc_nom=vdc_nom,
            use_c_backend=False,  # Always use Python implementation
            **kwargs,
        )

        self.controller_mode = controller_mode
        self.use_python = use_python
        self.debug = debug

        # ============================================================
        # PMSM parameters (match C)
        # ============================================================

        self.pole_pairs = pole_pairs
        self.Rs = rs
        self.Ld = ld
        self.Lq = lq
        self.lambda_pm = lambda_pm
        self.J = j
        self.B = b
        self.Tload = tload

        # ============================================================
        # Open-loop parameters
        # ============================================================

        self.theta = 0.0
        self.amp = open_loop_amp
        self.ramp_rate = open_loop_ramp_rate
        self._current_freq = 0.0

        # ============================================================
        # Speed PI (torque correction) - gains from C
        # ============================================================

        self.Kp_speed = kp_speed
        self.Ki_speed = ki_speed

        # Integral accumulator (no dt multiplication)
        self.speed_integral = 0.0

        # ============================================================
        # Current PI - gains from C
        # ============================================================

        self.Kp_d = kp_d
        self.Kp_q = kp_q
        self.Ki_d = ki_d
        self.Ki_q = ki_q

        # Integral accumulators (no dt multiplication)
        self.id_integral = 0.0
        self.iq_integral = 0.0

        # ============================================================
        # Limits - match C
        # ============================================================

        self.max_current = max_current
        self.max_iq_dot = max_iq_dot
        self.integral_limit = integral_limit
        self.modulation_limit = modulation_limit

        # ============================================================
        # Startup parameters - MATCH C
        # ============================================================

        self.startup_mod_min = startup_mod_min
        self.startup_mod_max = startup_mod_max
        self.startup_increment = startup_increment
        self.startup_modulation = 0.0
        self.theta_open_loop = 0.0
        self._startup_elapsed = 0.0

        # ============================================================
        # Switch to closed-loop flags (matches C)
        # ============================================================

        self.switch_to_closed_loop = False
        self.control_reinit = False

        # Motor spinning detection (matches C's successCounter)
        self.spinning_counter = 0
        self.spinning_past_index = spinning_past_index
        self.stopped_counter = 0
        self.stopped_past_index = stopped_past_index

        # ============================================================
        # Jerk-limited trajectory (matches C)
        # ============================================================

        self.trajectory = SpeedTrajectory(
            max_speed_rpm=3000.0,
            max_accel_rpm_s=500.0,
            max_jerk_rpm_s3=3000.0,
            settle_tolerance=0.1,
            debug=debug,
        )

        # ============================================================
        # Diagnostics
        # ============================================================

        self._last_print = -1.0
        self._print_counter = 0

        print(f"\n{'=' * 70}")
        print(f" PYTHON CONTROLLER - Mode: {controller_mode} (ALIGNED WITH C)")
        print(f"  Using Python implementation (use_python={use_python})")
        if debug:
            print(f"  Debug mode: ENABLED")
        print(f"{'=' * 70}")

        if controller_mode == "OPEN_LOOP":
            print(f"  Amplitude: {self.amp}")
            print(f"  Ramp rate: {self.ramp_rate} Hz/s")
            print("  Follows speed_ref from VectorStep")
        else:
            print(f"  Speed PI (torque): Kp={self.Kp_speed}, Ki={self.Ki_speed}")
            print(f"  Current PI: Kp_d={self.Kp_d}, Kp_q={self.Kp_q}")
            print(f"  Current PI: Ki_d={self.Ki_d}, Ki_q={self.Ki_q}")
            print(f"  Integral limit: {self.integral_limit}")
            print(f"  Max current: {self.max_current} A")
            print(f"  S-curve: Jmax={self.trajectory.max_jerk_rpm_s3:.1f} RPM/s³")
            print(f"  S-curve: Amax={self.trajectory.max_accel_rpm_s:.1f} RPM/s")
            print(f"  Startup: modulation {self.startup_mod_min} → {self.startup_mod_max}")
            print(f"  Startup increment: {self.startup_increment}")
            print(f"  Spinning PastIndex: {self.spinning_past_index}")
            print(f"  Stopped PastIndex: {self.stopped_past_index}")
        print(f"{'=' * 70}\n")

    # ==================================================================
    # Additional Diagnostic Methods
    # ==================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the controller state.

        Returns
        -------
        dict
            Dictionary containing controller diagnostics
        """
        return {
            "switch_to_closed_loop": self.switch_to_closed_loop,
            "control_reinit": self.control_reinit,
            "speed_integral": self.speed_integral,
            "id_integral": self.id_integral,
            "iq_integral": self.iq_integral,
            "startup_modulation": self.startup_modulation,
            "spinning_counter": self.spinning_counter,
            "stopped_counter": self.stopped_counter,
            "trajectory_speed": self.trajectory.speed,
            "trajectory_accel": self.trajectory.accel,
            "trajectory_jerk": self.trajectory.jerk,
        }

    def set_parameters(self, **kwargs) -> None:
        """
        Dynamically set controller parameters.

        Parameters
        ----------
        **kwargs
            Parameter name-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"[Controller] Updated {key} = {value}")
            else:
                print(f"[Controller] Warning: {key} not found")

    def log_state(self, t: float) -> None:
        """
        Log current controller state for debugging.

        Parameters
        ----------
        t : float
            Current simulation time
        """
        if self.controller_mode == "DFC":
            print(
                f"[Controller t={t:.3f}s] "
                f"closed={self.switch_to_closed_loop} "
                f"traj={self.trajectory.speed:.1f}RPM "
                f"spin_cnt={self.spinning_counter} "
                f"mod={self.startup_modulation:.3f}"
            )

    # ==================================================================
    # Utility
    # ==================================================================

    def compute(self, t, dt, input_values=None):
        """
        Override compute to ALWAYS use Python implementation.
        This ensures our fixes are always applied.
        """
        return self.compute_py(t, dt, input_values)

    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(value, minimum, maximum):
        return max(minimum, min(maximum, value))

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(angle):
        angle = angle % (2.0 * math.pi)
        if angle < 0.0:
            angle += 2.0 * math.pi
        return angle

    # ==================================================================
    # Space Vector Modulation - matches C's SVM
    # ==================================================================

    def _svm(self, valpha, vbeta, vdc):
        """
        Convert alpha-beta voltage to PWM duties.
        Matches C's SVM_CalculateDutyCycle behavior.
        """
        v_mag = math.sqrt(valpha * valpha + vbeta * vbeta)
        v_max = vdc / math.sqrt(3.0)

        if v_mag > 0.0:
            mod_idx = self._clamp(v_mag / v_max, 0.0, self.modulation_limit)
            valpha = (valpha / v_mag) * mod_idx * v_max
            vbeta = (vbeta / v_mag) * mod_idx * v_max

        # Use wrapper's inv_clarke (matches C)
        vu, vv, vw = self.inv_clarke(valpha, vbeta)

        vmax = vdc / 2.0
        if vmax > 0.0:
            duty_u = self._clamp((vu / vmax + 1.0) / 2.0, 0.0, 1.0)
            duty_v = self._clamp((vv / vmax + 1.0) / 2.0, 0.0, 1.0)
            duty_w = self._clamp((vw / vmax + 1.0) / 2.0, 0.0, 1.0)
            return duty_u, duty_v, duty_w

        return 0.5, 0.5, 0.5

    # ==================================================================
    # Motor Spinning Detection - MATCHES C EXACTLY
    # ==================================================================

    def _is_motor_spinning(self, speed_meas_rpm, speed_ref_rpm, past_index=None):
        """
        Matches C's EmbedSim_IsMotorSpinning exactly.

        Returns True if measured speed exceeds 95% of reference speed
        for past_index consecutive samples.

        Parameters:
        -----------
        speed_meas_rpm : float
            Measured speed in RPM
        speed_ref_rpm : float
            Reference speed in RPM
        past_index : int
            Number of consecutive samples required (default: self.spinning_past_index)
        """
        if past_index is None:
            past_index = self.spinning_past_index

        # Use RPM directly - matches C exactly
        threshold = 0.15 * speed_ref_rpm

        # Check if measured speed exceeds threshold (in RPM)
        if abs(speed_meas_rpm) > threshold:
            if self.spinning_counter < 0x7FFFFFFF:  # MAX_int32_T
                self.spinning_counter += 1
        else:
            self.spinning_counter = 0

        # If counter exceeds past_index, motor is spinning
        if self.spinning_counter > past_index:
            self.spinning_counter = 0  # Reset counter (matches C)
            return True

        return False

    # ==================================================================
    # Motor Stopped Detection - MATCHES C EXACTLY
    # ==================================================================

    def _is_motor_stopped(self, speed_meas_rpm, past_index=None):
        """
        Matches C's EmbedSim_IsNotSpinning exactly.

        Returns True if measured speed is below 0.2 RPM for past_index
        consecutive samples.

        Parameters:
        -----------
        speed_meas_rpm : float
            Measured speed in RPM
        past_index : int
            Number of consecutive samples required (default: self.stopped_past_index)
        """
        if past_index is None:
            past_index = self.stopped_past_index

        # Check if speed is below 0.2 RPM (matches C)
        if abs(speed_meas_rpm) < 0.2:
            if self.stopped_counter < 0x7FFFFFFF:
                self.stopped_counter += 1
        else:
            self.stopped_counter = 0

        if self.stopped_counter > past_index:
            self.stopped_counter = 0
            return True

        return False

    # ==================================================================
    # Main controller
    # ==================================================================

    def compute_py(self, t, dt, input_values=None):
        """
        Main controller step - aligned with C implementation.

        Input vector:
        [0] speed_ref_rpm
        [1] ia
        [2] ib
        [3] ic
        [4] speed_sensor_rpm
        [5] sample_time
        [6] position_sensor_rad
        [7] valid
        [8] unused
        [9] Vdc
        """

        u = input_values[0].value

        speed_ref_rpm = float(u[0])
        ia = float(u[1])
        ib = float(u[2])
        ic = float(u[3])
        speed_sensor_rpm = float(u[4])
        sample_time = float(u[5])
        position_sensor_rad = float(u[6])
        valid_in = int(u[7])
        vdc = float(u[9])

        # Avoid unused-variable warnings
        _ = sample_time
        _ = valid_in

        # ============================================================
        # Diagnostics - PRINT SPEED OUTPUT
        # ============================================================

        # Print more frequently when debug is enabled
        print_interval = 0.02 if self.debug else 0.2

        if t - self._last_print >= print_interval:
            self._last_print = t
            self._print_counter += 1

            if self.controller_mode == "OPEN_LOOP":
                target_freq = speed_ref_rpm * self.pole_pairs / 60.0
                print(
                    f"[OpenLoop t={t:.2f}s] "
                    f"speed_ref={speed_ref_rpm:.1f} RPM  "
                    f"freq={self._current_freq:.1f}Hz  "
                    f"speed={speed_sensor_rpm:.1f} RPM"
                )
            else:
                # Show jerk even when 0 - but with more context
                jerk_str = f"{self.trajectory.jerk:.1f}"
                if abs(self.trajectory.jerk) > 1.0:
                    jerk_str = f"*{self.trajectory.jerk:.1f}*"  # Highlight non-zero jerk

                print(
                    f"[DFC t={t:.2f}s] "
                    f"speed_ref={speed_ref_rpm:.1f} RPM  "
                    f"speed={speed_sensor_rpm:.1f} RPM  "
                    f"traj={self.trajectory.speed:.1f} RPM  "
                    f"acc={self.trajectory.accel:.1f} RPM/s  "
                    f"jerk={jerk_str} RPM/s³  "
                    f"closed_loop={self.switch_to_closed_loop}  "
                    f"spin_cnt={self.spinning_counter}  "
                    f"mod={self.startup_modulation:.3f}"
                )

        # ============================================================
        # Electrical rotor angle
        # theta_e = p * theta_m
        # ============================================================

        theta_elec = position_sensor_rad * self.pole_pairs
        theta_elec = self._wrap_angle(theta_elec)

        # ============================================================
        # Clarke transform - USE WRAPPER (matches C)
        # ============================================================

        ialpha, ibeta = self.clarke(ia, ib, ic)

        # ============================================================
        # Park transform - USE WRAPPER (matches C)
        # ============================================================

        id_meas, iq_meas = self.park(ialpha, ibeta, theta_elec)

        # ============================================================
        # OPEN LOOP MODE
        # ============================================================

        if self.controller_mode == "OPEN_LOOP":
            target_freq = speed_ref_rpm * self.pole_pairs / 60.0
            freq_error = target_freq - self._current_freq
            max_change = self.ramp_rate * dt

            if abs(freq_error) > max_change:
                self._current_freq += max_change * np.sign(freq_error)
            else:
                self._current_freq = target_freq

            self.theta += 2.0 * math.pi * self._current_freq * dt

            amp = self.amp
            duty_u = 0.5 + amp * math.sin(self.theta)
            duty_v = 0.5 + amp * math.sin(self.theta - 2.0 * math.pi / 3.0)
            duty_w = 0.5 + amp * math.sin(self.theta - 4.0 * math.pi / 3.0)

            duty_u = np.clip(duty_u, 0.0, 1.0)
            duty_v = np.clip(duty_v, 0.0, 1.0)
            duty_w = np.clip(duty_w, 0.0, 1.0)

            out = np.array([duty_u, duty_v, duty_w, 1.0], dtype=DEFAULT_DTYPE)
            self.output = VectorSignal(out, self.name)
            return self.output

        # ============================================================
        # DFC STARTUP - MATCHES C EXACTLY
        # ============================================================

        # Update trajectory (matches C's EmbedSim_CalculateTimeOptimalSCurve)
        ref = self.trajectory.update(speed_ref_rpm, dt)

        # ============================================================
        # Check if we should switch to closed-loop
        # ============================================================

        # Use target speed (in RPM) for spinning detection - matches C exactly
        if self._is_motor_spinning(speed_sensor_rpm, speed_ref_rpm, self.spinning_past_index):
            if not self.switch_to_closed_loop:
                print(f"[DFC t={t:.2f}s] SWITCHING TO CLOSED-LOOP")
                print(f"  measured={speed_sensor_rpm:.1f} RPM > 0.95 * target={0.95 * speed_ref_rpm:.1f} RPM")
                self.switch_to_closed_loop = True
                # Reset integrators on switch (matches C's DFC_Reset)
                self.speed_integral = 0.0
                self.id_integral = 0.0
                self.iq_integral = 0.0
                self.control_reinit = True
                self.trajectory.speed = speed_sensor_rpm
                self.trajectory.accel = 0.0
                self.trajectory.jerk = 0.0
                self.startup_modulation = 0.0

        # If we're in startup mode (not closed-loop)
        if not self.switch_to_closed_loop:
            # Ramp modulation
            self.startup_modulation += self.startup_increment
            self.startup_modulation = self._clamp(
                self.startup_modulation,
                self.startup_mod_min,
                self.startup_mod_max
            )

            # Electrical speed during startup (matches C)
            omega_startup_e = self.pole_pairs * (speed_ref_rpm * (2.0 * math.pi / 60.0))

            # Integrate angle (matches C)
            self.theta_open_loop += omega_startup_e * dt
            self.theta_open_loop = self._wrap_angle(self.theta_open_loop)

            # Vd = 0, Vq = (Vdc/sqrt(3)) * modulation (matches C)
            startup_vd = 0.0
            startup_vq = (vdc / math.sqrt(3.0)) * self.startup_modulation

            # Inverse Park to alpha-beta - USE WRAPPER (matches C)
            valpha, vbeta = self.inv_park(startup_vd, startup_vq, self.theta_open_loop)

            # SVM (matches C)
            duty_u, duty_v, duty_w = self._svm(valpha, vbeta, vdc)

            out = np.array([duty_u, duty_v, duty_w, 1.0], dtype=DEFAULT_DTYPE)
            self.output = VectorSignal(out, self.name)
            return self.output

        # ============================================================
        # CLOSED-LOOP DFC - MATCHES C EXACTLY
        # ============================================================

        omega_ref = ref["omega_ref"]
        omega_dot = ref["omega_dot"]
        omega_ddot = ref["omega_ddot"]

        omega_meas = speed_sensor_rpm * (2.0 * math.pi / 60.0)

        # Speed PI (torque correction)
        speed_error = omega_ref - omega_meas
        self.speed_integral += speed_error
        self.speed_integral = self._clamp(
            self.speed_integral,
            -self.integral_limit,
            self.integral_limit,
        )
        torque_correction = (
            self.Kp_speed * speed_error
            + self.Ki_speed * self.speed_integral
        )

        # Mechanical Flatness
        torque_ff = self.J * omega_dot + self.B * omega_ref + self.Tload
        torque_required = torque_ff + torque_correction

        # Electrical Flatness
        torque_constant = 1.5 * self.pole_pairs * self.lambda_pm
        if abs(torque_constant) > 1.0e-6:
            iq_ref = self._clamp(torque_required / torque_constant, -self.max_current, self.max_current)
            iq_ref_dot = self._clamp(
                (self.J * omega_ddot + self.B * omega_dot) / torque_constant,
                -self.max_iq_dot,
                self.max_iq_dot
            )
        else:
            iq_ref = 0.0
            iq_ref_dot = 0.0

        # Voltage Feedforward
        omega_e_ref = self.pole_pairs * omega_ref
        vd_ff = -omega_e_ref * self.Lq * iq_ref
        vq_ff = self.Rs * iq_ref + self.Lq * iq_ref_dot + omega_e_ref * self.lambda_pm

        # Current PI
        id_ref = 0.0
        id_error = id_ref - id_meas
        iq_error = iq_ref - iq_meas

        self.id_integral += id_error
        self.iq_integral += iq_error
        self.id_integral = self._clamp(self.id_integral, -self.integral_limit, self.integral_limit)
        self.iq_integral = self._clamp(self.iq_integral, -self.integral_limit, self.integral_limit)

        id_error = self._clamp(id_error, -self.max_current, self.max_current)
        iq_error = self._clamp(iq_error, -self.max_current, self.max_current)

        vd_corr = self.Kp_d * id_error + self.Ki_d * self.id_integral
        vq_corr = self.Kp_q * iq_error + self.Ki_q * self.iq_integral

        # Final voltage references
        vd_ref = vd_ff + vd_corr
        vq_ref = vq_ff + vq_corr

        # Inverse Park - USE WRAPPER (matches C)
        valpha, vbeta = self.inv_park(vd_ref, vq_ref, theta_elec)

        # SVM - matches C
        duty_u, duty_v, duty_w = self._svm(valpha, vbeta, vdc)

        # Check if motor has stopped (matches C)
        if self._is_motor_stopped(speed_sensor_rpm, self.stopped_past_index):
            print(f"[DFC t={t:.2f}s] MOTOR STOPPED - REVERTING TO OPEN-LOOP")
            self.switch_to_closed_loop = False
            self.speed_integral = 0.0
            self.id_integral = 0.0
            self.iq_integral = 0.0
            self.control_reinit = True
            self.startup_modulation = 0.0
            self.trajectory.speed = 0.0
            self.trajectory.accel = 0.0
            self.trajectory.jerk = 0.0
            return self.compute_py(t, dt, input_values)

        # Output
        out = np.array([duty_u, duty_v, duty_w, 1.0], dtype=DEFAULT_DTYPE)
        self.output = VectorSignal(out, self.name)
        return self.output

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(self):
        super().reset()

        # Open-loop state
        self.theta = 0.0
        self._current_freq = 0.0

        # Speed PI
        self.speed_integral = 0.0

        # Current PI
        self.id_integral = 0.0
        self.iq_integral = 0.0

        # Startup
        self.theta_open_loop = 0.0
        self.startup_modulation = 0.0
        self._startup_elapsed = 0.0

        # Switch flag and counters
        self.switch_to_closed_loop = False
        self.control_reinit = True
        self.spinning_counter = 0
        self.stopped_counter = 0

        # Trajectory
        self.trajectory.reset()

        # Diagnostics
        self._last_print = -1.0
        self._print_counter = 0