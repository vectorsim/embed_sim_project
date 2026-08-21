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


# ================================================================
# Canonical coordinate transforms
# ================================================================

from embedsim_control_wrapper import (
    clarke,
    park,
    inv_park,
    inv_clarke,
)


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
    ):

        self.max_speed_rpm = float(max_speed_rpm)
        self.max_accel_rpm_s = float(max_accel_rpm_s)
        self.max_jerk_rpm_s3 = float(max_jerk_rpm_s3)  # C: MAX_JERK_RPM (RPM/s^3)
        self.settle_tolerance = float(settle_tolerance)  # C: SPEED_SETTLE_TOL

        # Trajectory states (in RPM, RPM/s)
        self.speed = 0.0
        self.accel = 0.0
        self.jerk = 0.0  # RPM/s^3

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
            **kwargs,
    ):

        super().__init__(
            name=name,
            dt_s=dt_s,
            vdc_nom=vdc_nom,
            use_c_backend=False,
            **kwargs,
        )

        self.controller_mode = controller_mode

        # ============================================================
        # PMSM parameters (match C)
        # ============================================================

        self.pole_pairs = 4.0
        self.Rs = 0.19
        self.Ld = 0.125e-3
        self.Lq = 0.125e-3
        self.lambda_pm = 0.0014
        self.J = 2.4e-6
        self.B = 1.0e-6
        self.Tload = 0.0

        # ============================================================
        # Open-loop parameters
        # ============================================================

        self.theta = 0.0
        self.amp = 0.3
        self.ramp_rate = 200.0
        self._current_freq = 0.0

        # ============================================================
        # Speed PI (torque correction) – gains from C
        # ============================================================

        self.Kp_speed = 0.0021   # DFC_SPEED_KP_Q_F
        self.Ki_speed = 0.0001   # DFC_SPEED_KI_Q_F

        # Integral accumulator (no dt multiplication)
        self.speed_integral = 0.0

        # ============================================================
        # Current PI – gains from C
        # ============================================================

        self.Kp_d = 0.0001   # DFC_CURRENT_KP_D_F
        self.Kp_q = 0.0195   # DFC_CURRENT_KP_Q_F
        self.Ki_d = 0.0005   # DFC_CURRENT_KI_D_F
        self.Ki_q = 0.0002   # DFC_CURRENT_KI_Q_F

        # Integral accumulators (no dt multiplication)
        self.id_integral = 0.0
        self.iq_integral = 0.0

        # ============================================================
        # Limits – match C
        # ============================================================

        self.max_current = 100.0          # DFC_MAX_CURRENT
        self.max_iq_dot = 1000.0          # DFC_MAX_IQ_DOT_F
        self.integral_limit = 5.0         # DFC_INTEGRAL_LIMIT_F
        self.modulation_limit = 0.90      # as in C (clamped to 0.90)

        # ============================================================
        # Startup parameters - MATCH C
        # ============================================================

        self.startup_time = 0.8           # DFC_STARTUP_TIME_S (C uses 0.8s)
        self.startup_speed_rpm = 300.0    # DFC_STARTUP_SPEED_RPM
        self.startup_mod_min = 0.001      # DFC_STARTUP_MOD_MIN (increment)
        self.startup_mod_max = 0.25       # DFC_STARTUP_MOD_MAX
        self.startup_modulation = 0.0     # Current modulation (ramps from min to max)
        self.theta_open_loop = 0.0
        self._startup_elapsed = 0.0

        # ============================================================
        # Switch to closed-loop flag (matches C)
        # ============================================================

        self.switch_to_closed_loop = False
        self.control_reinit = False

        # Motor spinning detection counter (matches C's successCounter)
        self.spinning_counter = 0

        # ============================================================
        # Jerk-limited trajectory (matches C)
        # ============================================================

        self.trajectory = SpeedTrajectory(
            max_speed_rpm=3000.0,
            max_accel_rpm_s=500.0,
            max_jerk_rpm_s3=3000.0,  # C: MAX_JERK_RPM (RPM/s^3)
            settle_tolerance=0.1,     # C: SPEED_SETTLE_TOL
        )

        # ============================================================
        # Diagnostics
        # ============================================================

        self._last_print = -1.0

        print(f"\n{'=' * 70}")
        print(f" PYTHON CONTROLLER - Mode: {controller_mode} (ALIGNED WITH C)")
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
            print(f"  Startup: {self.startup_time * 1000:.0f}ms at {self.startup_speed_rpm:.0f} RPM")
            print(f"  Startup: modulation {self.startup_mod_min} → {self.startup_mod_max}")
        print(f"{'=' * 70}\n")

    # ==================================================================
    # Utility
    # ==================================================================

    def compute(self, t, dt, input_values=None):
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

        vu, vv, vw = inv_clarke(valpha, vbeta)

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
    #
    # C code (from embed_sim_control.c):
    #   uint32_T EmbedSim_IsMotorSpinning(const EmbedSimCtrlInput_T* const InputPtr, uint32_T PastIndex)
    #   {
    #       static uint32_T successCounter = 0U;
    #       uint32_T result = 0U;
    #
    #       if (fabs(CON_RPM_TO_RAD(InputPtr->RotorSpeedObsEstM)) > (InputPtr->RotorVelocityRefM))
    #       {
    #           if (successCounter < MAX_int32_T) successCounter++;
    #       }
    #       else
    #       {
    #           successCounter = 0U;
    #       }
    #
    #       if (successCounter > PastIndex)
    #       {
    #           result = 1U;
    #           successCounter = 0U;
    #       }
    #       return result;
    #   }
    #
    # KEY INSIGHT: In DFC mode, RotorVelocityRefM is overwritten by the
    # TRAJECTORY speed (from EmbedSim_CalculateTimeOptimalSCurve) before
    # DFC_Step is called. So during DFC mode, the comparison is:
    #   measured_speed > trajectory_speed
    #
    # During startup (before trajectory starts), RotorVelocityRefM is set
    # to the target speed by ExecuteObserver. So the comparison is:
    #   measured_speed > target_speed
    # ==================================================================

    def _is_motor_spinning(self, speed_meas_rpm, speed_ref_rad_s, past_index=80000):
        """
        Matches C's EmbedSim_IsMotorSpinning exactly.

        Returns True if measured speed (in rad/s) exceeds reference speed (in rad/s)
        for past_index consecutive samples.

        The condition is: |speed_meas_rad_s| > speed_ref_rad_s

        Parameters:
        -----------
        speed_meas_rpm : float
            Measured speed in RPM
        speed_ref_rad_s : float
            Reference speed in rad/s (this could be target OR trajectory speed)
        past_index : int
            Number of consecutive samples required (C uses 80000)
        """
        # Convert measured speed to rad/s
        speed_meas_rad_s = speed_meas_rpm * (2.0 * math.pi / 60.0)

        # Check if measured speed exceeds reference speed (in rad/s)
        # Matches C: fabs(CON_RPM_TO_RAD(measured)) > reference_in_rad_s
        if abs(speed_meas_rad_s) > speed_ref_rad_s:
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
        # Diagnostics
        # ============================================================

        if t - self._last_print >= 0.2:
            self._last_print = t

            if self.controller_mode == "OPEN_LOOP":
                target_freq = speed_ref_rpm * self.pole_pairs / 60.0
                print(
                    f"[OpenLoop t={t:.2f}s] "
                    f"speed_ref={speed_ref_rpm:.1f} RPM  "
                    f"freq={self._current_freq:.1f}Hz  "
                    f"speed={speed_sensor_rpm:.1f} RPM"
                )
            else:
                print(
                    f"[DFC t={t:.2f}s] "
                    f"speed_ref={speed_ref_rpm:.1f} RPM  "
                    f"speed={speed_sensor_rpm:.1f} RPM  "
                    f"traj={self.trajectory.speed:.1f} RPM  "
                    f"acc={self.trajectory.accel:.1f} RPM/s  "
                    f"jerk={self.trajectory.jerk:.1f} RPM/s³  "
                    f"closed_loop={self.switch_to_closed_loop}  "
                    f"spin_cnt={self.spinning_counter}"
                )

        # ============================================================
        # Electrical rotor angle
        # theta_e = p * theta_m
        # ============================================================

        theta_elec = position_sensor_rad * self.pole_pairs
        theta_elec = self._wrap_angle(theta_elec)

        # ============================================================
        # Clarke transform
        # ============================================================

        ialpha, ibeta = clarke(ia, ib, ic)

        # ============================================================
        # Park transform
        # ============================================================

        id_meas, iq_meas = park(ialpha, ibeta, theta_elec)

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
        #
        # C execution order for DFC mode:
        #   1. ExecuteObserver: sets RotorVelocityRefM = target speed (rad/s)
        #   2. CalculateTimeOptimalSCurve: overwrites RotorVelocityRefM with trajectory speed (rad/s)
        #   3. DFC_Step: uses RotorVelocityRefM (trajectory speed) for spinning detection
        #
        # KEY INSIGHT: During startup, the trajectory speed is 0 until the
        # trajectory starts generating. So the spinning detection compares:
        #   measured_speed > trajectory_speed (which is 0 during early startup)
        #
        # This means the motor will appear to be "spinning" as soon as it starts moving!
        #
        # In C, the spinning detection in DFC_Step happens AFTER CalculateTimeOptimalSCurve,
        # so it uses the trajectory speed (not the target speed).
        # ============================================================

        # Update trajectory (matches C's EmbedSim_CalculateTimeOptimalSCurve)
        # This is called BEFORE spinning detection, just like in C
        ref = self.trajectory.update(speed_ref_rpm, dt)

        # The trajectory speed in rad/s (this is what C uses for spinning detection)
        trajectory_speed_rad_s = ref["omega_ref"]

        # ============================================================
        # Check if we should switch to closed-loop
        # Matches C: if (EmbedSim_IsMotorSpinning(...) && SwitchToClosedLoop != 1)
        #
        # CRITICAL: In C, the spinning detection uses RotorVelocityRefM
        # which is the TRAJECTORY speed (after CalculateTimeOptimalSCurve),
        # NOT the target speed!
        #
        # During early startup, trajectory_speed_rad_s = 0, so any positive
        # measured speed will trigger the spinning detection.
        # ============================================================

        # Use trajectory speed (in rad/s) for spinning detection - matches C exactly
        if self._is_motor_spinning(speed_sensor_rpm, trajectory_speed_rad_s, 80000):
            if not self.switch_to_closed_loop:
                print(f"[DFC t={t:.2f}s] SWITCHING TO CLOSED-LOOP")
                print(f"  measured={speed_sensor_rpm:.1f} RPM > trajectory={self.trajectory.speed:.1f} RPM")
                self.switch_to_closed_loop = True
                # Reset integrators on switch (matches C's DFC_Reset in the spinning check)
                self.speed_integral = 0.0
                self.id_integral = 0.0
                self.iq_integral = 0.0
                # Re-initialize trajectory from measured speed (matches C's ControlReInit logic)
                self.trajectory.speed = speed_sensor_rpm
                self.trajectory.accel = 0.0
                self.trajectory.jerk = 0.0

        # If we're in startup mode (not closed-loop)
        if not self.switch_to_closed_loop:
            # Matches C: modulation += DFC_STARTUP_MOD_MIN
            # C uses modulation += DFC_STARTUP_MOD_MIN (0.001) each step
            self.startup_modulation += 0.001  # DFC_STARTUP_MOD_MIN as increment

            # Clamp to [DFC_STARTUP_MOD_MIN, DFC_STARTUP_MOD_MAX]
            self.startup_modulation = self._clamp(
                self.startup_modulation,
                0.001,   # DFC_STARTUP_MOD_MIN
                0.25     # DFC_STARTUP_MOD_MAX
            )

            # Electrical speed during startup (matches C)
            omega_startup_e = self.pole_pairs * (speed_ref_rpm * (2.0 * math.pi / 60.0))

            # Integrate angle
            self.theta_open_loop += omega_startup_e * dt
            self.theta_open_loop = self._wrap_angle(self.theta_open_loop)

            # Vd = 0, Vq = (Vdc/sqrt(3)) * modulation (matches C)
            startup_vd = 0.0
            startup_vq = (vdc / math.sqrt(3.0)) * self.startup_modulation

            # Inverse Park to alpha-beta
            valpha, vbeta = inv_park(startup_vd, startup_vq, self.theta_open_loop)

            # SVM (matches C)
            duty_u, duty_v, duty_w = self._svm(valpha, vbeta, vdc)

            out = np.array([duty_u, duty_v, duty_w, 1.0], dtype=DEFAULT_DTYPE)
            self.output = VectorSignal(out, self.name)
            return self.output

        # ============================================================
        # CLOSED-LOOP DFC - MATCHES C EXACTLY
        # ============================================================

        # We already have the trajectory reference from above
        omega_ref = ref["omega_ref"]
        omega_dot = ref["omega_dot"]
        omega_ddot = ref["omega_ddot"]  # This is jerk in rad/s^3

        # Convert measured speed to rad/s
        omega_meas = speed_sensor_rpm * (2.0 * math.pi / 60.0)

        # ============================================================
        # Speed PI (torque correction) - matches C exactly
        # ============================================================
        #
        # C code:
        #   speedError = omegaRef - omegaMeas;
        #   speedIntegralError += speedError;
        #   speedIntegralError = DFC_ClampValue(speedIntegralError, -limit, limit);
        #   torqueCorrection = (Kp_speed * speedError) + (Ki_speed * speedIntegralError);
        #

        speed_error = omega_ref - omega_meas

        # Accumulate integral (no dt multiplication - matches C)
        self.speed_integral += speed_error

        # Clamp integral to prevent windup
        self.speed_integral = self._clamp(
            self.speed_integral,
            -self.integral_limit,
            self.integral_limit,
        )

        # Compute torque correction
        torque_correction = (
            self.Kp_speed * speed_error
            + self.Ki_speed * self.speed_integral
        )

        # ============================================================
        # Mechanical Flatness - matches C exactly
        # ============================================================
        #
        # C code:
        #   torqueFeedforward = (J * omegaRefDot) + (B * omegaRef) + TorqueLoad;
        #   torqueRequired = torqueFeedforward + torqueCorrection;
        #

        torque_ff = self.J * omega_dot + self.B * omega_ref + self.Tload
        torque_required = torque_ff + torque_correction

        # ============================================================
        # Electrical Flatness - matches C exactly
        # ============================================================
        #
        # C code:
        #   torqueConstant = 1.5 * polePairs * FluxPm;
        #   iqRef = torqueRequired / torqueConstant;
        #   iqRef = clamp(iqRef, -DFC_MAX_CURRENT, DFC_MAX_CURRENT);
        #   iqRefDot = (J * omegaRefDDot + B * omegaRefDot) / torqueConstant;
        #   iqRefDot = clamp(iqRefDot, -DFC_MAX_IQ_DOT_F, DFC_MAX_IQ_DOT_F);
        #

        torque_constant = 1.5 * self.pole_pairs * self.lambda_pm

        if abs(torque_constant) > 1.0e-6:
            iq_ref = torque_required / torque_constant
            iq_ref = self._clamp(iq_ref, -self.max_current, self.max_current)

            # Derivative of iq reference
            iq_ref_dot = (self.J * omega_ddot + self.B * omega_dot) / torque_constant
            iq_ref_dot = self._clamp(iq_ref_dot, -self.max_iq_dot, self.max_iq_dot)
        else:
            iq_ref = 0.0
            iq_ref_dot = 0.0

        # ============================================================
        # Voltage Feedforward - matches C exactly
        # ============================================================
        #
        # C code:
        #   vdRef = -polePairs * omegaRef * Lq * iqRef;
        #   vqRef = (Rs * iqRef) + (Lq * iqRefDot) + (polePairs * omegaRef * FluxPm);
        #

        omega_e_ref = self.pole_pairs * omega_ref

        vd_ff = -omega_e_ref * self.Lq * iq_ref
        vq_ff = self.Rs * iq_ref + self.Lq * iq_ref_dot + omega_e_ref * self.lambda_pm

        # ============================================================
        # Current PI - matches C exactly
        # ============================================================
        #
        # C code:
        #   idError = 0.0 - dqCurrentMeas.D;
        #   iqError = iqRef - dqCurrentMeas.Q;
        #   idIntegralError += idError;
        #   iqIntegralError += iqError;
        #   idIntegralError = clamp(idIntegralError, -limit, limit);
        #   iqIntegralError = clamp(iqIntegralError, -limit, limit);
        #   vdCorr = (Kp_d * idError) + (Ki_d * idIntegralError);
        #   vqCorr = (Kp_q * iqError) + (Ki_q * iqIntegralError);
        #

        id_ref = 0.0  # Id_ref = 0 for surface PMSM

        id_error = id_ref - id_meas
        iq_error = iq_ref - iq_meas

        # Integrate (no dt multiplication - matches C)
        self.id_integral += id_error
        self.iq_integral += iq_error

        # Clamp integrals
        self.id_integral = self._clamp(self.id_integral, -self.integral_limit, self.integral_limit)
        self.iq_integral = self._clamp(self.iq_integral, -self.integral_limit, self.integral_limit)

        # Clamp errors to max current (matches C)
        id_error = self._clamp(id_error, -self.max_current, self.max_current)
        iq_error = self._clamp(iq_error, -self.max_current, self.max_current)

        # Compute voltage corrections
        vd_corr = self.Kp_d * id_error + self.Ki_d * self.id_integral
        vq_corr = self.Kp_q * iq_error + self.Ki_q * self.iq_integral

        # ============================================================
        # Final voltage references - matches C
        # ============================================================

        vd_ref = vd_ff + vd_corr
        vq_ref = vq_ff + vq_corr

        # ============================================================
        # Inverse Park - matches C
        # ============================================================

        valpha, vbeta = inv_park(vd_ref, vq_ref, theta_elec)

        # ============================================================
        # SVM - matches C
        # ============================================================

        duty_u, duty_v, duty_w = self._svm(valpha, vbeta, vdc)

        # ============================================================
        # Output
        # ============================================================

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

        # Switch flag and spinning counter
        self.switch_to_closed_loop = False
        self.control_reinit = True
        self.spinning_counter = 0

        # Trajectory
        self.trajectory.reset()

        # Diagnostics
        self._last_print = -1.0