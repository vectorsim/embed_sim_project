"""
pmsm_dfc.py
===========

PMSM Control - Python Controller **EXACTLY ALIGNED WITH C DFC IMPLEMENTATION**

- Linear startup ramp (no smoothstep)
- PI integral updates multiplied by dt
- Observer with direction enforcement
- Direct duty‑cycle clamping (no modulation index limit)
- Configurable startup duration (default 3.0s to match C)
"""

from __future__ import annotations

import sys
import math
from typing import Any

import numpy as np

# ================================================================
# Path setup (keep as in original)
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
# Jerk-Limited Speed Trajectory - ALIGNED WITH C
# =============================================================================

class SpeedTrajectory:
    """
    Jerk-limited speed trajectory generator - matches C exactly.
    Uses RPM for speed, RPM/s for acceleration, RPM/s³ for jerk.
    """

    def __init__(
            self,
            max_speed_rpm=3000.0,
            max_accel_rpm_s=800.0,      # C: MAX_ACCEL_RPM = 800
            max_jerk_rpm_s3=3500.0,     # C: MAX_JERK_RPM = 3500
            settle_tolerance=0.1,       # C: SPEED_SETTLE_TOL
    ):
        self.max_speed_rpm = float(max_speed_rpm)
        self.max_accel_rpm_s = float(max_accel_rpm_s)
        self.max_jerk_rpm_s3 = float(max_jerk_rpm_s3)
        self.settle_tolerance = float(settle_tolerance)

        self.speed = 0.0      # RPM
        self.accel = 0.0      # RPM/s
        self.jerk = 0.0       # RPM/s³

    def reset(self):
        self.speed = 0.0
        self.accel = 0.0
        self.jerk = 0.0

    @staticmethod
    def _clamp(value, minimum, maximum):
        return max(minimum, min(maximum, value))

    def update(self, target_rpm, dt, reinit=False):
        """
        Update trajectory by one step - matches C's EmbedSim_CalculateJerkLimitedTrajectory.
        If reinit=True, set accel and jerk to zero (keep speed).
        """
        if dt <= 0.0:
            return self._output()

        # If reinit, reset accel and jerk but keep speed (C does this when ControlReInit=1)
        if reinit:
            self.accel = 0.0
            self.jerk = 0.0

        target = self._clamp(float(target_rpm), -self.max_speed_rpm, self.max_speed_rpm)
        error = target - self.speed
        distance = abs(error)

        # Settle if very close
        if distance < self.settle_tolerance and abs(self.accel) < 0.01:
            self.speed = target
            self.accel = 0.0
            self.jerk = 0.0
            return self._output()

        direction = 1.0 if error >= 0.0 else -1.0

        # Stopping acceleration: a_stop = sqrt(2 * Jmax * |error|)
        stopping_accel = math.sqrt(max(0.0, 2.0 * self.max_jerk_rpm_s3 * distance))
        desired_accel = direction * min(self.max_accel_rpm_s, stopping_accel)

        # Jerk needed to reach desired accel in one step
        jerk_request = (desired_accel - self.accel) / dt
        self.jerk = self._clamp(jerk_request, -self.max_jerk_rpm_s3, self.max_jerk_rpm_s3)

        # Integrate accel and speed (second‑order)
        new_accel = self.accel + self.jerk * dt
        new_accel = self._clamp(new_accel, -self.max_accel_rpm_s, self.max_accel_rpm_s)

        new_speed = self.speed + self.accel * dt + 0.5 * self.jerk * dt * dt
        new_speed = self._clamp(new_speed, -self.max_speed_rpm, self.max_speed_rpm)

        # Prevent overshoot
        if (direction > 0.0 and new_speed > target) or (direction < 0.0 and new_speed < target):
            new_speed = target
            new_accel = 0.0
            self.jerk = 0.0

        self.speed = new_speed
        self.accel = new_accel

        return self._output()

    def _output(self):
        rpm_to_rad_s = 2.0 * math.pi / 60.0
        return {
            "omega_ref": self.speed * rpm_to_rad_s,          # rad/s
            "omega_dot": self.accel * rpm_to_rad_s,          # rad/s²
            "omega_ddot": self.jerk * rpm_to_rad_s,          # rad/s³
        }


# =============================================================================
# Python Controller - EXACTLY ALIGNED WITH C
# =============================================================================

class PythonController(GenericControlBlock):
    """
    PMSM controller that exactly replicates the C DFC behaviour.
    """

    def __init__(
            self,
            name="ctrl",
            dt_s=50e-6,
            vdc_nom=12.0,
            controller_mode="DFC",
            # C gains (from embed_sim_control.h)
            kp_speed=0.00092,
            ki_speed=0.00091,
            kp_d=0.00999,
            kp_q=0.019995,
            ki_d=0.00000025,
            ki_q=0.00000025,
            integral_limit=25.0,
            max_current=100.0,
            max_iq_dot=1000.0,
            # Startup (linear ramp over `startup_duration` seconds)
            startup_duration=3.0,        # <-- now configurable, default matches C
            startup_mod_min=0.05,
            startup_mod_max=0.20,
            # PMSM parameters
            pole_pairs=4.0,
            rs=0.19,
            ld=0.125e-3,
            lq=0.125e-3,
            lambda_pm=0.0014,
            j=2.4e-6,
            b=1.0e-6,
            tload=0.0,
            debug=False,
            **kwargs,
    ):
        # Force Python mode
        super().__init__(
            name=name,
            dt_s=dt_s,
            vdc_nom=vdc_nom,
            use_c_backend=False,
            **kwargs,
        )

        self.controller_mode = controller_mode
        self.debug = debug

        # Motor parameters
        self.pole_pairs = pole_pairs
        self.Rs = rs
        self.Ld = ld
        self.Lq = lq
        self.lambda_pm = lambda_pm
        self.J = j
        self.B = b
        self.Tload = tload

        # PI gains (C values)
        self.Kp_speed = kp_speed
        self.Ki_speed = ki_speed
        self.Kp_d = kp_d
        self.Kp_q = kp_q
        self.Ki_d = ki_d
        self.Ki_q = ki_q

        # Integrals (accumulate error * dt)
        self.speed_integral = 0.0
        self.id_integral = 0.0
        self.iq_integral = 0.0

        # Limits
        self.max_current = max_current
        self.max_iq_dot = max_iq_dot
        self.integral_limit = integral_limit

        # Startup (timer‑based linear ramp)
        self.startup_duration = float(startup_duration)
        self.startup_timer = 0.0
        self.startup_mod_min = startup_mod_min
        self.startup_mod_max = startup_mod_max
        self.startup_modulation = 0.0
        self.theta_open_loop = 0.0

        # Switch flag and reinit
        self.switch_to_closed_loop = False
        self.control_reinit = False

        # Trajectory (uses C limits)
        self.trajectory = SpeedTrajectory(
            max_speed_rpm=3000.0,
            max_accel_rpm_s=800.0,
            max_jerk_rpm_s3=3500.0,
            settle_tolerance=0.1,
        )

        # Observer state (model angle)
        self.theta_model = 0.0

        # Diagnostics
        self._last_print = -1.0
        self._print_counter = 0

        print(f"\n{'=' * 70}")
        print(f" PYTHON CONTROLLER - Mode: {controller_mode} (ALIGNED WITH C)")
        print(f"  Using Python implementation (exact C behaviour)")
        print(f"{'=' * 70}")
        print(f"  Speed PI: Kp={self.Kp_speed}, Ki={self.Ki_speed}")
        print(f"  Current PI: Kp_d={self.Kp_d}, Kp_q={self.Kp_q}, Ki_d={self.Ki_d}, Ki_q={self.Ki_q}")
        print(f"  Integral limit: {self.integral_limit}")
        print(f"  Startup: linear modulation {self.startup_mod_min} → {self.startup_mod_max} over {self.startup_duration}s")
        print(f"  S-curve: Jmax={self.trajectory.max_jerk_rpm_s3:.1f} RPM/s³, Amax={self.trajectory.max_accel_rpm_s:.1f} RPM/s")
        print(f"{'=' * 70}\n")

    # ------------------------------------------------------------------
    # Helper methods (matches C)
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(value, minimum, maximum):
        return max(minimum, min(maximum, value))

    @staticmethod
    def _wrap_angle(angle):
        angle = angle % (2.0 * math.pi)
        if angle < 0.0:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _angle_distance(angle1, angle2):
        """Shortest signed angular distance in [-pi, pi)."""
        diff = angle1 - angle2
        if diff >= math.pi:
            diff -= 2.0 * math.pi
        elif diff < -math.pi:
            diff += 2.0 * math.pi
        return diff

    # ------------------------------------------------------------------
    # Observer (matches C's EmbedSim_ExecuteObserver)
    # ------------------------------------------------------------------

    def _observer_update(self, position_mech_rad, speed_rpm, ref_speed_rpm, dt):
        """
        Update the model angle theta_model to track the sensor angle.
        Matches C's direction enforcement, gain scheduling, and half‑sample feedforward.
        """
        # Electrical angle from sensor
        sensor_elec = position_mech_rad * self.pole_pairs
        sensor_elec = self._wrap_angle(sensor_elec)

        # Raw difference
        diff = self._angle_distance(sensor_elec, self.theta_model)

        # ===== DIRECTION ENFORCEMENT (matches C) =====
        if ref_speed_rpm > 0.0:
            if diff < 0.0:
                diff += 2.0 * math.pi
        elif ref_speed_rpm < 0.0:
            if diff > 0.0:
                diff -= 2.0 * math.pi

        abs_err = abs(diff)

        # Gain scheduling (same as C)
        threshold = 0.1  # ES_ANGLE_CORR_THRESHOLD_RAD
        if abs_err < threshold:
            gain = 1.0
        else:
            gain = 0.1 + 0.9 * (threshold / abs_err)
            gain = self._clamp(gain, 0.1, 1.0)

        # Apply correction
        self.theta_model += gain * diff

        # Half‑sample delay feedforward (when locked)
        if abs_err < threshold:
            omega_e = (speed_rpm * (2.0 * math.pi / 60.0)) * self.pole_pairs
            feedforward = omega_e * 0.5 * dt
            feedforward = self._clamp(feedforward, -0.01, 0.01)
            self.theta_model += feedforward

        # Wrap
        self.theta_model = self._wrap_angle(self.theta_model)
        return self.theta_model

    # ------------------------------------------------------------------
    # Direct PWM generation (matches C's direct clamping)
    # ------------------------------------------------------------------

    def _duty_from_alpha_beta(self, valpha, vbeta, vdc):
        """Inverse Clarke + clamping – no modulation index limit."""
        vu, vv, vw = self.inv_clarke(valpha, vbeta)

        vhalf = vdc / 2.0
        if vhalf <= 0.0:
            return 0.5, 0.5, 0.5

        duty_u = self._clamp((vu / vhalf + 1.0) / 2.0, 0.0, 1.0)
        duty_v = self._clamp((vv / vhalf + 1.0) / 2.0, 0.0, 1.0)
        duty_w = self._clamp((vw / vhalf + 1.0) / 2.0, 0.0, 1.0)
        return duty_u, duty_v, duty_w

    # ------------------------------------------------------------------
    # Main compute method
    # ------------------------------------------------------------------

    def compute_py(self, t, dt, input_values=None):
        """
        Main controller step - EXACTLY ALIGNED WITH C.
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

        # ================================================================
        # Diagnostic prints (rate‑limited)
        # ================================================================
        print_interval = 0.2
        if t - self._last_print >= print_interval:
            self._last_print = t
            self._print_counter += 1
            mode_str = "CLOSED" if self.switch_to_closed_loop else "OPEN"
            print(
                f"[C‑Aligned DFC t={t:.2f}s] "
                f"mode={mode_str}  "
                f"ref={speed_ref_rpm:.1f} RPM  "
                f"meas={speed_sensor_rpm:.1f} RPM  "
                f"traj={self.trajectory.speed:.1f} RPM  "
                f"mod={self.startup_modulation:.3f}  "
                f"timer={self.startup_timer:.2f}s"
            )

        # ================================================================
        # 1. Update trajectory (only in closed‑loop, matches C)
        # ================================================================
        if self.switch_to_closed_loop:
            ref = self.trajectory.update(speed_ref_rpm, dt, reinit=self.control_reinit)
            self.control_reinit = False
        else:
            ref = self.trajectory._output()

        # ================================================================
        # 2. Measure currents in dq
        # ================================================================
        ialpha, ibeta = self.clarke(ia, ib, ic)
        theta_elec_sensor = position_sensor_rad * self.pole_pairs
        theta_elec_sensor = self._wrap_angle(theta_elec_sensor)
        id_meas, iq_meas = self.park(ialpha, ibeta, theta_elec_sensor)

        # ================================================================
        # 3. Startup phase – LINEAR RAMP (matches C)
        # ================================================================
        if not self.switch_to_closed_loop:
            self.startup_timer += dt
            tau = self.startup_timer / self.startup_duration   # <-- now uses configurable duration
            tau = self._clamp(tau, 0.0, 1.0)

            # LINEAR interpolation (NO smoothstep)
            self.startup_modulation = self.startup_mod_min + tau * (self.startup_mod_max - self.startup_mod_min)
            self.startup_modulation = self._clamp(self.startup_modulation, self.startup_mod_min, self.startup_mod_max)

            omega_e_startup = self.pole_pairs * (speed_ref_rpm * (2.0 * math.pi / 60.0))
            self.theta_open_loop += omega_e_startup * dt
            self.theta_open_loop = self._wrap_angle(self.theta_open_loop)

            startup_vd = 0.0
            startup_vq = (vdc / math.sqrt(3.0)) * self.startup_modulation
            valpha, vbeta = self.inv_park(startup_vd, startup_vq, self.theta_open_loop)
            duty_u, duty_v, duty_w = self._duty_from_alpha_beta(valpha, vbeta, vdc)

            if self.startup_timer >= self.startup_duration:   # <-- now uses configurable duration
                self.switch_to_closed_loop = True
                print(f"[C‑Aligned DFC t={t:.2f}s] SWITCHING TO CLOSED‑LOOP (timer)")
                self.speed_integral = 0.0
                self.id_integral = 0.0
                self.iq_integral = 0.0
                self.control_reinit = True
                self.startup_modulation = 0.0
                self.theta_model = position_sensor_rad * self.pole_pairs
                self.theta_model = self._wrap_angle(self.theta_model)

            out = np.array([duty_u, duty_v, duty_w, 1.0], dtype=DEFAULT_DTYPE)
            self.output = VectorSignal(out, self.name)
            return self.output

        # ================================================================
        # 4. Closed‑loop DFC
        # ================================================================

        # Observer update
        ref_speed_rpm = self.trajectory.speed
        self._observer_update(position_sensor_rad, speed_sensor_rpm, ref_speed_rpm, dt)
        theta_elec = self.theta_model

        omega_ref = ref["omega_ref"]
        omega_dot = ref["omega_dot"]
        omega_ddot = ref["omega_ddot"]
        omega_meas = speed_sensor_rpm * (2.0 * math.pi / 60.0)

        # Speed PI
        speed_error = omega_ref - omega_meas
        self.speed_integral += speed_error * dt
        self.speed_integral = self._clamp(self.speed_integral, -self.integral_limit, self.integral_limit)
        torque_correction = self.Kp_speed * speed_error + self.Ki_speed * self.speed_integral

        # Mechanical flatness
        torque_ff = self.J * omega_dot + self.B * omega_ref + self.Tload
        torque_required = torque_ff + torque_correction

        # Electrical flatness
        torque_constant = 1.5 * self.pole_pairs * self.lambda_pm
        if abs(torque_constant) > 1.0e-6:
            iq_ref = self._clamp(torque_required / torque_constant, -self.max_current, self.max_current)
            iq_ref_dot = self._clamp(
                (self.J * omega_ddot + self.B * omega_dot) / torque_constant,
                -self.max_iq_dot, self.max_iq_dot
            )
        else:
            iq_ref = 0.0
            iq_ref_dot = 0.0

        # Voltage feedforward
        omega_e_ref = self.pole_pairs * omega_ref
        vd_ff = -omega_e_ref * self.Lq * iq_ref
        vq_ff = self.Rs * iq_ref + self.Lq * iq_ref_dot + omega_e_ref * self.lambda_pm

        # Current PI
        id_ref = 0.0
        id_error = id_ref - id_meas
        iq_error = iq_ref - iq_meas

        self.id_integral += id_error * dt
        self.iq_integral += iq_error * dt
        self.id_integral = self._clamp(self.id_integral, -self.integral_limit, self.integral_limit)
        self.iq_integral = self._clamp(self.iq_integral, -self.integral_limit, self.integral_limit)

        id_error = self._clamp(id_error, -self.max_current, self.max_current)
        iq_error = self._clamp(iq_error, -self.max_current, self.max_current)

        vd_corr = self.Kp_d * id_error + self.Ki_d * self.id_integral
        vq_corr = self.Kp_q * iq_error + self.Ki_q * self.iq_integral

        vd_ref = vd_ff + vd_corr
        vq_ref = vq_ff + vq_corr

        # Inverse Park + direct PWM
        valpha, vbeta = self.inv_park(vd_ref, vq_ref, theta_elec)
        duty_u, duty_v, duty_w = self._duty_from_alpha_beta(valpha, vbeta, vdc)

        out = np.array([duty_u, duty_v, duty_w, 1.0], dtype=DEFAULT_DTYPE)
        self.output = VectorSignal(out, self.name)
        return self.output

    # ------------------------------------------------------------------
    def reset(self):
        super().reset()
        self.speed_integral = 0.0
        self.id_integral = 0.0
        self.iq_integral = 0.0
        self.startup_timer = 0.0
        self.startup_modulation = 0.0
        self.theta_open_loop = 0.0
        self.theta_model = 0.0
        self.switch_to_closed_loop = False
        self.control_reinit = False
        self.trajectory.reset()
        self._last_print = -1.0
        self._print_counter = 0

    # ------------------------------------------------------------------
    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)