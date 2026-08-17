"""
pmsm_dfc1.py
===========

PMSM Control - Single Python Controller with Mode Switching

Architecture
------------

    Speed Reference
          |
          v
    Jerk-Limited S-Curve
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
    Small Speed Feedback Correction
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
# Jerk-Limited Speed Trajectory
# =============================================================================

class SpeedTrajectory:
    """
    Jerk-limited speed trajectory generator.

    State:

        speed  [RPM]
        accel  [RPM/s]
        jerk   [RPM/s^2]

    Outputs:

        omega_ref       [rad/s]
        omega_dot       [rad/s^2]
        omega_ddot      [rad/s^3]

    The trajectory is generated recursively:

        jerk
          |
          v
        acceleration
          |
          v
        speed

    The algorithm limits both acceleration and jerk and
    automatically reduces acceleration when approaching
    the target speed.
    """

    def __init__(
            self,
            max_speed_rpm=3000.0,
            max_accel_rpm_s=500.0,
            max_jerk_rpm_s2=3000.0,
    ):

        self.max_speed_rpm = float(max_speed_rpm)
        self.max_accel_rpm_s = float(max_accel_rpm_s)
        self.max_jerk_rpm_s2 = float(max_jerk_rpm_s2)

        # Trajectory states
        self.speed = 0.0
        self.accel = 0.0
        self.jerk = 0.0

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
        Update trajectory by one controller sample.

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
        # Speed error
        # ------------------------------------------------------------

        error = target - self.speed

        # ------------------------------------------------------------
        # Target reached
        # ------------------------------------------------------------

        if (
                abs(error) < 0.01
                and abs(self.accel) < 0.01
        ):
            self.speed = target
            self.accel = 0.0
            self.jerk = 0.0

            return self._output()

        # ------------------------------------------------------------
        # Direction toward target
        # ------------------------------------------------------------

        if error >= 0.0:
            direction = 1.0
        else:
            direction = -1.0

        # ------------------------------------------------------------
        # Estimate acceleration required to stop the velocity error.
        #
        # This creates the braking portion of the S-curve.
        #
        # a_stop^2 = 2 * J * |error|
        # ------------------------------------------------------------

        stopping_accel = math.sqrt(
            max(
                0.0,
                2.0
                * self.max_jerk_rpm_s2
                * abs(error),
            )
        )

        # ------------------------------------------------------------
        # Desired acceleration
        # ------------------------------------------------------------

        desired_accel = direction * min(
            self.max_accel_rpm_s,
            stopping_accel,
        )

        # ------------------------------------------------------------
        # Required jerk
        # ------------------------------------------------------------

        jerk_request = (
                               desired_accel - self.accel
                       ) / dt

        self.jerk = self._clamp(
            jerk_request,
            -self.max_jerk_rpm_s2,
            self.max_jerk_rpm_s2,
        )

        # ------------------------------------------------------------
        # Integrate jerk -> acceleration
        # ------------------------------------------------------------

        self.accel += self.jerk * dt

        self.accel = self._clamp(
            self.accel,
            -self.max_accel_rpm_s,
            self.max_accel_rpm_s,
        )

        # ------------------------------------------------------------
        # Integrate acceleration -> speed
        # ------------------------------------------------------------

        self.speed += self.accel * dt

        self.speed = self._clamp(
            self.speed,
            -self.max_speed_rpm,
            self.max_speed_rpm,
        )

        # ------------------------------------------------------------
        # Prevent overshoot
        # ------------------------------------------------------------

        new_error = target - self.speed

        if (
                (error > 0.0 and new_error < 0.0)
                or
                (error < 0.0 and new_error > 0.0)
        ):
            self.speed = target
            self.accel = 0.0
            self.jerk = 0.0

        return self._output()

    # ------------------------------------------------------------------
    def _output(self):

        rpm_to_rad_s = (
                2.0 * math.pi / 60.0
        )

        return {
            "omega_ref":
                self.speed * rpm_to_rad_s,

            "omega_dot":
                self.accel * rpm_to_rad_s,

            "omega_ddot":
                self.jerk * rpm_to_rad_s,
        }


# =============================================================================
# Universal Python Controller
# =============================================================================

class PythonController(GenericControlBlock):
    """
    Universal PMSM controller.

    Modes:

        OPEN_LOOP
        DFC

    DFC architecture:

        S-Curve
            |
            +--> omega_ref
            +--> omega_dot
            +--> omega_ddot
                    |
                    v
              Differential Flatness
                    |
                    +--> iq_ff
                    +--> iq_dot_ff
                    +--> vd_ff
                    +--> vq_ff
                    |
                    v
              Speed correction
                    |
                    v
                  iq_ref
                    |
                    v
                Current PI
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
        # PMSM parameters
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
        # Speed feedback correction
        # ============================================================

        self.Kp_speed =  0.2    #0.2
        self.Ki_speed =  1.0    #1.0

        self.speed_integral = 0.0

        self.speed_limit = 12.0

        # ============================================================
        # Current PI
        # ============================================================

        self.Kp_d = 0.1
        self.Kp_q = 0.5

        self.Ki_d = 0.5
        self.Ki_q = 2.0

        self.id_integral = 0.0
        self.iq_integral = 0.0

        # ============================================================
        # Limits
        # ============================================================

        self.max_current = 50.0

        self.max_iq_dot = 2000.0

        self.integral_limit = 15.0

        self.modulation_limit = 0.90

        # ============================================================
        # Jerk-limited trajectory
        # ============================================================

        self.trajectory = SpeedTrajectory(
            max_speed_rpm=3000.0,
            max_accel_rpm_s=500.0,
            max_jerk_rpm_s2=3000.0,
        )

        # ============================================================
        # Startup
        # ============================================================

        self.startup_time = 0.3

        self.startup_speed = 300.0

        self.theta_open_loop = 0.0

        # ============================================================
        # Diagnostics
        # ============================================================

        self._last_print = -1.0

        print(f"\n{'=' * 70}")
        print(
            f" PYTHON CONTROLLER - Mode: "
            f"{controller_mode}"
        )
        print(f"{'=' * 70}")

        if controller_mode == "OPEN_LOOP":

            print(
                f"  Amplitude: "
                f"{self.amp}"
            )

            print(
                f"  Ramp rate: "
                f"{self.ramp_rate} Hz/s"
            )

            print(
                "  Follows speed_ref "
                "from VectorStep"
            )

        else:

            print(
                f"  Speed correction: "
                f"Kp={self.Kp_speed}, "
                f"Ki={self.Ki_speed}"
            )

            print(
                f"  Current PI: "
                f"Kp_d={self.Kp_d}, "
                f"Kp_q={self.Kp_q}"
            )

            print(
                f"  Current PI: "
                f"Ki_d={self.Ki_d}, "
                f"Ki_q={self.Ki_q}"
            )

            print(
                f"  S-curve: "
                f"Jmax={self.trajectory.max_jerk_rpm_s2:.1f} RPM/s²"
            )

            print(
                f"  S-curve: "
                f"Amax={self.trajectory.max_accel_rpm_s:.1f} RPM/s"
            )

            print(
                f"  Startup: "
                f"{self.startup_time * 1000:.0f}ms "
                f"at {self.startup_speed:.0f} RPM"
            )

        print(f"{'=' * 70}\n")

    # ==================================================================
    # Utility
    # ==================================================================

    def compute(self, t, dt, input_values=None):
        return self.compute_py(
            t,
            dt,
            input_values,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(value, minimum, maximum):
        return max(
            minimum,
            min(maximum, value),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(angle):

        angle = angle % (
                2.0 * math.pi
        )

        if angle < 0.0:
            angle += 2.0 * math.pi

        return angle

    # ==================================================================
    # Space Vector Modulation
    # ==================================================================

    def _svm(
            self,
            valpha,
            vbeta,
            vdc,
    ):
        """
        Convert alpha-beta voltage to PWM duties.
        """

        v_mag = math.sqrt(
            valpha * valpha
            + vbeta * vbeta
        )

        v_max = (
                vdc / math.sqrt(3.0)
        )

        if v_mag > 0.0:
            mod_idx = self._clamp(
                v_mag / v_max,
                0.0,
                self.modulation_limit,
            )

            valpha = (
                    valpha
                    / v_mag
                    * mod_idx
                    * v_max
            )

            vbeta = (
                    vbeta
                    / v_mag
                    * mod_idx
                    * v_max
            )

        vu, vv, vw = inv_clarke(
            valpha,
            vbeta,
        )

        vmax = vdc / 2.0

        if vmax > 0.0:
            duty_u = self._clamp(
                (vu / vmax + 1.0) / 2.0,
                0.0,
                1.0,
            )

            duty_v = self._clamp(
                (vv / vmax + 1.0) / 2.0,
                0.0,
                1.0,
            )

            duty_w = self._clamp(
                (vw / vmax + 1.0) / 2.0,
                0.0,
                1.0,
            )

            return (
                duty_u,
                duty_v,
                duty_w,
            )

        return (
            0.5,
            0.5,
            0.5,
        )

    # ==================================================================
    # Main controller
    # ==================================================================

    def compute_py(
            self,
            t,
            dt,
            input_values=None,
    ):

        # ============================================================
        # Input vector
        #
        # [0] speed_ref_rpm
        # [1] ia
        # [2] ib
        # [3] ic
        # [4] speed_sensor_rpm
        # [5] sample_time
        # [6] position_sensor_rad
        # [7] valid
        # [8] unused
        # [9] Vdc
        # ============================================================

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

        # Avoid unused-variable warnings in some linters
        _ = sample_time
        _ = valid_in

        # ============================================================
        # Diagnostics
        # ============================================================

        if (
                t - self._last_print
                >= 0.2
        ):

            self._last_print = t

            if (
                    self.controller_mode
                    == "OPEN_LOOP"
            ):

                target_freq = (
                        speed_ref_rpm
                        * self.pole_pairs
                        / 60.0
                )

                print(
                    f"[OpenLoop t={t:.2f}s] "
                    f"speed_ref="
                    f"{speed_ref_rpm:.1f} RPM  "
                    f"freq="
                    f"{self._current_freq:.1f}Hz  "
                    f"speed="
                    f"{speed_sensor_rpm:.1f} RPM"
                )

            else:

                print(
                    f"[DFC t={t:.2f}s] "
                    f"speed_ref="
                    f"{speed_ref_rpm:.1f} RPM  "
                    f"speed="
                    f"{speed_sensor_rpm:.1f} RPM  "
                    f"traj="
                    f"{self.trajectory.speed:.1f} RPM  "
                    f"acc="
                    f"{self.trajectory.accel:.1f} "
                    f"RPM/s  "
                    f"jerk="
                    f"{self.trajectory.jerk:.1f} "
                    f"RPM/s²"
                )

        # ============================================================
        # Electrical rotor angle
        #
        # Controller receives mechanical rotor angle.
        #
        # theta_e = p * theta_m
        # ============================================================

        theta_elec = (
                position_sensor_rad
                * self.pole_pairs
        )

        theta_elec %= (
                2.0 * math.pi
        )

        # ============================================================
        # Clarke transform
        # ============================================================

        ialpha, ibeta = clarke(
            ia,
            ib,
            ic,
        )

        # ============================================================
        # Park transform
        # ============================================================

        id_meas, iq_meas = park(
            ialpha,
            ibeta,
            theta_elec,
        )

        # ============================================================
        # OPEN LOOP MODE
        # ============================================================

        if (
                self.controller_mode
                == "OPEN_LOOP"
        ):

            target_freq = (
                    speed_ref_rpm
                    * self.pole_pairs
                    / 60.0
            )

            freq_error = (
                    target_freq
                    - self._current_freq
            )

            max_change = (
                    self.ramp_rate
                    * dt
            )

            if abs(freq_error) > max_change:

                self._current_freq += (
                        max_change
                        * np.sign(freq_error)
                )

            else:

                self._current_freq = (
                    target_freq
                )

            self.theta += (
                    2.0
                    * math.pi
                    * self._current_freq
                    * dt
            )

            amp = self.amp

            duty_u = (
                    0.5
                    + amp
                    * math.sin(self.theta)
            )

            duty_v = (
                    0.5
                    + amp
                    * math.sin(
                self.theta
                - 2.0 * math.pi / 3.0
            )
            )

            duty_w = (
                    0.5
                    + amp
                    * math.sin(
                self.theta
                - 4.0 * math.pi / 3.0
            )
            )

            duty_u = np.clip(
                duty_u,
                0.0,
                1.0,
            )

            duty_v = np.clip(
                duty_v,
                0.0,
                1.0,
            )

            duty_w = np.clip(
                duty_w,
                0.0,
                1.0,
            )

            out = np.array(
                [
                    duty_u,
                    duty_v,
                    duty_w,
                    1.0,
                ],
                dtype=DEFAULT_DTYPE,
            )

            self.output = VectorSignal(
                out,
                self.name,
            )

            return self.output

        # ============================================================
        # DFC STARTUP
        # ============================================================

        if t < self.startup_time:
            ramp = min(
                t / self.startup_time,
                1.0,
            )

            modulation = (
                    0.05
                    + ramp * 0.15
            )

            self.theta_open_loop += (
                    2.0
                    * math.pi
                    * self.startup_speed
                    / 60.0
                    * self.pole_pairs
                    * dt
            )

            vd_ref = 0.0

            vq_ref = (
                    vdc
                    / math.sqrt(3.0)
                    * modulation
            )

            valpha, vbeta = inv_park(
                vd_ref,
                vq_ref,
                self.theta_open_loop,
            )

            duty_u, duty_v, duty_w = (
                self._svm(
                    valpha,
                    vbeta,
                    vdc,
                )
            )

            out = np.array(
                [
                    duty_u,
                    duty_v,
                    duty_w,
                    1.0,
                ],
                dtype=DEFAULT_DTYPE,
            )

            self.output = VectorSignal(
                out,
                self.name,
            )

            return self.output

        # ============================================================
        # S-CURVE TRAJECTORY
        # ============================================================

        ref = self.trajectory.update(
            speed_ref_rpm,
            dt,
        )

        omega_ref = ref[
            "omega_ref"
        ]

        omega_dot = ref[
            "omega_dot"
        ]

        omega_ddot = ref[
            "omega_ddot"
        ]

        # ============================================================
        # MECHANICAL FLATNESS
        #
        # J*domega/dt =
        #       T_em - B*omega - Tload
        #
        # Therefore:
        #
        # T_required =
        #       J*omega_dot
        #       + B*omega
        #       + Tload
        # ============================================================

        torque_required = (
                self.J * omega_dot
                + self.B * omega_ref
                + self.Tload
        )

        # ============================================================
        # Electrical Flatness
        #
        # T = 1.5*p*lambda_pm*iq
        #
        # Therefore:
        #
        # iq_ff =
        #       T_required /
        #       (1.5*p*lambda_pm)
        # ============================================================

        torque_constant = (
                1.5
                * self.pole_pairs
                * self.lambda_pm
        )

        if abs(torque_constant) > 1.0e-6:

            iq_ref_ff = (
                    torque_required
                    / torque_constant
            )

            iq_ref_ff = self._clamp(
                iq_ref_ff,
                -self.max_current,
                self.max_current,
            )

            # --------------------------------------------------------
            # derivative of iq reference
            #
            # diq/dt =
            #   (J*omega_ddot + B*omega_dot)
            #   / Kt
            # --------------------------------------------------------

            iq_ref_dot = (
                                 self.J * omega_ddot
                                 + self.B * omega_dot
                         ) / torque_constant

            iq_ref_dot = self._clamp(
                iq_ref_dot,
                -self.max_iq_dot,
                self.max_iq_dot,
            )

        else:

            iq_ref_ff = 0.0
            iq_ref_dot = 0.0

        # ============================================================
        # SPEED FEEDBACK CORRECTION
        #
        # IMPORTANT:
        #
        # The speed PI does NOT create the main trajectory.
        #
        # The S-curve + DFC creates iq_ref_ff.
        #
        # Speed feedback provides only a correction.
        # ============================================================

        speed_error = (
                speed_ref_rpm
                - speed_sensor_rpm
        )

        self.speed_integral += (
                self.Ki_speed
                * speed_error
                * dt
        )

        self.speed_integral = self._clamp(
            self.speed_integral,
            -self.speed_limit,
            self.speed_limit,
        )

        iq_correction = (
                self.Kp_speed
                * speed_error
                + self.speed_integral
        )

        iq_correction = self._clamp(
            iq_correction,
            -self.speed_limit,
            self.speed_limit,
        )

        # ============================================================
        # Final q-axis current reference
        #
        # iq_ref =
        #       feedforward
        #       +
        #       feedback correction
        # ============================================================

        iq_ref = (
                iq_ref_ff
                + iq_correction
        )

        iq_ref = self._clamp(
            iq_ref,
            -self.max_current,
            self.max_current,
        )

        # d-axis reference
        id_ref = 0.0

        # ============================================================
        # FLATNESS VOLTAGE FEEDFORWARD
        #
        # Electrical speed:
        #
        # omega_e = p * omega_m
        #
        # For Ld = Lq:
        #
        # vd_ff =
        #       -omega_e * Lq * iq
        #
        # vq_ff =
        #       Rs*iq
        #       + Lq*diq/dt
        #       + omega_e*lambda_pm
        # ============================================================

        omega_e_ref = (
                self.pole_pairs
                * omega_ref
        )

        vd_ff = (
                -omega_e_ref
                * self.Lq
                * iq_ref_ff
        )

        vq_ff = (
                self.Rs * iq_ref_ff
                + self.Lq * iq_ref_dot
                + omega_e_ref
                * self.lambda_pm
        )

        # ============================================================
        # CURRENT PI
        # ============================================================

        id_error = (
                id_ref
                - id_meas
        )

        iq_error = (
                iq_ref
                - iq_meas
        )

        # ------------------------------------------------------------
        # d-axis integrator
        # ------------------------------------------------------------

        self.id_integral += (
                self.Ki_d
                * id_error
                * dt
        )

        self.id_integral = self._clamp(
            self.id_integral,
            -self.integral_limit,
            self.integral_limit,
        )

        # ------------------------------------------------------------
        # q-axis integrator
        # ------------------------------------------------------------

        self.iq_integral += (
                self.Ki_q
                * iq_error
                * dt
        )

        self.iq_integral = self._clamp(
            self.iq_integral,
            -self.integral_limit,
            self.integral_limit,
        )

        # ============================================================
        # Final voltage references
        # ============================================================

        vd_ref = (
                vd_ff
                + self.Kp_d * id_error
                + self.id_integral
        )

        vq_ref = (
                vq_ff
                + self.Kp_q * iq_error
                + self.iq_integral
        )

        # ============================================================
        # Inverse Park
        # ============================================================

        valpha, vbeta = inv_park(
            vd_ref,
            vq_ref,
            theta_elec,
        )

        # ============================================================
        # SVM
        # ============================================================

        duty_u, duty_v, duty_w = (
            self._svm(
                valpha,
                vbeta,
                vdc,
            )
        )

        # ============================================================
        # Output
        # ============================================================

        out = np.array(
            [
                duty_u,
                duty_v,
                duty_w,
                1.0,
            ],
            dtype=DEFAULT_DTYPE,
        )

        self.output = VectorSignal(
            out,
            self.name,
        )

        return self.output

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(self):

        super().reset()

        # Open-loop state
        self.theta = 0.0
        self._current_freq = 0.0

        # Speed feedback
        self.speed_integral = 0.0

        # Current PI
        self.id_integral = 0.0
        self.iq_integral = 0.0

        # Startup
        self.theta_open_loop = 0.0

        # Trajectory
        self.trajectory.reset()

        # Diagnostics
        self._last_print = -1.0