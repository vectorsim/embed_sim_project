"""
pmsm_dfc.py  -  PMSM Control - Single Controller with Mode Switching
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

# ================================================================
# Path setup
# ================================================================
from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE = get_current_parent()
_ROOT = get_project_root()
_PMSM = _ROOT / "pmsm"
_C_SRC = _PMSM / "c_src"

for _p in (get_embedsim_import_path(), str(_PMSM), str(_C_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ================================================================
# Imports
# ================================================================

import numpy as np
from embedsim.core_blocks import VectorSignal, DEFAULT_DTYPE
from embedsim_generic_control import GenericControlBlock

# Import transforms from the C wrapper
# These are used by both the Python controller and the plant
from embedsim_control_wrapper import clarke, park, inv_park, inv_clarke


# =============================================================================
# Universal Python Controller - Handles both Open Loop and DFC
# =============================================================================

class PythonController(GenericControlBlock):
    """
    Universal Python controller.
    Mode is set via controller_mode parameter.

    Uses the same transform functions as the C backend (from embedsim_control_wrapper)
    to ensure consistency between Python and C implementations.
    """

    def __init__(self, name="ctrl", dt_s=50e-6, vdc_nom=12.0,
                 controller_mode="DFC", **kwargs):
        super().__init__(name=name, dt_s=dt_s, vdc_nom=vdc_nom,
                        use_c_backend=False, **kwargs)

        self.controller_mode = controller_mode  # "OPEN_LOOP" or "DFC"

        # Motor parameters (must match C backend parameters)
        self.pole_pairs = 4.0
        self.Rs = 0.19
        self.Ld = 0.125e-3
        self.Lq = 0.125e-3
        self.lambda_pm = 0.0014
        self.J = 2.4e-6
        self.B = 1.0e-6
        self.Tload = 0.0

        # Open-loop parameters
        self.theta = 0.0
        self.amp = 0.3
        self.ramp_rate = 200.0  # Hz/s - reaches target in ~0.33 seconds

        # Speed PI (for DFC)
        self.Kp_speed = 0.2
        self.Ki_speed = 1.0
        self.speed_integral = 0.0
        self.speed_limit = 12.0

        # Current PI (for DFC)
        self.Kp_d = 0.1
        self.Kp_q = 0.5
        self.Ki_d = 0.5
        self.Ki_q = 2.0
        self.id_integral = 0.0
        self.iq_integral = 0.0

        # Limits
        self.max_current = 50.0
        self.max_iq_dot = 2000.0
        self.integral_limit = 15.0
        self.modulation_limit = 0.90

        # S-curve parameters (for DFC)
        self.max_jerk_rpm = 3000.0
        self.jerk_smoothing = 0.6
        self.closed_loop_min_speed = 50.0

        # Startup parameters (for DFC)
        self.startup_time = 0.3
        self.startup_speed = 300.0
        self.theta_open_loop = 0.0

        # S-curve state (for DFC)
        self.current_speed_rpm = 0.0
        self.current_accel_rpm = 0.0
        self.current_jerk_rpm = 0.0
        self.current_position_rad = 0.0
        self.is_rolling = False

        self._last_print = -1.0
        self._current_freq = 0.0

        print(f"\n{'='*70}")
        print(f" PYTHON CONTROLLER - Mode: {controller_mode}")
        print(f"{'='*70}")
        if controller_mode == "OPEN_LOOP":
            print(f"  Amplitude: {self.amp}")
            print(f"  Ramp rate: {self.ramp_rate} Hz/s")
            print(f"  Follows speed_ref from VectorStep")
        else:
            print(f"  Speed PI: Kp={self.Kp_speed}, Ki={self.Ki_speed}")
            print(f"  Current PI: Kp_d={self.Kp_d}, Kp_q={self.Kp_q}")
            print(f"  Current PI: Ki_d={self.Ki_d}, Ki_q={self.Ki_q}")
            print(f"  Startup: {self.startup_time*1000:.0f}ms at {self.startup_speed:.0f} RPM")
        print(f"{'='*70}\n")

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)

    def _clamp(self, val, min_val, max_val):
        return max(min_val, min(max_val, val))

    def _wrap_angle(self, angle):
        angle = angle % (2.0 * math.pi)
        return angle if angle >= 0 else angle + 2.0 * math.pi

    def _smooth_jerk(self, raw_jerk, prev_jerk):
        return self.jerk_smoothing * raw_jerk + (1.0 - self.jerk_smoothing) * prev_jerk

    def _svm(self, valpha, vbeta, vdc):
        """Space Vector Modulation - matches C backend implementation"""
        v_mag = np.sqrt(valpha**2 + vbeta**2)
        v_max = vdc / math.sqrt(3.0)

        if v_mag > 0:
            mod_idx = self._clamp(v_mag / v_max, 0.0, self.modulation_limit)
            valpha = valpha / v_mag * mod_idx * v_max
            vbeta = vbeta / v_mag * mod_idx * v_max

        vu, vv, vw = inv_clarke(valpha, vbeta)
        vmax = vdc / 2.0

        if vmax > 0:
            return (self._clamp((vu / vmax + 1.0) / 2.0, 0.0, 1.0),
                    self._clamp((vv / vmax + 1.0) / 2.0, 0.0, 1.0),
                    self._clamp((vw / vmax + 1.0) / 2.0, 0.0, 1.0))
        return 0.5, 0.5, 0.5

    def _s_curve(self, speed_ref_rpm, speed_sensor_rpm, dt, position_sensor):
        """S-curve trajectory generator - matches C backend implementation"""
        target_speed = self._clamp(speed_ref_rpm, -3000.0, 3000.0)
        speed_error = target_speed - self.current_speed_rpm
        abs_speed_error = abs(speed_error)

        if abs_speed_error < 0.1:
            self.current_speed_rpm = target_speed
            self.current_accel_rpm = 0.0
            self.current_jerk_rpm = 0.0
        else:
            accel_target = speed_error * 0.2
            raw_jerk = (accel_target - self.current_accel_rpm) / dt if dt > 0 else 0.0
            raw_jerk = self._clamp(raw_jerk, -self.max_jerk_rpm, self.max_jerk_rpm)
            self.current_jerk_rpm = self._smooth_jerk(raw_jerk, self.current_jerk_rpm)
            self.current_accel_rpm += self.current_jerk_rpm * dt
            self.current_speed_rpm += self.current_accel_rpm * dt

            if (speed_error > 0.0 and self.current_speed_rpm > target_speed) or \
               (speed_error < 0.0 and self.current_speed_rpm < target_speed):
                self.current_speed_rpm = target_speed
                self.current_accel_rpm = 0.0
                self.current_jerk_rpm = 0.0

        current_speed_rad = self.current_speed_rpm * 2.0 * math.pi / 60.0

        if self.is_rolling:
            if abs(speed_sensor_rpm) > self.closed_loop_min_speed:
                self.current_position_rad = position_sensor
                switch_to_closed_loop = True
            else:
                switch_to_closed_loop = False
                self.current_position_rad += current_speed_rad * dt
                self.current_position_rad = self._wrap_angle(self.current_position_rad)
        else:
            if abs(self.current_speed_rpm) > self.closed_loop_min_speed:
                self.is_rolling = True
                switch_to_closed_loop = True
                self.current_position_rad = position_sensor
            else:
                switch_to_closed_loop = False
                self.current_position_rad += current_speed_rad * dt
                self.current_position_rad = self._wrap_angle(self.current_position_rad)

        return {
            'omega_ref': current_speed_rad,
            'omega_dot': self.current_accel_rpm * 2.0 * math.pi / 60.0,
            'omega_ddot': self.current_jerk_rpm * 2.0 * math.pi / 60.0,
            'switch_to_closed_loop': switch_to_closed_loop
        }

    def compute_py(self, t, dt, input_values=None):
        """
        Main computation method.
        Uses transforms from embedsim_control_wrapper for consistency with C backend.
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

        # Rate-limited debug prints
        if (t - self._last_print) >= 0.2:
            self._last_print = t
            if self.controller_mode == "OPEN_LOOP":
                target_freq = (speed_ref_rpm * self.pole_pairs) / 60.0
                print(f"[OpenLoop t={t:.2f}s] speed_ref={speed_ref_rpm:.1f} RPM  freq={self._current_freq:.1f}Hz  speed={speed_sensor_rpm:.1f} RPM")
            else:
                mode = "STARTUP" if t < self.startup_time else "DFC"
                print(f"[DFC t={t:.2f}s] mode={mode} speed_ref={speed_ref_rpm:.1f}  speed={speed_sensor_rpm:.1f}")

        # Compute electrical angle
        theta_elec = position_sensor_rad * self.pole_pairs
        theta_elec = theta_elec % (2.0 * math.pi)

        # Clarke and Park transforms using C wrapper functions
        ialpha, ibeta = clarke(ia, ib, ic)
        id_meas, iq_meas = park(ialpha, ibeta, theta_elec)

        # ================================================================
        # OPEN-LOOP MODE - Follows speed_ref with fast ramp
        # ================================================================
        if self.controller_mode == "OPEN_LOOP":
            # Calculate target frequency from speed reference
            target_freq = (speed_ref_rpm * self.pole_pairs) / 60.0

            # Ramp frequency to avoid sudden changes
            freq_error = target_freq - self._current_freq
            max_change = self.ramp_rate * dt
            if abs(freq_error) > max_change:
                self._current_freq += max_change * np.sign(freq_error)
            else:
                self._current_freq = target_freq

            # Update angle
            self.theta += 2 * np.pi * self._current_freq * dt

            # Generate sinusoidal voltages
            amp = self.amp
            duty_u = 0.5 + amp * np.sin(self.theta)
            duty_v = 0.5 + amp * np.sin(self.theta - 2*np.pi/3)
            duty_w = 0.5 + amp * np.sin(self.theta - 4*np.pi/3)

            duty_u = np.clip(duty_u, 0.0, 1.0)
            duty_v = np.clip(duty_v, 0.0, 1.0)
            duty_w = np.clip(duty_w, 0.0, 1.0)

            out = np.array([duty_u, duty_v, duty_w, 1.0], dtype=DEFAULT_DTYPE)
            self.output = VectorSignal(out, self.name)
            return self.output

        # ================================================================
        # DFC MODE
        # ================================================================

        # OPEN-LOOP STARTUP
        if t < self.startup_time:
            ramp = min(t / self.startup_time, 1.0)
            modulation = 0.05 + ramp * 0.15

            self.theta_open_loop += 2 * np.pi * self.startup_speed / 60 * self.pole_pairs * dt

            vd_ref = 0.0
            vq_ref = (vdc / math.sqrt(3.0)) * modulation

            valpha, vbeta = inv_park(vd_ref, vq_ref, self.theta_open_loop)
            duty_u, duty_v, duty_w = self._svm(valpha, vbeta, vdc)

            out = np.array([duty_u, duty_v, duty_w, 1.0], dtype=DEFAULT_DTYPE)
            self.output = VectorSignal(out, self.name)
            return self.output

        # CLOSED-LOOP DFC
        ref = self._s_curve(speed_ref_rpm, speed_sensor_rpm, dt, position_sensor_rad)

        omega_ref = ref['omega_ref']
        omega_dot = ref['omega_dot']
        omega_ddot = ref['omega_ddot']
        switch_to_closed_loop = ref['switch_to_closed_loop']

        if not switch_to_closed_loop:
            modulation = 0.1
            self.theta_open_loop += 2 * np.pi * self.startup_speed / 60 * self.pole_pairs * dt
            vd_ref = 0.0
            vq_ref = (vdc / math.sqrt(3.0)) * modulation
            valpha, vbeta = inv_park(vd_ref, vq_ref, self.theta_open_loop)
            duty_u, duty_v, duty_w = self._svm(valpha, vbeta, vdc)
            out = np.array([duty_u, duty_v, duty_w, 1.0], dtype=DEFAULT_DTYPE)
            self.output = VectorSignal(out, self.name)
            return self.output

        # ---- SPEED PI CONTROLLER ----
        speed_error = speed_ref_rpm - speed_sensor_rpm

        self.speed_integral += self.Ki_speed * speed_error * dt
        self.speed_integral = self._clamp(self.speed_integral, -self.speed_limit, self.speed_limit)

        iq_ref = self.Kp_speed * speed_error + self.speed_integral
        iq_ref = self._clamp(iq_ref, -self.speed_limit, self.speed_limit)
        id_ref = 0.0

        # ---- Mechanical Flatness ----
        torque_required = self.J * omega_dot + self.B * omega_ref + self.Tload

        # ---- Electrical Flatness ----
        torque_constant = 1.5 * self.pole_pairs * self.lambda_pm

        if abs(torque_constant) > 1e-6:
            iq_ref_ff = torque_required / torque_constant
            iq_ref_ff = self._clamp(iq_ref_ff, -self.max_current, self.max_current)
            iq_ref_dot = (self.J * omega_ddot + self.B * omega_dot) / torque_constant
            iq_ref_dot = self._clamp(iq_ref_dot, -self.max_iq_dot, self.max_iq_dot)
        else:
            iq_ref_ff = 0.0
            iq_ref_dot = 0.0

        # ---- Flatness Voltage Mapping ----
        vd_ff = -self.pole_pairs * omega_ref * self.Lq * iq_ref_ff
        vq_ff = (self.Rs * iq_ref_ff) + (self.Lq * iq_ref_dot) + (self.pole_pairs * omega_ref * self.lambda_pm)

        # ---- PI Current Correction ----
        id_error = id_ref - id_meas
        iq_error = iq_ref - iq_meas

        self.id_integral += self.Ki_d * id_error * dt
        self.iq_integral += self.Ki_q * iq_error * dt
        self.id_integral = self._clamp(self.id_integral, -self.integral_limit, self.integral_limit)
        self.iq_integral = self._clamp(self.iq_integral, -self.integral_limit, self.integral_limit)

        vd_ref = vd_ff + self.Kp_d * id_error + self.id_integral
        vq_ref = vq_ff + self.Kp_q * iq_error + self.iq_integral

        # ---- Inverse Park -> Alpha-Beta using C wrapper ----
        valpha, vbeta = inv_park(vd_ref, vq_ref, theta_elec)

        # ---- SVM ----
        duty_u, duty_v, duty_w = self._svm(valpha, vbeta, vdc)

        out = np.array([duty_u, duty_v, duty_w, 1.0], dtype=DEFAULT_DTYPE)
        self.output = VectorSignal(out, self.name)
        return self.output

    def reset(self):
        super().reset()
        self.theta = 0.0
        self._current_freq = 0.0
        self.speed_integral = 0.0
        self.id_integral = 0.0
        self.iq_integral = 0.0
        self.theta_open_loop = 0.0
        self.current_speed_rpm = 0.0
        self.current_accel_rpm = 0.0
        self.current_jerk_rpm = 0.0
        self.current_position_rad = 0.0
        self.is_rolling = False
        self._last_print = -1.0