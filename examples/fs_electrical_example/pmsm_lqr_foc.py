"""
pmsm_lqr_kalman_foc.py
===============================================================================
EmbedSim — PMSM FOC  |  LQR + Kalman Filter  |  DB42S02  |  AURIX TC3xx
===============================================================================

LQR replaces PI speed loop + current loop in one gain matrix.
Kalman Filter provides state estimation for sensorless or noisy measurement operation.

State  x = [id, iq, omega_m, theta_e, xi_d, xi_q, xi_omega]   7 states
Input  u = [vd, vq]                                             2 inputs
Meas   y = [id, iq, theta_e] (or [ia, ib, ic] for sensorless)  3 measurements

Plant (linearised at we_op):
  did/dt      = -Rs/Ld · id  + we · iq            + vd/Ld
  diq/dt      = -we    · id  - Rs/Lq · iq          + vq/Lq
  domega_m/dt =  KT/J  · iq  - B/J  · omega_m
  dtheta_e/dt =  p · omega_m

Integrators (zero steady-state error):
  dxi_d/dt     = id    - 0
  dxi_q/dt     = iq    - 0
  dxi_omega/dt = omega_m - omega_ref

Motor: NANOTEC DB42S02  Rs=0.19  Ld=Lq=125µH  λpm=0.0014  p=4  J=2.4e-6  Vdc=17V
"""

from __future__ import annotations
import math
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE
for _c in (_HERE, _HERE.parent, _HERE.parent.parent):
    if (_c / "embedsim").is_dir():
        _ROOT = _c
        break
for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_FS = _ROOT / "fs_electrical_machines"
if _FS.is_dir() and str(_FS) not in sys.path:
    sys.path.insert(0, str(_FS))

from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep
from embedsim.dynamic_blocks import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver, VectorDelay
from fs_electrical_machines.pmsm_python_plant import PMSM_Python_Plant


# ==============================================================================
#  Motor constants — NANOTEC DB42S02
# ==============================================================================
class _DB42S02:
    p = 4
    R_s = 0.19
    L_d = 0.125e-3
    L_q = 0.125e-3
    lam_pm = 0.0014
    J = 2.4e-6
    B_fric = 7e-5  # Renamed to avoid confusion
    I_max = 3.57
    V_dc = 17.0
    V_max = 17.0 / 2.0  # 8.5 V half-bridge limit
    KT = 1.5 * p * lam_pm  # 0.0084 N·m/A
    RPM_OP = 400.0
    WM_OP = RPM_OP * math.pi / 30.0
    WE_OP = WM_OP * p


# ==============================================================================
#  LQR + Kalman Filter Design
# ==============================================================================
class LQRKalmanDesign:
    """Design LQR controller and Kalman filter for PMSM"""

    def __init__(self):
        self.Rs = _DB42S02.R_s
        self.Ld = _DB42S02.L_d
        self.Lq = _DB42S02.L_q
        self.KT = _DB42S02.KT
        self.J = _DB42S02.J
        self.B_fric = _DB42S02.B_fric  # Friction coefficient
        self.p = _DB42S02.p
        self.we_op = _DB42S02.WE_OP
        self.Imax = _DB42S02.I_max
        self.Vmax = _DB42S02.V_max
        self.Wmax = _DB42S02.WM_OP

        # System matrices
        self.A, self.B_mat = self._build_system_matrices()  # Renamed B_mat to avoid confusion
        self.C = self._build_measurement_matrix()

        # Process and measurement noise covariances
        self.Q_kf = self._build_process_noise_covariance()
        self.R_kf = self._build_measurement_noise_covariance()

        # LQR weights
        self.Q_lqr = self._build_lqr_weights()
        self.R_lqr = self._build_control_weights()

        # Solve for gains
        self.K_lqr = self._solve_lqr()
        self.K_kf = self._solve_kalman_gain()

    def _build_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build continuous-time system matrices (7 states)"""
        # State order: [id, iq, omega_m, theta_e, xi_d, xi_q, xi_omega]
        A = np.array([
            # id        iq        omega_m   theta_e   xi_d    xi_q    xi_omega
            [-self.Rs / self.Ld, self.we_op, 0, 0, 0, 0, 0],  # did/dt
            [-self.we_op, -self.Rs / self.Lq, 0, 0, 0, 0, 0],  # diq/dt
            [0, self.KT / self.J, -self.B_fric / self.J, 0, 0, 0, 0],  # domega_m/dt
            [0, 0, self.p, 0, 0, 0, 0],  # dtheta_e/dt
            [1, 0, 0, 0, 0, 0, 0],  # dxi_d/dt
            [0, 1, 0, 0, 0, 0, 0],  # dxi_q/dt
            [0, 0, 1, 0, 0, 0, 0],  # dxi_omega/dt
        ], dtype=float)

        B = np.zeros((7, 2))
        B[0, 0] = 1.0 / self.Ld
        B[1, 1] = 1.0 / self.Lq

        return A, B

    def _build_measurement_matrix(self) -> np.ndarray:
        """C matrix for measurements [id, iq, theta_e]"""
        # Measured states: id, iq, theta_e
        C = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # id
            [0, 1, 0, 0, 0, 0, 0],  # iq
            [0, 0, 0, 1, 0, 0, 0],  # theta_e
        ], dtype=float)
        return C

    def _build_process_noise_covariance(self) -> np.ndarray:
        """Process noise covariance matrix Q_kf"""
        # Current noise (5% of max current)
        current_noise = (0.05 * self.Imax) ** 2
        # Speed noise (10 rad/s)
        speed_noise = 10.0 ** 2
        # Angle noise (0.1 rad)
        angle_noise = 0.1 ** 2
        # Integrator noise (small)
        integrator_noise = 1e-6

        Q = np.diag([
            current_noise,  # id
            current_noise,  # iq
            speed_noise,  # omega_m
            angle_noise,  # theta_e
            integrator_noise,  # xi_d
            integrator_noise,  # xi_q
            integrator_noise,  # xi_omega
        ])
        return Q

    def _build_measurement_noise_covariance(self) -> np.ndarray:
        """Measurement noise covariance matrix R_kf"""
        # Current measurement noise (2% of max current)
        current_meas_noise = (0.02 * self.Imax) ** 2
        # Angle measurement noise (0.05 rad)
        angle_meas_noise = 0.05 ** 2

        R = np.diag([
            current_meas_noise,  # id measurement
            current_meas_noise,  # iq measurement
            angle_meas_noise,  # theta_e measurement
        ])
        return R

    def _build_lqr_weights(self) -> np.ndarray:
        """Build LQR Q matrix (state weights)"""
        # Safe calculations with bounds
        xi_i_max = self.Imax * self.Ld / max(self.Rs, 1e-6)

        # Use friction coefficient for integrator limit
        xi_w_max = self.Imax * self.KT / max(self.B_fric, 1e-6)

        # If xi_w_max is too large, use a reasonable default
        if xi_w_max > 1e6 or xi_w_max < 0 or not np.isfinite(xi_w_max):
            xi_w_max = 1000.0  # Reasonable default

        # Use scalar values, not arrays
        Q_diag = [
            10.0 / (self.Imax ** 2 + 1e-6),      # id
            10.0 / (self.Imax ** 2 + 1e-6),      # iq
            1000.0 / (self.Wmax ** 2 + 1e-6),    # omega_m
            10.0 / (np.pi ** 2 + 1e-6),          # theta_e
            100.0 / (xi_i_max ** 2 + 1e-6),      # xi_d
            100.0 / (xi_i_max ** 2 + 1e-6),      # xi_q
            1000.0 / (xi_w_max ** 2 + 1e-6),     # xi_omega
        ]

        # Ensure all values are finite and positive
        Q_diag = [float(x) if np.isfinite(x) and x > 0 else 1.0 for x in Q_diag]

        Q = np.diag(Q_diag)
        return Q

    def _build_control_weights(self) -> np.ndarray:
        """Build LQR R matrix (control effort weights)"""
        R_val1 = 1.0 / (self.Vmax ** 2 + 1e-6)
        R_val2 = 1.0 / (self.Vmax ** 2 + 1e-6)
        R = np.diag([R_val1, R_val2])
        return R

    def _solve_lqr(self) -> np.ndarray:
        """Solve continuous-time LQR problem"""
        # Add regularization
        epsilon = 1e-6
        Q_reg = self.Q_lqr + epsilon * np.eye(7)
        R_reg = self.R_lqr + epsilon * np.eye(2)

        try:
            from scipy.linalg import solve_continuous_are

            # Check controllability
            controllability = np.linalg.matrix_rank(
                np.linalg.matrix_power(self.A, 6) @ self.B_mat
            )
            print(f"  Controllability rank: {controllability}/7")

            if controllability < 7:
                print("  Warning: System is not fully controllable")

            # Solve CARE
            P = solve_continuous_are(self.A, self.B_mat, Q_reg, R_reg)
            K = np.linalg.solve(R_reg, self.B_mat.T @ P)

            # Verify stability
            A_cl = self.A - self.B_mat @ K
            eigs = np.linalg.eigvals(A_cl)

            print("\n" + "=" * 60)
            print("  LQR Design Results:")
            print(f"  K(vd): {np.round(K[0], 3)}")
            print(f"  K(vq): {np.round(K[1], 3)}")
            print("  Closed-loop eigenvalues:")
            stable_count = 0
            for i, ev in enumerate(eigs):
                bw = -ev.real / (2 * math.pi) if ev.real < 0 else 0
                status = "✅" if ev.real < -1e-6 else "⚠️"
                if ev.real < -1e-6:
                    stable_count += 1
                if i < 5:  # Show first 5 eigenvalues
                    print(f"    λ{i + 1} = {ev.real:+.1f}{ev.imag:+.1f}j   BW≈{bw:.0f} Hz  {status}")
            print(f"  Stable eigenvalues: {stable_count}/7")
            print("=" * 60 + "\n")

            return K

        except Exception as e:
            print(f"  Continuous LQR failed: {e}")
            print("  Using discrete-time LQR...")
            return self._solve_lqr_discrete()

    def _solve_lqr_discrete(self, dt=50e-6) -> np.ndarray:
        """Discrete-time LQR fallback"""
        # Discretize
        n = self.A.shape[0]
        Ad = np.eye(n) + self.A * dt
        Bd = self.B_mat * dt

        # Add regularization
        R_reg = self.R_lqr + 1e-6 * np.eye(2)

        # Solve discrete ARE
        P = self.Q_lqr.copy()
        max_iter = 10000
        tolerance = 1e-6

        for i in range(max_iter):
            try:
                K_temp = np.linalg.solve(R_reg + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
                P_next = Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K_temp + self.Q_lqr
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                K_temp = np.linalg.pinv(R_reg + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
                P_next = Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K_temp + self.Q_lqr

            if np.max(np.abs(P_next - P)) < tolerance:
                print(f"  Discrete ARE converged in {i + 1} iterations")
                break
            P = P_next

        # Final gain
        try:
            K = np.linalg.solve(R_reg + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        except np.linalg.LinAlgError:
            K = np.linalg.pinv(R_reg + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

        print(f"  Discrete LQR gains: vd={np.round(K[0], 3)}, vq={np.round(K[1], 3)}\n")
        return K

    def _solve_kalman_gain(self, dt=50e-6) -> np.ndarray:
        """Solve continuous-time Kalman filter gain"""
        try:
            from scipy.linalg import solve_continuous_are

            # Check observability
            obs_matrix = np.vstack([self.C, self.C @ self.A, self.C @ self.A @ self.A])
            observability = np.linalg.matrix_rank(obs_matrix)
            print(f"  Observability rank: {observability}/7")

            if observability < 7:
                print("  Warning: System is not fully observable")

            # Add regularization
            Q_reg = self.Q_kf + 1e-6 * np.eye(7)
            R_reg = self.R_kf + 1e-6 * np.eye(3)

            # Solve continuous ARE for Kalman filter
            P_inf = solve_continuous_are(self.A.T, self.C.T, Q_reg, R_reg)
            K_kf = P_inf @ self.C.T @ np.linalg.inv(R_reg)

            print("  Kalman filter gains:")
            for i, name in enumerate(['id', 'iq', 'omega', 'theta', 'xi_d', 'xi_q', 'xi_w']):
                print(f"    K_{name}: {np.round(K_kf[i], 4)}")

            return K_kf

        except Exception as e:
            print(f"  Continuous Kalman failed: {e}")
            print("  Using discrete-time Kalman...")
            return self._solve_kalman_discrete(dt)

    def _solve_kalman_discrete(self, dt=50e-6) -> np.ndarray:
        """Discrete-time Kalman filter fallback"""
        # Discretize
        n = self.A.shape[0]
        Ad = np.eye(n) + self.A * dt
        Gd = np.eye(n) * np.sqrt(dt)

        # Add regularization
        Q_reg = self.Q_kf + 1e-6 * np.eye(7)
        R_reg = self.R_kf + 1e-6 * np.eye(3)

        # Discrete ARE for Kalman
        P = Q_reg.copy()
        max_iter = 10000
        tolerance = 1e-6

        for i in range(max_iter):
            P_pred = Ad @ P @ Ad.T + Gd @ Q_reg @ Gd.T
            S = self.C @ P_pred @ self.C.T + R_reg
            try:
                K = P_pred @ self.C.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = P_pred @ self.C.T @ np.linalg.pinv(S)
            P_next = (np.eye(n) - K @ self.C) @ P_pred

            if np.max(np.abs(P_next - P)) < tolerance:
                print(f"  Discrete Kalman converged in {i + 1} iterations")
                break
            P = P_next

        # Final gain
        P_pred = Ad @ P @ Ad.T + Gd @ Q_reg @ Gd.T
        S = self.C @ P_pred @ self.C.T + R_reg
        try:
            K_final = P_pred @ self.C.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K_final = P_pred @ self.C.T @ np.linalg.pinv(S)

        print("  Using discrete-time Kalman gains")
        return K_final


# ==============================================================================
#  DB42S02PlantBlock with measurement noise
# ==============================================================================
class DB42S02PlantBlock(PMSM_Python_Plant):
    TOPO_CATEGORY = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label = "[rpm,ia,ib,ic,theta_m,Tem,id,iq]"

    def __init__(self, name, add_noise=False, noise_std=0.05):
        super().__init__(name=name,
                         R=_DB42S02.R_s, L_d=_DB42S02.L_d, L_q=_DB42S02.L_q,
                         lambda_pm=_DB42S02.lam_pm, J=_DB42S02.J, B_fric=_DB42S02.B_fric,
                         p=float(_DB42S02.p), v_dc=_DB42S02.V_dc)
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.rng = np.random.RandomState(42)  # Fixed seed for reproducibility

    def compute_py(self, t, dt, input_values=None):
        va = vb = vc = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                va, vb, vc = float(v[0]), float(v[1]), float(v[2])

        vdc = _DB42S02.V_dc
        ta = np.clip(va / vdc + 0.5, 0.0, 1.0)
        tb = np.clip(vb / vdc + 0.5, 0.0, 1.0)
        tc = np.clip(vc / vdc + 0.5, 0.0, 1.0)

        result = super().compute_py(t, dt, [VectorSignal(
            np.array([ta, tb, tc, vdc, 0.0], dtype=DEFAULT_DTYPE))])

        if self.add_noise and result is not None:
            # Add measurement noise to outputs
            noisy_result = result.value.copy()
            # Add noise to currents (indices 1,2,3 for ia,ib,ic)
            noisy_result[1:4] += self.rng.normal(0, self.noise_std, 3)
            # Add noise to position (index 4)
            noisy_result[4] += self.rng.normal(0, self.noise_std * 0.1)
            # Add noise to id,iq (indices 6,7)
            noisy_result[6:8] += self.rng.normal(0, self.noise_std, 2)
            result = VectorSignal(noisy_result, self.name)

        return result

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# ==============================================================================
#  Kalman Filter Block
# ==============================================================================
class KalmanFilterBlock(VectorBlock):
    """
    Discrete-time Kalman filter for state estimation
    Input: [id_meas, iq_meas, theta_e_meas]
    Output: [id_est, iq_est, omega_m_est, theta_e_est, xi_d_est, xi_q_est, xi_omega_est]
    """

    def __init__(self, name="kalman_filter", dt=50e-6):
        super().__init__(name)
        self.dt = dt
        self.vector_size = 7
        self.output_label = "[id,iq,omega_m,theta_e,xi_d,xi_q,xi_omega]"
        self.is_dynamic = True

        # Get system matrices
        self.design = LQRKalmanDesign()
        self.A = self.design.A
        self.B = self.design.B_mat
        self.C = self.design.C
        self.Q = self.design.Q_kf
        self.R = self.design.R_kf

        # Discretize
        self.Ad = np.eye(7) + self.A * dt
        self.Bd = self.B * dt
        self.Cd = self.C

        # Process noise input matrix (simplified)
        self.Gd = np.eye(7) * np.sqrt(dt)

        # Initial state and covariance
        self.x_est = np.zeros(7, dtype=float)
        self.P = np.eye(7) * 0.1

        # Kalman gain (precomputed)
        self.K = self.design.K_kf

    def compute_py(self, t, dt, input_values=None):
        # Prediction step
        u = np.zeros(2)
        if input_values and len(input_values) > 1 and input_values[1] is not None:
            u_val = input_values[1].value
            if len(u_val) >= 2:
                u = u_val[:2]

        # State prediction
        x_pred = self.Ad @ self.x_est + self.Bd @ u

        # Covariance prediction
        P_pred = self.Ad @ self.P @ self.Ad.T + self.Gd @ self.Q @ self.Gd.T

        # Update step (if measurements available)
        if input_values and input_values[0] is not None:
            y_meas = input_values[0].value[:3]  # [id, iq, theta_e]

            # Innovation
            y_pred = self.Cd @ x_pred
            y_err = y_meas - y_pred

            # Innovation covariance with regularization
            S = self.Cd @ P_pred @ self.Cd.T + self.R
            S = S + 1e-6 * np.eye(3)  # Add small regularization

            # Kalman gain (adaptive)
            try:
                K = P_pred @ self.Cd.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = self.K  # Use precomputed gain if singular

            # State update
            self.x_est = x_pred + K @ y_err

            # Covariance update (Joseph form for numerical stability)
            IKC = np.eye(7) - K @ self.Cd
            self.P = IKC @ P_pred @ IKC.T + K @ self.R @ K.T
        else:
            self.x_est = x_pred
            self.P = P_pred

        # Ensure physical bounds
        self.x_est[2] = np.clip(self.x_est[2], -_DB42S02.WM_OP * 2, _DB42S02.WM_OP * 2)
        self.x_est[3] = self.x_est[3] % (2 * np.pi)
        self.x_est[4] = np.clip(self.x_est[4], -10, 10)  # xi_d bounds
        self.x_est[5] = np.clip(self.x_est[5], -10, 10)  # xi_q bounds
        self.x_est[6] = np.clip(self.x_est[6], -100, 100)  # xi_omega bounds

        self.output = VectorSignal(self.x_est.copy(), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)

    def reset(self):
        super().reset()
        self.x_est = np.zeros(7, dtype=float)
        self.P = np.eye(7) * 0.1


# ==============================================================================
#  LQR Controller Block (using estimated states)
# ==============================================================================
class LQRBlock(VectorBlock):
    """
    LQR controller using estimated states from Kalman filter
    port 0: estimated states from Kalman filter
    port 1: omega_ref (RPM)
    output: [v_alpha, v_beta]
    """

    C_CODEGEN_EXCLUDE = True

    def __init__(self, name="lqr", v_max=_DB42S02.V_max):
        super().__init__(name)
        self.v_max = float(v_max)
        self.output_label = "[v_alpha,v_beta]"
        self.is_dynamic = False
        self.vector_size = 2

        # Get LQR gain
        design = LQRKalmanDesign()
        self.K = design.K_lqr  # (2×7)

        # Store integrator limits
        self.xi_i_lim = v_max * _DB42S02.L_d / max(_DB42S02.R_s, 1e-6)
        self.xi_w_lim = _DB42S02.I_max * _DB42S02.KT / max(_DB42S02.B_fric, 1e-6)

    def compute_py(self, t, dt, input_values=None):
        # Get estimated states and reference
        x_est = np.zeros(7)
        omega_ref = 0.0

        if input_values:
            if input_values[0] is not None:
                x_est = input_values[0].value[:7]
            if len(input_values) > 1 and input_values[1] is not None:
                omega_ref = float(input_values[1].value[0]) * math.pi / 30.0

        # Extract states
        id_m = x_est[0]
        iq_m = x_est[1]
        omega_m = x_est[2]
        theta_e = x_est[3]

        # State vector for LQR: [id, iq, omega_m - omega_ref, theta_e, xi_d, xi_q, xi_omega]
        x = np.array([
            id_m,
            iq_m,
            omega_m - omega_ref,
            theta_e,
            x_est[4],  # xi_d from Kalman
            x_est[5],  # xi_q from Kalman
            x_est[6],  # xi_omega from Kalman
        ], dtype=float)

        # LQR control law
        u = -self.K @ x
        vd = float(u[0])
        vq = float(u[1])

        # Feedforward decoupling
        we = omega_m * float(_DB42S02.p)
        vd += -we * _DB42S02.L_q * iq_m
        vq += we * (_DB42S02.L_d * id_m + _DB42S02.lam_pm)

        # Voltage limiting with anti-windup
        v_norm = math.sqrt(vd ** 2 + vq ** 2)
        if v_norm > self.v_max and v_norm > 0:
            scale = self.v_max / v_norm
            vd *= scale
            vq *= scale

        # Inverse Park transform
        c, s = math.cos(theta_e), math.sin(theta_e)
        v_alpha = vd * c - vq * s
        v_beta = vd * s + vq * c

        self.output = VectorSignal(
            np.array([v_alpha, v_beta], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# ==============================================================================
#  Measurement Extraction Block
# ==============================================================================
class MeasurementExtractor(VectorBlock):
    """Extract measurements from motor bus for Kalman filter"""

    def __init__(self, name="meas_extract"):
        super().__init__(name)
        self.output_label = "[id,iq,theta_e]"
        self.is_dynamic = False
        self.vector_size = 3

    def compute_py(self, t, dt, input_values=None):
        id_m = iq_m = theta_e = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 8:
                theta_m = float(v[4])
                theta_e = (theta_m * float(_DB42S02.p)) % (2.0 * math.pi)
                id_m = float(v[6])
                iq_m = float(v[7])

        self.output = VectorSignal(
            np.array([id_m, iq_m, theta_e], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# ==============================================================================
#  InvClarkeBlock
# ==============================================================================
class InvClarkeBlock(VectorBlock):
    def __init__(self, name="inv_clarke"):
        super().__init__(name)
        self.output_label = "[va,vb,vc]"
        self.is_dynamic = False
        self.vector_size = 3

    def compute_py(self, t, dt, input_values=None):
        a = b = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 2:
                a, b = float(v[0]), float(v[1])
        s3 = math.sqrt(3.0) / 2.0
        self.output = VectorSignal(
            np.array([a, -0.5 * a + s3 * b, -0.5 * a - s3 * b], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# ==============================================================================
#  Main simulation
# ==============================================================================
def main():
    print("=" * 65)
    print("  EmbedSim — LQR + Kalman Filter FOC  |  DB42S02")
    print("=" * 65)

    # Simulation parameters
    DT = 50e-6
    T_SIM = 2.0
    RPM_REF = 400.0
    ADD_NOISE = True
    NOISE_STD = 0.05

    # Create blocks
    omega_ref = VectorStep("omega_ref", step_time=0.05,
                           before_value=0.0, after_value=float(RPM_REF))

    motor = DB42S02PlantBlock("motor", add_noise=ADD_NOISE, noise_std=NOISE_STD)
    meas_extract = MeasurementExtractor("meas_extract")
    kalman = KalmanFilterBlock("kalman", dt=DT)
    lqr = LQRBlock("lqr")
    inv_clarke = InvClarkeBlock("inv_clarke")
    motor_dly = VectorDelay("motor_dly", initial=[0.0] * 8)
    sink = VectorEnd("sink")

    # Wiring
    motor >> motor_dly  # Delay for feedback
    motor_dly >> meas_extract  # Extract measurements from delayed motor bus
    motor_dly >> sink  # Log to sink

    # Kalman filter gets measurements and control inputs
    meas_extract >> kalman  # port 0: measurements
    lqr >> kalman  # port 1: control inputs (for prediction)

    # LQR uses estimated states from Kalman
    kalman >> lqr  # port 0: estimated states
    omega_ref >> lqr  # port 1: speed reference

    # Voltage generation
    lqr >> inv_clarke  # [v_alpha, v_beta]
    inv_clarke >> motor  # [va, vb, vc]

    # Run simulation
    sim = EmbedSim(sinks=[sink], T=T_SIM, dt=DT, solver=ODESolver.EULER)

    print(f"\n  Target: {RPM_REF:.0f} RPM  |  dt={DT * 1e6:.0f} µs  |  T={T_SIM:.1f}s")
    print(f"  Measurement noise: {'ON' if ADD_NOISE else 'OFF'} (std={NOISE_STD}A)")
    sim.topo.print_console()

    # Add signals to scope
    sim.scope.add(motor, indices=[0, 1, 2, 3, 4], label="Motor")
    sim.scope.add(omega_ref, indices=[0], label="SpeedRef")
    sim.scope.add(kalman, indices=[0, 1, 2, 3], label="Kalman")

    print(f"\nRunning ({T_SIM}s @ {1 / DT:.0f} Hz)...")
    sim.run()
    print(f"  Steps: {len(sim.scope.t)}")

    # Plot results
    sc = sim.scope
    t = np.array(sc.t, dtype=np.float32)

    def _g(lbl, pos):
        s = sc.get_signal(lbl, pos)
        return s if s is not None else np.zeros(len(t), dtype=np.float32)

    # Motor measurements
    rpm_meas = _g("Motor", 0)
    ia = _g("Motor", 1)
    ib = _g("Motor", 2)
    ic = _g("Motor", 3)
    theta_m = _g("Motor", 4)
    rpm_ref = _g("SpeedRef", 0)

    # Kalman estimates
    id_est = _g("Kalman", 0)
    iq_est = _g("Kalman", 1)
    omega_est = _g("Kalman", 2)
    theta_e_est = _g("Kalman", 3)

    # Calculate electrical angles
    p = float(_DB42S02.p)
    theta_e_meas = (theta_m * p) % (2.0 * math.pi)
    theta_e_ref = (np.cumsum(rpm_ref * (math.pi / 30.0) * DT) * p) % (2.0 * math.pi)

    # Create plots
    fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=True)
    fig.suptitle("EmbedSim — LQR + Kalman Filter FOC  |  DB42S02  400 RPM\n"
                 "State estimation with noisy measurements",
                 fontsize=13, fontweight="bold")

    # Speed tracking
    axes[0].plot(t, rpm_ref, "k--", lw=1.2, label="ω_ref [RPM]")
    axes[0].plot(t, rpm_meas, "C0", lw=1.5, alpha=0.7, label="ω_meas (noisy)")
    axes[0].plot(t, omega_est * 30 / np.pi, "C1", lw=1.8, label="ω_est (Kalman)")
    axes[0].set_ylabel("Speed [RPM]", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.35)

    # Phase currents
    axes[1].plot(t, ia, "C3", lw=0.7, alpha=0.6, label="ia [A] (noisy)")
    axes[1].plot(t, ib, "C2", lw=0.7, alpha=0.6, label="ib [A] (noisy)")
    axes[1].plot(t, ic, "C0", lw=0.7, alpha=0.6, label="ic [A] (noisy)")
    axes[1].set_ylabel("Phase currents [A]", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.35)

    # dq currents estimation
    axes[2].plot(t, _g("Motor", 6), "C3", lw=1.0, alpha=0.5, label="id_meas (noisy)")
    axes[2].plot(t, _g("Motor", 7), "C2", lw=1.0, alpha=0.5, label="iq_meas (noisy)")
    axes[2].plot(t, id_est, "C3", lw=1.8, label="id_est (Kalman)")
    axes[2].plot(t, iq_est, "C2", lw=1.8, label="iq_est (Kalman)")
    axes[2].set_ylabel("dq currents [A]", fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.35)

    # Electrical angle estimation
    axes[3].plot(t, theta_e_meas, "C4", lw=1.0, alpha=0.5, label="θe_meas (noisy)")
    axes[3].plot(t, theta_e_est, "C1", lw=1.8, label="θe_est (Kalman)")
    axes[3].plot(t, theta_e_ref, "k--", lw=1.0, alpha=0.7, label="θe_ref")
    axes[3].set_ylabel("Electrical angle [rad]", fontsize=12)
    axes[3].legend(fontsize=10)
    axes[3].grid(alpha=0.35)

    # Estimation errors
    error_speed = (rpm_meas - omega_est * 30 / np.pi)
    error_id = (_g("Motor", 6) - id_est)
    error_iq = (_g("Motor", 7) - iq_est)

    axes[4].plot(t, error_speed, "C0", lw=1.0, label="Speed error [RPM]")
    axes[4].plot(t, error_id, "C1", lw=1.0, label="id error [A]")
    axes[4].plot(t, error_iq, "C2", lw=1.0, label="iq error [A]")
    axes[4].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[4].set_ylabel("Estimation error", fontsize=12)
    axes[4].set_xlabel("Time [s]", fontsize=12)
    axes[4].legend(fontsize=10)
    axes[4].grid(alpha=0.35)

    # Calculate and print performance metrics
    steady_state_idx = int(0.5 * len(t))  # After 0.5s

    if steady_state_idx < len(t):
        # RMS errors
        speed_rms_error = np.sqrt(np.mean(error_speed[steady_state_idx:] ** 2))
        id_rms_error = np.sqrt(np.mean(error_id[steady_state_idx:] ** 2))
        iq_rms_error = np.sqrt(np.mean(error_iq[steady_state_idx:] ** 2))

        # Settling time (find when speed stays within 2% of reference)
        target_speed = RPM_REF
        tolerance = 0.02 * target_speed
        settling_idx = len(t) - 1
        for i in range(len(t)):
            if abs(rpm_meas[i] - target_speed) < tolerance:
                # Check if it stays within tolerance for next 100 samples
                end_idx = min(i + 100, len(t))
                if all(abs(rpm_meas[i:end_idx] - target_speed) < tolerance):
                    settling_idx = i
                    break
        settling_time = t[settling_idx] if settling_idx < len(t) else t[-1]

        # Overshoot
        overshoot = max(0, (max(rpm_meas[settling_idx:]) - target_speed) / target_speed * 100) if settling_idx < len(t) else 0

        print("\n" + "=" * 65)
        print("  Performance Metrics:")
        print(f"    Settling time (2%): {settling_time:.3f} s")
        print(f"    Overshoot: {overshoot:.1f}%")
        print(f"    Speed RMS error: {speed_rms_error:.2f} RPM")
        print(f"    Id RMS error:    {id_rms_error:.3f} A")
        print(f"    Iq RMS error:    {iq_rms_error:.3f} A")

        # Add Kalman filter performance
        if ADD_NOISE:
            speed_noise_reduction = np.std(rpm_meas[steady_state_idx:]) / max(np.std(omega_est[steady_state_idx:]*30/np.pi), 1e-6)
            id_noise_reduction = np.std(_g('Motor',6)[steady_state_idx:]) / max(np.std(id_est[steady_state_idx:]), 1e-6)
            iq_noise_reduction = np.std(_g('Motor',7)[steady_state_idx:]) / max(np.std(iq_est[steady_state_idx:]), 1e-6)

            print(f"\n  Kalman Filter Performance:")
            print(f"    Speed noise reduction: {speed_noise_reduction:.1f}x")
            print(f"    Id noise reduction:    {id_noise_reduction:.1f}x")
            print(f"    Iq noise reduction:    {iq_noise_reduction:.1f}x")
        print("=" * 65)

    plt.tight_layout()
    out = _HERE / "pmsm_lqr_kalman_foc_results.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    print(f"\n💾  Saved: {out}")
    print("=" * 65)


if __name__ == "__main__":
    main()