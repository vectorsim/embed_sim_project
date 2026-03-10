"""
smo_block.py
============

Sliding Mode Observer (SMO) block for sensorless PMSM FOC.
Estimates rotor position and speed from phase currents and voltages.

Ports
-----
    port 0 : [i_alpha, i_beta]    — measured currents in stationary frame [A]
    port 1 : [v_alpha, v_beta]    — applied voltages in stationary frame [V]

Output
------
    [theta_e_hat, omega_m_hat, id_hat, iq_hat]
        - theta_e_hat : estimated electrical angle [rad]
        - omega_m_hat : estimated mechanical speed [rad/s]
        - id_hat      : estimated d-axis current [A]
        - iq_hat      : estimated q-axis current [A]

FIX (v1.0.1)
------------
    When emf_mag < threshold the angle was held (theta_new = theta_e_hat)
    but theta_prev was still updated to theta_new.  On the step where emf_mag
    first crosses the threshold the angle suddenly jumps to the arctan2 value,
    and d_theta = theta_new - theta_prev can be large → spurious speed spike.

    Fix: only update theta_prev when the angle estimate is actually computed
    from back-EMF (not from the hold path).  During the hold phase theta_prev
    stays fixed so the first valid d_theta is computed against the last real
    reference, not the held value.
"""

import sys
import numpy as np
from typing import List, Optional

from _path_utils import get_embedsim_import_path

sys.path.insert(0, get_embedsim_import_path())

from embedsim.code_generator import SimBlockBase
from embedsim.core_blocks import VectorSignal

# Debug flag
DEBUG = True


# ==============================================================================
# Pure-Python SMO implementation (no C dependency)
# ==============================================================================

class _PySMO:
    """
    Pure Python implementation of Sliding Mode Observer.
    Used when C wrapper is not available or use_c_backend=False.
    """

    def __init__(self,
                 R: float = 0.5,
                 L: float = 0.0055,
                 K_smo: float = 50.0,
                 wc_emf: float = 2.0 * np.pi * 1000.0,
                 wc_spd: float = 2.0 * np.pi * 200.0,
                 phi_smo: float = 0.1,
                 p: int = 2):

        self.R       = float(R)
        self.L       = float(L)
        self.K_smo   = float(K_smo)
        self.wc_emf  = float(wc_emf)
        self.wc_spd  = float(wc_spd)
        self.phi_smo = float(max(phi_smo, 1e-6))
        self.p       = int(p)

        # Observer states
        self.i_alpha_hat  = 0.0
        self.i_beta_hat   = 0.0
        self.emf_alpha    = 0.0
        self.emf_beta     = 0.0
        self.theta_e_hat  = 0.0
        self.omega_e_hat  = 0.0
        # FIX: theta_prev is only updated when a real back-EMF angle is
        # computed.  During the hold phase it is left unchanged so that
        # the first valid d_theta is measured against the correct reference.
        self.theta_prev   = 0.0
        self._angle_valid = False   # True once emf_mag has exceeded threshold

        self.step_count = 0

    @staticmethod
    def _sat(x: float, phi: float) -> float:
        """Saturation function with boundary layer."""
        v = x / phi
        if v >  1.0: return  1.0
        if v < -1.0: return -1.0
        return v

    @staticmethod
    def _unwrap_angle(delta: float) -> float:
        """Unwrap angle difference to [-π, π]."""
        if delta >  np.pi: delta -= 2.0 * np.pi
        elif delta < -np.pi: delta += 2.0 * np.pi
        return delta

    def compute(self, i_alpha: float, i_beta: float,
                v_alpha: float, v_beta: float, dt: float):
        """
        Execute one SMO step.
        Returns: (theta_e_hat, omega_m_hat, id_hat, iq_hat)
        """
        self.step_count += 1

        # ── Current estimation errors ─────────────────────────────────────
        err_alpha = i_alpha - self.i_alpha_hat
        err_beta  = i_beta  - self.i_beta_hat

        # ── Sliding injection ─────────────────────────────────────────────
        z_alpha = self.K_smo * self._sat(err_alpha, self.phi_smo)
        z_beta  = self.K_smo * self._sat(err_beta,  self.phi_smo)

        # ── Current observer (Euler) ──────────────────────────────────────
        d_ialpha = (v_alpha - self.R * self.i_alpha_hat + z_alpha) / self.L
        d_ibeta  = (v_beta  - self.R * self.i_beta_hat  + z_beta)  / self.L

        self.i_alpha_hat += d_ialpha * dt
        self.i_beta_hat  += d_ibeta  * dt

        # ── Back-EMF low-pass filter ──────────────────────────────────────
        self.emf_alpha += self.wc_emf * (z_alpha - self.emf_alpha) * dt
        self.emf_beta  += self.wc_emf * (z_beta  - self.emf_beta)  * dt

        # ── Angle extraction ──────────────────────────────────────────────
        emf_mag = np.sqrt(self.emf_alpha ** 2 + self.emf_beta ** 2)

        if emf_mag > 0.1:
            theta_new = np.arctan2(-self.emf_alpha, self.emf_beta)

            # Speed estimation via angle differentiation + LPF
            d_theta = self._unwrap_angle(theta_new - self.theta_prev)

            # Rate limiter: cap at 2000 rad/s electrical
            max_d_theta = 2000.0 * dt
            if abs(d_theta) > max_d_theta:
                d_theta   = np.sign(d_theta) * max_d_theta
                theta_new = self.theta_prev + d_theta

            omega_e_raw = d_theta / max(dt, 1e-9)
            omega_e_raw = np.clip(omega_e_raw, -2000.0, 2000.0)

            self.omega_e_hat += self.wc_spd * (omega_e_raw - self.omega_e_hat) * dt
            self.theta_e_hat  = theta_new

            # FIX: only advance theta_prev when we have a valid angle measurement.
            # Previously theta_prev was always updated to theta_new (even when
            # theta_new == old theta_e_hat from the hold path), so the first
            # real d_theta after emf_mag crossed threshold was near zero, then
            # jumped, producing a speed spike.
            self.theta_prev   = theta_new
            self._angle_valid = True

        else:
            # Hold angle; do NOT update theta_prev so the next valid d_theta
            # is computed against the last real measurement.
            theta_new = self.theta_e_hat
            # Allow omega to decay gently toward zero during hold
            self.omega_e_hat += self.wc_spd * (0.0 - self.omega_e_hat) * dt

        # ── Mechanical speed ──────────────────────────────────────────────
        omega_m_hat = self.omega_e_hat / self.p

        # ── Estimated dq currents (Park on observer currents) ─────────────
        cos_t  = np.cos(self.theta_e_hat)
        sin_t  = np.sin(self.theta_e_hat)
        id_hat =  self.i_alpha_hat * cos_t + self.i_beta_hat * sin_t
        iq_hat = -self.i_alpha_hat * sin_t + self.i_beta_hat * cos_t

        return self.theta_e_hat, omega_m_hat, id_hat, iq_hat

    def reset(self):
        """Reset all observer states."""
        self.i_alpha_hat  = 0.0
        self.i_beta_hat   = 0.0
        self.emf_alpha    = 0.0
        self.emf_beta     = 0.0
        self.theta_e_hat  = 0.0
        self.omega_e_hat  = 0.0
        self.theta_prev   = 0.0
        self._angle_valid = False
        self.step_count   = 0


# ==============================================================================
# SMOBlock  —  EmbedSim VectorBlock
# ==============================================================================

class SMOBlock(SimBlockBase):
    """
    Sliding Mode Observer block for sensorless PMSM FOC.

    Estimates electrical angle, mechanical speed, and dq currents from
    measured αβ currents and applied αβ voltages.

    Parameters
    ----------
    name         : str   — unique block identifier
    R            : float — stator resistance [Ω] (default 0.5)
    L            : float — average inductance (Ld+Lq)/2 [H] (default 0.0055)
    K_smo        : float — sliding mode gain [V] (default 50.0)
    wc_emf       : float — back-EMF LPF cutoff [rad/s] (default 2π·1000)
    wc_spd       : float — speed LPF cutoff [rad/s] (default 2π·200)
    phi_smo      : float — boundary layer thickness [A] (default 0.1)
    p            : int   — number of pole pairs (default 2)
    use_c_backend: bool  — use compiled smo_wrapper.pyd
    """

    # ── CodeGen marker attributes ─────────────────────────────────────────
    import pathlib as _pl
    PYX_FILE:    str  = str(_pl.Path(__file__).parent / 'c_src' / 'smo_wrapper.pyx')
    step_func:   str  = 'SMO_Compute'
    state_struct: str = 'SMO_Block_T'
    NUM_INPUTS:  int  = 2
    OUTPUT_SIZE: int  = 4
    C_SOURCES:   list = ['smo.c']
    C_HEADERS:   list = ['smo.h', 'Sys_Types.h']

    def __init__(
            self,
            name:          str,
            R:             float = 0.5,
            L:             float = 0.0055,
            K_smo:         float = 50.0,
            wc_emf:        float = 2.0 * np.pi * 1000.0,
            wc_spd:        float = 2.0 * np.pi * 200.0,
            phi_smo:       float = 0.1,
            p:             int   = 2,
            use_c_backend: bool  = False,
            dtype                = None,
    ) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        self.output_label = "[theta_e_hat, omega_m_hat, id_hat, iq_hat]"
        self.is_dynamic   = False
        self.vector_size  = 4
        self.state        = None

        self._debug_count = 0

        self._params = dict(
            R=R, L=L, K_smo=K_smo, wc_emf=wc_emf, wc_spd=wc_spd,
            phi_smo=phi_smo, p=p
        )

        if use_c_backend:
            self._impl = self._load_c_wrapper(R, L, K_smo, wc_emf, wc_spd, phi_smo, p)
        else:
            self._impl = _PySMO(R, L, K_smo, wc_emf, wc_spd, phi_smo, p)

    # ── C loader ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_c_wrapper(R, L, K_smo, wc_emf, wc_spd, phi_smo, p):
        try:
            import smo_wrapper as sw
            w = sw.SMOWrapper()
            w.set_params(R=R, L=L, K_smo=K_smo,
                         wc_emf=wc_emf, wc_spd=wc_spd,
                         phi=phi_smo, p=p)
            return w
        except ImportError:
            raise ImportError(
                "Cython wrapper 'smo_wrapper' not found.\n"
                "Compile with: python setup_smo.py build_ext --inplace\n"
                "Or set use_c_backend=False to use the Python backend."
            )

    # ── Input parsing ────────────────────────────────────────────────────

    def _parse_inputs(self, input_values):
        """Extract (i_alpha, i_beta, v_alpha, v_beta) safely."""
        i_alpha = i_beta = v_alpha = v_beta = 0.0

        if not input_values or len(input_values) < 2:
            return i_alpha, i_beta, v_alpha, v_beta

        # Port 0: [i_alpha, i_beta]
        if input_values[0] is not None:
            val = input_values[0].value
            if len(val) >= 2:
                i_alpha = float(val[0])
                i_beta  = float(val[1])

        # Port 1: [v_alpha, v_beta]
        if input_values[1] is not None:
            val = input_values[1].value
            if len(val) >= 2:
                v_alpha = float(val[0])
                v_beta  = float(val[1])

        return i_alpha, i_beta, v_alpha, v_beta

    # ── Compute dispatch ─────────────────────────────────────────────────

    def compute(self, t, dt, input_values=None):
        if self.use_c_backend:
            return self.compute_c(t, dt, input_values)
        return self.compute_py(t, dt, input_values)

    # -- Python backend -------------------------------------------------------

    def compute_py(
            self,
            t:  float,
            dt: float,
            input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        self._debug_count += 1
        i_alpha, i_beta, v_alpha, v_beta = self._parse_inputs(input_values)

        if DEBUG and self._debug_count % 1000 == 0:
            print(f"SMO t={t:.3f}: i_αβ=({i_alpha:.2f}, {i_beta:.2f}), "
                  f"v_αβ=({v_alpha:.2f}, {v_beta:.2f})")

        theta_e, omega_m, id_hat, iq_hat = self._impl.compute(
            i_alpha, i_beta, v_alpha, v_beta, dt
        )

        if DEBUG and self._debug_count % 1000 == 0:
            print(f"SMO t={t:.3f}: θ_e={theta_e:.3f}, ω_m={omega_m:.1f}")

        out = np.array([theta_e, omega_m, id_hat, iq_hat], dtype=np.float32)
        self.output = VectorSignal(out, self.name, dtype=self.dtype)
        return self.output

    # -- C backend ------------------------------------------------------------

    def compute_c(
            self,
            t:  float,
            dt: float,
            input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        self._debug_count += 1
        i_alpha, i_beta, v_alpha, v_beta = self._parse_inputs(input_values)

        if DEBUG and self._debug_count % 1000 == 0:
            print(f"SMO t={t:.3f}: i_αβ=({i_alpha:.2f}, {i_beta:.2f})")

        theta_e, omega_m, id_hat, iq_hat = self._impl.compute(
            i_alpha, i_beta, v_alpha, v_beta, float(dt)
        )

        if DEBUG and self._debug_count % 1000 == 0:
            print(f"SMO t={t:.3f}: θ_e={theta_e:.3f}, ω_m={omega_m:.1f}")

        out = np.array([theta_e, omega_m, id_hat, iq_hat], dtype=np.float32)
        self.output = VectorSignal(out, self.name, dtype=self.dtype)
        return self.output

    # ── Block lifecycle ─────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self._impl.reset()
        self._debug_count = 0

    # ── Runtime parameter update ────────────────────────────────────────

    def set_params(self, R=None, L=None, K_smo=None,
                   wc_emf=None, wc_spd=None, phi_smo=None, p=None):
        if R       is not None: self._params['R']       = float(R)
        if L       is not None: self._params['L']       = float(L)
        if K_smo   is not None: self._params['K_smo']   = float(K_smo)
        if wc_emf  is not None: self._params['wc_emf']  = float(wc_emf)
        if wc_spd  is not None: self._params['wc_spd']  = float(wc_spd)
        if phi_smo is not None: self._params['phi_smo'] = float(max(phi_smo, 1e-6))
        if p       is not None: self._params['p']       = int(p)

        if self.use_c_backend:
            self._impl.set_params(
                R=self._params['R'],       L=self._params['L'],
                K_smo=self._params['K_smo'], wc_emf=self._params['wc_emf'],
                wc_spd=self._params['wc_spd'], phi=self._params['phi_smo'],
                p=self._params['p']
            )
        else:
            self._impl.R       = self._params['R']
            self._impl.L       = self._params['L']
            self._impl.K_smo   = self._params['K_smo']
            self._impl.wc_emf  = self._params['wc_emf']
            self._impl.wc_spd  = self._params['wc_spd']
            self._impl.phi_smo = self._params['phi_smo']
            self._impl.p       = self._params['p']

    # ── Diagnostics ─────────────────────────────────────────────────────

    @property
    def estimated_angle(self) -> float:
        return self._impl.theta_e_hat if hasattr(self._impl, 'theta_e_hat') else 0.0

    @property
    def estimated_speed(self) -> float:
        if hasattr(self._impl, 'omega_e_hat'):
            return self._impl.omega_e_hat / self._params['p']
        return 0.0

    @property
    def back_emf_magnitude(self) -> float:
        if hasattr(self._impl, 'emf_alpha') and hasattr(self._impl, 'emf_beta'):
            return np.sqrt(self._impl.emf_alpha ** 2 + self._impl.emf_beta ** 2)
        return 0.0

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        p  = self._params
        return (
            f"SMOBlock('{self.name}', "
            f"R={p['R']}, L={p['L']}, K_smo={p['K_smo']}, "
            f"φ={p['phi_smo']}, p={p['p']}, backend={be})"
        )
