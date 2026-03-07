"""
smc_block.py
============

Sliding Mode Controller (SMC) block for EmbedSim.

Operates on the **d-q rotating reference frame** — receives d-q current
references and measurements, outputs d-q voltage commands.

Typical position in a PMSM FOC chain
--------------------------------------

  Speed   →  [SpeedController]  →  iq_ref
                                   id_ref = 0 (MTPA)
                                        │
                    id_meas ────────────┤
                    iq_meas ────────────┤
                    (from Park block)   ▼
                               [SMCBlock]
                                   │
                              [v_d, v_q]
                                   │
                            [InvParkBlock]
                                   │
                              [v_α, v_β]
                                   │
                           [InvClarkeBlock]
                                   │
                            [va, vb, vc]  → PWM


Ports
-----
    port 0 : [id_ref, iq_ref]   — d/q current references   [A]
    port 1 : [id_meas, iq_meas] — d/q measured currents     [A]

Output : [v_d, v_q]             — voltage commands           [V]

Backends
--------
    Python  : Pure-Python SMC (always available, for simulation)
    C       : Cython-compiled smc_wrapper.pyd (matches embedded firmware)

Author : EmbedSim Framework
Version: 1.0.0
"""

import sys
import numpy as np
from typing import List, Optional

from _path_utils import get_embedsim_import_path
sys.path.insert(0, get_embedsim_import_path())

from embedsim.code_generator import SimBlockBase
from embedsim.core_blocks    import VectorSignal


# ==============================================================================
# Pure-Python SMC implementation  (no C dependency)
# ==============================================================================

class _PySMC:
    """
    Minimal Python re-implementation of SMC_Block_T.
    Used when C wrapper is not available or use_c_backend=False.
    """

    def __init__(self,
                 lambda_d:  float = 500.0,
                 K_sw_d:    float = 24.0,
                 phi_d:     float = 5.0,
                 lambda_q:  float = 500.0,
                 K_sw_q:    float = 24.0,
                 phi_q:     float = 5.0,
                 out_min:   float = -24.0,
                 out_max:   float =  24.0):

        self.params = [
            dict(lam=lambda_d, K=K_sw_d, phi=max(phi_d, 1e-9),
                 lo=out_min, hi=out_max),
            dict(lam=lambda_q, K=K_sw_q, phi=max(phi_q, 1e-9),
                 lo=out_min, hi=out_max),
        ]
        self.state = [
            dict(integral=0.0, surface=0.0, output=0.0),
            dict(integral=0.0, surface=0.0, output=0.0),
        ]

    @staticmethod
    def _sat(v: float) -> float:
        if v >  1.0: return  1.0
        if v < -1.0: return -1.0
        return v

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _channel(self, ch: int, ref: float, meas: float, dt: float) -> float:
        p = self.params[ch]
        s = self.state[ch]
        e = ref - meas
        s['integral'] += e * dt
        surface = e + p['lam'] * s['integral']
        s['surface'] = surface
        raw = p['K'] * self._sat(surface / p['phi'])
        clamped = self._clamp(raw, p['lo'], p['hi'])
        if clamped != raw:                    # anti-windup
            s['integral'] -= e * dt
        s['output'] = clamped
        return clamped

    def compute(self, ref_d, ref_q, meas_d, meas_q, dt):
        v_d = self._channel(0, ref_d, meas_d, dt)
        v_q = self._channel(1, ref_q, meas_q, dt)
        return float(v_d), float(v_q)

    def reset(self):
        for s in self.state:
            s['integral'] = 0.0
            s['surface']  = 0.0
            s['output']   = 0.0

    def get_surface(self, ch): return self.state[ch]['surface']
    def get_integral(self, ch): return self.state[ch]['integral']


# ==============================================================================
# SMCBlock  —  EmbedSim VectorBlock
# ==============================================================================

class SMCBlock(SimBlockBase):
    """
    Sliding Mode Controller block for PMSM FOC d-q axis inner loop.

    Parameters
    ----------
    name        : str   — unique block identifier
    lambda_d    : float — d-axis surface slope     (default 500)
    K_sw_d      : float — d-axis switching gain    (default 24 V)
    phi_d       : float — d-axis boundary layer    (default 5 A)
    lambda_q    : float — q-axis surface slope     (default 500)
    K_sw_q      : float — q-axis switching gain    (default 24 V)
    phi_q       : float — q-axis boundary layer    (default 5 A)
    out_min     : float — output clamp low  [V]    (default -24)
    out_max     : float — output clamp high [V]    (default +24)
    use_c_backend: bool — use compiled smc_wrapper.pyd

    Ports
    -----
    port 0 : [id_ref, iq_ref]    — current references  [A]
    port 1 : [id_meas, iq_meas]  — measured currents   [A]

    Output
    ------
    [v_d, v_q]  — voltage commands  [V]
    """

    # ── CodeGen marker attributes (read by PYXInspector feature 05121967) ────
    import pathlib as _pl
    #: Absolute path to .pyx — works regardless of working directory
    PYX_FILE:    str  = str(_pl.Path(__file__).parent / 'c_src' / 'smc_wrapper.pyx')
    #: step_func / state_struct filled by PYXInspector via __init_subclass__
    step_func:   str  = 'SMC_Compute'
    state_struct: str = 'SMC_Block_T'
    #: Input port count
    NUM_INPUTS:  int  = 2
    #: Output vector size
    OUTPUT_SIZE: int  = 2
    #: C source files required by this block
    C_SOURCES:   list = ['sliding_mode_controller.c']
    #: C header files
    C_HEADERS:   list = ['sliding_mode_controller.h', 'Sys_Types.h']

    def __init__(
        self,
        name:          str,
        lambda_d:      float = 500.0,
        K_sw_d:        float = 24.0,
        phi_d:         float = 5.0,
        lambda_q:      float = 500.0,
        K_sw_q:        float = 24.0,
        phi_q:         float = 5.0,
        out_min:       float = -24.0,
        out_max:       float =  24.0,
        use_c_backend: bool  = False,
        dtype                = None,
    ) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        self.output_label = "[v_d,v_q]"
        self.is_dynamic   = False         # state managed internally by C wrapper / _PySMC
        self.vector_size  = 2
        # RK4 guard: engine checks b.state on all is_dynamic blocks
        # SMC state is owned by _impl (C wrapper or _PySMC) — not by RK4
        self.state = None

        # Store tuning params for reset/repr
        self._params = dict(
            lambda_d=lambda_d, K_sw_d=K_sw_d, phi_d=phi_d,
            lambda_q=lambda_q, K_sw_q=K_sw_q, phi_q=phi_q,
            out_min=out_min,   out_max=out_max,
        )

        if use_c_backend:
            self._impl = self._load_c_wrapper(
                lambda_d, K_sw_d, phi_d,
                lambda_q, K_sw_q, phi_q,
                out_min, out_max,
            )
        else:
            self._impl = _PySMC(
                lambda_d, K_sw_d, phi_d,
                lambda_q, K_sw_q, phi_q,
                out_min, out_max,
            )

    # ── C loader ─────────────────────────────────────────────────────────────

    @staticmethod
    def _load_c_wrapper(ld, Kd, pd, lq, Kq, pq, lo, hi):
        try:
            import smc_wrapper as sw
            w = sw.SMCWrapper()
            w.set_params_d(lambda_=ld, K_sw=Kd, phi=pd, out_min=lo, out_max=hi)
            w.set_params_q(lambda_=lq, K_sw=Kq, phi=pq, out_min=lo, out_max=hi)
            return w
        except ImportError:
            raise ImportError(
                "Cython wrapper 'smc_wrapper' not found.\n"
                "Compile with: python setup_smc.py build_ext --inplace\n"
                "Or set use_c_backend=False to use the Python backend."
            )

    # ── Compute dispatch ──────────────────────────────────────────────────────

    def compute(self, t, dt, input_values=None):
        if self.use_c_backend:
            return self.compute_c(t, dt, input_values)
        return self.compute_py(t, dt, input_values)

    def _parse_inputs(self, input_values):
        """Extract (id_ref, iq_ref, id_meas, iq_meas) safely."""
        zero2 = np.zeros(2, dtype=np.float32)
        if not input_values or len(input_values) < 2:
            return 0.0, 0.0, 0.0, 0.0
        ref  = input_values[0].value if input_values[0] is not None else zero2
        meas = input_values[1].value if input_values[1] is not None else zero2
        id_ref  = float(ref[0])  if len(ref)  >= 1 else 0.0
        iq_ref  = float(ref[1])  if len(ref)  >= 2 else 0.0
        id_meas = float(meas[0]) if len(meas) >= 1 else 0.0
        iq_meas = float(meas[1]) if len(meas) >= 2 else 0.0
        return id_ref, iq_ref, id_meas, iq_meas

    # -- Python backend -------------------------------------------------------

    def compute_py(
        self,
        t:  float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        id_ref, iq_ref, id_meas, iq_meas = self._parse_inputs(input_values)
        v_d, v_q = self._impl.compute(id_ref, iq_ref, id_meas, iq_meas, dt)
        self.output = VectorSignal(
            np.array([v_d, v_q], dtype=np.float32), self.name, dtype=self.dtype
        )
        return self.output

    # -- C backend ------------------------------------------------------------

    def compute_c(
        self,
        t:  float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        id_ref, iq_ref, id_meas, iq_meas = self._parse_inputs(input_values)
        v_d, v_q = self._impl.compute(
            id_ref, iq_ref, id_meas, iq_meas, float(dt)
        )
        self.output = VectorSignal(
            np.array([v_d, v_q], dtype=np.float32), self.name, dtype=self.dtype
        )
        return self.output

    # ── Block lifecycle (EmbedSim engine hooks) ───────────────────────────────

    def reset(self) -> None:
        """Reset SMC integrators (called by EmbedSim at sim start)."""
        super().reset()
        self._impl.reset()

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @property
    def surface_d(self) -> float:
        """Current d-axis sliding surface value s(e_d)."""
        return self._impl.get_surface(0)

    @property
    def surface_q(self) -> float:
        """Current q-axis sliding surface value s(e_q)."""
        return self._impl.get_surface(1)

    @property
    def integral_d(self) -> float:
        """Current d-axis integrator state."""
        return self._impl.get_integral(0)

    @property
    def integral_q(self) -> float:
        """Current q-axis integrator state."""
        return self._impl.get_integral(1)

    def set_params_d(self, lambda_=None, K_sw=None, phi=None,
                     out_min=None, out_max=None) -> None:
        """Update d-axis SMC parameters at runtime."""
        p = self._params
        lam = lambda_  if lambda_  is not None else p['lambda_d']
        K   = K_sw     if K_sw     is not None else p['K_sw_d']
        ph  = phi      if phi      is not None else p['phi_d']
        lo  = out_min  if out_min  is not None else p['out_min']
        hi  = out_max  if out_max  is not None else p['out_max']
        if self.use_c_backend:
            self._impl.set_params_d(lambda_=lam, K_sw=K, phi=ph, out_min=lo, out_max=hi)
        else:
            self._impl.params[0] = dict(lam=lam, K=K, phi=max(ph,1e-9), lo=lo, hi=hi)

    def set_params_q(self, lambda_=None, K_sw=None, phi=None,
                     out_min=None, out_max=None) -> None:
        """Update q-axis SMC parameters at runtime."""
        p = self._params
        lam = lambda_  if lambda_  is not None else p['lambda_q']
        K   = K_sw     if K_sw     is not None else p['K_sw_q']
        ph  = phi      if phi      is not None else p['phi_q']
        lo  = out_min  if out_min  is not None else p['out_min']
        hi  = out_max  if out_max  is not None else p['out_max']
        if self.use_c_backend:
            self._impl.set_params_q(lambda_=lam, K_sw=K, phi=ph, out_min=lo, out_max=hi)
        else:
            self._impl.params[1] = dict(lam=lam, K=K, phi=max(ph,1e-9), lo=lo, hi=hi)

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        p  = self._params
        return (
            f"SMCBlock('{self.name}', "
            f"λ_d={p['lambda_d']}, K_d={p['K_sw_d']}, φ_d={p['phi_d']}, "
            f"λ_q={p['lambda_q']}, K_q={p['K_sw_q']}, φ_q={p['phi_q']}, "
            f"backend={be})"
        )
