# svpwm_block.py
# =============================================================================
# EmbedSim VectorBlock wrapper for the SVPWM C implementation.
# Location: fs_electrical_machines/foc_generator/svpwm_block.py
#
# Inputs  (3 scalars):
#   [0]  Vref   — reference voltage magnitude [V]
#   [1]  alpha  — reference angle [rad]  (electrical, 0..2*pi)
#   [2]  Vdc    — DC bus voltage [V]
#
# Outputs (4 scalars):
#   [0]  T1     — active vector 1 on-time [s]
#   [1]  T2     — active vector 2 on-time [s]
#   [2]  T0     — zero vector on-time [s]
#   [3]  sector — active sector 1..6  (cast to real32_T for signal bus)
#
# CodeGen notes:
#   C_CUSTOM_EMIT is used because SVPWM_Step() takes typed structs
#   (SVPWM_Input*, SVPWM_Output*) — incompatible with the standard
#   flat real32_T u[]/y[] auto-emission path in LoopGenerator._emit_block().
#
#   C_INPUT_MAP declares the three upstream signal sources explicitly so
#   LoopGenerator does not have to infer them from block.inputs, which
#   would be ambiguous when one of the sources is a LoopBreaker (theta_e).
# =============================================================================

from typing import List, Optional
import numpy as np
from embedsim.core_blocks import VectorBlock, VectorSignal


class SVPWMBlock(VectorBlock):
    """
    Space Vector PWM switching time calculator.

    Pure-Python path implements the same T1/T2/T0 mathematics as svpwm.c
    so the simulation result is bit-comparable with the C build.

    CodeGen (C) path emits a typed-struct call via C_CUSTOM_EMIT.
    """

    # ── CodeGen class attributes ─────────────────────────────────────────────
    NUM_INPUTS  = 3
    OUTPUT_SIZE = 4

    C_SOURCES = ["svpwm.c"]
    C_HEADERS = ["svpwm.h"]

    # Explicit port → upstream-signal wiring.
    # Format: list of (source_block_name, output_index_within_that_block)
    # One entry per element of the C u[] input array.
    #
    # Adjust source block names to match your simulation wiring:
    #   "vref_block"  → output of the speed/current PI that produces |Vref|
    #   "theta_e"     → LoopBreaker / unit-delay carrying electrical angle
    #   "vdc_block"   → constant or measured DC bus voltage
    C_INPUT_MAP = [
        ("vref_block", 0),   # u[0] ← Vref   magnitude
        ("theta_e",    0),   # u[1] ← alpha  (electrical angle, rad)
        ("vdc_block",  0),   # u[2] ← Vdc
    ]

    # Struct-pointer ABI — bypass auto-emission.
    # 'sn' is the sanitised block name injected by _emit_block().
    # 'dt' is the loop step parameter already in scope inside embedsim_loop_step().
    C_CUSTOM_EMIT = """\
    /* --- svpwm (SVPWMBlock) --- */
    SVPWM_Input  u_svpwm;
    SVPWM_Output y_svpwm;
    u_svpwm.Vref  = y_vref_block[0];
    u_svpwm.alpha = y_theta_e[0];
    u_svpwm.Vdc   = y_vdc_block[0];
    u_svpwm.Ts    = dt;
    SVPWM_Step(&u_svpwm, &y_svpwm);
    real32_T y_svpwm[4];
    y_svpwm[0] = y_svpwm_out.T1;
    y_svpwm[1] = y_svpwm_out.T2;
    y_svpwm[2] = y_svpwm_out.T0;
    y_svpwm[3] = (real32_T)y_svpwm_out.sector;
"""

    # ── Constants (mirror svpwm.c) ───────────────────────────────────────────
    _SQRT3      = np.sqrt(3.0, dtype=np.float32)
    _PI_OVER_3  = np.float32(np.pi / 3.0)
    _TWO_PI     = np.float32(2.0 * np.pi)

    # ── Constructor ──────────────────────────────────────────────────────────
    def __init__(self, name: str = "svpwm",
                 use_c_backend: bool = False,
                 dtype=np.float32):
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)
        self.vector_size = self.OUTPUT_SIZE

    # ── Python step ──────────────────────────────────────────────────────────
    def compute_py(self,
                   t: float,
                   dt: float,
                   input_values: Optional[List[VectorSignal]] = None
                   ) -> VectorSignal:
        """
        Compute SVPWM dwell times in pure Python.

        Expects input_values packed as [Vref, alpha, Vdc] across
        the connected upstream signals.
        """
        # ── Unpack inputs ────────────────────────────────────────────────────
        if input_values is None or len(input_values) == 0:
            y = np.zeros(self.OUTPUT_SIZE, dtype=np.float32)
            self.output = VectorSignal(y, self.name)
            return self.output

        flat = np.concatenate(
            [np.atleast_1d(sig.value).astype(np.float32) for sig in input_values]
        )

        Vref  = float(flat[0]) if len(flat) > 0 else 0.0
        alpha = float(flat[1]) if len(flat) > 1 else 0.0
        Vdc   = float(flat[2]) if len(flat) > 2 else 1.0
        Ts    = float(dt)

        # ── Guard ────────────────────────────────────────────────────────────
        if Vdc < 1.0e-6:
            y = np.array([0.0, 0.0, Ts, 1.0], dtype=np.float32)
            self.output = VectorSignal(y, self.name)
            return self.output

        # ── Normalise alpha to [0, 2*pi) ─────────────────────────────────────
        alpha_norm = alpha % float(self._TWO_PI)
        if alpha_norm < 0.0:
            alpha_norm += float(self._TWO_PI)

        # ── Modulation index ─────────────────────────────────────────────────
        modulation = (float(self._SQRT3) * Vref) / Vdc

        # ── Sector 1..6 ──────────────────────────────────────────────────────
        sector = int(alpha_norm / float(self._PI_OVER_3)) + 1
        sector = max(1, min(6, sector))

        # ── alpha relative to sector start ───────────────────────────────────
        alpha_local = alpha_norm - (sector - 1) * float(self._PI_OVER_3)

        # ── Dwell times ───────────────────────────────────────────────────────
        T1 = Ts * modulation * np.sin(float(self._PI_OVER_3) - alpha_local)
        T2 = Ts * modulation * np.sin(alpha_local)
        T0 = max(0.0, Ts - T1 - T2)

        y = np.array([T1, T2, T0, float(sector)], dtype=np.float32)
        self.output = VectorSignal(y, self.name)
        return self.output

    def reset(self):
        """Stateless — no state to reset."""
        self.output = None
