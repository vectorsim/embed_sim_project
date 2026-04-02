"""
coordinate_transform_blocks.py  --  fs_electrical_machines
==========================================================
Clarke / Park / InvClarke / InvPark as EmbedSim VectorBlock subclasses.

PYX_FILE is resolved using Path(__file__) so it is always absolute.

C_CUSTOM_EMIT is used for all four transforms because their C functions
take individual scalar arguments and pointer outputs — incompatible with
the flat real32_T u[]/y[] auto-emission path in LoopGenerator._emit_block().

Actual signatures (from Coordinate_Transform.h):
    Clarke_Step (Clarke_T*, ia, ib, ic, alpha_out*, beta_out*)
    Park_Step   (Park_T*,   alpha, beta, theta, d_out*, q_out*)
    InvPark_Step(InvPark_T*, d, q, theta, alpha_out*, beta_out*)
    InvClarke_Step(InvClarke_T*, alpha, beta, va_out*, vb_out*, vc_out*)
"""

from __future__ import annotations

import math
import sys
import numpy as np
from pathlib import Path

# -- path bootstrap ------------------------------------------------------------
_HERE  = Path(__file__).resolve().parent        # always fs_electrical_machines/
_C_SRC = _HERE / "c_src"

_root = _HERE.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if str(_C_SRC) not in sys.path:
    sys.path.insert(0, str(_C_SRC))

from embedsim.core_blocks import VectorBlock, VectorSignal

# Absolute path to the .pyx — PYXInspector fills C_SOURCES, C_HEADERS,
# step_func, init_func, state_struct at class-definition time.
_PYX = str(_C_SRC / "coordinate_transform_wrapper.pyx")

# -- try to import Cython wrapper ----------------------------------------------
try:
    from coordinate_transform_wrapper import (
        ClarkeWrapper,
        ParkWrapper,
        InvParkWrapper,
        InvClarkeWrapper,
    )
    _HAS_C = True
except ImportError:
    _HAS_C = False


# ==============================================================================
# ClarkeTransformBlock
# ==============================================================================

class ClarkeTransformBlock(VectorBlock):
    """
    Clarke transform: [i_a, i_b, i_c] -> [i_alpha, i_beta]

    C signature:
        void Clarke_Step(Clarke_T* pS,
                         MatrixFloat ia, MatrixFloat ib, MatrixFloat ic,
                         MatrixFloat* alpha_out, MatrixFloat* beta_out);

    C_CUSTOM_EMIT bypasses flat-array auto-emission because the C function
    takes individual scalar args and pointer outputs, not u[]/y[] arrays.
    State struct Clarke_T is stateful per the header (has Clarke_Init).
    """

    PYX_FILE     = _PYX
    NUM_INPUTS   = 1          # single port: [ia, ib, ic]
    OUTPUT_SIZE  = 2          # [i_alpha, i_beta]
    C_SOURCES    = ["Coordinate_Transform.c", "Matrix.c"]
    C_HEADERS    = ["Coordinate_Transform.h"]
    init_func    = "Clarke_Init"
    step_func    = "Clarke_Step"
    state_struct = "Clarke_T"

    # Custom C emission — matches the actual scalar-arg pointer-output signature
    C_CUSTOM_EMIT = """\
    /* --- Clarke (ClarkeTransformBlock) --- */
    {
        MatrixFloat clarke_alpha, clarke_beta;
        Clarke_Step(&Clarke_state,
                    u_cg_start[0], u_cg_start[1], u_cg_start[2],
                    &clarke_alpha, &clarke_beta);
        real32_T y_Clarke[2];
        y_Clarke[0] = clarke_alpha;
        y_Clarke[1] = clarke_beta;
    }"""

    def __init__(self, name: str, use_c_backend: bool = True, dtype=None):
        super().__init__(name, dtype=dtype)
        self.is_dynamic   = False
        self.vector_size  = 2
        self.output_label = "[i_alpha,i_beta]"
        self._use_c       = use_c_backend and _HAS_C
        self._wrapper     = ClarkeWrapper() if self._use_c else None

    def compute_py(self, t, dt, input_values=None):
        i_a = i_b = i_c = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                i_a, i_b, i_c = float(v[0]), float(v[1]), float(v[2])
        # Amplitude-invariant Clarke -- matches iLLD Clarke() and C embed_sim_coordinate_transform.c
        # i_alpha = ia
        # i_beta  = (ia + 2*ib) / sqrt(3)
        i_alpha = i_a
        i_beta  = (i_a + 2.0 * i_b) / math.sqrt(3.0)
        self.output = VectorSignal(np.array([i_alpha, i_beta], dtype=np.float32),
                                   self.name, dtype=self.dtype)
        return self.output

    def compute(self, t, dt, input_values=None):
        if not self._use_c:
            return self.compute_py(t, dt, input_values)
        u = np.zeros(3, dtype=np.float32)
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            u[:min(3, len(v))] = v[:3]
        self._wrapper.step(float(u[0]), float(u[1]), float(u[2]))
        y = self._wrapper.get_outputs()
        self.output = VectorSignal(np.array(y, dtype=np.float32),
                                   self.name, dtype=self.dtype)
        return self.output

    def reset(self):
        super().reset()
        if self._wrapper is not None:
            self._wrapper.reset()


# ==============================================================================
# ParkTransformBlock
# ==============================================================================

class ParkTransformBlock(VectorBlock):
    """
    Park transform: [i_alpha, i_beta] + theta_e -> [i_d, i_q]

    C signature:
        void Park_Step(Park_T* pS,
                       MatrixFloat alpha, MatrixFloat beta, MatrixFloat theta,
                       MatrixFloat* d_out, MatrixFloat* q_out);

    theta comes from the upstream VectorConstant (Theta block).
    In generated C it is a static const defined in embedsim_loop.c.
    """

    PYX_FILE     = _PYX
    NUM_INPUTS   = 2          # port 0: [alpha, beta]  port 1: theta scalar
    OUTPUT_SIZE  = 2          # [i_d, i_q]
    C_SOURCES    = ["Coordinate_Transform.c", "Matrix.c"]
    C_HEADERS    = ["Coordinate_Transform.h"]
    init_func    = "Park_Init"
    step_func    = "Park_Step"
    state_struct = "Park_T"

    C_CUSTOM_EMIT = """\
    /* --- Park (ParkTransformBlock) --- */
    {
        MatrixFloat park_d, park_q;
        Park_Step(&Park_state,
                  y_Clarke[0], y_Clarke[1],
                  THETA_E,
                  &park_d, &park_q);
        real32_T y_Park[2];
        y_Park[0] = park_d;
        y_Park[1] = park_q;
    }"""

    def __init__(self, name: str, use_c_backend: bool = True, dtype=None):
        super().__init__(name, dtype=dtype)
        self.is_dynamic   = False
        self.vector_size  = 2
        self.output_label = "[i_d,i_q]"
        self._use_c       = use_c_backend and _HAS_C
        self._wrapper     = ParkWrapper() if self._use_c else None

    def compute_py(self, t, dt, input_values=None):
        alpha = beta = theta = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 2:
                alpha, beta = float(v[0]), float(v[1])
        if input_values and len(input_values) > 1 and input_values[1] is not None:
            theta = float(input_values[1].value[0])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        i_d =  alpha * cos_t + beta * sin_t
        i_q = -alpha * sin_t + beta * cos_t
        self.output = VectorSignal(np.array([i_d, i_q], dtype=np.float32),
                                   self.name, dtype=self.dtype)
        return self.output

    def compute(self, t, dt, input_values=None):
        if not self._use_c:
            return self.compute_py(t, dt, input_values)
        alpha = beta = theta = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 2:
                alpha, beta = float(v[0]), float(v[1])
        if input_values and len(input_values) > 1 and input_values[1] is not None:
            theta = float(input_values[1].value[0])
        self._wrapper.step(alpha, beta, theta)
        y = self._wrapper.get_outputs()
        self.output = VectorSignal(np.array(y, dtype=np.float32),
                                   self.name, dtype=self.dtype)
        return self.output

    def reset(self):
        super().reset()
        if self._wrapper is not None:
            self._wrapper.reset()


# ==============================================================================
# InvParkTransformBlock
# ==============================================================================

class InvParkTransformBlock(VectorBlock):
    """
    Inverse Park: [v_d, v_q] + theta_e -> [v_alpha, v_beta]

    C signature:
        void InvPark_Step(InvPark_T* pS,
                          MatrixFloat d, MatrixFloat q, MatrixFloat theta,
                          MatrixFloat* alpha_out, MatrixFloat* beta_out);
    """

    PYX_FILE     = _PYX
    NUM_INPUTS   = 2
    OUTPUT_SIZE  = 2
    C_SOURCES    = ["Coordinate_Transform.c", "Matrix.c"]
    C_HEADERS    = ["Coordinate_Transform.h"]
    init_func    = "InvPark_Init"
    step_func    = "InvPark_Step"
    state_struct = "InvPark_T"

    C_CUSTOM_EMIT = """\
    /* --- inv_park (InvParkTransformBlock) --- */
    real32_T y_inv_park[2];
    {
        MatrixFloat invpark_alpha, invpark_beta;
        InvPark_Step(&inv_park_state,
                     y_vf_dq[0],    /* v_d    */
                     y_vf_dq[1],    /* v_q    */
                     y_vf_theta[0], /* theta_e */
                     &invpark_alpha, &invpark_beta);
        y_inv_park[0] = (real32_T)invpark_alpha;
        y_inv_park[1] = (real32_T)invpark_beta;
    }"""

    def __init__(self, name: str, use_c_backend: bool = True, dtype=None):
        super().__init__(name, dtype=dtype)
        self.is_dynamic   = False
        self.vector_size  = 2
        self.output_label = "[v_alpha,v_beta]"
        self._use_c       = use_c_backend and _HAS_C
        self._wrapper     = InvParkWrapper() if self._use_c else None

    def compute_py(self, t, dt, input_values=None):
        v_d = v_q = theta = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 2:
                v_d, v_q = float(v[0]), float(v[1])
        if input_values and len(input_values) > 1 and input_values[1] is not None:
            theta = float(input_values[1].value[0])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        v_alpha = v_d * cos_t - v_q * sin_t
        v_beta  = v_d * sin_t + v_q * cos_t
        self.output = VectorSignal(np.array([v_alpha, v_beta], dtype=np.float32),
                                   self.name, dtype=self.dtype)
        return self.output

    def compute(self, t, dt, input_values=None):
        if not self._use_c:
            return self.compute_py(t, dt, input_values)
        v_d = v_q = theta = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 2:
                v_d, v_q = float(v[0]), float(v[1])
        if input_values and len(input_values) > 1 and input_values[1] is not None:
            theta = float(input_values[1].value[0])
        self._wrapper.step(v_d, v_q, theta)
        y = self._wrapper.get_outputs()
        self.output = VectorSignal(np.array(y, dtype=np.float32),
                                   self.name, dtype=self.dtype)
        return self.output

    def reset(self):
        super().reset()
        if self._wrapper is not None:
            self._wrapper.reset()


# ==============================================================================
# InvClarkeTransformBlock
# ==============================================================================

class InvClarkeTransformBlock(VectorBlock):
    """
    Inverse Clarke: [v_alpha, v_beta] -> [v_a, v_b, v_c]

    C signature:
        void InvClarke_Step(InvClarke_T* pS,
                            MatrixFloat alpha, MatrixFloat beta,
                            MatrixFloat* va_out, MatrixFloat* vb_out,
                            MatrixFloat* vc_out);
    """

    PYX_FILE     = _PYX
    NUM_INPUTS   = 1
    OUTPUT_SIZE  = 3
    C_SOURCES    = ["Coordinate_Transform.c", "Matrix.c"]
    C_HEADERS    = ["Coordinate_Transform.h"]
    init_func    = "InvClarke_Init"
    step_func    = "InvClarke_Step"
    state_struct = "InvClarke_T"

    C_CUSTOM_EMIT = """\
    /* --- InvClarke (InvClarkeTransformBlock) --- */
    {
        MatrixFloat invclarke_va, invclarke_vb, invclarke_vc;
        InvClarke_Step(&InvClarke_state,
                       y_upstream_alpha[0], y_upstream_beta[0],
                       &invclarke_va, &invclarke_vb, &invclarke_vc);
        real32_T y_InvClarke[3];
        y_InvClarke[0] = invclarke_va;
        y_InvClarke[1] = invclarke_vb;
        y_InvClarke[2] = invclarke_vc;
    }"""

    _HALF_SQRT3 = math.sqrt(3.0) / 2.0

    def __init__(self, name: str, use_c_backend: bool = True, dtype=None):
        super().__init__(name, dtype=dtype)
        self.is_dynamic   = False
        self.vector_size  = 3
        self.output_label = "[v_a,v_b,v_c]"
        self._use_c       = use_c_backend and _HAS_C
        self._wrapper     = InvClarkeWrapper() if self._use_c else None

    def compute_py(self, t, dt, input_values=None):
        v_alpha = v_beta = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 2:
                v_alpha, v_beta = float(v[0]), float(v[1])
        v_a =  v_alpha
        v_b = -0.5 * v_alpha + self._HALF_SQRT3 * v_beta
        v_c = -0.5 * v_alpha - self._HALF_SQRT3 * v_beta
        self.output = VectorSignal(np.array([v_a, v_b, v_c], dtype=np.float32),
                                   self.name, dtype=self.dtype)
        return self.output

    def compute(self, t, dt, input_values=None):
        if not self._use_c:
            return self.compute_py(t, dt, input_values)
        v_alpha = v_beta = 0.0
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 2:
                v_alpha, v_beta = float(v[0]), float(v[1])
        self._wrapper.step(v_alpha, v_beta)
        y = self._wrapper.get_outputs()
        self.output = VectorSignal(np.array(y, dtype=np.float32),
                                   self.name, dtype=self.dtype)
        return self.output

    def reset(self):
        super().reset()
        if self._wrapper is not None:
            self._wrapper.reset()
