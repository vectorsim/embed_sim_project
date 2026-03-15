"""
coordinate_transform_blocks.py  —  fs_electrical_machines
==========================================================
Clarke / Park / InvClarke / InvPark as EmbedSim VectorBlock subclasses.

Each block:
  - has a pure-Python compute_py() fallback (works before Cython build)
  - tries to use the Cython/C wrapper in compute() when available
  - carries full CodeGen attributes for Feature 05121967 (PYXInspector)

Signal conventions
------------------
  Clarke    : input  [i_a, i_b, i_c]  → output [i_alpha, i_beta]
  Park      : input  [i_alpha, i_beta, theta_e]  → output [i_d, i_q]
  InvPark   : input  [v_d, v_q, theta_e]  → output [v_alpha, v_beta]
  InvClarke : input  [v_alpha, v_beta]  → output [v_a, v_b, v_c]
"""

from __future__ import annotations

import math
import sys
import numpy as np
from pathlib import Path

# ── path bootstrap ────────────────────────────────────────────────────────────
from _path_utils import get_embedsim_import_path, get_current_parent

sys.path.insert(0, get_embedsim_import_path())

from embedsim.core_blocks import VectorBlock, VectorSignal

_C_SRC = get_current_parent() / "c_src"
if str(_C_SRC) not in sys.path:
    sys.path.insert(0, str(_C_SRC))

# ── try to import Cython wrapper ──────────────────────────────────────────────
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


# =============================================================================
# ClarkeTransformBlock
# =============================================================================

class ClarkeTransformBlock(VectorBlock):
    """
    Clarke transform: [i_a, i_b, i_c] → [i_alpha, i_beta]

    Power-invariant form:
        i_alpha = (2/3) * i_a  −  (1/3) * i_b  −  (1/3) * i_c
        i_beta  = (1/√3) * (i_b − i_c)
    """

    NUM_INPUTS:   int  = 1
    OUTPUT_SIZE:  int  = 2
    C_SOURCES:    list = ["Coordinate_Transform.c", "Matrix.c"]
    C_HEADERS:    list = ["Coordinate_Transform.h"]
    step_func:    str  = "Clarke_Step"
    init_func:    str  = "Clarke_Init"
    state_struct: str  = "Clarke_T"

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
        i_alpha = (2.0 / 3.0) * i_a - (1.0 / 3.0) * i_b - (1.0 / 3.0) * i_c
        i_beta  = (i_b - i_c) / math.sqrt(3.0)
        out = np.array([i_alpha, i_beta], dtype=np.float32)
        self.output = VectorSignal(out, self.name, dtype=self.dtype)
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
        self.output = VectorSignal(np.array(y, dtype=np.float32), self.name, dtype=self.dtype)
        return self.output

    def reset(self):
        super().reset()
        if self._wrapper is not None:
            self._wrapper.reset()


# =============================================================================
# ParkTransformBlock
# =============================================================================

class ParkTransformBlock(VectorBlock):
    """
    Park transform: [i_alpha, i_beta, theta_e] → [i_d, i_q]

        i_d =  i_alpha * cos(θ) + i_beta * sin(θ)
        i_q = −i_alpha * sin(θ) + i_beta * cos(θ)
    """

    NUM_INPUTS:   int  = 2          # port 0: [alpha, beta],  port 1: theta scalar
    OUTPUT_SIZE:  int  = 2
    C_SOURCES:    list = ["Coordinate_Transform.c", "Matrix.c"]
    C_HEADERS:    list = ["Coordinate_Transform.h"]
    step_func:    str  = "Park_Step"
    init_func:    str  = "Park_Init"
    state_struct: str  = "Park_T"

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
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        i_d =  alpha * cos_t + beta * sin_t
        i_q = -alpha * sin_t + beta * cos_t
        out = np.array([i_d, i_q], dtype=np.float32)
        self.output = VectorSignal(out, self.name, dtype=self.dtype)
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
        self.output = VectorSignal(np.array(y, dtype=np.float32), self.name, dtype=self.dtype)
        return self.output

    def reset(self):
        super().reset()
        if self._wrapper is not None:
            self._wrapper.reset()


# =============================================================================
# InvParkTransformBlock
# =============================================================================

class InvParkTransformBlock(VectorBlock):
    """
    Inverse Park transform: [v_d, v_q, theta_e] → [v_alpha, v_beta]

        v_alpha = v_d * cos(θ) − v_q * sin(θ)
        v_beta  = v_d * sin(θ) + v_q * cos(θ)
    """

    NUM_INPUTS:   int  = 2          # port 0: [d, q],  port 1: theta scalar
    OUTPUT_SIZE:  int  = 2
    C_SOURCES:    list = ["Coordinate_Transform.c", "Matrix.c"]
    C_HEADERS:    list = ["Coordinate_Transform.h"]
    step_func:    str  = "InvPark_Step"
    init_func:    str  = "InvPark_Init"
    state_struct: str  = "InvPark_T"

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
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        v_alpha = v_d * cos_t - v_q * sin_t
        v_beta  = v_d * sin_t + v_q * cos_t
        out = np.array([v_alpha, v_beta], dtype=np.float32)
        self.output = VectorSignal(out, self.name, dtype=self.dtype)
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
        self.output = VectorSignal(np.array(y, dtype=np.float32), self.name, dtype=self.dtype)
        return self.output

    def reset(self):
        super().reset()
        if self._wrapper is not None:
            self._wrapper.reset()


# =============================================================================
# InvClarkeTransformBlock
# =============================================================================

class InvClarkeTransformBlock(VectorBlock):
    """
    Inverse Clarke transform: [v_alpha, v_beta] → [v_a, v_b, v_c]

        v_a =  v_alpha
        v_b = −(1/2) * v_alpha + (√3/2) * v_beta
        v_c = −(1/2) * v_alpha − (√3/2) * v_beta
    """

    NUM_INPUTS:   int  = 1
    OUTPUT_SIZE:  int  = 3
    C_SOURCES:    list = ["Coordinate_Transform.c", "Matrix.c"]
    C_HEADERS:    list = ["Coordinate_Transform.h"]
    step_func:    str  = "InvClarke_Step"
    init_func:    str  = "InvClarke_Init"
    state_struct: str  = "InvClarke_T"

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
        out = np.array([v_a, v_b, v_c], dtype=np.float32)
        self.output = VectorSignal(out, self.name, dtype=self.dtype)
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
        self.output = VectorSignal(np.array(y, dtype=np.float32), self.name, dtype=self.dtype)
        return self.output

    def reset(self):
        super().reset()
        if self._wrapper is not None:
            self._wrapper.reset()
