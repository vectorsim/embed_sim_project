"""
dynamic_blocks.py
=================

Dynamic system blocks with continuous and discrete state evolution.

Part of the EmbedSim framework — lightweight block-diagram simulation
targeting 32-bit embedded platforms.

All numeric arrays default to float32 for MCU compatibility. Pass
dtype=np.float64 to individual blocks when double precision is required.

Classes:
    VectorIntegrator:      Continuous integrator (ẋ = u)
    StateSpaceBlock:       General state-space model (ẋ = Ax + Bu, y = Cx + Du)
    TransferFunctionBlock: Transfer function using state-space representation
    VectorEnd:             Sink block for recording simulation outputs

Author: EmbedSim Framework
Version: 1.0.0
"""

import numpy as np
from typing import List, Optional
from .core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE, validate_inputs_exist


# =========================
# Integrator Block
# =========================

class VectorIntegrator(VectorBlock):
    """
    Continuous integrator block implementing ẋ = u.

    Attributes:
        state (np.ndarray): Current integrator state (float32 by default)
        derivative (np.ndarray): Current rate of change (input value)
        is_dynamic (bool): Always True

    Example:
        >>> integrator = VectorIntegrator("int", initial_state=[0, 0, 0])
    """

    def __init__(self, name: str, initial_state: Optional[List[float]] = None,
                 dim: int = 3, use_c_backend: bool = False, dtype=None) -> None:
        """
        Initialize a VectorIntegrator block.

        Args:
            name:          Unique identifier for this block
            initial_state: Initial state x(0). If None, zeros of dimension dim.
            dim:           Dimension (used only when initial_state is None). Default: 3
            dtype:         Override dtype (default: float32).
        """
        _dtype = dtype if dtype is not None else DEFAULT_DTYPE
        super().__init__(name, use_c_backend=use_c_backend, dtype=_dtype)
        self.is_dynamic = True

        if initial_state is not None:
            self.state: np.ndarray = np.array(initial_state, dtype=_dtype)
        else:
            self.state = np.zeros(dim, dtype=_dtype)

        self.derivative: np.ndarray = np.zeros_like(self.state)

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        validate_inputs_exist(input_values, self.name, min_inputs=1)
        self.derivative = input_values[0].value.copy()
        self.output = VectorSignal(self.state.copy(), self.name, dtype=self.dtype)
        return self.output

    def get_derivative(self, t: float, input_values: Optional[List[VectorSignal]] = None) -> np.ndarray:
        if not input_values:
            return np.zeros_like(self.state)
        return input_values[0].value.astype(self.dtype)

    def integrate_state(self, dt: float, solver: str = 'euler') -> None:
        if solver == 'euler':
            self.state = self.state + self.derivative * dt
        elif solver == 'rk4':
            pass  # RK4 handled externally by simulation engine


# =========================
# State-Space Block
# =========================

class StateSpaceBlock(VectorBlock):
    """
    Continuous state-space model.

    Implements:
        ẋ(t) = A·x(t) + B·u(t)
        y(t) = C·x(t) + D·u(t)

    Attributes:
        A, B, C, D (np.ndarray): State-space matrices (float32 by default)
        state (np.ndarray): Current state vector x
        n_states (int): Number of states
        is_dynamic (bool): Always True

    Example:
        >>> rl_circuit = StateSpaceBlock("RL", A, B, C, D, initial_state=[0.0])
    """

    def __init__(self, name: str, A: np.ndarray, B: np.ndarray,
                 C: np.ndarray, D: np.ndarray,
                 initial_state: Optional[List[float]] = None,
                 use_c_backend: bool = False,
                 dtype=None) -> None:
        """
        Initialize a StateSpaceBlock.

        Args:
            name:          Unique identifier for this block
            A, B, C, D:   State-space matrices. Stored as float32 by default.
            initial_state: Initial state x(0). If None, zeros.
            dtype:         Override dtype (default: float32).
        """
        _dtype = dtype if dtype is not None else DEFAULT_DTYPE
        super().__init__(name, use_c_backend=use_c_backend, dtype=_dtype)
        self.is_dynamic = True

        self.A: np.ndarray = np.array(A, dtype=_dtype)
        self.B: np.ndarray = np.array(B, dtype=_dtype)
        self.C: np.ndarray = np.array(C, dtype=_dtype)
        self.D: np.ndarray = np.array(D, dtype=_dtype)

        self.n_states: int = self.A.shape[0]

        if initial_state is not None:
            self.state: np.ndarray = np.array(initial_state, dtype=_dtype)
        else:
            self.state = np.zeros(self.n_states, dtype=_dtype)

        self.derivative: np.ndarray = np.zeros_like(self.state)

        self.k1: Optional[np.ndarray] = None
        self.k2: Optional[np.ndarray] = None
        self.k3: Optional[np.ndarray] = None
        self.k4: Optional[np.ndarray] = None

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        if not input_values:
            u = np.zeros(self.B.shape[1], dtype=self.dtype)
        else:
            u = input_values[0].value.astype(self.dtype)

        self.derivative = self.A @ self.state + self.B @ u
        y = self.C @ self.state + self.D @ u

        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def get_derivative(self, t: float, input_values: Optional[List[VectorSignal]] = None) -> np.ndarray:
        if not input_values:
            u = np.zeros(self.B.shape[1], dtype=self.dtype)
        else:
            u = input_values[0].value.astype(self.dtype)
        return self.A @ self.state + self.B @ u

    def integrate_state(self, dt: float, solver: str = 'euler') -> None:
        if solver == 'euler':
            self.state = self.state + self.derivative * dt


# =========================
# Transfer Function Block
# =========================

class TransferFunctionBlock(VectorBlock):
    """
    Continuous transfer function using state-space (controllable canonical) form.

    Implements H(s) = num(s)/den(s). Each vector element is processed
    independently through the same transfer function.

    Example:
        >>> # First-order low-pass filter: H(s) = 100/(s + 100)
        >>> lpf = TransferFunctionBlock("lpf", num=[100], den=[1, 100], dim=3)
    """

    def __init__(self, name: str, num: List[float], den: List[float], use_c_backend: bool = False,
                 dim: int = 3, initial_state: Optional[np.ndarray] = None,
                 dtype=None) -> None:
        """
        Initialize a TransferFunctionBlock.

        Args:
            name:          Unique identifier for this block
            num:           Numerator coefficients (descending powers of s)
            den:           Denominator coefficients (descending powers of s)
            dim:           Number of independent input channels. Default: 3
            initial_state: Initial internal states, shape (dim, n_states).
                          If None, zeros.
            dtype:         Override dtype (default: float32).
        """
        _dtype = dtype if dtype is not None else DEFAULT_DTYPE
        super().__init__(name, use_c_backend=use_c_backend, dtype=_dtype)
        self.is_dynamic = True
        self.dim: int = dim

        self.num: np.ndarray = np.array(num, dtype=_dtype)
        self.den: np.ndarray = np.array(den, dtype=_dtype)

        a0 = self.den[0]
        self.num = self.num / a0
        self.den = self.den / a0

        n = len(self.den) - 1

        if n > 0:
            A = np.zeros((n, n), dtype=_dtype)
            if n > 1:
                A[:-1, 1:] = np.eye(n - 1, dtype=_dtype)
            A[-1, :] = -self.den[1:][::-1]

            B = np.zeros((n, 1), dtype=_dtype)
            B[-1, 0] = 1.0

            C = np.zeros((1, n), dtype=_dtype)
            if len(self.num) == len(self.den):
                diff = self.num - self.num[0] * self.den
                if n > 0 and len(diff) > 1:
                    C[0, :] = diff[1:][::-1]
            else:
                if n > 0 and len(self.num) > 1:
                    C[0, :] = self.num[1:][::-1]
                elif n > 0 and len(self.num) == 1:
                    C[0, -1] = self.num[0]

            D = (np.array([[self.num[0]]], dtype=_dtype)
                 if len(self.num) == len(self.den)
                 else np.array([[0.0]], dtype=_dtype))

            self.A = A
            self.B = B
            self.C = C
            self.D = D
        else:
            self.A = np.zeros((0, 0), dtype=_dtype)
            self.B = np.zeros((0, 1), dtype=_dtype)
            self.C = np.zeros((1, 0), dtype=_dtype)
            self.D = np.array([[self.num[0] / self.den[0]]], dtype=_dtype)

        self.n_states: int = self.A.shape[0]

        if initial_state is not None:
            self.state: np.ndarray = np.array(initial_state, dtype=_dtype)
        else:
            self.state = np.zeros((self.dim, self.n_states), dtype=_dtype)

        self.derivative: np.ndarray = np.zeros_like(self.state)
        self.k1 = np.zeros_like(self.state)
        self.k2 = np.zeros_like(self.state)
        self.k3 = np.zeros_like(self.state)
        self.k4 = np.zeros_like(self.state)

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        if not input_values:
            u = np.zeros(self.dim, dtype=self.dtype)
        else:
            u = input_values[0].value.astype(self.dtype)

        y = np.zeros(self.dim, dtype=self.dtype)

        for i in range(self.dim):
            if self.n_states > 0:
                self.derivative[i, :] = (self.A @ self.state[i, :]
                                         + (self.B @ u[i].reshape(1)).flatten())
                y[i] = (self.C @ self.state[i, :] + self.D @ u[i].reshape(1)).flatten()[0]
            else:
                y[i] = (self.D @ u[i].reshape(1)).flatten()[0]

        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def get_derivative(self, t: float, input_values: Optional[List[VectorSignal]] = None) -> np.ndarray:
        if not input_values:
            u = np.zeros(self.dim, dtype=self.dtype)
        else:
            u = input_values[0].value.astype(self.dtype)

        derivative = np.zeros_like(self.state)
        for i in range(self.dim):
            if self.n_states > 0:
                derivative[i, :] = (self.A @ self.state[i, :]
                                    + (self.B @ u[i].reshape(1)).flatten())
        return derivative

    def integrate_state(self, dt: float, solver: str = 'euler') -> None:
        if self.n_states == 0:
            return
        if solver == 'euler':
            self.state = self.state + self.derivative * dt


# =========================
# Sink Block
# =========================

class VectorEnd(VectorBlock):
    """
    Sink block to record simulation outputs.

    Terminates a signal path and records the history of values received.

    Attributes:
        history (List[np.ndarray]): Recorded values from each time step

    Example:
        >>> sink = VectorEnd("output")
        >>> source >> sink
    """

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None) -> None:
        _dtype = dtype if dtype is not None else DEFAULT_DTYPE
        super().__init__(name, use_c_backend=use_c_backend, dtype=_dtype)
        self.history: List[np.ndarray] = []

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        if input_values:
            self.output = input_values[0]
            self.history.append(self.output.value.copy())
        return self.output


# =========================
# Module Metadata
# =========================

__all__ = [
    'VectorIntegrator',
    'StateSpaceBlock',
    'TransferFunctionBlock',
    'VectorEnd',
]

__version__ = '1.0.0'
__author__ = 'EmbedSim Framework'