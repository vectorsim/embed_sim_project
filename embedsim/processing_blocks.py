"""
processing_blocks.py
====================

Signal processing and transformation blocks for vector signals.

Part of the EmbedSim framework — lightweight block-diagram simulation
targeting 32-bit embedded platforms.

All numeric arrays default to float32 for MCU compatibility. Pass
dtype=np.float64 to individual blocks when double precision is required.

Classes:
    VectorGain:       Multiplies input by a scalar or matrix gain
    VectorSum:        Sums multiple input signals with optional sign control
    VectorDelay:      Implements a discrete one-step delay (z^-1)
    VectorProduct:    Element-wise multiplication of two vectors
    VectorAbs:        Absolute value of each vector element
    VectorSaturation: Limits vector elements to specified bounds

Author: EmbedSim Framework
Version: 1.0.0
"""

import numpy as np
from typing import List, Optional, Union
from .core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE, validate_inputs_exist


# =========================
# Gain Block
# =========================

class VectorGain(VectorBlock):
    """
    Multiplies the input signal by a gain (scalar or matrix).

    Attributes:
        gain (Union[float, np.ndarray]): Gain value (scalar or matrix; float32 by default)

    Example:
        >>> gain1 = VectorGain("amplifier", gain=2.0)
        >>> gain2 = VectorGain("transform", gain=K)
    """

    def __init__(self, name: str, gain: Union[float, np.ndarray],
                 use_c_backend: bool = False, dtype=None) -> None:
        """
        Initialize a VectorGain block.

        Args:
            name:          Unique identifier for this block
            gain:          Scalar or matrix gain (converted to float32 if array).
            use_c_backend: False = Python (default), True = compiled C.
            dtype:         Override dtype (default: float32).
        """
        _dtype = dtype if dtype is not None else DEFAULT_DTYPE
        super().__init__(name, use_c_backend=use_c_backend, dtype=_dtype)
        if np.isscalar(gain):
            self.gain: Union[float, np.ndarray] = gain
        else:
            self.gain = np.array(gain, dtype=_dtype)

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        validate_inputs_exist(input_values, self.name, min_inputs=1)
        val = input_values[0].value * self.gain
        self.output = VectorSignal(val, self.name, dtype=self.dtype)
        return self.output


# =========================
# Sum Block
# =========================

class VectorSum(VectorBlock):
    """
    Sums multiple input signals with optional sign control.

    Example:
        >>> sum1 = VectorSum("adder")           # all +1
        >>> sum2 = VectorSum("error", signs=[1, -1])  # subtraction
    """

    def __init__(self, name: str, signs: Optional[List[float]] = None,
                 use_c_backend: bool = False, dtype=None) -> None:
        """
        Initialize a VectorSum block.

        Args:
            name:  Unique identifier for this block
            signs: Optional list of signs/weights per input. None means all +1.
            dtype: Override dtype (default: float32).
        """
        _dtype = dtype if dtype is not None else DEFAULT_DTYPE
        super().__init__(name, use_c_backend=use_c_backend, dtype=_dtype)
        self.signs: Optional[List[float]] = signs

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        validate_inputs_exist(input_values, self.name, min_inputs=1)
        if self.signs is None:
            val = sum(inp.value for inp in input_values)
        else:
            val = sum(sign * inp.value for sign, inp in zip(self.signs, input_values))
        self.output = VectorSignal(val, self.name, dtype=self.dtype)
        return self.output


# =========================
# Delay Block
# =========================

class VectorDelay(VectorBlock):
    """
    Implements a discrete one-step delay (z^-1 in z-domain).

    The first output (before any input is stored) is the initial value.

    Attributes:
        last_output (VectorSignal): Stored previous input value (float32 by default)

    Example:
        >>> delay = VectorDelay("delay_1step", initial=[0.0, 0.0, 0.0])
    """

    def __init__(self, name: str, initial: Optional[List[float]] = None,
                 use_c_backend: bool = False, dtype=None) -> None:
        """
        Initialize a VectorDelay block.

        Args:
            name:    Unique identifier for this block
            initial: Initial output value. If None, zeros (same dim as input).
            dtype:   Override dtype (default: float32).
        """
        _dtype = dtype if dtype is not None else DEFAULT_DTYPE
        super().__init__(name, use_c_backend=use_c_backend, dtype=_dtype)
        if initial is not None:
            self.last_output = VectorSignal(np.array(initial, dtype=_dtype))
        else:
            self.last_output = None

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        validate_inputs_exist(input_values, self.name, min_inputs=1)
        if self.last_output is not None:
            val = self.last_output.value.copy()
        else:
            val = np.zeros_like(input_values[0].value)
        self.last_output = VectorSignal(input_values[0].value.copy(), dtype=self.dtype)
        self.output = VectorSignal(val, self.name, dtype=self.dtype)
        return self.output


# =========================
# Product Block
# =========================

class VectorProduct(VectorBlock):
    """
    Element-wise multiplication of two vector inputs.

    Computes: output[i] = input1[i] * input2[i]

    Example:
        >>> product = VectorProduct("power")
        >>> voltage >> product
        >>> current >> product
    """

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None) -> None:
        _dtype = dtype if dtype is not None else DEFAULT_DTYPE
        super().__init__(name, use_c_backend=use_c_backend, dtype=_dtype)

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        validate_inputs_exist(input_values, self.name, min_inputs=2)
        val = input_values[0].value * input_values[1].value
        self.output = VectorSignal(val, self.name, dtype=self.dtype)
        return self.output


# =========================
# Absolute Value Block
# =========================

class VectorAbs(VectorBlock):
    """
    Computes the absolute value of each vector element.

    Computes: output[i] = |input[i]|
    """

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None) -> None:
        _dtype = dtype if dtype is not None else DEFAULT_DTYPE
        super().__init__(name, use_c_backend=use_c_backend, dtype=_dtype)

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        validate_inputs_exist(input_values, self.name, min_inputs=1)
        val = np.abs(input_values[0].value)
        self.output = VectorSignal(val, self.name, dtype=self.dtype)
        return self.output


# =========================
# Saturation Block
# =========================

class VectorSaturation(VectorBlock):
    """
    Limits each vector element to specified lower and upper bounds.

    Computes: output[i] = clip(input[i], lower, upper)

    Useful for actuator limits and anti-windup protection.

    Attributes:
        lower (float): Lower saturation limit
        upper (float): Upper saturation limit

    Example:
        >>> sat = VectorSaturation("limiter", lower=-10.0, upper=10.0)
    """

    def __init__(self, name: str, lower: float = -1.0, upper: float = 1.0,
                 use_c_backend: bool = False, dtype=None) -> None:
        """
        Initialize a VectorSaturation block.

        Args:
            name:  Unique identifier for this block
            lower: Lower saturation limit. Default: -1.0
            upper: Upper saturation limit. Default: 1.0
            dtype: Override dtype (default: float32).

        Raises:
            ValueError: If lower >= upper
        """
        _dtype = dtype if dtype is not None else DEFAULT_DTYPE
        super().__init__(name, use_c_backend=use_c_backend, dtype=_dtype)
        if lower >= upper:
            raise ValueError(f"{name}: lower limit must be < upper limit")
        self.lower: float = lower
        self.upper: float = upper

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        validate_inputs_exist(input_values, self.name, min_inputs=1)
        val = np.clip(input_values[0].value, self.lower, self.upper)
        self.output = VectorSignal(val, self.name, dtype=self.dtype)
        return self.output


# =========================
# Module Metadata
# =========================

__all__ = [
    'VectorGain',
    'VectorSum',
    'VectorDelay',
    'VectorProduct',
    'VectorAbs',
    'VectorSaturation',
]

__version__ = '1.0.0'
__author__ = 'EmbedSim Framework'