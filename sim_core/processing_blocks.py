"""
processing_blocks.py
====================

Signal processing and transformation blocks for vector signals.

This module provides blocks that process, transform, or combine vector signals.
These blocks operate on their inputs to produce modified outputs.

Classes:
    VectorGain: Multiplies input by a scalar or matrix gain
    VectorSum: Sums multiple input signals with optional sign control
    VectorDelay: Implements a discrete one-step delay (z^-1)
    VectorProduct: Element-wise multiplication of two vectors
    VectorAbs: Absolute value of each vector element
    VectorSaturation: Limits vector elements to specified bounds
    
Author: Vector Simulation Framework  
Version: 1.0.0
"""

import numpy as np
from typing import List, Optional, Union
import ast
import textwrap
from pathlib import Path
from .core_blocks import VectorBlock, VectorSignal, validate_inputs_exist


# =========================
# Gain Block
# =========================

class VectorGain(VectorBlock):
    """
    Multiplies the input signal by a gain (scalar or matrix).
    
    This block implements:
        - Scalar gain: output = gain * input (all elements scaled equally)
        - Matrix gain: output = gain @ input (linear transformation)
    
    Attributes:
        gain (Union[float, np.ndarray]): Gain value (scalar or matrix)
    
    Example:
        >>> # Scalar gain: multiply by 2.0
        >>> gain1 = VectorGain("amplifier", gain=2.0)
        >>> 
        >>> # Matrix gain: coordinate transformation
        >>> K = np.array([[1, 2, 0], [0, 1, 1], [1, 0, 1]])
        >>> gain2 = VectorGain("transform", gain=K)
    """
    
    def __init__(self, name: str, gain: Union[float, np.ndarray]) -> None:
        """
        Initialize a VectorGain block.
        
        Args:
            name: Unique identifier for this block
            gain: Gain value. Can be:
                 - Scalar (float): Multiplies all elements by same value
                 - Matrix (np.ndarray): Matrix multiplication for transformation
        
        Example:
            >>> # Simple scalar gain
            >>> g1 = VectorGain("gain_2x", gain=2.0)
            >>> 
            >>> # Diagonal scaling matrix
            >>> g2 = VectorGain("scale_xyz", gain=np.diag([1.0, 2.0, 3.0]))
            >>> 
            >>> # Full transformation matrix (e.g., Clarke transform)
            >>> clarke = (2/3) * np.array([
            ...     [1, -0.5, -0.5],
            ...     [0, np.sqrt(3)/2, -np.sqrt(3)/2],
            ...     [0.5, 0.5, 0.5]
            ... ])
            >>> g3 = VectorGain("clarke_transform", gain=clarke)
        """
        super().__init__(name)
        if np.isscalar(gain):
            self.gain: Union[float, np.ndarray] = gain
        else:
            self.gain = np.array(gain, dtype=float)

    def compute(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Multiply input signal by gain.
        
        Args:
            t: Current simulation time (unused)
            dt: Time step (unused)
            input_values: List containing the input signal
        
        Returns:
            VectorSignal: Gained output signal
        
        Raises:
            ValueError: If no input is provided
        
        Example:
            >>> gain = VectorGain("g", gain=2.0)
            >>> input_sig = VectorSignal([1.0, 2.0, 3.0])
            >>> output = gain.compute(t=0, dt=0.01, input_values=[input_sig])
            >>> print(output.value)  # [2. 4. 6.]
        """
        validate_inputs_exist(input_values, self.name, min_inputs=1)
        
        val = input_values[0].value * self.gain
        self.output = VectorSignal(val, self.name)
        return self.output


# =========================
# Sum Block
# =========================

class VectorSum(VectorBlock):
    """
    Sums multiple input signals with optional sign control.
    
    This block can perform:
        - Simple addition: sum all inputs
        - Subtraction: use signs=[1, -1] for input1 - input2
        - Weighted sum: signs=[a, b, c] for a*input1 + b*input2 + c*input3
    
    Attributes:
        signs (Optional[List[float]]): Sign/weight for each input. None means all +1.
    
    Example:
        >>> # Add two signals
        >>> sum1 = VectorSum("adder", signs=None)  # or signs=[1, 1]
        >>> signal1 >> sum1
        >>> signal2 >> sum1
        >>> 
        >>> # Subtract: signal1 - signal2
        >>> sum2 = VectorSum("subtractor", signs=[1, -1])
        >>> signal1 >> sum2
        >>> signal2 >> sum2
    """
    
    def __init__(self, name: str, signs: Optional[List[float]] = None) -> None:
        """
        Initialize a VectorSum block.
        
        Args:
            name: Unique identifier for this block
            signs: Optional list of signs/weights for each input.
                  - None: All inputs are added (equivalent to all +1)
                  - [1, -1]: First input + second input × (-1) = subtraction
                  - [a, b, c]: Weighted sum a*in1 + b*in2 + c*in3
        
        Note:
            The length of signs should match the number of connected inputs.
            If signs is shorter, remaining inputs use default +1.
            If signs is longer, extra values are ignored.
        
        Example:
            >>> # Simple addition
            >>> adder = VectorSum("add")
            >>> 
            >>> # Subtraction  
            >>> subtractor = VectorSum("subtract", signs=[1, -1])
            >>> 
            >>> # Weighted combination
            >>> mixer = VectorSum("mix", signs=[0.5, 0.3, 0.2])
        """
        super().__init__(name)
        self.signs: Optional[List[float]] = signs

    def compute(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Sum the input signals with optional sign/weight control.
        
        Args:
            t: Current simulation time (unused)
            dt: Time step (unused)
            input_values: List of input signals to sum
        
        Returns:
            VectorSignal: Sum of inputs with signs applied
        
        Raises:
            ValueError: If no inputs are provided
        
        Example:
            >>> sum_block = VectorSum("s", signs=[1, -1])
            >>> in1 = VectorSignal([5.0, 6.0, 7.0])
            >>> in2 = VectorSignal([1.0, 2.0, 3.0])
            >>> output = sum_block.compute(t=0, dt=0.01, input_values=[in1, in2])
            >>> print(output.value)  # [4. 4. 4.] = [5-1, 6-2, 7-3]
        """
        validate_inputs_exist(input_values, self.name, min_inputs=1)
        
        if self.signs is None:
            # Simple addition of all inputs
            val = sum(inp.value for inp in input_values)
        else:
            # Weighted sum with signs
            val = sum(sign * inp.value for sign, inp in zip(self.signs, input_values))
        
        self.output = VectorSignal(val, self.name)
        return self.output


# =========================
# Delay Block
# =========================

class VectorDelay(VectorBlock):
    """
    Implements a discrete one-step delay (z^-1 in z-domain).
    
    This block outputs the previous time step's input value. Useful for:
        - Implementing discrete-time systems
        - Breaking algebraic loops
        - Creating history buffers
    
    The first output (before any input is stored) is the initial value.
    
    Attributes:
        last_output (VectorSignal): Stored previous input value
    
    Example:
        >>> # Delay with zero initial condition
        >>> delay1 = VectorDelay("delay_1step", initial=[0.0, 0.0, 0.0])
        >>> 
        >>> # Delay with non-zero initial condition
        >>> delay2 = VectorDelay("delay_ic", initial=[1.0, 2.0, 3.0])
    """
    
    def __init__(self, name: str, initial: Optional[List[float]] = None) -> None:
        """
        Initialize a VectorDelay block.
        
        Args:
            name: Unique identifier for this block
            initial: Initial output value (before first input arrives).
                    If None, first output will be zeros with same dimension as input.
        
        Example:
            >>> # Zero initial condition (dimension inferred from input)
            >>> delay1 = VectorDelay("d1", initial=None)
            >>> 
            >>> # Specific initial condition
            >>> delay2 = VectorDelay("d2", initial=[1.0, 0.0, -1.0])
        """
        super().__init__(name)
        if initial is not None:
            self.last_output = VectorSignal(np.array(initial, dtype=float))
        else:
            self.last_output = None

    def compute(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Output the previous time step's input, then store current input.
        
        Args:
            t: Current simulation time (unused)
            dt: Time step (unused)
            input_values: List containing the current input signal
        
        Returns:
            VectorSignal: Previous input value (or initial value on first call)
        
        Raises:
            ValueError: If no input is provided
        
        Example:
            >>> delay = VectorDelay("d", initial=[0.0, 0.0])
            >>> in1 = VectorSignal([1.0, 2.0])
            >>> out1 = delay.compute(t=0, dt=0.01, input_values=[in1])
            >>> print(out1.value)  # [0. 0.] (initial value)
            >>> in2 = VectorSignal([3.0, 4.0])
            >>> out2 = delay.compute(t=0.01, dt=0.01, input_values=[in2])
            >>> print(out2.value)  # [1. 2.] (previous input)
        """
        validate_inputs_exist(input_values, self.name, min_inputs=1)
        
        # Output is previous input (or zeros if first call)
        if self.last_output is not None:
            val = self.last_output.value.copy()
        else:
            val = np.zeros_like(input_values[0].value)
        
        # Store current input for next time step
        self.last_output = VectorSignal(input_values[0].value.copy())
        
        self.output = VectorSignal(val, self.name)
        return self.output


# =========================
# Product Block
# =========================

class VectorProduct(VectorBlock):
    """
    Element-wise multiplication of two vector inputs.
    
    Computes: output[i] = input1[i] * input2[i] for all i
    
    Useful for:
        - Modulation
        - Power calculations (voltage × current)
        - Nonlinear systems
    
    Example:
        >>> # Multiply two signals element-wise
        >>> product = VectorProduct("multiplier")
        >>> voltage >> product
        >>> current >> product
        >>> product >> power  # power = V * I
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize a VectorProduct block.
        
        Args:
            name: Unique identifier for this block
        
        Example:
            >>> mult = VectorProduct("power_calc")
        """
        super().__init__(name)

    def compute(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Compute element-wise product of two inputs.
        
        Args:
            t: Current simulation time (unused)
            dt: Time step (unused)
            input_values: List containing exactly two input signals
        
        Returns:
            VectorSignal: Element-wise product
        
        Raises:
            ValueError: If fewer than 2 inputs provided
        
        Example:
            >>> prod = VectorProduct("p")
            >>> in1 = VectorSignal([2.0, 3.0, 4.0])
            >>> in2 = VectorSignal([5.0, 6.0, 7.0])
            >>> output = prod.compute(t=0, dt=0.01, input_values=[in1, in2])
            >>> print(output.value)  # [10. 18. 28.]
        """
        validate_inputs_exist(input_values, self.name, min_inputs=2)
        
        val = input_values[0].value * input_values[1].value
        self.output = VectorSignal(val, self.name)
        return self.output


# =========================
# Absolute Value Block
# =========================

class VectorAbs(VectorBlock):
    """
    Computes the absolute value of each vector element.
    
    Computes: output[i] = |input[i]| for all i
    
    Useful for:
        - Magnitude calculations
        - Rectification
        - Envelope detection
    
    Example:
        >>> # Compute magnitude of AC signal
        >>> abs_block = VectorAbs("magnitude")
        >>> ac_signal >> abs_block
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize a VectorAbs block.
        
        Args:
            name: Unique identifier for this block
        """
        super().__init__(name)

    def compute(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Compute absolute value of input.
        
        Args:
            t: Current simulation time (unused)
            dt: Time step (unused)
            input_values: List containing the input signal
        
        Returns:
            VectorSignal: Absolute value of input
        
        Raises:
            ValueError: If no input is provided
        """
        validate_inputs_exist(input_values, self.name, min_inputs=1)
        
        val = np.abs(input_values[0].value)
        self.output = VectorSignal(val, self.name)
        return self.output


# =========================
# Saturation Block
# =========================

class VectorSaturation(VectorBlock):
    """
    Limits each vector element to specified lower and upper bounds.
    
    Computes: output[i] = clip(input[i], lower, upper)
    
    Useful for:
        - Actuator limits
        - Anti-windup protection
        - Physical constraints
    
    Attributes:
        lower (float): Lower saturation limit
        upper (float): Upper saturation limit
    
    Example:
        >>> # Limit signal between -10 and +10
        >>> sat = VectorSaturation("limiter", lower=-10.0, upper=10.0)
        >>> signal >> sat
    """
    
    def __init__(self, name: str, lower: float = -1.0, upper: float = 1.0) -> None:
        """
        Initialize a VectorSaturation block.
        
        Args:
            name: Unique identifier for this block
            lower: Lower saturation limit. Default: -1.0
            upper: Upper saturation limit. Default: 1.0
        
        Raises:
            ValueError: If lower >= upper
        
        Example:
            >>> sat1 = VectorSaturation("volt_limit", lower=0.0, upper=100.0)
            >>> sat2 = VectorSaturation("bipolar", lower=-5.0, upper=5.0)
        """
        super().__init__(name)
        if lower >= upper:
            raise ValueError(f"{name}: lower limit must be < upper limit")
        self.lower: float = lower
        self.upper: float = upper

    def compute(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Apply saturation limits to input.
        
        Args:
            t: Current simulation time (unused)
            dt: Time step (unused)
            input_values: List containing the input signal
        
        Returns:
            VectorSignal: Saturated output
        
        Raises:
            ValueError: If no input is provided
        """
        validate_inputs_exist(input_values, self.name, min_inputs=1)
        
        val = np.clip(input_values[0].value, self.lower, self.upper)
        self.output = VectorSignal(val, self.name)
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
__author__ = 'Vector Simulation Framework'
