"""
core_blocks.py
==============

Core base classes and fundamental structures for the vector block simulation framework.

This module provides the foundation for building block-based simulations with vector signals.
It includes the base classes that all other blocks inherit from.

Classes:
    VectorSignal: Represents a vector-valued signal in the simulation
    VectorBlock: Abstract base class for all simulation blocks
    
Author: Vector Simulation Framework
Version: 1.0.0
"""

import numpy as np
from typing import List, Optional, Union


# =========================
# Signal Representation
# =========================

class VectorSignal:
    """
    Represents a vector-valued signal in the simulation.
    
    A VectorSignal encapsulates a multi-dimensional signal value that flows between
    blocks in the simulation. Each signal has a value (numpy array) and an optional
    name for identification in plotting and logging.
    
    Attributes:
        value (np.ndarray): The current vector value as a numpy array
        name (str): Optional identifier for the signal (used in plots/logs)
    
    Example:
        >>> sig = VectorSignal([1.0, 2.0, 3.0], name="three_phase_current")
        >>> print(sig.value)
        [1. 2. 3.]
    """
    
    def __init__(self, value: Union[List[float], np.ndarray], name: str = "") -> None:
        """
        Initialize a VectorSignal.
        
        Args:
            value: Initial vector value, can be a Python list or numpy array.
                   Will be converted to float64 numpy array internally.
            name: Optional name for the signal. Used for plotting and debugging.
                  Default is empty string.
        
        Raises:
            ValueError: If value cannot be converted to a numeric array
        
        Example:
            >>> sig1 = VectorSignal([1.0, 2.0, 3.0], name="current")
            >>> sig2 = VectorSignal(np.array([4.0, 5.0, 6.0]))
        """
        self.value: np.ndarray = np.array(value, dtype=float)
        self.name: str = name
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        name_str = f"'{self.name}'" if self.name else "unnamed"
        return f"VectorSignal({name_str}, shape={self.value.shape}, value={self.value})"


# =========================
# Base Block Class
# =========================

class VectorBlock:
    """
    Abstract base class for all vector-based simulation blocks.
    
    VectorBlock provides the fundamental interface and functionality that all
    simulation blocks must implement. It handles:
    - Connection management between blocks
    - Output signal storage
    - State management for dynamic blocks
    - Integration interface for continuous dynamics
    
    Blocks can be connected using the >> operator to create signal flow paths.
    
    Attributes:
        name (str): Unique identifier for this block
        inputs (List[VectorBlock]): List of upstream blocks connected to this block
        output (Optional[VectorSignal]): Current output signal (None until computed)
        last_output (Optional[VectorSignal]): Previous output (for delays/history)
        is_dynamic (bool): True if block has internal states requiring integration
    
    Example:
        >>> source = SomeSourceBlock("source")
        >>> processor = SomeProcessorBlock("processor")
        >>> sink = SomeSinkBlock("sink")
        >>> source >> processor >> sink  # Connect blocks
    """
    
    def __init__(self, name: str = "") -> None:
        """
        Initialize a VectorBlock.
        
        Args:
            name: Unique identifier for this block. Used in hierarchy display,
                  error messages, and code generation. Should be descriptive.
        
        Example:
            >>> block = VectorBlock("my_block")
        """
        self.name: str = name
        self.inputs: List[VectorBlock] = []  # Upstream blocks feeding this block
        self.output: Optional[VectorSignal] = None  # Current computed output
        self.last_output: Optional[VectorSignal] = None  # Previous output value
        self.is_dynamic: bool = False  # True for blocks with states (integrators, etc.)

    def __rshift__(self, other: "VectorBlock") -> "VectorBlock":
        """
        Connect this block's output to another block's input using >> operator.
        
        This operator creates a signal flow connection from this block to the
        other block. Multiple blocks can feed into a single block (for sum operations),
        but this creates a one-to-one connection.
        
        Args:
            other: The downstream block to connect to
        
        Returns:
            The downstream block (allows chaining: a >> b >> c)
        
        Example:
            >>> source = ThreePhaseGenerator("gen", amplitude=5, freq=50)
            >>> gain = VectorGain("gain", 2.0)
            >>> sink = VectorEnd("sink")
            >>> source >> gain >> sink  # Chain connections
        """
        other.inputs.append(self)
        return other

    def reset(self) -> None:
        """
        Reset the block's output signals.
        
        Called before starting a new simulation run. Clears output and last_output
        to ensure clean state. Subclasses with internal states should override
        this method to also reset their state variables.
        
        Example:
            >>> block.reset()  # Prepare for new simulation
        """
        self.output = None
        self.last_output = None

    def compute(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Compute this block's output based on current time and inputs.
        
        This is the core method that must be implemented by all subclasses.
        It defines the block's transfer function or behavior.
        
        Args:
            t: Current simulation time (seconds)
            dt: Time step duration (seconds)
            input_values: List of input signals from connected upstream blocks.
                         None or empty list if this is a source block.
        
        Returns:
            VectorSignal: The computed output signal for this time step
        
        Raises:
            NotImplementedError: If subclass doesn't implement this method
        
        Note:
            - The output should be stored in self.output before returning
            - For source blocks, input_values is typically None or ignored
            - For processing blocks, input_values contains signals from self.inputs
        
        Example:
            >>> class MyBlock(VectorBlock):
            ...     def compute(self, t, dt, input_values):
            ...         result = some_computation(input_values)
            ...         self.output = VectorSignal(result, self.name)
            ...         return self.output
        """
        raise NotImplementedError(f"compute() must be implemented by {self.__class__.__name__}")

    def get_derivative(self, t: float, input_values: Optional[List[VectorSignal]] = None) -> Optional[np.ndarray]:
        """
        Get the derivative of this block's internal state (for continuous dynamics).
        
        For blocks with continuous internal states (integrators, state-space models),
        this method returns dx/dt. Used by ODE solvers (Euler, RK4) to integrate
        the state forward in time.
        
        Args:
            t: Current simulation time (seconds)
            input_values: List of input signals from connected upstream blocks
        
        Returns:
            np.ndarray or None: Derivative of the internal state vector.
                               None if the block has no continuous state.
        
        Note:
            - Only relevant for blocks with is_dynamic = True
            - Default implementation returns None (no continuous state)
            - Override in dynamic blocks (integrators, state-space models)
        
        Example:
            >>> class Integrator(VectorBlock):
            ...     def get_derivative(self, t, input_values):
            ...         return input_values[0].value  # dx/dt = u
        """
        return None

    def integrate_state(self, dt: float, solver: str = 'euler') -> None:
        """
        Integrate the internal state forward in time using the specified ODE solver.
        
        For continuous blocks, this advances the state by one time step using
        numerical integration. The derivative is computed by get_derivative().
        
        Args:
            dt: Time step duration (seconds)
            solver: Integration method to use:
                   - 'euler': First-order Euler method (fast, less accurate)
                   - 'rk4': Fourth-order Runge-Kutta (slower, more accurate)
                   - 'heun': Second-order Heun's method (compromise)
        
        Note:
            - Only called for blocks with is_dynamic = True
            - Default implementation does nothing (no state to integrate)
            - Override in dynamic blocks to implement integration logic
            - For RK4, integration may be handled externally by the simulation engine
        
        Example:
            >>> class SimpleIntegrator(VectorBlock):
            ...     def integrate_state(self, dt, solver='euler'):
            ...         if solver == 'euler':
            ...             self.state = self.state + self.derivative * dt
        """
        pass

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}('{self.name}')"


# =========================
# Utility Functions
# =========================

def validate_vector_dimension(signal: VectorSignal, expected_dim: int, block_name: str) -> None:
    """
    Validate that a signal has the expected vector dimension.
    
    Args:
        signal: The signal to validate
        expected_dim: Expected number of elements in the vector
        block_name: Name of the block performing validation (for error messages)
    
    Raises:
        ValueError: If signal dimension doesn't match expected_dim
    
    Example:
        >>> sig = VectorSignal([1.0, 2.0, 3.0])
        >>> validate_vector_dimension(sig, 3, "my_block")  # OK
        >>> validate_vector_dimension(sig, 2, "my_block")  # Raises ValueError
    """
    if len(signal.value) != expected_dim:
        raise ValueError(
            f"{block_name}: Expected signal dimension {expected_dim}, "
            f"but got {len(signal.value)}"
        )


def validate_inputs_exist(input_values: Optional[List[VectorSignal]], 
                         block_name: str,
                         min_inputs: int = 1) -> None:
    """
    Validate that a block has received the minimum required number of inputs.
    
    Args:
        input_values: List of input signals (may be None or empty)
        block_name: Name of the block performing validation (for error messages)
        min_inputs: Minimum number of required inputs (default: 1)
    
    Raises:
        ValueError: If insufficient inputs are provided
    
    Example:
        >>> validate_inputs_exist(None, "my_block")  # Raises ValueError
        >>> validate_inputs_exist([sig1], "my_block")  # OK
        >>> validate_inputs_exist([sig1], "my_block", min_inputs=2)  # Raises ValueError
    """
    if not input_values or len(input_values) < min_inputs:
        actual = len(input_values) if input_values else 0
        raise ValueError(
            f"{block_name}: Expected at least {min_inputs} input(s), "
            f"but got {actual}"
        )




# =========================
# Module Metadata
# =========================

__all__ = [
    'VectorSignal',
    'VectorBlock',
    'validate_vector_dimension',
    'validate_inputs_exist',
]

__version__ = '1.0.0'
__author__ = 'Vector Simulation Framework'
