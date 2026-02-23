"""
dynamic_blocks.py
=================

Dynamic system blocks with continuous and discrete state evolution.

This module provides blocks that have internal states and implement differential
or difference equations. These blocks are the core of continuous and discrete
dynamical system simulation.

Classes:
    VectorIntegrator: Continuous integrator (ẋ = u)
    StateSpaceBlock: General state-space model (ẋ = Ax + Bu, y = Cx + Du)
    TransferFunctionBlock: Transfer function using state-space representation
    VectorEnd: Sink block for recording simulation outputs
    
Author: Vector Simulation Framework
Version: 1.0.0
"""

import numpy as np
from typing import List, Optional
from .core_blocks import VectorBlock, VectorSignal, validate_inputs_exist


# =========================
# Integrator Block
# =========================

class VectorIntegrator(VectorBlock):
    """
    Continuous integrator block implementing ẋ = u.
    
    This block performs time integration of its input signal. The output is the
    accumulated (integrated) value of the input over time. Mathematically:
    
        dx/dt = u(t)
        x(t) = x(0) + ∫u(τ)dτ from 0 to t
    
    Attributes:
        state (np.ndarray): Current integrator state (output value)
        derivative (np.ndarray): Current rate of change (input value)
        is_dynamic (bool): Always True for integrators
    
    Example:
        >>> # Integrate a constant to get a ramp
        >>> const = VectorConstant("input", [1.0, 2.0, 3.0])
        >>> integrator = VectorIntegrator("int", initial_state=[0, 0, 0])
        >>> const >> integrator
        >>> # Output will be [t, 2t, 3t]
    """
    
    def __init__(self, name: str, initial_state: Optional[List[float]] = None, 
                 dim: int = 3) -> None:
        """
        Initialize a VectorIntegrator block.
        
        Args:
            name: Unique identifier for this block
            initial_state: Initial value of the integrator state x(0).
                          If None, defaults to zeros with dimension dim.
            dim: Dimension of the state vector. Default: 3
                Only used if initial_state is None.
        
        Example:
            >>> # Zero initial condition
            >>> int1 = VectorIntegrator("integrator1", initial_state=None, dim=3)
            >>> 
            >>> # Non-zero initial condition
            >>> int2 = VectorIntegrator("integrator2", initial_state=[1.0, 2.0, 3.0])
        """
        super().__init__(name)
        self.is_dynamic = True  # Mark as having internal state
        
        if initial_state is not None:
            self.state: np.ndarray = np.array(initial_state, dtype=float)
        else:
            self.state = np.zeros(dim, dtype=float)
        
        self.derivative: np.ndarray = np.zeros_like(self.state)

    def compute(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Compute integrator output and store derivative.
        
        The output is the current state (integrated value).
        The derivative (for integration) is the input value.
        
        Args:
            t: Current simulation time (unused)
            dt: Time step (unused in compute, used in integrate_state)
            input_values: List containing the input signal to integrate
        
        Returns:
            VectorSignal: Current integrator state (integrated value)
        
        Raises:
            ValueError: If no input is provided
        
        Example:
            >>> integrator = VectorIntegrator("int", initial_state=[0, 0])
            >>> input_sig = VectorSignal([5.0, 10.0])
            >>> output = integrator.compute(t=0, dt=0.01, input_values=[input_sig])
            >>> print(output.value)  # [0. 0.] (current state before integration)
            >>> integrator.integrate_state(dt=0.01)
            >>> output = integrator.compute(t=0.01, dt=0.01, input_values=[input_sig])
            >>> print(output.value)  # [0.05 0.1] (state after one step)
        """
        validate_inputs_exist(input_values, self.name, min_inputs=1)
        
        # Store derivative (which is the input)
        self.derivative = input_values[0].value.copy()
        
        # Output is current state
        self.output = VectorSignal(self.state.copy(), self.name)
        return self.output

    def get_derivative(self, t: float, input_values: Optional[List[VectorSignal]] = None) -> np.ndarray:
        """
        Get the derivative of the integrator state.
        
        For an integrator, the derivative is simply the input: dx/dt = u
        
        Args:
            t: Current simulation time (unused)
            input_values: List containing the input signal
        
        Returns:
            np.ndarray: Derivative (equals input value)
        
        Example:
            >>> integrator = VectorIntegrator("int", initial_state=[0, 0])
            >>> input_sig = VectorSignal([3.0, 4.0])
            >>> deriv = integrator.get_derivative(t=0, input_values=[input_sig])
            >>> print(deriv)  # [3. 4.]
        """
        if not input_values:
            return np.zeros_like(self.state)
        return input_values[0].value

    def integrate_state(self, dt: float, solver: str = 'euler') -> None:
        """
        Integrate the state forward in time using the specified solver.
        
        Args:
            dt: Time step for integration
            solver: Integration method:
                   - 'euler': x(t+dt) = x(t) + dx/dt * dt
                   - 'rk4': Fourth-order Runge-Kutta (handled externally)
        
        Note:
            For RK4 integration, the simulation engine handles the multi-stage
            integration externally, so this method may not be called.
        
        Example:
            >>> integrator = VectorIntegrator("int", initial_state=[0.0])
            >>> integrator.derivative = np.array([10.0])
            >>> integrator.integrate_state(dt=0.01, solver='euler')
            >>> print(integrator.state)  # [0.1]
        """
        if solver == 'euler':
            self.state = self.state + self.derivative * dt
        elif solver == 'rk4':
            # RK4 integration handled externally in simulation loop
            pass


# =========================
# State-Space Block
# =========================

class StateSpaceBlock(VectorBlock):
    """
    Continuous state-space model.
    
    Implements the standard continuous-time state-space representation:
    
        ẋ(t) = A·x(t) + B·u(t)  (state equation)
        y(t) = C·x(t) + D·u(t)  (output equation)
    
    where:
        x: State vector (n × 1)
        u: Input vector (m × 1)
        y: Output vector (p × 1)
        A: State matrix (n × n) - system dynamics
        B: Input matrix (n × m) - input influence
        C: Output matrix (p × n) - state to output mapping
        D: Feedthrough matrix (p × m) - direct input to output
    
    This is a general representation that can model many physical systems
    including mechanical, electrical, thermal, and control systems.
    
    Attributes:
        A, B, C, D (np.ndarray): State-space matrices
        state (np.ndarray): Current state vector x
        derivative (np.ndarray): Current state derivative ẋ
        n_states (int): Number of states (dimension of x)
        is_dynamic (bool): Always True for state-space models
    
    Example:
        >>> # RL circuit: L(di/dt) = v - Ri
        >>> # State: i (current), Input: v (voltage), Output: i
        >>> R, L = 10.0, 0.1  # Resistance and inductance
        >>> A = np.array([[-R/L]])
        >>> B = np.array([[1/L]])
        >>> C = np.array([[1.0]])
        >>> D = np.array([[0.0]])
        >>> rl_circuit = StateSpaceBlock("RL", A, B, C, D, initial_state=[0.0])
    """
    
    def __init__(self, name: str, A: np.ndarray, B: np.ndarray, 
                 C: np.ndarray, D: np.ndarray, 
                 initial_state: Optional[List[float]] = None) -> None:
        """
        Initialize a StateSpaceBlock.
        
        Args:
            name: Unique identifier for this block
            A: State matrix (n × n) - defines system dynamics
            B: Input matrix (n × m) - defines input influence on state
            C: Output matrix (p × n) - maps state to output
            D: Feedthrough matrix (p × m) - direct input to output
            initial_state: Initial state vector x(0). If None, defaults to zeros.
        
        Note:
            Matrix dimensions must be compatible:
            - A is square (n × n)
            - B has n rows (n × m)
            - C has n columns (p × n)
            - D matches (p × m)
        
        Example:
            >>> # Mass-spring-damper: mẍ + cẋ + kx = f
            >>> # States: [x, ẋ], Input: f, Output: x
            >>> m, c, k = 1.0, 0.5, 10.0
            >>> A = np.array([[0, 1], [-k/m, -c/m]])
            >>> B = np.array([[0], [1/m]])
            >>> C = np.array([[1, 0]])  # Output is position
            >>> D = np.array([[0]])
            >>> system = StateSpaceBlock("mass_spring", A, B, C, D, 
            ...                          initial_state=[0.0, 0.0])
        """
        super().__init__(name)
        self.is_dynamic = True
        
        # Store matrices
        self.A: np.ndarray = np.array(A, dtype=float)
        self.B: np.ndarray = np.array(B, dtype=float)
        self.C: np.ndarray = np.array(C, dtype=float)
        self.D: np.ndarray = np.array(D, dtype=float)
        
        # Get state dimension from A matrix
        self.n_states: int = self.A.shape[0]
        
        # Initialize state
        if initial_state is not None:
            self.state: np.ndarray = np.array(initial_state, dtype=float)
        else:
            self.state = np.zeros(self.n_states, dtype=float)
        
        self.derivative: np.ndarray = np.zeros_like(self.state)
        
        # For RK4 integration (k1, k2, k3, k4 stages)
        self.k1: Optional[np.ndarray] = None
        self.k2: Optional[np.ndarray] = None
        self.k3: Optional[np.ndarray] = None
        self.k4: Optional[np.ndarray] = None

    def compute(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Compute state-space output and derivative.
        
        Calculates:
            ẋ = A·x + B·u
            y = C·x + D·u
        
        Args:
            t: Current simulation time (unused)
            dt: Time step (unused in compute)
            input_values: List containing input signal u. If None, u = 0.
        
        Returns:
            VectorSignal: Output y = C·x + D·u
        
        Example:
            >>> A = np.array([[-1]])
            >>> B = np.array([[1]])
            >>> C = np.array([[1]])
            >>> D = np.array([[0]])
            >>> block = StateSpaceBlock("sys", A, B, C, D, initial_state=[0.0])
            >>> input_sig = VectorSignal([5.0])
            >>> output = block.compute(t=0, dt=0.01, input_values=[input_sig])
        """
        # Get input vector u
        if not input_values:
            u = np.zeros(self.B.shape[1])
        else:
            u = input_values[0].value
        
        # Calculate derivative: ẋ = Ax + Bu
        self.derivative = self.A @ self.state + self.B @ u
        
        # Calculate output: y = Cx + Du
        y = self.C @ self.state + self.D @ u
        
        self.output = VectorSignal(y, self.name)
        return self.output

    def get_derivative(self, t: float, input_values: Optional[List[VectorSignal]] = None) -> np.ndarray:
        """
        Get state derivative for ODE integration.
        
        Returns ẋ = A·x + B·u
        
        Args:
            t: Current simulation time (unused)
            input_values: List containing input signal u
        
        Returns:
            np.ndarray: State derivative ẋ
        """
        if not input_values:
            u = np.zeros(self.B.shape[1])
        else:
            u = input_values[0].value
        return self.A @ self.state + self.B @ u

    def integrate_state(self, dt: float, solver: str = 'euler') -> None:
        """
        Integrate the state forward in time.
        
        Args:
            dt: Time step for integration
            solver: Integration method ('euler' or 'rk4')
        
        Note:
            For RK4, the multi-stage integration is handled externally.
        """
        if solver == 'euler':
            self.state = self.state + self.derivative * dt
        elif solver == 'rk4':
            # RK4 integration handled externally
            pass


# =========================
# Transfer Function Block
# =========================

class TransferFunctionBlock(VectorBlock):
    """
    Continuous transfer function using state-space representation.
    
    Implements a transfer function H(s) = num(s)/den(s) by converting it to
    state-space form (controllable canonical form). Each vector element is
    processed independently through the same transfer function.
    
    For example, a first-order low-pass filter:
        H(s) = ωc / (s + ωc)
        num = [ωc], den = [1, ωc]
    
    Attributes:
        num (np.ndarray): Numerator coefficients (descending powers of s)
        den (np.ndarray): Denominator coefficients (descending powers of s)
        A, B, C, D (np.ndarray): Internal state-space matrices
        state (np.ndarray): State for each vector dimension
        dim (int): Number of vector elements
        n_states (int): Order of the transfer function
    
    Example:
        >>> # First-order low-pass filter: H(s) = 100/(s + 100)
        >>> num = [100]
        >>> den = [1, 100]
        >>> lpf = TransferFunctionBlock("lpf", num, den, dim=3)
        >>> 
        >>> # Second-order system: H(s) = 1/(s² + 2s + 1)
        >>> num2 = [1]
        >>> den2 = [1, 2, 1]
        >>> system = TransferFunctionBlock("2nd_order", num2, den2, dim=2)
    """
    
    def __init__(self, name: str, num: List[float], den: List[float], 
                 dim: int = 3, initial_state: Optional[np.ndarray] = None) -> None:
        """
        Initialize a TransferFunctionBlock.
        
        Args:
            name: Unique identifier for this block
            num: Numerator polynomial coefficients [n0, n1, n2, ...] for
                 H(s) = (n0·s^k + n1·s^(k-1) + ... + nk) / (d0·s^n + d1·s^(n-1) + ... + dn)
            den: Denominator polynomial coefficients [d0, d1, d2, ...]
            dim: Number of independent signals (each filtered identically). Default: 3
            initial_state: Initial internal states. Shape (dim, n_states) where
                          n_states = len(den) - 1. If None, defaults to zeros.
        
        Note:
            - Coefficients are in descending order (highest power first)
            - The denominator's first coefficient is used for normalization
            - System order = len(den) - 1
        
        Example:
            >>> # Low-pass filter: cutoff = 50 Hz
            >>> fc = 50
            >>> wc = 2 * np.pi * fc
            >>> num = [wc]
            >>> den = [1, wc]
            >>> lpf = TransferFunctionBlock("50Hz_lpf", num, den, dim=3)
        """
        super().__init__(name)
        self.is_dynamic = True
        self.dim: int = dim
        
        # Store and normalize coefficients
        self.num: np.ndarray = np.array(num, dtype=float)
        self.den: np.ndarray = np.array(den, dtype=float)
        
        # Normalize by leading denominator coefficient
        a0 = self.den[0]
        self.num = self.num / a0
        self.den = self.den / a0
        
        n = len(self.den) - 1  # System order
        
        # Convert to state-space (controllable canonical form)
        if n > 0:
            # A matrix (companion form)
            A = np.zeros((n, n))
            if n > 1:
                A[:-1, 1:] = np.eye(n-1)
            A[-1, :] = -self.den[1:][::-1]
            
            # B matrix
            B = np.zeros((n, 1))
            B[-1, 0] = 1
            
            # C matrix
            C = np.zeros((1, n))
            if len(self.num) == len(self.den):
                # Proper transfer function
                diff = self.num - self.num[0] * self.den
                if n > 0 and len(diff) > 1:
                    C[0, :] = diff[1:][::-1]
            else:
                # Strictly proper
                if n > 0 and len(self.num) > 1:
                    C[0, :] = self.num[1:][::-1]
                elif n > 0 and len(self.num) == 1:
                    C[0, -1] = self.num[0]
            
            # D matrix
            D = np.array([[self.num[0]]]) if len(self.num) == len(self.den) else np.array([[0.0]])
            
            self.A = A
            self.B = B
            self.C = C
            self.D = D
        else:
            # Pure gain (no dynamics)
            self.A = np.zeros((0, 0))
            self.B = np.zeros((0, 1))
            self.C = np.zeros((1, 0))
            self.D = np.array([[self.num[0] / self.den[0]]])
        
        # State for each dimension
        self.n_states: int = self.A.shape[0]
        if initial_state is not None:
            self.state: np.ndarray = np.array(initial_state, dtype=float)
        else:
            self.state = np.zeros((self.dim, self.n_states), dtype=float)
        
        self.derivative: np.ndarray = np.zeros_like(self.state)
        
        # For RK4 integration
        self.k1: np.ndarray = np.zeros_like(self.state)
        self.k2: np.ndarray = np.zeros_like(self.state)
        self.k3: np.ndarray = np.zeros_like(self.state)
        self.k4: np.ndarray = np.zeros_like(self.state)

    def compute(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Compute transfer function output.
        
        Each input dimension is filtered independently through the same
        transfer function.
        
        Args:
            t: Current simulation time (unused)
            dt: Time step (unused in compute)
            input_values: List containing input signal u
        
        Returns:
            VectorSignal: Filtered output y
        """
        if not input_values:
            u = np.zeros(self.dim)
        else:
            u = input_values[0].value
        
        y = np.zeros(self.dim)
        
        for i in range(self.dim):
            if self.n_states > 0:
                # ẋ = Ax + Bu
                self.derivative[i, :] = self.A @ self.state[i, :] + (self.B @ u[i].reshape(1)).flatten()
                # y = Cx + Du
                y[i] = (self.C @ self.state[i, :] + self.D @ u[i].reshape(1)).flatten()[0]
            else:
                # Pure gain
                y[i] = (self.D @ u[i].reshape(1)).flatten()[0]
        
        self.output = VectorSignal(y, self.name)
        return self.output

    def get_derivative(self, t: float, input_values: Optional[List[VectorSignal]] = None) -> np.ndarray:
        """Get state derivative for RK4 integration."""
        if not input_values:
            u = np.zeros(self.dim)
        else:
            u = input_values[0].value
        
        derivative = np.zeros_like(self.state)
        
        for i in range(self.dim):
            if self.n_states > 0:
                derivative[i, :] = self.A @ self.state[i, :] + (self.B @ u[i].reshape(1)).flatten()
        
        return derivative

    def integrate_state(self, dt: float, solver: str = 'euler') -> None:
        """Integrate states forward in time."""
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
    
    This block terminates a signal path and records the history of values
    it receives. It simply passes through its input as output while storing
    all values for later analysis.
    
    Attributes:
        history (List[np.ndarray]): Recorded values from each time step
    
    Example:
        >>> source = ThreePhaseGenerator("gen", amplitude=5, freq=50)
        >>> sink = VectorEnd("output")
        >>> source >> sink
        >>> # After simulation, sink.history contains all recorded values
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize a VectorEnd sink block.
        
        Args:
            name: Unique identifier for this block
        
        Example:
            >>> sink1 = VectorEnd("output")
            >>> sink2 = VectorEnd("measured_current")
        """
        super().__init__(name)
        self.history: List[np.ndarray] = []

    def compute(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Record input value and pass it through as output.
        
        Args:
            t: Current simulation time (unused)
            dt: Time step (unused)
            input_values: List containing the signal to record
        
        Returns:
            VectorSignal: Same as input (pass-through)
        
        Example:
            >>> sink = VectorEnd("s")
            >>> input_sig = VectorSignal([1.0, 2.0, 3.0])
            >>> output = sink.compute(t=0, dt=0.01, input_values=[input_sig])
            >>> print(sink.history[-1])  # [1. 2. 3.]
        """
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
__author__ = 'Vector Simulation Framework'
