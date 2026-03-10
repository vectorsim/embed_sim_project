"""
source_blocks.py
================

Signal source blocks for generating various types of input signals.

This module provides blocks that generate signals without requiring inputs.
Sources include constant signals, step functions, sinusoidal generators, and
specialized three-phase generators for power system simulations.

Classes:
    VectorConstant: Outputs a constant vector signal
    VectorStep: Outputs a step change at a specified time
    ThreePhaseGenerator: Generates balanced three-phase sinusoidal signals
    VectorRamp: Generates a linearly increasing signal

Author: Vector Simulation Framework
Version: 1.0.0
"""

import numpy as np
from typing import List, Optional, Union
from .core_blocks import VectorBlock, VectorSignal


# =========================
# Constant Source
# =========================

class VectorConstant(VectorBlock):
    """
    Outputs a constant vector signal regardless of simulation time.

    This block generates a fixed vector value that never changes. Useful for
    setpoints, reference signals, or bias terms.

    Attributes:
        value (np.ndarray): The constant vector value to output

    Example:
        >>> # Create a constant 3-phase reference
        >>> ref = VectorConstant("reference", [1.0, 1.0, 1.0])
        >>> ref >> controller  # Feed to controller
    """

    def __init__(self, name: str, value: Union[List[float], np.ndarray],
                 use_c_backend: bool = False, dtype=None, **kwargs) -> None:
        """
        Initialize a VectorConstant block.

        Args:
            name: Unique identifier for this block
            value: Constant vector value to output. Can be a list or numpy array.
                   Will be converted to float64 internally.

        Example:
            >>> const1 = VectorConstant("setpoint", [5.0, 5.0, 5.0])
            >>> const2 = VectorConstant("bias", np.array([1.0, 2.0, 3.0]))
        """
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype, **kwargs)
        self.value: np.ndarray = np.array(value, dtype=float)

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Return the constant vector value.

        Args:
            t: Current simulation time (unused for constant source)
            dt: Time step (unused for constant source)
            input_values: Ignored for source blocks

        Returns:
            VectorSignal: The constant vector value

        Example:
            >>> const = VectorConstant("c", [1.0, 2.0, 3.0])
            >>> output = const.compute(t=0.5, dt=0.01)
            >>> print(output.value)  # [1. 2. 3.]
        """
        self.output = VectorSignal(self.value.copy(), self.name)
        return self.output


# =========================
# Step Source
# =========================

class VectorStep(VectorBlock):
    """
    Outputs a step vector signal that changes value at a specified time.

    This block outputs one value before the step time and switches to another
    value after the step time. All vector elements step simultaneously.
    Useful for simulating disturbances, load changes, or switching events.

    Attributes:
        step_time (float): Time when the step occurs (seconds)
        before_value (np.ndarray): Vector value before step time
        after_value (np.ndarray): Vector value after step time

    Example:
        >>> # Voltage step at t=0.01s: 0V → 100V
        >>> voltage = VectorStep("v_step", step_time=0.01,
        ...                      before_value=0.0, after_value=100.0, dim=3)
    """

    def __init__(self, name: str, step_time: float = 0.0,
                 before_value: float = 0.0, after_value: float = 1.0,
                 dim: int = 3, use_c_backend: bool = False, dtype=None, **kwargs) -> None:
        """
        Initialize a VectorStep block.

        Args:
            name: Unique identifier for this block
            step_time: Time when the step occurs (seconds). Default: 0.0
            before_value: Scalar value for all elements before step. Default: 0.0
            after_value: Scalar value for all elements after step. Default: 1.0
            dim: Dimension of the output vector. Default: 3

        Note:
            All vector elements use the same before/after values. For different
            values per element, create multiple VectorStep blocks or use VectorConstant
            with time-based switching logic.

        Example:
            >>> # Step from 0 to 10 at t=0.5s
            >>> step1 = VectorStep("load_change", step_time=0.5,
            ...                    before_value=0.0, after_value=10.0, dim=3)
            >>>
            >>> # Step from -5 to +5 at t=0.1s
            >>> step2 = VectorStep("disturbance", step_time=0.1,
            ...                    before_value=-5.0, after_value=5.0, dim=2)
        """
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype, **kwargs)
        self.step_time: float = step_time
        self.before_value: np.ndarray = np.array([before_value] * dim, dtype=float)
        self.after_value: np.ndarray = np.array([after_value] * dim, dtype=float)

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Compute the step signal value at current time.

        Args:
            t: Current simulation time (seconds)
            dt: Time step (unused for step source)
            input_values: Ignored for source blocks

        Returns:
            VectorSignal: before_value if t < step_time, otherwise after_value

        Example:
            >>> step = VectorStep("s", step_time=1.0, before_value=0.0,
            ...                   after_value=5.0, dim=2)
            >>> out1 = step.compute(t=0.5, dt=0.01)
            >>> print(out1.value)  # [0. 0.]
            >>> out2 = step.compute(t=1.5, dt=0.01)
            >>> print(out2.value)  # [5. 5.]
        """
        if t < self.step_time:
            val = self.before_value.copy()
        else:
            val = self.after_value.copy()

        self.output = VectorSignal(val, self.name)
        return self.output

# =========================
# Sinusoidal Generator
# =========================
class SinusoidalGenerator(VectorBlock):
    """

    Generates sinusoidal signal.

    This block generates sinusoidal signals with a constant phase dependent

    Attributes:
        amplitude (float): Peak value of the signal
        freq (float): Frequency in Hz
        phase (float): Phase shift of phase A in radians

    Example:
        >>> # 50 Hz current with 5A amplitude
        >>> i_abc = SinusoidalGenerator("s_current", amplitude=5.0, freq=50.0)
        >>>
        >>> # 60 Hz voltage with 45° phase shift
        >>> v_abc = SinusoidalGenerator("s_voltage", amplitude=100.0, freq=60.0,
        ...                             phase=np.pi/4)
    """

    def __init__(self, name: str, amplitude: float = 1.0,
                 freq: float = 50.0, phase: float = 0.0,
                 use_c_backend: bool = False, dtype=None, **kwargs) -> None:
        """
        Initialize a ThreePhaseGenerator block.

        Args:
            name: Unique identifier for this block
            amplitude: Peak value of each phase (same for all phases). Default: 1.0
            freq: Fundamental frequency in Hz. Default: 50.0 (European standard)
            phase: Phase shift of phase A in radians. Default: 0.0

        Note:
            For 60 Hz systems (North America), use freq=60.0
            Phase shift affects all three phases equally (maintains 120° separation)

        Example:
            >>> # European standard: 50 Hz
            >>> gen_eu = SinusoidalGenerator("eu_gen", amplitude=230.0, freq=50.0)
            >>>
            >>> # North American standard: 60 Hz
            >>> gen_na = SinusoidalGenerator("na_gen", amplitude=120.0, freq=60.0)
            >>>
            >>> # Custom phase shift (lead by 30°)
            >>> gen_shifted = SinusoidalGenerator("shifted", amplitude=1.0,
            ...                                    freq=50.0, phase=np.pi/6)
        """
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype, **kwargs)
        self.amplitude: float = amplitude
        self.freq: float = freq
        self.phase: float = phase

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Compute the sinusoidal output at current time.

        Args:
            t: Current simulation time (seconds)
            dt: Time step (unused for sinusoidal source)
            input_values: Ignored for source blocks

        Returns:
            VectorSignal: [signal]

        Example:
            >>> gen = SinusoidalGenerator("gen", amplitude=10.0, freq=50.0)
            >>> output = gen.compute(t=0.005, dt=0.0001)  # At 0.005s (quarter cycle)
            >>> print(output.value)  # Approximately [10., -5., -5.]
        """
        omega = 2 * np.pi * self.freq

        val = [self.amplitude * np.sin(omega * t + self.phase)]

        self.output = VectorSignal(val, self.name)
        return self.output


# =========================
# Three-Phase Sinusoidal Generator
# =========================

class ThreePhaseGenerator(VectorBlock):
    """
    Generates a balanced three-phase sinusoidal signal.

    This block produces three sinusoidal waveforms with 120° phase separation,
    commonly used in AC power systems. The output is a 3-element vector
    representing phases A, B, and C.

    Phase relationships:
        - Phase A: amplitude * sin(ωt + φ)
        - Phase B: amplitude * sin(ωt - 2π/3 + φ)  (lags A by 120°)
        - Phase C: amplitude * sin(ωt + 2π/3 + φ)  (leads A by 120°)

    where ω = 2πf is the angular frequency.

    Attributes:
        amplitude (float): Peak value of each phase
        freq (float): Frequency in Hz
        phase (float): Phase shift of phase A in radians

    Example:
        >>> # 50 Hz three-phase current with 5A amplitude
        >>> i_abc = ThreePhaseGenerator("i_3ph", amplitude=5.0, freq=50.0)
        >>>
        >>> # 60 Hz voltage with 45° phase shift
        >>> v_abc = ThreePhaseGenerator("v_3ph", amplitude=100.0, freq=60.0,
        ...                             phase=np.pi/4)
    """

    def __init__(self, name: str, amplitude: float = 1.0,
                 freq: float = 50.0, phase: float = 0.0,
                 use_c_backend: bool = False, dtype=None, **kwargs) -> None:
        """
        Initialize a ThreePhaseGenerator block.

        Args:
            name: Unique identifier for this block
            amplitude: Peak value of each phase (same for all phases). Default: 1.0
            freq: Fundamental frequency in Hz. Default: 50.0 (European standard)
            phase: Phase shift of phase A in radians. Default: 0.0

        Note:
            For 60 Hz systems (North America), use freq=60.0
            Phase shift affects all three phases equally (maintains 120° separation)

        Example:
            >>> # European standard: 50 Hz
            >>> gen_eu = ThreePhaseGenerator("eu_gen", amplitude=230.0, freq=50.0)
            >>>
            >>> # North American standard: 60 Hz
            >>> gen_na = ThreePhaseGenerator("na_gen", amplitude=120.0, freq=60.0)
            >>>
            >>> # Custom phase shift (lead by 30°)
            >>> gen_shifted = ThreePhaseGenerator("shifted", amplitude=1.0,
            ...                                    freq=50.0, phase=np.pi/6)
        """
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype, **kwargs)
        self.amplitude: float = amplitude
        self.freq: float = freq
        self.phase: float = phase
        self.output_label: str = "[a,b,c]"

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Compute the three-phase sinusoidal output at current time.

        Args:
            t: Current simulation time (seconds)
            dt: Time step (unused for sinusoidal source)
            input_values: Ignored for source blocks

        Returns:
            VectorSignal: Three-element vector [phase_a, phase_b, phase_c]

        Example:
            >>> gen = ThreePhaseGenerator("gen", amplitude=10.0, freq=50.0)
            >>> output = gen.compute(t=0.005, dt=0.0001)  # At 0.005s (quarter cycle)
            >>> print(output.value)  # Approximately [10., -5., -5.]
        """
        omega = 2 * np.pi * self.freq

        val = [
            self.amplitude * np.sin(omega * t + self.phase),
            self.amplitude * np.sin(omega * t - 2*np.pi/3 + self.phase),
            self.amplitude * np.sin(omega * t + 2*np.pi/3 + self.phase)
        ]

        self.output = VectorSignal(val, self.name)
        return self.output


# =========================
# Ramp Source
# =========================

class VectorRamp(VectorBlock):
    """
    Generates a linearly increasing (or decreasing) vector signal.

    This block outputs a signal that increases linearly with time at a specified
    slope (rate of change). Useful for testing system response to gradual changes.

    The output is: initial_value + slope * (t - start_time) for t >= start_time
                   initial_value for t < start_time

    Attributes:
        slope (float): Rate of change (units per second)
        initial_value (np.ndarray): Starting value vector
        start_time (float): Time when ramp begins

    Example:
        >>> # Ramp from 0 to 10 over 5 seconds (slope = 2.0)
        >>> ramp = VectorRamp("load_ramp", slope=2.0, initial_value=0.0,
        ...                   start_time=1.0, dim=3)
    """

    def __init__(self, name: str, slope: float = 1.0,
                 initial_value: float = 0.0, start_time: float = 0.0,
                 dim: int = 3, use_c_backend: bool = False, dtype=None, **kwargs) -> None:
        """
        Initialize a VectorRamp block.

        Args:
            name: Unique identifier for this block
            slope: Rate of change in units per second. Default: 1.0
                   Use negative value for decreasing ramp.
            initial_value: Starting value for all vector elements. Default: 0.0
            start_time: Time when ramp begins (seconds). Default: 0.0
            dim: Dimension of the output vector. Default: 3

        Example:
            >>> # Increase from 0 at 5 units/sec, starting at t=2s
            >>> ramp1 = VectorRamp("increasing", slope=5.0, initial_value=0.0,
            ...                    start_time=2.0, dim=3)
            >>>
            >>> # Decrease from 100 at -10 units/sec, starting at t=0s
            >>> ramp2 = VectorRamp("decreasing", slope=-10.0, initial_value=100.0,
            ...                    start_time=0.0, dim=2)
        """
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype, **kwargs)
        self.slope: float = slope
        self.initial_value: np.ndarray = np.array([initial_value] * dim, dtype=float)
        self.start_time: float = start_time

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Compute the ramp signal value at current time.

        Args:
            t: Current simulation time (seconds)
            dt: Time step (unused for ramp source)
            input_values: Ignored for source blocks

        Returns:
            VectorSignal: Ramped vector value

        Example:
            >>> ramp = VectorRamp("r", slope=2.0, initial_value=0.0,
            ...                   start_time=1.0, dim=2)
            >>> out1 = ramp.compute(t=0.5, dt=0.01)
            >>> print(out1.value)  # [0. 0.] (before start)
            >>> out2 = ramp.compute(t=3.0, dt=0.01)
            >>> print(out2.value)  # [4. 4.] (2 seconds after start)
        """
        if t < self.start_time:
            val = self.initial_value.copy()
        else:
            val = self.initial_value + self.slope * (t - self.start_time)

        self.output = VectorSignal(val, self.name)
        return self.output


# =========================
# Gaussian White Noise Source
# =========================

class GaussianNoiseBlock(VectorBlock):
    """
    Outputs a vector of independent Gaussian (white) noise samples.

    Each call to compute_py draws a fresh independent sample from
    N(mean, std²) for every element of the output vector.  The noise
    is statistically white in discrete time: samples at different time
    steps are uncorrelated.

    A fixed ``seed`` makes runs reproducible — useful for unit-testing
    and for comparing controller designs under identical disturbance
    sequences.  Leave ``seed=None`` for non-deterministic operation.

    Attributes:
        mean  (float):       Mean of the Gaussian distribution.
        std   (float):       Standard deviation (σ).  Power = σ².
        dim   (int):         Output vector length.
        _rng  (np.random.Generator): Internal numpy Generator instance.

    Mathematical description:
        y(k) ~ N(mean, std²)  ∀ k,  components i.i.d.

    Typical use-cases in EmbedSim:
        • Sensor noise added to current/speed measurements
        • Process disturbance injected into a torque or voltage path
        • Monte-Carlo robustness sweeps (vary seed per run)

    Example:
        >>> # 1-D speed-sensor noise, σ = 0.05 rad/s
        >>> speed_noise = GaussianNoiseBlock("spd_noise", std=0.05, dim=1)
        >>> speed_noise >> speed_sum          # add to measured speed
        >>>
        >>> # 2-D current noise, reproducible
        >>> iq_noise = GaussianNoiseBlock("iq_noise", std=0.01, dim=2, seed=42)
    """

    def __init__(self, name: str, mean: float = 0.0, std: float = 1.0,
                 dim: int = 1, seed: Optional[int] = None,
                 use_c_backend: bool = False, dtype=None, **kwargs) -> None:
        """
        Initialize a GaussianNoiseBlock.

        Args:
            name:  Unique identifier for this block.
            mean:  Mean of the distribution.  Default: 0.0
            std:   Standard deviation σ ≥ 0.  Default: 1.0
            dim:   Number of independent noise channels (output vector length).
                   Default: 1
            seed:  Integer seed for the numpy Generator, or None for
                   non-deterministic operation.  Default: None

        Raises:
            ValueError: If std < 0.

        Example:
            >>> n1 = GaussianNoiseBlock("noise", mean=0.0, std=0.1, dim=2)
            >>> n2 = GaussianNoiseBlock("rep_noise", std=0.05, dim=1, seed=0)
        """
        if std < 0:
            raise ValueError(f"GaussianNoiseBlock '{name}': std must be >= 0, got {std}")
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype, **kwargs)
        self.mean: float = mean
        self.std:  float = std
        self.dim:  int   = dim
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def compute_py(self, t: float, dt: float,
                   input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Draw one vector of i.i.d. Gaussian samples.

        Args:
            t:            Current simulation time (seconds) — unused.
            dt:           Time step — unused.
            input_values: Ignored; this is a source block.

        Returns:
            VectorSignal: ``dim``-element vector sampled from N(mean, std²).

        Example:
            >>> noise = GaussianNoiseBlock("n", std=1.0, dim=3, seed=7)
            >>> out = noise.compute(t=0.0, dt=0.01)
            >>> print(out.value.shape)   # (3,)
            >>> print(out.value)         # e.g. [ 0.27  -1.13   0.62 ]
        """
        val = self._rng.normal(loc=self.mean, scale=self.std, size=self.dim)
        self.output = VectorSignal(val, self.name)
        return self.output


# =========================
# Module Metadata
# =========================

__all__ = [
    'VectorConstant',
    'VectorStep',
    'ThreePhaseGenerator',
    'SinusoidalGenerator',
    'VectorRamp',
    'GaussianNoiseBlock',
]

__version__ = '1.0.0'
__author__ = 'Vector Simulation Framework'