
"""
vehicle_fmu.py
===============

ControlForge — VehicleKinematicBicycle FMU Wrapper and Path Generator

Purpose:
    This module provides tools for integrating a Vehicle Kinematic Bicycle FMU
    into ControlForge simulations. It contains:

    1. VehicleKinematicFMU — a wrapper around the FMU for use with VectorSim.
    2. VehiclePathGenerator — generates steering and acceleration commands.

About FMUs (Functional Mock-up Units):
    • FMU is a standardized, self-contained simulation model that can be
      imported into different simulation environments using the FMI standard.
    • It encapsulates a system’s dynamics, equations, or control logic,
      including inputs, outputs, and internal states.
    • FMUs can be co-simulated (own solver) or model-exchange (solved externally).
    • Here, the VehicleKinematicBicycle FMU models the kinematic behavior
      of a simple vehicle with inputs: steering angle (delta) and acceleration (a),
      and outputs: position (x, y), heading (theta), and velocity (v).

Purpose of the Wrapper:
    • Provides convenient access to FMU outputs as world-frame coordinates.
    • Simplifies integration with VectorSim, a Python-based simulation framework.
    • Supports steering and speed control through VehiclePathGenerator.
    • Keeps interface consistent with other VectorSim blocks for easy chaining.

Usage:
    - Use VehiclePathGenerator to produce delta and acceleration commands.
    - Connect to VehicleKinematicFMU to simulate vehicle motion.
    - Collect outputs via VectorEnd or monitor using sim.scope.

References:
    - FMI Standard: https://fmi-standard.org/
    - ControlForge VectorSim Framework
    - https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html

"""


import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing                     import List, Optional
from embedsim.fmu_blocks        import FMUBlock
from embedsim.core_blocks       import VectorSignal, VectorBlock
from embedsim.simulation_engine import  DEFAULT_DTYPE


# =========================
# Vehicle FMU Wrapper
# =========================
class VehicleKinematicFMU(FMUBlock):
    """
    Wrapper for the VehicleKinematicBicycle FMU.

    Provides world-frame outputs and integrates with VectorSim.
    Inputs: delta (steering), a (acceleration)
    Outputs: x_out, y_out, theta_out, v_out
    """

    def __init__(self, name: str, fmu_path: str = 'VehicleKinematicBicycle.fmu', use_c_backend: bool = False, dtype=None) -> None:
        _dtype = dtype if dtype is not None else DEFAULT_DTYPE
        super().__init__(
            name=name,
            fmu_path=fmu_path,
            input_names=['delta', 'a'],
            output_names=['x_out', 'y_out', 'theta_out', 'v_out'],
            use_c_backend=use_c_backend,
            dtype=dtype
        )
        # Initialize convenience output properties
        self.out_x = 0.0
        self.out_y = 0.0
        self.out_theta = 0.0
        self.out_v = 0.0

    def compute_py(self, t, dt, input_values=None) -> VectorSignal:
        """
        Compute FMU outputs for the given inputs at time t.

        Args:
            t: Current simulation time
            dt: Time step
            input_values: List of VectorSignal inputs [delta, a]

        Returns:
            VectorSignal containing [x, y, theta, v]
        """
        result = super().compute_py(t, dt, input_values)

        self.out_x     = result.value[0]
        self.out_y     = result.value[1]
        self.out_theta = result.value[2]
        self.out_v     = result.value[3]

        return result


# =========================
# Path / Command Generator
# =========================
class VehiclePathGenerator(VectorBlock):
    """
    Generates constant steering and acceleration commands.

    Outputs:
        VectorSignal([delta, a]) at each simulation step.

    Notes:
        - For circular motion, keep delta constant and non-zero.
        - Acceleration ramps up until cruise speed, then maintains cruise_accel.
    """

    def __init__(self, name: str,
                 delta: float = 0.3,
                 accel: float = 2.0,
                 cruise_speed: float = 5.0,
                 cruise_accel: float = 0.0,
                 use_c_backend: bool = False,
                 dtype=None) -> None:
        """
        Args:
            name: Block name
            delta: Constant steering angle [rad] (positive = left)
            accel: Acceleration until cruise speed [m/s²]
            cruise_speed: Target speed [m/s]
            cruise_accel: Acceleration to maintain cruise speed
        """
        _dtype = dtype if dtype is not None else DEFAULT_DTYPE

        super().__init__(name = name, use_c_backend=use_c_backend, dtype=_dtype)
        self.delta = delta
        self.accel = accel
        self.cruise_speed = cruise_speed
        self.cruise_accel = cruise_accel
        self.current_speed = 0.0  # Updated externally by FMU after each step

    def compute_py(self, t: float, dt: float,
                input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Compute steering and acceleration commands.

        Args:
            t: Current simulation time
            dt: Time step
            input_values: Ignored

        Returns:
            VectorSignal([delta, a_cmd])
        """
        if self.current_speed < self.cruise_speed:
            a_cmd = self.accel
        else:
            a_cmd = self.cruise_accel

        self.output = VectorSignal([self.delta, a_cmd], self.name)
        return self.output


# =========================
# Module Metadata
# =========================
__all__ = [
    'VehicleKinematicFMU',
    'VehiclePathGenerator'
]

__version__ = '1.0.0'
__author__ = 'Vector Simulation Framework'
