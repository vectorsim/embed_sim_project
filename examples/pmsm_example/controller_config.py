"""
controller_config.py - Centralized configuration for PMSM controllers
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ControllerConfig:
    """Configuration for the Python DFC controller."""

    # Current PI gains (matches C: DFC_CURRENT_KP_D_F, etc.)
    kp_d: float = 0.0001
    kp_q: float = 0.0195
    ki_d: float = 0.0005
    ki_q: float = 0.0002

    # Speed PI gains (matches C: DFC_SPEED_KP_Q_F, DFC_SPEED_KI_Q_F)
    kp_speed: float = 0.0039
    ki_speed: float = 0.0002

    # Limits (matches C: DFC_INTEGRAL_LIMIT_F)
    integral_limit: float = 25.0
    max_current: float = 100.0
    max_iq_dot: float = 1000.0
    modulation_limit: float = 0.90

    # Startup parameters (matches C)
    startup_mod_min: float = 0.05
    startup_mod_max: float = 0.25
    startup_increment: float = 0.001

    # Spinning detection (matches C)
    spinning_past_index: int = 89500
    stopped_past_index: int = 2000


@dataclass
class PlantConfig:
    """Configuration for the PMSM plant."""

    # Electrical parameters
    rs: float = 0.19
    ld: float = 0.125e-3
    lq: float = 0.125e-3
    lambda_pm: float = 0.0014

    # Mechanical parameters
    j: float = 2.4e-6
    b: float = 1.0e-6

    # Machine parameters
    pole_pairs: float = 4.0
    vdc: float = 12.0


@dataclass
class SimulationConfig:
    """Simulation configuration."""

    # Timing
    t_sim: float = 2.5
    dt: float = 50e-6

    # Reference
    target_rpm: float = 850.0
    step_time: float = 0.1

    # Paths
    fmu_path: Optional[str] = None

    # Flags
    save_plot: bool = True
    show_plot: bool = False
    verbose: bool = True


# Default configurations
DEFAULT_CONTROLLER_CONFIG = ControllerConfig()
DEFAULT_PLANT_CONFIG = PlantConfig()
DEFAULT_SIM_CONFIG = SimulationConfig()