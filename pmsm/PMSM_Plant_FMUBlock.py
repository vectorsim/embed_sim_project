from __future__ import annotations
import sys
from typing import Dict, List
from _path_utils import get_embedsim_import_path

# Ensure EmbedSim is discoverable BEFORE import
sys.path.insert(0, get_embedsim_import_path())
from embedsim.fmu_blocks import FMUBlock

class PMSM_Plant_FMUBlock(FMUBlock):
    """
    Auto-generated FMUBlock wrapper for Modelica model: PMSM_Plant_FMU

    This class allows EmbedSim to interact with the FMU
    using named inputs/outputs and parameter mapping.
    """

    # Input variable names (must match FMU exactly)
    INPUT_VARS: List[str] = ['duty_a', 'duty_b', 'duty_c', 'v_dc', 'T_load']

    # Output variable names (must match FMU exactly)
    OUTPUT_VARS: List[str] = ['rpm', 'ia', 'ib', 'ic', 'theta_m', 'T_em', 'id_out', 'iq_out']

    # Default parameter values extracted from Modelica
    DEFAULT_PARAMS: Dict[str, float] = {}

    def __init__(self, name: str, fmu_path: str):
        """
        Initialize FMU block instance.

        Parameters
        ----------
        name : str
            Unique name in simulation graph
        fmu_path : str
            Path to compiled FMU file
        """

        super().__init__(
            name=name,
            fmu_path=fmu_path,
            input_names=self.INPUT_VARS,
            output_names=self.OUTPUT_VARS,
            parameters=self.DEFAULT_PARAMS,
            instance_name=name,
        )
