
# =============================================================================
# FMU CO-SIMULATION BLOCK
# =============================================================================
#
# OVERVIEW
# -----------------------------------------------------------------------------
# This module implements FMUBlock, a wrapper that integrates a
# Co-Simulation FMU (Functional Mock-up Unit) into the Vector Simulation
# Framework.
#
# An FMU is a standardized simulation model container defined by the
# FMI (Functional Mock-up Interface) standard.
#
# FMUs allow models created in one tool (Modelica, Simulink, etc.)
# to be executed in another simulation environment.
#
#
# WHAT IS AN FMU?
# -----------------------------------------------------------------------------
# An FMU (.fmu file) is a ZIP archive containing:
#
#   modelDescription.xml   → Model metadata (variables, GUID, structure)
#   binaries/              → Compiled C binaries of the model
#   resources/             → Optional supporting data
#
# There are TWO FMI types:
#
# 1) Model Exchange (ME)
#    - Requires external ODE solver
#    - Only equations are exported
#
# 2) Co-Simulation (CS)  ← Used in this implementation
#    - FMU contains its own internal solver
#    - Simulation environment only calls doStep(dt)
#    - FMU advances its own internal state
#
#
# WHY FMUBLOCK EXISTS
# -----------------------------------------------------------------------------
# FMUBlock acts as an adapter between:
#
#   VectorSignal-based simulation framework
#   AND
#   FMI Co-Simulation runtime (fmpy.FMU2Slave)
#
# It abstracts:
#   - FMU extraction
#   - Variable reference lookup
#   - Parameter setting
#   - Input/output mapping
#   - Time stepping
#   - Proper termination and cleanup
#
#
# HIGH-LEVEL ARCHITECTURE
# -----------------------------------------------------------------------------
#
#        +----------------------------------+
#        |   Vector Simulation Framework    |
#        +----------------+-----------------+
#                         |
#                         v
#        +----------------------------------+
#        |            FMUBlock              |
#        |----------------------------------|
#        | - maps inputs to FMU             |
#        | - calls doStep(dt)               |
#        | - reads outputs from FMU         |
#        +----------------+-----------------+
#                         |
#                         v
#        +----------------------------------+
#        |        Co-Simulation FMU         |
#        |----------------------------------|
#        | - Internal solver                |
#        | - Continuous states              |
#        | - Compiled binary model          |
#        +----------------------------------+
#
#
# DATA FLOW PER TIME STEP
# -----------------------------------------------------------------------------
#
#   VectorSignal Inputs
#           │
#           ▼
#   FMUBlock.setReal(valueReference, value)
#           │
#           ▼
#   FMU.doStep(currentTime, dt)
#           │
#           ▼
#   FMU.getReal(valueReference)
#           │
#           ▼
#   VectorSignal Outputs
#
#
# FMU LIFECYCLE (MANAGED BY THIS CLASS)
# -----------------------------------------------------------------------------
#
#   1. Extract FMU
#   2. Read model description
#   3. Instantiate FMU2Slave
#   4. Set parameters
#   5. Setup experiment
#   6. Enter initialization mode
#   7. Exit initialization mode
#   8. Repeated doStep(dt) calls
#   9. Terminate and free instance
#
#
# STATE MACHINE
# -----------------------------------------------------------------------------
#
#             +----------------------+
#             |   Not Initialized    |
#             +----------+-----------+
#                        |
#                        v
#             +----------------------+
#             |     Initialized      |
#             +----------+-----------+
#                        |
#                        v
#             +----------------------+
#             |       Running        |
#             +----------+-----------+
#                        |
#                        v
#             +----------------------+
#             |     Terminated       |
#             +----------------------+
#
#
# VALUE REFERENCES EXPLAINED
# -----------------------------------------------------------------------------
# FMI does NOT access variables by string name during simulation.
# Each variable is assigned an integer ID called a "value reference".
#
# Example:
#   "u"  → valueReference = 12
#   "w"  → valueReference = 27
#
# Runtime calls:
#   fmu.setReal([12], [value])
#   fmu.getReal([27])
#
# This class caches value references for efficiency.
#
#
# TIME MANAGEMENT
# -----------------------------------------------------------------------------
# The FMU maintains internal continuous states.
# The framework controls time using:
#
#   currentCommunicationPoint = current_time
#   communicationStepSize     = dt
#
# After doStep(), the FMU advances its internal state to:
#
#   t_new = t_old + dt
#
#
# ERROR HANDLING
# -----------------------------------------------------------------------------
# This implementation validates:
#   - FMU file existence
#   - Variable name correctness
#   - Input dimension matching
#   - Proper initialization before use
#   - Safe cleanup on termination
#
# =============================================================================
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import os
from .core_blocks import VectorBlock, VectorSignal, validate_inputs_exist


# =========================
# FMU Co-Simulation Block
# =========================

class FMUBlock(VectorBlock):
    """
    FMU Co-Simulation block wrapper.
    
    This block wraps an FMU (Functional Mock-up Unit) model and integrates it
    into the vector block simulation framework. The FMU can have multiple inputs
    and outputs, which are mapped to/from vector signals.
    
    Features:
    - Automatic FMU extraction and initialization
    - Configurable input/output mappings
    - Support for parameters and initial values
    - Proper FMU lifecycle management (instantiate, initialize, step, terminate)
    - Error handling and validation
    
    Attributes:
        fmu_path (str): Path to the FMU file
        input_names (List[str]): Names of FMU input variables
        output_names (List[str]): Names of FMU output variables
        parameters (Dict): FMU parameters to set
        fmu (FMU2Slave): The instantiated FMU instance
        

    """
    
    def __init__(self, 
                 name: str,
                 fmu_path: str,
                 input_names: List[str],
                 output_names: List[str],
                 parameters: Optional[Dict[str, float]] = None,
                 instance_name: Optional[str] = None) -> None:
        """
        Initialize an FMU block.
        
        Args:
            name: Unique identifier for this block
            fmu_path: Path to the .fmu file (absolute or relative)
            input_names: List of FMU input variable names (in order)
            output_names: List of FMU output variable names (in order)
            parameters: Dictionary of parameter names and values to set
            instance_name: FMU instance name (default: same as name)
        
        Raises:
            FileNotFoundError: If FMU file doesn't exist
            ValueError: If input/output names are invalid
            

        """
        super().__init__(name)
        
        # Validate FMU file exists
        if not os.path.exists(fmu_path):
            raise FileNotFoundError(f"FMU file not found: {fmu_path}")
        
        self.fmu_path = fmu_path
        self.input_names = input_names
        self.output_names = output_names
        self.parameters = parameters if parameters is not None else {}
        self.instance_name = instance_name if instance_name is not None else name
        
        # FMU internals
        self.fmu: Optional[FMU2Slave] = None
        self.model_description = None
        self.unzip_dir = None
        self.value_references: Dict[str, int] = {}
        
        # Track if FMU is initialized
        self._initialized = False
        self._simulation_started = False
        
        # Current time tracking
        self._current_time = 0.0
        
    def _extract_and_load_fmu(self) -> None:
        """Extract FMU and read model description."""
        if self.model_description is None:
            self.model_description = read_model_description(self.fmu_path)
            self.unzip_dir = extract(self.fmu_path)
            
    def _get_value_reference(self, var_name: str) -> int:
        """Get the value reference for a variable name."""
        if var_name not in self.value_references:
            # Find variable in model description
            var = None
            for v in self.model_description.modelVariables:
                if v.name == var_name:
                    var = v
                    break
            
            if var is None:
                available_vars = [v.name for v in self.model_description.modelVariables]
                raise ValueError(
                    f"Variable '{var_name}' not found in FMU '{self.fmu_path}'. "
                    f"Available variables: {available_vars}"
                )
            
            self.value_references[var_name] = var.valueReference
        
        return self.value_references[var_name]
    
    def initialize_fmu(self, t_start: float = 0.0) -> None:
        """
        Initialize the FMU for simulation.
        
        This should be called before the first compute() call, typically
        by the simulation engine's reset() method.
        
        Args:
            t_start: Starting time for simulation
        """
        if self._initialized:
            return  # Already initialized
        
        # Extract and load FMU
        self._extract_and_load_fmu()
        
        # Create FMU instance
        self.fmu = FMU2Slave(
            guid=self.model_description.guid,
            unzipDirectory=self.unzip_dir,
            modelIdentifier=self.model_description.coSimulation.modelIdentifier,
            instanceName=self.instance_name
        )
        
        # Instantiate
        self.fmu.instantiate()
        
        # Set parameters before initialization
        for param_name, param_value in self.parameters.items():
            vr = self._get_value_reference(param_name)
            self.fmu.setReal([vr], [param_value])
        
        # Setup experiment
        self.fmu.setupExperiment(startTime=t_start)
        
        # Enter initialization mode
        self.fmu.enterInitializationMode()
        
        # Exit initialization mode
        self.fmu.exitInitializationMode()
        
        self._initialized = True
        self._current_time = t_start
        
    def reset(self) -> None:
        """
        Reset the block and terminate FMU if running.
        
        Called before starting a new simulation run.
        """
        super().reset()
        
        # Terminate existing FMU instance
        if self.fmu is not None and self._initialized:
            try:
                self.fmu.terminate()
                self.fmu.freeInstance()
            except:
                pass  # Ignore errors during cleanup
        
        self.fmu = None
        self._initialized = False
        self._simulation_started = False
        self._current_time = 0.0
        
    def compute(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Compute FMU outputs for current timestep.
        
        This method:
        1. Sets FMU inputs from input_values
        2. Advances FMU simulation by dt
        3. Reads FMU outputs
        4. Returns outputs as VectorSignal
        
        Args:
            t: Current simulation time
            dt: Time step
            input_values: List of input signals (must match input_names)
        
        Returns:
            VectorSignal: FMU outputs (ordered by output_names)
            
        Raises:
            RuntimeError: If FMU not initialized
            ValueError: If input dimensions don't match
        """
        # Initialize on first compute
        if not self._initialized:
            self.initialize_fmu(t_start=t)
        
        # Validate inputs if FMU expects them
        if len(self.input_names) > 0:
            if not input_values or len(input_values) == 0:
                raise ValueError(
                    f"{self.name}: FMU expects {len(self.input_names)} input(s), "
                    f"but got 0. Input names: {self.input_names}"
                )
            
            # Combine all input signals into a single vector
            combined_inputs = []
            for sig in input_values:
                if isinstance(sig.value, np.ndarray):
                    combined_inputs.extend(sig.value.flatten())
                else:
                    combined_inputs.append(float(sig.value))
            
            # Check dimension match
            if len(combined_inputs) != len(self.input_names):
                raise ValueError(
                    f"{self.name}: Expected {len(self.input_names)} input value(s), "
                    f"got {len(combined_inputs)}. Input names: {self.input_names}"
                )
            
            # Set FMU inputs
            for i, input_name in enumerate(self.input_names):
                vr = self._get_value_reference(input_name)
                self.fmu.setReal([vr], [combined_inputs[i]])
        
        # Perform FMU time step
        self.fmu.doStep(
            currentCommunicationPoint=self._current_time,
            communicationStepSize=dt
        )
        
        # Update current time
        self._current_time = t + dt
        
        # Read FMU outputs
        output_values = []
        for output_name in self.output_names:
            vr = self._get_value_reference(output_name)
            value = self.fmu.getReal([vr])[0]
            output_values.append(value)
        
        # Create output signal
        self.output = VectorSignal(np.array(output_values), self.name)
        return self.output
    
    def get_output_by_name(self, var_name: str) -> float:
        """
        Get a specific output value by variable name.
        
        Args:
            var_name: Name of the output variable
            
        Returns:
            float: Current value of the variable

        """
        if not self._initialized:
            raise RuntimeError(f"{self.name}: FMU not initialized")
        
        vr = self._get_value_reference(var_name)
        return self.fmu.getReal([vr])[0]
    
    def set_parameter(self, param_name: str, value: float) -> None:
        """
        Set an FMU parameter value.
        
        Can be called before initialization or during simulation.
        
        Args:
            param_name: Name of the parameter
            value: New parameter value
        """
        self.parameters[param_name] = value
        
        if self._initialized:
            vr = self._get_value_reference(param_name)
            self.fmu.setReal([vr], [value])
    
    def get_all_outputs(self) -> Dict[str, float]:
        """
        Get all output values as a dictionary.
        
        Returns:
            Dict mapping output names to current values
        """
        if not self._initialized:
            raise RuntimeError(f"{self.name}: FMU not initialized")
        
        outputs = {}
        for output_name in self.output_names:
            vr = self._get_value_reference(output_name)
            outputs[output_name] = self.fmu.getReal([vr])[0]
        
        return outputs
    
    def __del__(self):
        """Cleanup FMU on deletion."""
        if self.fmu is not None and self._initialized:
            try:
                self.fmu.terminate()
                self.fmu.freeInstance()
            except:
                pass

    def terminate(self) -> None:
        """
        Properly terminate and free the FMU instance.
        """

        if self.fmu is None:
            return

        try:
            self.fmu.terminate()
            self.fmu.freeInstance()
        except Exception as e:
            raise RuntimeError(f"FMU termination failed: {e}")

# =========================
# Module Metadata
# =========================

__all__ = [
    'FMUBlock'
]

__version__ = '1.0.0'
__author__ = 'Vector Simulation Framework'
