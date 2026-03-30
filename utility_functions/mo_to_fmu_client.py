# ============================================================
# Fixed Modelica → FMU Python Generator (Educational Version)
# ============================================================
#
# This script parses a simplified Modelica (.mo) model file and
# generates a Python FMUBlock wrapper compatible with EmbedSim.
#
# It is designed for:
#   - Learning Modelica structure
#   - Understanding FMU interfaces
#   - Building automated simulation pipelines
#
# Key features:
#   ✔ Extracts inputs, outputs, and parameters
#   ✔ Supports Real and Integer parameters
#   ✔ Avoids unsafe variable exposure (STRICT mode)
#   ✔ Generates clean, ready-to-use Python block code
#
# ============================================================

import re
from dataclasses import dataclass
from typing import List, Optional

# ============================================================
# Data Structures (Intermediate Representation)
# ============================================================

@dataclass
class Variable:
    """
    Represents a Modelica variable (input/output).
    """
    name: str
    default: Optional[float] = None

@dataclass
class Parameter:
    """
    Represents a Modelica parameter.

    type:
        "Real"    → floating-point parameter
        "Integer" → integer parameter (e.g., pole pairs)
    """
    name: str
    default: Optional[float] = None
    type: str = "Real"

@dataclass
class ModelInfo:
    """
    Container holding all extracted model information.
    """
    name: str
    inputs: List[Variable]
    outputs: List[Variable]
    parameters: List[Parameter]

# ============================================================
# Modelica Parser
# ============================================================

class MoParser:
    """
    Lightweight Modelica parser using regex.

    NOTE:
    This is NOT a full Modelica parser.
    It supports a restricted but practical subset:
        - parameter Real
        - parameter Integer
        - input Real
        - output Real

    It assumes one declaration per line.
    """

    # --- Regex patterns for supported Modelica constructs ---

    # Matches: parameter Real R = 0.19;
    _RE_PARAM_REAL = re.compile(
        r"^\s*parameter\s+Real\s+(\w+)"
        r"(?:\s*=\s*([0-9eE+\-.]+))?.*;"
    )

    # Matches: parameter Integer p = 4;
    _RE_PARAM_INT = re.compile(
        r"^\s*parameter\s+Integer\s+(\w+)"
        r"(?:\s*=\s*([0-9+\-]+))?.*;"
    )

    # Matches: input Real duty_a;
    _RE_INPUT = re.compile(r"^\s*input\s+Real\s+(\w+).*")

    # Matches: output Real rpm;
    _RE_OUTPUT = re.compile(r"^\s*output\s+Real\s+(\w+).*")

    # Matches: Real internalVar;
    _RE_REAL = re.compile(r"^\s*Real\s+(\w+).*")

    # Strict mode prevents exposing internal variables as outputs
    STRICT_OUTPUTS = True

    def __init__(self, text: str):
        """
        Parameters
        ----------
        text : str
            Full contents of a Modelica (.mo) file
        """
        self.lines = text.splitlines()

    def parse(self) -> ModelInfo:
        """
        Parse the Modelica file and extract model structure.

        Returns
        -------
        ModelInfo
            Structured representation of the model
        """

        params: List[Parameter] = []
        inputs: List[Variable] = []
        outputs: List[Variable] = []

        model_name = "UnknownModel"

        for line in self.lines:
            line = line.strip()

            # --- Detect model name ---
            if line.startswith("model "):
                model_name = line.split()[1]

            # --- Parse Real parameters ---
            m = self._RE_PARAM_REAL.match(line)
            if m:
                name, val = m.groups()
                params.append(Parameter(
                    name=name,
                    default=float(val) if val else None,
                    type="Real"
                ))
                continue

            # --- Parse Integer parameters ---
            m = self._RE_PARAM_INT.match(line)
            if m:
                name, val = m.groups()
                params.append(Parameter(
                    name=name,
                    default=int(val) if val else None,
                    type="Integer"
                ))
                continue

            # --- Parse inputs ---
            m = self._RE_INPUT.match(line)
            if m:
                inputs.append(Variable(m.group(1)))
                continue

            # --- Parse outputs ---
            m = self._RE_OUTPUT.match(line)
            if m:
                outputs.append(Variable(m.group(1)))
                continue

            # --- Optional: promote plain Real variables to outputs ---
            # This is disabled by default for safety
            if not self.STRICT_OUTPUTS:
                m = self._RE_REAL.match(line)
                if m:
                    outputs.append(Variable(m.group(1)))

        # --- Remove duplicate outputs safely ---
        seen = set()
        unique_outputs = []
        for v in outputs:
            if v.name not in seen:
                seen.add(v.name)
                unique_outputs.append(v)

        return ModelInfo(model_name, inputs, unique_outputs, params)

# ============================================================
# Python Code Generator
# ============================================================

class ClientGenerator:
    """
    Generates a Python FMUBlock wrapper from ModelInfo.

    Output:
        A ready-to-use EmbedSim block class.
    """

    def __init__(self, model: ModelInfo):
        self.model = model

    def generate(self) -> str:
        """
        Generate Python code for FMUBlock wrapper.
        """

        inputs = [v.name for v in self.model.inputs]
        outputs = [v.name for v in self.model.outputs]

        # Only include parameters that actually have defaults
        params_dict = {}
        for p in self.model.parameters:
            if p.default is not None:
                params_dict[p.name] = p.default

        return f'''from __future__ import annotations
import sys
from typing import Dict, List
from _path_utils import get_embedsim_import_path

# Ensure EmbedSim is discoverable BEFORE import
sys.path.insert(0, get_embedsim_import_path())
from embedsim.fmu_blocks import FMUBlock

class {self.model.name}Block(FMUBlock):
    """
    Auto-generated FMUBlock wrapper for Modelica model: {self.model.name}

    This class allows EmbedSim to interact with the FMU
    using named inputs/outputs and parameter mapping.
    """

    # Input variable names (must match FMU exactly)
    INPUT_VARS: List[str] = {inputs}

    # Output variable names (must match FMU exactly)
    OUTPUT_VARS: List[str] = {outputs}

    # Default parameter values extracted from Modelica
    DEFAULT_PARAMS: Dict[str, float] = {params_dict}

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
'''

# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    """
    Example workflow:

    1. Load Modelica file
    2. Parse structure
    3. Generate Python FMU wrapper
    """

    with open("model.mo") as f:
        text = f.read()

    parser = MoParser(text)
    model = parser.parse()

    generator = ClientGenerator(model)
    python_code = generator.generate()

    print(python_code)
