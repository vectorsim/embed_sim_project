"""
mo_to_fmu_client.py
===================
Parses OpenModelica (.mo) files and automatically generates
typed FMUBlock subclasses for the EmbedSim VectorBlock framework.

MAIN FUNCTION:
    generate_fmu_block(mo_path, output_dir=None)
        - Parses a .mo file and generates a Python FMUBlock class
        - Returns the path to the generated file

    generate_fmu_blocks_from_folder(folder_path, output_dir=None)
        - Processes all .mo files in a folder
        - Returns list of generated file paths

USAGE:
    from mo_to_fmu_client import generate_fmu_block

    # Generate a single block
    output_file = generate_fmu_block("ThreePhaseMotor.mo")

    # Generate with custom output directory
    output_file = generate_fmu_block("ThreePhaseMotor.mo", output_dir="./blocks")

    # Generate all models in a folder
    generated_files = generate_fmu_blocks_from_folder("./models/")
"""

from __future__ import annotations

import re
import os
import glob
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ModelVariable:
    name:        str
    description: str  = ""
    unit:        str  = ""
    default:     Optional[float] = None

@dataclass
class ModelInfo:
    name:        str
    inputs:      List[ModelVariable] = field(default_factory=list)
    outputs:     List[ModelVariable] = field(default_factory=list)
    parameters:  List[ModelVariable] = field(default_factory=list)
    description: str = ""


# =============================================================================
# Parser
# =============================================================================

class MoParser:
    """
    Parses a Modelica (.mo) file and extracts model structure.

    Handles:
      - parameter Real <name> (start=<val>) "<description>"
      - input  Real <name> (start=<val>) "<description>"
      - output Real <name>               "<description>"
      - Real <name> (start=<val>)        "<description>"  → state/algebraic (outputs)
    """

    # Regex patterns
    _RE_MODEL = re.compile(
        r'^\s*model\s+(\w+)\s*(?:"([^"]*)")?', re.MULTILINE
    )
    _RE_PARAM = re.compile(
        r'^\s*parameter\s+Real\s+(\w+)'
        r'(?:\s*=\s*([0-9eE+\-\.]+))?'      # = default value
        r'(?:\s*\([^)]*\))?'                  # (annotations)
        r'(?:\s*"([^"]*)")?'                  # "description"
        r'\s*;',
        re.MULTILINE
    )
    _RE_INPUT = re.compile(
        r'^\s*input\s+Real\s+(\w+)'
        r'(?:\s*\([^)]*start\s*=\s*([0-9eE+\-\.]+)[^)]*\))?'
        r'(?:\s*"([^"]*)")?'
        r'\s*;',
        re.MULTILINE
    )
    _RE_OUTPUT = re.compile(
        r'^\s*output\s+Real\s+(\w+)'
        r'(?:\s*\([^)]*start\s*=\s*([0-9eE+\-\.]+)[^)]*\))?'
        r'(?:\s*"([^"]*)")?'
        r'\s*;',
        re.MULTILINE
    )
    # State variables and algebraic outputs declared as plain Real
    _RE_STATE = re.compile(
        r'^\s*Real\s+(\w+)'
        r'(?:\s*\(([^)]*)\))?'               # (start=x, fixed=true, ...)
        r'(?:\s*"([^"]*)")?'
        r'\s*;',
        re.MULTILINE
    )
    _RE_START = re.compile(r'start\s*=\s*([0-9eE+\-\.]+)')
    _RE_UNIT  = re.compile(r'\[([^\]]+)\]')

    def __init__(self, mo_path: str):
        if not os.path.exists(mo_path):
            raise FileNotFoundError(f"File not found: {mo_path}")
        self.mo_path = mo_path
        with open(mo_path, "r", encoding="utf-8") as f:
            self._src = f.read()

    def parse(self) -> ModelInfo:
        src = self._src

        # ── Model name ───────────────────────────────────────────────
        m = self._RE_MODEL.search(src)
        if not m:
            raise ValueError(f"No 'model' declaration found in {self.mo_path}")
        model_name  = m.group(1)
        model_desc  = m.group(2) or ""

        info = ModelInfo(name=model_name, description=model_desc)

        # ── Parameters ──────────────────────────────────────────────
        for pm in self._RE_PARAM.finditer(src):
            name    = pm.group(1)
            default = float(pm.group(2)) if pm.group(2) else None
            desc    = pm.group(3) or ""
            unit    = self._extract_unit(desc)
            info.parameters.append(ModelVariable(name, desc, unit, default))

        # ── Inputs ──────────────────────────────────────────────────
        for im in self._RE_INPUT.finditer(src):
            name    = im.group(1)
            default = float(im.group(2)) if im.group(2) else 0.0
            desc    = im.group(3) or ""
            unit    = self._extract_unit(desc)
            info.inputs.append(ModelVariable(name, desc, unit, default))

        # ── Explicit outputs ─────────────────────────────────────────
        for om in self._RE_OUTPUT.finditer(src):
            name    = om.group(1)
            default = float(om.group(2)) if om.group(2) else 0.0
            desc    = om.group(3) or ""
            unit    = self._extract_unit(desc)
            info.outputs.append(ModelVariable(name, desc, unit, default))

        # ── State/algebraic variables → also outputs ─────────────────
        # (only add if not already in outputs)
        existing_out_names = {v.name for v in info.outputs}
        input_names        = {v.name for v in info.inputs}
        param_names        = {v.name for v in info.parameters}

        for sm in self._RE_STATE.finditer(src):
            name = sm.group(1)
            if name in existing_out_names or name in input_names or name in param_names:
                continue
            annot   = sm.group(2) or ""
            desc    = sm.group(3) or ""
            sm_val  = self._RE_START.search(annot)
            default = float(sm_val.group(1)) if sm_val else 0.0
            unit    = self._extract_unit(desc)
            info.outputs.append(ModelVariable(name, desc, unit, default))

        return info

    @staticmethod
    def _extract_unit(description: str) -> str:
        m = MoParser._RE_UNIT.search(description)
        return m.group(1) if m else ""


# =============================================================================
# Code generator
# =============================================================================

class ClientGenerator:
    """
    Generates a typed FMUBlock subclass from a ModelInfo.

    The generated class:
      - Subclasses FMUBlock (from embedsim.fmu_blocks)
      - Pre-wires INPUT_VARS, OUTPUT_VARS, DEFAULT_PARAMS as class constants
      - Provides a clean __init__ with named parameter kwargs
      - Adds typed read_<output>() helper methods for each output variable
      - Is directly usable as a VectorBlock inside any EmbedSim simulation
    """

    def __init__(self, model: ModelInfo):
        self.model = model

    # ----------------------------------------------------------------
    # Public
    # ----------------------------------------------------------------

    def generate(self) -> str:
        m   = self.model
        cls = f"{m.name}Block"
        lines = []
        lines += self._file_header(m, cls)
        lines += self._imports()
        lines += self._class_body(m, cls)
        return "\n".join(lines)

    def write(self, output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = f"{self.model.name}Block.py"
        code = self.generate()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        return output_path

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    def _file_header(self, m: ModelInfo, cls: str) -> List[str]:
        inp_names = [v.name for v in m.inputs]
        out_names = [v.name for v in m.outputs]
        par_names = [v.name for v in m.parameters]
        return [
            '"""',
            f"Auto-generated FMUBlock subclass for: {m.name}",
            f"Source model: {m.description}" if m.description else "",
            "",
            "Generated by mo_to_fmu_client.py",
            "DO NOT EDIT — re-run generator to update.",
            "",
            "USAGE:",
            f"    from {cls} import {cls}",
            "",
            f"    block = {cls}(",
            f'        name="{m.name.lower()}",',
            f'        fmu_path="{m.name}.fmu",',
            "    )",
            "    # Drop into any EmbedSim simulation engine as a VectorBlock.",
            "",
            "INPUTS  : " + (", ".join(inp_names) if inp_names else "(none)"),
            "OUTPUTS : " + (", ".join(out_names) if out_names else "(none)"),
            "PARAMS  : " + (", ".join(par_names) if par_names else "(none)"),
            '"""',
            "",
        ]



    def _imports(self) -> List[str]:
        return [
            "from __future__ import annotations",
            "import sys",
            "from typing import Dict, List, Optional",
            "from embedsim.fmu_blocks import FMUBlock",
            "from _path_utils import get_embedsim_import_path",
            "sys.path.insert(0, get_embedsim_import_path())",
            ""
        ]

    def _class_body(self, m: ModelInfo, cls: str) -> List[str]:
        inp = m.inputs
        out = m.outputs
        par = m.parameters

        L = []

        # ── Class declaration — subclasses FMUBlock ──────────────────
        L.append(f'class {cls}(FMUBlock):')
        L.append(f'    """')
        L.append(f'    Typed FMUBlock for the {m.name} OpenModelica model.')
        if m.description:
            L.append(f'    {m.description}')
        L.append(f'    ')
        L.append(f'    Subclasses FMUBlock — all VectorBlock lifecycle methods')
        L.append(f'    (reset, compute_py, terminate) are inherited.')
        L.append(f'    """')
        L.append('')

        # ── Class-level constants (documentation + introspection) ────
        inp_list = repr([v.name for v in inp])
        out_list = repr([v.name for v in out])
        L.append(f'    # FMU variable lists — passed to FMUBlock automatically')
        L.append(f'    INPUT_VARS:  List[str] = {inp_list}')
        L.append(f'    OUTPUT_VARS: List[str] = {out_list}')
        L.append('')

        # Default parameters dict
        if par:
            L.append('    DEFAULT_PARAMS: Dict[str, float] = {')
            for p in par:
                val     = p.default if p.default is not None else 0.0
                comment = f"  # [{p.unit}]" if p.unit else (f"  # {p.description}" if p.description else "")
                L.append(f'        {repr(p.name)}: {val},{comment}')
            L.append('    }')
            L.append('')

        # ── __init__ ─────────────────────────────────────────────────
        # Named kwargs for every parameter so IDEs show them
        L.append('    def __init__(')
        L.append('        self,')
        L.append('        name: str,')
        L.append(f'        fmu_path: str = "{m.name}.fmu",')
        # Parameter overrides as kwargs
        for p in par:
            val = p.default if p.default is not None else 0.0
            unit_hint = f"  # [{p.unit}]" if p.unit else ""
            L.append(f'        {p.name}: float = {val},{unit_hint}')
        L.append('        use_c_backend: bool = False,')
        L.append('        dtype=None,')
        L.append('    ) -> None:')
        L.append(f'        """')
        L.append(f'        Create a {cls} block.')
        L.append(f'        ')
        L.append(f'        Parameters')
        L.append(f'        ----------')
        L.append(f'        name      : Unique block identifier within the simulation graph.')
        L.append(f'        fmu_path  : Path to {m.name}.fmu (produced by OpenModelica).')
        for p in par:
            unit = f" [{p.unit}]" if p.unit else ""
            desc = p.description.split("[")[0].strip() if p.description else p.name
            L.append(f'        {p.name:<12}: {desc}{unit}')
        L.append(f'        """')

        # Build parameters dict from kwargs
        if par:
            L.append('        _params = {')
            for p in par:
                L.append(f'            {repr(p.name)}: {p.name},')
            L.append('        }')
        else:
            L.append('        _params = {}')

        # Call FMUBlock.__init__ — wires everything
        L.append('        super().__init__(')
        L.append('            name=name,')
        L.append('            fmu_path=fmu_path,')
        L.append('            input_names=self.INPUT_VARS,')
        L.append('            output_names=self.OUTPUT_VARS,')
        L.append('            parameters=_params,')
        L.append('            instance_name=name,')
        L.append('            use_c_backend=use_c_backend,')
        L.append('            dtype=dtype,')
        L.append('        )')
        L.append('')

        # ── set_<param> helpers ───────────────────────────────────────
        if par:
            L.append('    # ── Parameter setters (callable after instantiation) ──────')
            for p in par:
                unit = f" [{p.unit}]" if p.unit else ""
                L.append(f'    def set_{p.name}(self, value: float) -> None:')
                L.append(f'        """Set {p.name}{unit} — delegates to FMUBlock.set_parameter."""')
                L.append(f'        self.set_parameter({repr(p.name)}, value)')
                L.append('')

        # ── read_<output> helpers ─────────────────────────────────────
        L.append('    # ── Output readers (typed convenience accessors) ────────────')
        for v in out:
            unit    = f" [{v.unit}]" if v.unit else ""
            desc    = v.description if v.description else v.name
            L.append(f'    def read_{v.name}(self) -> float:')
            L.append(f'        """Read {desc}{unit}"""')
            L.append(f'        return self.get_output_by_name({repr(v.name)})')
            L.append('')

        # ── get_all helper ────────────────────────────────────────────
        L.append('    def get_all(self) -> Dict[str, float]:')
        L.append('        """Return all output variables as a named dict."""')
        L.append('        return self.get_all_outputs()')
        L.append('')

        # ── __repr__ ─────────────────────────────────────────────────
        L.append('    def __repr__(self) -> str:')
        L.append(f'        return (')
        L.append(f'            f"{cls}(name={{self.name!r}}, "')
        L.append(f'            f"fmu={{self.fmu_path!r}}, "')
        L.append(f'            f"init={{self._initialized}})"')
        L.append(f'        )')
        L.append('')

        return L


# =============================================================================
# Summary printer (optional, can be disabled)
# =============================================================================

def print_model_summary(model: ModelInfo, verbose: bool = True) -> None:
    """Print a summary of the parsed model if verbose is True."""
    if not verbose:
        return

    w = 60
    print(f"\n{'─'*w}")
    print(f"  Model  : {model.name}")
    if model.description:
        print(f"  Desc   : {model.description}")
    print(f"{'─'*w}")

    def _row(label, vars_):
        if not vars_:
            return
        print(f"  {label}:")
        for v in vars_:
            unit  = f" [{v.unit}]" if v.unit else ""
            deflt = f" = {v.default}" if v.default is not None else ""
            desc  = f"  ← {v.description}" if v.description else ""
            print(f"      {v.name:<20}{unit:<12}{deflt}{desc}")

    _row("PARAMETERS", model.parameters)
    _row("INPUTS",     model.inputs)
    _row("OUTPUTS",    model.outputs)
    print(f"{'─'*w}\n")


# =============================================================================
# Main API Functions
# =============================================================================

def generate_fmu_block(mo_path: str, output_dir: Optional[str] = None,
                      verbose: bool = True) -> str:
    """
    Generate an FMUBlock Python class from a .mo file.

    Args:
        mo_path: Path to the .mo file to parse
        output_dir: Optional directory to write the output file
        verbose: Whether to print progress information

    Returns:
        Path to the generated Python file

    Raises:
        FileNotFoundError: If the .mo file doesn't exist
        ValueError: If the .mo file can't be parsed correctly
    """
    if verbose:
        print(f"\nParsing: {mo_path}")

    # Parse the model
    parser = MoParser(mo_path)
    model = parser.parse()

    # Print summary if verbose
    print_model_summary(model, verbose)

    # Generate the code
    generator = ClientGenerator(model)

    # Determine output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model.name}Block.py")
    else:
        output_path = f"{model.name}Block.py"

    # Write the file
    generator.write(output_path)

    if verbose:
        print(f"  ✓  Generated: {output_path}")

    return output_path


def generate_fmu_blocks_from_folder(folder_path: str, output_dir: Optional[str] = None,
                                   verbose: bool = True) -> List[str]:
    """
    Generate FMUBlock Python classes for all .mo files in a folder.

    Args:
        folder_path: Path to folder containing .mo files
        output_dir: Optional directory to write the output files
        verbose: Whether to print progress information

    Returns:
        List of paths to generated Python files
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    # Find all .mo files
    mo_files = glob.glob(os.path.join(folder_path, "*.mo"))
    if not mo_files:
        if verbose:
            print(f"  No .mo files found in: {folder_path}")
        return []

    if verbose:
        print(f"\nFound {len(mo_files)} .mo files in {folder_path}")

    # Process each file
    generated_files = []
    for mo_file in mo_files:
        try:
            output_path = generate_fmu_block(mo_file, output_dir, verbose)
            generated_files.append(output_path)
        except Exception as e:
            if verbose:
                print(f"  [ERROR] Failed to process {mo_file}: {e}")

    if verbose:
        print(f"\nGenerated {len(generated_files)} FMUBlock classes")

    return generated_files


# Simple function for basic use
def mo_to_fmu_block(mo_path: str, output_path: Optional[str] = None) -> str:
    """
    Simple wrapper for basic use cases.

    Args:
        mo_path: Path to the .mo file
        output_path: Optional full output path

    Returns:
        Path to the generated Python file
    """
    if output_path:
        output_dir = os.path.dirname(output_path)
    else:
        output_dir = None

    return generate_fmu_block(mo_path, output_dir, verbose=False)


# =============================================================================
# Legacy CLI support (optional)
# =============================================================================

def main_cli() -> None:
    """Legacy CLI entry point - kept for backward compatibility."""
    import sys

    args = sys.argv[1:]

    if not args:
        print(__doc__)
        print("\nCLI USAGE (legacy):")
        print("  python mo_to_fmu_client.py <file.mo>")
        print("  python mo_to_fmu_client.py <models_dir/>")
        print("  python mo_to_fmu_client.py <file.mo> --out <output_dir>")
        print("\nOr use the function API:")
        print("  from mo_to_fmu_client import generate_fmu_block")
        print("  generate_fmu_block('model.mo')")
        sys.exit(0)

    output_dir = None
    paths = []

    i = 0
    while i < len(args):
        if args[i] == "--out" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        else:
            paths.append(args[i])
            i += 1

    for p in paths:
        if os.path.isdir(p):
            generate_fmu_blocks_from_folder(p, output_dir)
        else:
            generate_fmu_block(p, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    # When run directly, use the CLI
    #main_cli()
    output_path = mo_to_fmu_block("ThreePhaseMotor.mo", "ThreePhaseFmuPmsm.py")
    print(output_path)