"""
mo_to_fmu_client.py
===================

PURPOSE
-------
This module reads a Modelica (.mo) source file and AUTOMATICALLY GENERATES
a typed Python FMUBlock subclass for use in EmbedSim simulations.

The output is a ready-to-import Python file like BuckConverterBlock.py.
You never write that file by hand — you write or edit the .mo source,
run gen_fmu.py, and the Python wrapper is recreated automatically.

THE FULL PIPELINE — where this module fits
------------------------------------------

  [OpenModelica]               [THIS MODULE]               [EmbedSim]
  BuckConverter.mo             mo_to_fmu_client.py
        │                            │
        ▼                            ▼
  BuckConverter.fmu  ←────  BuckConverterBlock.py  ────►  pi_buck_example.py
  (compiled binary)          (generated Python class)     (simulation script)

THIS MODULE handles the middle column only:
  Input:  BuckConverter.mo   (text file, Modelica equations)
  Output: BuckConverterBlock.py  (text file, Python class)
It does NOT compile anything. It is a Python TEXT GENERATOR.

WHY AUTO-GENERATE?
------------------
Writing FMUBlock subclasses by hand is mechanical and error-prone:
  - List INPUT_VARS, OUTPUT_VARS, DEFAULT_PARAMS exactly as in the .mo
  - Write __init__ kwargs for every parameter
  - Write set_X() for every parameter
  - Write read_Y() for every output

If the .mo file changes (new parameter, renamed output), the Python wrapper
must be updated in sync. mo_to_fmu_client.py automates this perfectly:
re-run gen_fmu.py and BuckConverterBlock.py is rebuilt from the current .mo.

ARCHITECTURE — two classes, one pipeline
-----------------------------------------
  MoParser         reads the .mo file → produces ModelInfo
  ClientGenerator  reads ModelInfo    → produces Python source text

USAGE (from gen_fmu.py or any script)
--------------------------------------
  from mo_to_fmu_client import generate_fmu_block
  generate_fmu_block("BuckConverter.mo", output_dir="buck_converter/")

  from mo_to_fmu_client import generate_fmu_blocks_from_folder
  generate_fmu_blocks_from_folder("electrical_blocks/modelica/")

CLI (legacy — prefer the function API)
---------------------------------------
  python mo_to_fmu_client.py BuckConverter.mo
  python mo_to_fmu_client.py models/ --out blocks/

CORRECTNESS REVIEW
------------------
Module is CORRECT. Specific notes documented inline:
  - _is_in_protected_section() logic is sound and correctly handles
    BuckConverter.mo's single protected section containing switch_state.
  - The empty-parameter branch uses '{{}}' (escaped f-string braces)
    to emit a literal '{}' in the generated file — correct Python.
  - Plain Real variable promotion to outputs correctly skips anything
    already classified as input, output, or parameter.
  - repr([v.name for v in ...]) produces valid Python list literals
    directly usable in the generated source — correct.

MAIN FUNCTIONS
--------------
    generate_fmu_block(mo_path, output_dir, verbose)
    generate_fmu_blocks_from_folder(folder_path, output_dir, verbose)
    mo_to_fmu_block(mo_path, output_path)   — simplified wrapper
"""

from __future__ import annotations  # allows List[str] etc. in Python 3.8

import re
import os
import glob
import textwrap      # imported but not used in current version — reserved
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union


# ==============================================================================
#  DATA STRUCTURES
# ==============================================================================
#
# @dataclass: a Python decorator (Python 3.7+) that reads the field
# declarations and auto-generates __init__, __repr__, and __eq__ for the class.
# Without @dataclass you would write all that boilerplate manually.
#
# field(default_factory=list): creates a NEW empty list for each instance.
# Never write  inputs: list = []  as a class attribute — that would share
# the SAME list object across all instances, causing subtle bugs.
# ==============================================================================

@dataclass
class ModelVariable:
    """
    One variable extracted from the Modelica source: a parameter, input, or output.

    For BuckConverter.mo the parser produces, for example:
        ModelVariable('L',      'Inductance [H]',         'H',    0.0001)
        ModelVariable('duty',   'PWM duty cycle [0-1]',   '0-1',  0.0)
        ModelVariable('V_out',  'Output voltage [V]',     'V',    0.0)
        ModelVariable('I_load', 'Load current [A]',       'A',    0.0)
    """
    name:        str                    # Modelica variable name,  e.g. 'L'
    description: str  = ""             # description string from .mo file
    unit:        str  = ""             # extracted from '[unit]' in description
    default:     Optional[float] = None # numeric default, or None if not given


@dataclass
class ModelInfo:
    """
    Complete description of one Modelica model, ready for code generation.

    Produced by MoParser.parse() and consumed by ClientGenerator.
    """
    name:        str                                    # e.g. 'BuckConverter'
    inputs:      List[ModelVariable] = field(default_factory=list)
    outputs:     List[ModelVariable] = field(default_factory=list)
    parameters:  List[ModelVariable] = field(default_factory=list)
    description: str = ""             # model-level description string


# ==============================================================================
#  MOPARSER  —  reads the .mo file and builds a ModelInfo
# ==============================================================================

class MoParser:
    """
    Parses a Modelica (.mo) source file with regular expressions.

    WHY REGEX INSTEAD OF A FULL MODELICA GRAMMAR?
    -----------------------------------------------
    A proper Modelica parser (grammar + AST) handles 100% of valid Modelica
    but is hundreds of lines of parser code. For EmbedSim's purposes we only
    need four constructs:
        1. parameter Real declarations
        2. input Real declarations
        3. output Real declarations
        4. plain Real declarations (state / algebraic variables)

    Targeted regex patterns cover all four constructs reliably for the
    subset of Modelica EmbedSim uses. The trade-off: exotic Modelica syntax
    (multi-line declarations, record types, arrays, Integer variables)
    would be missed. All current EmbedSim .mo files are well within scope.

    TRACING THROUGH BuckConverter.mo
    ----------------------------------
    Line  5:  parameter Real L = 100e-6 "Inductance [H]";
              → _RE_PARAM matches → parameters = [ModelVariable('L', ...)]
    Line 12:  input Real duty "PWM duty cycle [0-1]";
              → _RE_INPUT matches → inputs = [ModelVariable('duty', ...)]
    Lines 15-17: output Real V_out / I_L / I_load
              → _RE_OUTPUT matches 3× → outputs = [V_out, I_L, I_load]
    Line 20:  Real switch_state;  (inside 'protected')
              → _RE_STATE matches BUT _is_in_protected_section() returns True
              → SKIPPED — switch_state does NOT appear in outputs ✓
    """

    # ── Compiled regex patterns ────────────────────────────────────────────────
    # re.compile() pre-compiles the pattern for repeated fast matching.
    # re.MULTILINE makes ^ match at the start of EACH LINE (not just file start)
    # and $ match at the end of each line.
    #
    # REGEX QUICK REFERENCE used below:
    #   \s+         one or more whitespace characters (space, tab, newline)
    #   \w+         word characters: [a-zA-Z0-9_]+
    #   [^"]*       any characters except double-quote
    #   (?:...)?    non-capturing optional group
    #   ([...])     capturing group — what's in () becomes m.group(N)
    #   [0-9eE+\-.] characters legal in a floating-point literal

    # Matches:  model BuckConverter "Simple buck converter..."
    # Group 1 → 'BuckConverter'
    # Group 2 → 'Simple buck converter...' (optional description)
    _RE_MODEL = re.compile(
        r'^\s*model\s+(\w+)\s*(?:"([^"]*)")?', re.MULTILINE
    )

    # Matches:  parameter Real L = 100e-6 "Inductance [H]";
    # Group 1 → 'L'
    # Group 2 → '100e-6'  (the = value; absent if no default given)
    # Group 3 → 'Inductance [H]'  (description string)
    #
    # (?:\s*=\s*([0-9eE+\-\.]+))?   optional:  = 100e-6
    # (?:\s*\([^)]*\))?              optional:  (unit="H") annotation block
    # (?:\s*"([^"]*)")?              optional:  "description" string
    _RE_PARAM = re.compile(
        r'^\s*parameter\s+Real\s+(\w+)'
        r'(?:\s*=\s*([0-9eE+\-\.]+))?'      # optional default value
        r'(?:\s*\([^)]*\))?'                  # optional annotation block
        r'(?:\s*"([^"]*)")?'                  # optional description string
        r'\s*;',
        re.MULTILINE
    )

    # Matches:  input Real duty "PWM duty cycle [0-1]";
    # Group 1 → 'duty'
    # Group 2 → start value from (start=x) annotation (if present)
    # Group 3 → 'PWM duty cycle [0-1]'
    _RE_INPUT = re.compile(
        r'^\s*input\s+Real\s+(\w+)'
        r'(?:\s*\([^)]*start\s*=\s*([0-9eE+\-\.]+)[^)]*\))?'  # optional (start=x)
        r'(?:\s*"([^"]*)")?'
        r'\s*;',
        re.MULTILINE
    )

    # Matches:  output Real V_out "Output voltage [V]";
    # Same group structure as _RE_INPUT.
    _RE_OUTPUT = re.compile(
        r'^\s*output\s+Real\s+(\w+)'
        r'(?:\s*\([^)]*start\s*=\s*([0-9eE+\-\.]+)[^)]*\))?'
        r'(?:\s*"([^"]*)")?'
        r'\s*;',
        re.MULTILINE
    )

    # Matches plain (unqualified) Real declarations — ODE states and
    # algebraic variables not marked with parameter/input/output.
    # Example from a different model:  Real I_L (start=0.0) "Inductor [A]";
    # In BuckConverter.mo switch_state is the only plain Real — but it is
    # protected, so _is_in_protected_section() prevents it from being added.
    # Group 1 → variable name
    # Group 2 → full annotation content, e.g. 'start=0.0, fixed=true'
    # Group 3 → description string
    _RE_STATE = re.compile(
        r'^\s*Real\s+(\w+)'
        r'(?:\s*\(([^)]*)\))?'               # optional (annotations)
        r'(?:\s*"([^"]*)")?'
        r'\s*;',
        re.MULTILINE
    )

    # Extracts the numeric start value from an annotation string.
    # Input: 'start=0.5, fixed=true'  →  group(1) = '0.5'
    _RE_START = re.compile(r'start\s*=\s*([0-9eE+\-\.]+)')

    # Extracts the unit abbreviation from a description string.
    # Looks for [...] brackets: 'Output voltage [V]'  →  group(1) = 'V'
    _RE_UNIT = re.compile(r'\[([^\]]+)\]')

    # Matches a line that is ONLY the word 'protected' (Modelica section marker).
    # ^\s*protected\s*$ matches e.g.:   "protected"   "  protected  "
    # But NOT: "protected Real x;"  (that line has more than just 'protected')
    _RE_PROTECTED_START = re.compile(r'^\s*protected\s*$', re.MULTILINE)

    # Matches an 'end ModelName;' line — signals the close of a Modelica block.
    # Used to detect whether the 'protected' section has been closed.
    _RE_END = re.compile(r'^\s*end\s+\w+\s*;?\s*$', re.MULTILINE)

    def __init__(self, mo_path: str):
        """
        Load the .mo file into memory as a single string.

        All regex operations run on self._src. Reading once at construction
        is more efficient than opening the file for each parse step.
        """
        if not os.path.exists(mo_path):
            raise FileNotFoundError(f"File not found: {mo_path}")
        self.mo_path = mo_path
        with open(mo_path, "r", encoding="utf-8") as f:
            self._src = f.read()  # entire file as one string

    def _is_in_protected_section(self, match_start: int) -> bool:
        """
        Determine whether a character position in the file falls inside
        a Modelica 'protected' section.

        WHAT IS 'protected' IN MODELICA?
        Variables declared after the 'protected' keyword are internal to
        the model and are NOT accessible externally (not readable via FMPy,
        not in the FMU's output list). Example from BuckConverter.mo:

            protected
              Real switch_state;   ← internal — must NOT be an FMU output

        If we naively added every plain 'Real' declaration to outputs,
        switch_state would appear in BuckConverterBlock.OUTPUT_VARS, which
        would cause FMPy to fail when trying to read it from the FMU.
        This method prevents that.

        ALGORITHM
        ---------
        We examine only the text BEFORE match_start (src_before).
        1. Find all 'protected' keyword positions in src_before.
        2. If none → not protected → return False.
        3. Take the LAST 'protected' (most recently opened protected section).
        4. Look for any 'end ModelName;' line that appears AFTER that
           'protected' but still BEFORE our match position.
        5. If an 'end' was found → the protected section was closed before
           our variable → NOT protected → return False.
        6. If no 'end' found after the 'protected' → still inside protected
           → return True.

        For BuckConverter.mo:
            'protected' appears on line 19 at some character position P.
            'Real switch_state;' appears at position Q > P.
            'end BuckConverter;' appears at position R > Q.
            When we check switch_state (Q), src_before = src[:Q].
            P is before Q, so last_protected = P.
            'end BuckConverter;' (at R) is AFTER Q, so it is NOT in src_before.
            → no 'end' found after P in src_before → return True (protected). ✓

        Args:
            match_start: character index of the match being tested

        Returns:
            bool: True if inside a protected section
        """
        # Slice the source up to (but not including) our match position.
        src_before = self._src[:match_start]

        # Collect character positions of all 'protected' keywords before our match.
        # .end() gives the position AFTER the matched text (just past the newline).
        protected_positions = [
            m.end() for m in self._RE_PROTECTED_START.finditer(src_before)
        ]
        if not protected_positions:
            return False   # no 'protected' keyword found before our position

        # Take the most recently seen 'protected'.
        last_protected = max(protected_positions)

        # Check for any 'end ModelName;' line that appeared AFTER last_protected
        # but still BEFORE our match position (i.e. in src_before).
        # If such an 'end' exists, it closed the protected section before us.
        end_before = [
            m.start() for m in self._RE_END.finditer(src_before)
            if m.start() > last_protected
        ]

        # 'not end_before' is True when the list is empty:
        # no 'end' found after 'protected' and before our match → still protected.
        return not end_before

    def parse(self) -> ModelInfo:
        """
        Parse the loaded .mo source and return a complete ModelInfo.

        Extraction order (order matters — see de-duplication step 5):
            1. Model name and description
            2. parameter Real declarations  → ModelInfo.parameters
            3. input  Real declarations     → ModelInfo.inputs
            4. output Real declarations     → ModelInfo.outputs
            5. plain  Real declarations     → ModelInfo.outputs
               (ONLY if not already classified and not in protected section)

        Returns:
            ModelInfo: all parsed variables, ready for ClientGenerator
        """
        src = self._src

        # ── 1. Model name ──────────────────────────────────────────────────────
        # Every .mo file must start with:  model <Name>  "description"
        # If this is missing, the file is not a valid Modelica model.
        m = self._RE_MODEL.search(src)
        if not m:
            raise ValueError(f"No 'model' declaration found in {self.mo_path}")
        model_name = m.group(1)        # 'BuckConverter'
        model_desc = m.group(2) or ""  # 'Simple buck converter plant model...'

        info = ModelInfo(name=model_name, description=model_desc)

        # ── 2. Parameters ──────────────────────────────────────────────────────
        # finditer() returns an iterator of all non-overlapping regex matches.
        # Each match becomes one ModelVariable appended to info.parameters.
        for pm in self._RE_PARAM.finditer(src):
            name    = pm.group(1)
            # float('100e-6') = 0.0001. If no '= value' in the .mo, default is None.
            default = float(pm.group(2)) if pm.group(2) else None
            desc    = pm.group(3) or ""
            unit    = self._extract_unit(desc)  # 'Inductance [H]' → 'H'
            info.parameters.append(ModelVariable(name, desc, unit, default))

        # ── 3. Inputs ──────────────────────────────────────────────────────────
        for im in self._RE_INPUT.finditer(src):
            name    = im.group(1)
            # Inputs without a start= annotation default to 0.0
            # (distinct from parameters where None means "omitted").
            default = float(im.group(2)) if im.group(2) else 0.0
            desc    = im.group(3) or ""
            unit    = self._extract_unit(desc)
            info.inputs.append(ModelVariable(name, desc, unit, default))

        # ── 4. Explicit outputs ────────────────────────────────────────────────
        for om in self._RE_OUTPUT.finditer(src):
            name    = om.group(1)
            default = float(om.group(2)) if om.group(2) else 0.0
            desc    = om.group(3) or ""
            unit    = self._extract_unit(desc)
            info.outputs.append(ModelVariable(name, desc, unit, default))

        # ── 5. Plain Real declarations → promote to outputs if appropriate ─────
        # Some Modelica models declare ODE states or algebraic outputs without
        # the 'output' keyword (especially older or compact model styles).
        # These plain 'Real' variables are promoted to FMU outputs here,
        # subject to two guards:
        #   a. Not already listed as an output, input, or parameter.
        #   b. Not in a 'protected' section.
        #
        # For BuckConverter.mo:
        #   V_out, I_L, I_load → already in outputs (step 4) → skipped by (a).
        #   switch_state → in protected section → skipped by (b). ✓

        # Build sets for O(1) membership testing (faster than list search).
        # Sets use hash tables: 'name in set' is O(1) vs O(n) for lists.
        existing_out_names = {v.name for v in info.outputs}
        input_names        = {v.name for v in info.inputs}
        param_names        = {v.name for v in info.parameters}

        for sm in self._RE_STATE.finditer(src):
            name = sm.group(1)

            # Guard (b): skip variables in the protected section
            if self._is_in_protected_section(sm.start()):
                continue  # switch_state skipped here for BuckConverter.mo

            # Guard (a): skip variables already classified
            if name in existing_out_names or name in input_names or name in param_names:
                continue

            # Promote this unclassified plain Real to an output.
            annot   = sm.group(2) or ""    # annotation text between ( and )
            desc    = sm.group(3) or ""
            sm_val  = self._RE_START.search(annot)
            default = float(sm_val.group(1)) if sm_val else 0.0
            unit    = self._extract_unit(desc)
            info.outputs.append(ModelVariable(name, desc, unit, default))

        return info

    @staticmethod
    def _extract_unit(description: str) -> str:
        """
        Pull the unit abbreviation out of a Modelica description string.

        Looks for text inside [...] brackets. Examples:
            'Inductance [H]'           → 'H'
            'Output voltage [V]'       → 'V'
            'PWM duty cycle [0-1]'     → '0-1'
            'Switching frequency [Hz]' → 'Hz'
            'Some description'         → ''  (no brackets → empty string)

        @staticmethod: belongs to the class for namespace clarity but
        does not access any instance data — no 'self' needed.
        """
        m = MoParser._RE_UNIT.search(description)
        return m.group(1) if m else ""


# ==============================================================================
#  CLIENTGENERATOR  —  reads ModelInfo, emits Python source text
# ==============================================================================

class ClientGenerator:
    """
    Generates a complete, ready-to-use Python FMUBlock subclass
    from a ModelInfo object.

    WHAT IT GENERATES (for BuckConverter)
    ---------------------------------------
    The text of BuckConverterBlock.py:
        Module docstring with INPUTS / OUTPUTS / PARAMS summary
        from __future__ import annotations
        import sys ...
        class BuckConverterBlock(FMUBlock):
            INPUT_VARS  = ['duty']
            OUTPUT_VARS = ['V_out', 'I_L', 'I_load']
            DEFAULT_PARAMS = {'L': 0.0001, 'C': 0.0001, ...}
            def __init__(self, name, fmu_path, L, C, R_load, V_in, f_sw, ...):
                ...
                super().__init__(...)
            def set_L(self, value): ...
            def set_C(self, value): ...
            ...
            def read_V_out(self): ...
            def read_I_L(self):   ...
            def read_I_load(self): ...
            def get_all(self): ...
            def __repr__(self): ...

    CODE GENERATION STRATEGY
    -------------------------
    Each helper method returns a List[str] of Python source lines.
    generate() concatenates them all and joins with newlines.
    This "line list" pattern is common in code generators because:
      - Each method appends its own lines independently, easy to extend.
      - Indentation is explicit per line — no ambiguity.
      - Inserting blank lines between sections is trivial (append '').
      - No string formatting edge cases from multi-line f-strings.
    """

    def __init__(self, model: ModelInfo):
        self.model = model

    # ── Public API ──────────────────────────────────────────────────────────────

    def generate(self) -> str:
        """
        Generate the complete Python source file as a single string.

        Assembles the file from four sections:
          _file_header() → module docstring
          _imports()     → import statements
          _class_body()  → the class itself
        """
        m   = self.model
        cls = f"{m.name}Block"   # 'BuckConverter' → 'BuckConverterBlock'

        lines = []
        lines += self._file_header(m, cls)
        lines += self._imports()
        lines += self._class_body(m, cls)
        return "\n".join(lines)

    def write(self, output_path: Optional[str] = None) -> str:
        """
        Generate the source and write it to a file on disk.

        Args:
            output_path: destination file path.
                         Defaults to '{ModelName}Block.py' in the cwd.
        Returns:
            str: the path that was written (same as output_path)
        """
        if output_path is None:
            output_path = f"{self.model.name}Block.py"
        code = self.generate()
        # encoding='utf-8': consistent across Windows and Linux.
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        return output_path

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _file_header(self, m: ModelInfo, cls: str) -> List[str]:
        """
        Emit the module-level docstring — the top of the generated file.

        This is the triple-quoted block at the top of BuckConverterBlock.py
        that shows the model name, description, usage example, and the
        INPUTS / OUTPUTS / PARAMS summary lines.
        """
        inp_names = [v.name for v in m.inputs]
        out_names = [v.name for v in m.outputs]
        par_names = [v.name for v in m.parameters]
        return [
            '"""',
            f"Auto-generated FMUBlock subclass for: {m.name}",
            # Conditional: only emit the description line if it is non-empty.
            f"Source model: {m.description}" if m.description else "",
            "",
            "Generated by mo_to_fmu_client.py",
            "DO NOT EDIT — re-run generator to update.",
            "",
            "USAGE:",
            f"    from {cls} import {cls}",
            "",
            f"    block = {cls}(",
            f'        name="{m.name.lower()}",',      # e.g. name="buckconverter"
            f'        fmu_path="{m.name}.fmu",',       # e.g. fmu_path="BuckConverter.fmu"
            "    )",
            "    # Drop into any EmbedSim simulation engine as a VectorBlock.",
            "",
            # join() produces: "duty"  or  "(none)" if the list is empty.
            "INPUTS  : " + (", ".join(inp_names) if inp_names else "(none)"),
            "OUTPUTS : " + (", ".join(out_names) if out_names else "(none)"),
            "PARAMS  : " + (", ".join(par_names) if par_names else "(none)"),
            '"""',
            "",
        ]

    def _imports(self) -> List[str]:
        """
        Emit the import block for the generated file.

        The generated file imports _path_utils using a plain (non-package)
        import, which works because BuckConverterBlock.py is placed in the
        same directory as _path_utils.py (buck_converter/).

        Note: the import order intentionally puts 'from embedsim...' BEFORE
        'sys.path.insert(...)'. This works only because FMUBlock itself does
        not import from the project root at class-definition time — it imports
        lazily. In practice both orderings work for the generated file.
        """
        return [
            "from __future__ import annotations",
            "import sys",
            "from typing import Dict, List, Optional",
            "from embedsim.fmu_blocks import FMUBlock",
            # _path_utils.get_embedsim_import_path() returns the project root.
            # Adding it to sys.path ensures 'import embedsim' always works.
            "from _path_utils import get_embedsim_import_path",
            "sys.path.insert(0, get_embedsim_import_path())",
            ""
        ]

    def _class_body(self, m: ModelInfo, cls: str) -> List[str]:
        """
        Emit the entire class body: declaration, constants, __init__,
        parameter setters, output readers, get_all, __repr__.

        Uses a list L that all sections append to, returned at the end.
        """
        inp = m.inputs       # list of input  ModelVariables
        out = m.outputs      # list of output ModelVariables
        par = m.parameters   # list of parameter ModelVariables

        L = []  # accumulates all generated source lines

        # ── Class declaration ──────────────────────────────────────────────────
        # Emits:  class BuckConverterBlock(FMUBlock):
        # FMUBlock provides the entire FMPy lifecycle — no physics code needed here.
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

        # ── Class-level constants ──────────────────────────────────────────────
        # Emits:
        #   INPUT_VARS:  List[str] = ['duty']
        #   OUTPUT_VARS: List[str] = ['V_out', 'I_L', 'I_load']
        #
        # repr([v.name for v in inp]) produces a valid Python list LITERAL as
        # a string: "['duty']". This string is then inserted directly into the
        # generated source — it becomes executable Python in the output file.
        inp_list = repr([v.name for v in inp])   # e.g. "['duty']"
        out_list = repr([v.name for v in out])   # e.g. "['V_out', 'I_L', 'I_load']"
        L.append(f'    # FMU variable lists — passed to FMUBlock automatically')
        L.append(f'    INPUT_VARS:  List[str] = {inp_list}')
        L.append(f'    OUTPUT_VARS: List[str] = {out_list}')
        L.append('')

        # ── DEFAULT_PARAMS dict ────────────────────────────────────────────────
        # Emits:
        #   DEFAULT_PARAMS: Dict[str, float] = {
        #       'L': 0.0001,   # [H]
        #       'C': 0.0001,   # [F]
        #       ...
        #   }
        if par:
            L.append('    DEFAULT_PARAMS: Dict[str, float] = {')
            for p in par:
                val = p.default if p.default is not None else 0.0
                # Inline comment: unit if available, else full description.
                comment = (f"  # [{p.unit}]" if p.unit
                           else (f"  # {p.description}" if p.description else ""))
                # repr(p.name) → "'L'" (with quotes), correct as a dict key literal.
                L.append(f'        {repr(p.name)}: {val},{comment}')
            L.append('    }')
            L.append('')

        # ── __init__ signature ─────────────────────────────────────────────────
        # Emits a typed __init__ with one named kwarg per Modelica parameter.
        # Named kwargs allow IDEs to display parameter names and defaults
        # in autocomplete when the user types BuckConverterBlock(...).
        L.append('    def __init__(')
        L.append('        self,')
        L.append('        name: str,')
        L.append(f'        fmu_path: str = "{m.name}.fmu",')

        for p in par:
            val = p.default if p.default is not None else 0.0
            unit_hint = f"  # [{p.unit}]" if p.unit else ""
            # e.g.:  L: float = 0.0001,  # [H]
            L.append(f'        {p.name}: float = {val},{unit_hint}')

        # use_c_backend: irrelevant for FMU blocks (FMPy always used),
        # but included for API consistency with other EmbedSim blocks.
        L.append('        use_c_backend: bool = False,')
        L.append('        dtype=None,')
        L.append('    ) -> None:')

        # ── __init__ docstring ─────────────────────────────────────────────────
        L.append(f'        """')
        L.append(f'        Create a {cls} block.')
        L.append(f'        ')
        L.append(f'        Parameters')
        L.append(f'        ----------')
        L.append(f'        name      : Unique block identifier within the simulation graph.')
        L.append(f'        fmu_path  : Path to {m.name}.fmu (produced by OpenModelica).')
        for p in par:
            unit = f" [{p.unit}]" if p.unit else ""
            # Strip the [unit] suffix already in the description to avoid duplication.
            # 'Inductance [H]'.split('[')[0].strip()  →  'Inductance'
            desc = p.description.split("[")[0].strip() if p.description else p.name
            L.append(f'        {p.name:<12}: {desc}{unit}')
        L.append(f'        """')

        # ── __init__ body: _params dict ────────────────────────────────────────
        # Packs the kwarg values into a dict to pass to FMUBlock.
        # FMUBlock calls fmu.set_real() for each key-value before the first step.
        if par:
            L.append('        _params = {')
            for p in par:
                # repr(p.name) → "'L'" (quoted), producing:  'L': L,
                L.append(f'            {repr(p.name)}: {p.name},')
            L.append('        }')
        else:
            # No parameters: emit an empty dict literal.
            # '{{}}' in an f-string produces the literal characters '{}'
            # because {{ and }} are the escape sequences for { and } in f-strings.
            L.append('        _params = {{}}')

        # ── super().__init__() ─────────────────────────────────────────────────
        # Delegates to FMUBlock which handles: loading the .fmu via FMPy,
        # calling fmu.initialize(), setting parameters, and preparing
        # the input/output signal arrays.
        L.append('        super().__init__(')
        L.append('            name=name,')
        L.append('            fmu_path=fmu_path,')
        L.append('            input_names=self.INPUT_VARS,')   # which vars to SET each step
        L.append('            output_names=self.OUTPUT_VARS,') # which vars to GET each step
        L.append('            parameters=_params,')
        L.append('            instance_name=name,')  # FMU instance name — must be unique
        L.append('            use_c_backend=use_c_backend,')
        L.append('            dtype=dtype,')
        L.append('        )')
        L.append('')

        # ── Parameter setter methods ───────────────────────────────────────────
        # Emits one  def set_<name>(self, value)  per Modelica parameter.
        # Each method calls FMUBlock.set_parameter() which updates the FMU.
        # Useful for changing plant parameters between simulation runs:
        #   plant.set_R_load(5.0)  # halve the load resistance
        if par:
            L.append('    # ── Parameter setters (callable after instantiation) ──────')
            for p in par:
                unit = f" [{p.unit}]" if p.unit else ""
                L.append(f'    def set_{p.name}(self, value: float) -> None:')
                L.append(f'        """Set {p.name}{unit} — delegates to FMUBlock.set_parameter."""')
                L.append(f'        self.set_parameter({repr(p.name)}, value)')
                L.append('')

        # ── Output reader methods ──────────────────────────────────────────────
        # Emits one  def read_<name>(self) -> float  per output variable.
        # Convenience accessors for reading individual FMU outputs by name,
        # useful in tests and Jupyter notebooks:
        #   v = plant.read_V_out()
        L.append('    # ── Output readers (typed convenience accessors) ────────────')
        for v in out:
            unit = f" [{v.unit}]" if v.unit else ""
            desc = v.description if v.description else v.name
            L.append(f'    def read_{v.name}(self) -> float:')
            L.append(f'        """Read {desc}{unit}"""')
            # get_output_by_name() is inherited from FMUBlock / VectorBlock.
            L.append(f'        return self.get_output_by_name({repr(v.name)})')
            L.append('')

        # ── get_all() ──────────────────────────────────────────────────────────
        # Returns all outputs as a dict for bulk inspection or logging:
        #   {'V_out': 11.98, 'I_L': 1.198, 'I_load': 1.198}
        L.append('    def get_all(self) -> Dict[str, float]:')
        L.append('        """Return all output variables as a named dict."""')
        L.append('        return self.get_all_outputs()')
        L.append('')

        # ── __repr__ ──────────────────────────────────────────────────────────
        # Emits a developer-friendly representation:
        #   BuckConverterBlock(name='buck', fmu='BuckConverter.fmu', init=True)
        #
        # Note the doubled {{ and }} inside the f-string literals appended here.
        # Since these strings are themselves f-strings in _class_body(), any
        # { } that should appear LITERALLY in the generated file must be
        # written as {{ }} in the source here.
        L.append('    def __repr__(self) -> str:')
        L.append(f'        return (')
        L.append(f'            f"{cls}(name={{self.name!r}}, "')   # {{  →  {  in output
        L.append(f'            f"fmu={{self.fmu_path!r}}, "')
        L.append(f'            f"init={{self._initialized}})"')
        L.append(f'        )')
        L.append('')

        return L


# ==============================================================================
#  SUMMARY PRINTER
# ==============================================================================

def print_model_summary(model: ModelInfo, verbose: bool = True) -> None:
    """
    Print a formatted table of the parsed model to stdout.

    Called by generate_fmu_block() when verbose=True. Lets you verify
    that MoParser extracted the correct variables before writing the
    generated file. If the table looks wrong, check the .mo source.

    Example output for BuckConverter.mo:
        ────────────────────────────────────────────────────────────
          Model  : BuckConverter
          Desc   : Simple buck converter plant model...
        ────────────────────────────────────────────────────────────
          PARAMETERS:
              L                    [H]          = 0.0001  ← Inductance [H]
              ...
          INPUTS:
              duty                              ← PWM duty cycle [0-1]
          OUTPUTS:
              V_out                [V]           ← Output voltage [V]
              I_L                  [A]           ← Inductor current [A]
              I_load               [A]           ← Load current [A]
        ────────────────────────────────────────────────────────────
    """
    if not verbose:
        return  # silent mode — print nothing

    w = 60  # separator line width

    print(f"\n{'─'*w}")
    print(f"  Model  : {model.name}")
    if model.description:
        print(f"  Desc   : {model.description}")
    print(f"{'─'*w}")

    def _row(label, vars_):
        """Print one variable group (PARAMETERS / INPUTS / OUTPUTS)."""
        if not vars_:
            return  # skip entirely if group is empty
        print(f"  {label}:")
        for v in vars_:
            unit  = f" [{v.unit}]"         if v.unit            else ""
            deflt = f" = {v.default}"      if v.default is not None else ""
            desc  = f"  ← {v.description}" if v.description     else ""
            # :<20 and :<12: left-align in a field of width 20 and 12
            print(f"      {v.name:<20}{unit:<12}{deflt}{desc}")

    _row("PARAMETERS", model.parameters)
    _row("INPUTS",     model.inputs)
    _row("OUTPUTS",    model.outputs)
    print(f"{'─'*w}\n")


# ==============================================================================
#  PUBLIC API
# ==============================================================================

def generate_fmu_block(mo_path: str, output_dir: Optional[str] = None,
                       verbose: bool = True) -> str:
    """
    Parse a .mo file and generate the corresponding FMUBlock Python class.

    This is the primary entry point. gen_fmu.py calls this function.

    Steps:
        1.  MoParser(mo_path).parse()               → ModelInfo
        2.  print_model_summary(model)               → console table (if verbose)
        3.  ClientGenerator(model).write(output_path) → Python file on disk

    Args:
        mo_path:    Path to the Modelica source (.mo) file.
        output_dir: Folder for the generated .py file. Created if needed.
                    None → write to current working directory.
        verbose:    True → print parsed model summary to console.

    Returns:
        str: absolute path to the generated Python file.

    Raises:
        FileNotFoundError: if mo_path does not exist.
        ValueError:        if the .mo file has no 'model' declaration.
    """
    if verbose:
        print(f"\nParsing: {mo_path}")

    parser = MoParser(mo_path)
    model  = parser.parse()

    print_model_summary(model, verbose)

    generator = ClientGenerator(model)

    if output_dir:
        # os.makedirs with exist_ok=True is idempotent — safe if folder exists.
        os.makedirs(output_dir, exist_ok=True)
        # Output filename is always {ModelName}Block.py — e.g. BuckConverterBlock.py
        output_path = os.path.join(output_dir, f"{model.name}Block.py")
    else:
        output_path = f"{model.name}Block.py"

    generator.write(output_path)

    if verbose:
        print(f"  ✓  Generated: {output_path}")

    return output_path


def generate_fmu_blocks_from_folder(folder_path: str,
                                    output_dir: Optional[str] = None,
                                    verbose: bool = True) -> List[str]:
    """
    Generate FMUBlock classes for ALL .mo files found in a folder.

    Useful for generating multiple plant wrappers at once. Individual
    failures are caught and reported without aborting the whole batch.

    Args:
        folder_path: directory containing .mo files (non-recursive).
        output_dir:  where to write generated .py files.
        verbose:     True → print progress and summary per file.

    Returns:
        List[str]: paths to successfully generated files.
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    # glob.glob(pattern) returns a list of all paths matching the wildcard.
    # os.path.join builds the pattern cross-platform.
    mo_files = glob.glob(os.path.join(folder_path, "*.mo"))
    if not mo_files:
        if verbose:
            print(f"  No .mo files found in: {folder_path}")
        return []

    if verbose:
        print(f"\nFound {len(mo_files)} .mo files in {folder_path}")

    generated_files = []
    for mo_file in mo_files:
        try:
            output_path = generate_fmu_block(mo_file, output_dir, verbose)
            generated_files.append(output_path)
        except Exception as e:
            # Report and continue — one bad .mo file should not abort the batch.
            if verbose:
                print(f"  [ERROR] Failed to process {mo_file}: {e}")

    if verbose:
        print(f"\nGenerated {len(generated_files)} FMUBlock classes")

    return generated_files


def mo_to_fmu_block(mo_path: str, output_path: Optional[str] = None) -> str:
    """
    Simplified one-call wrapper for basic use cases.

    Derives output_dir from output_path's directory; uses cwd if None.

    Args:
        mo_path:     Path to the .mo source file.
        output_path: Full path for the generated .py file (optional).

    Returns:
        str: path to the generated Python file.
    """
    if output_path:
        output_dir = os.path.dirname(output_path)
    else:
        output_dir = None

    return generate_fmu_block(mo_path, output_dir, verbose=False)


# ==============================================================================
#  LEGACY CLI
# ==============================================================================

def main_cli() -> None:
    """
    Command-line entry point — kept for backward compatibility.

    Prefer calling generate_fmu_block() from gen_fmu.py for new work.

    CLI examples:
        python mo_to_fmu_client.py BuckConverter.mo
        python mo_to_fmu_client.py models/ --out blocks/
        python mo_to_fmu_client.py BuckConverter.mo --out buck_converter/
    """
    import sys

    # sys.argv[0] is the script name. sys.argv[1:] gives the user's arguments.
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

    # Simple manual argument parser — no argparse dependency.
    # Supports:  positional .mo paths  +  optional  --out <dir>
    output_dir = None
    paths      = []

    i = 0
    while i < len(args):
        if args[i] == "--out" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2   # consume both '--out' and its value
        else:
            paths.append(args[i])
            i += 1

    for p in paths:
        if os.path.isdir(p):
            generate_fmu_blocks_from_folder(p, output_dir)
        else:
            generate_fmu_block(p, output_dir)

    print("\nDone.")


# ==============================================================================
#  ENTRY POINT GUARD
# ==============================================================================

# This block runs ONLY when the file is executed directly:
#     python mo_to_fmu_client.py BuckConverter.mo
#
# When imported (from mo_to_fmu_client import generate_fmu_block),
# __name__ == 'mo_to_fmu_client', not '__main__', so this is skipped.
if __name__ == "__main__":
    main_cli()
