"""
gen_fmu.py
==========

PURPOSE
-------
One-shot utility script — run this manually, once, whenever BuckConverter.mo
changes. It does exactly two things:

    1. Calls mo_to_fmu_client.generate_fmu_block() which:
          a. Parses BuckConverter.mo with MoParser
          b. Generates BuckConverterBlock.py with ClientGenerator

    2. Prints the path to the generated file.

THIS SCRIPT DOES NOT:
  - Compile the FMU binary (.fmu / .dll)  — that is OpenModelica's job.
  - Run any simulation.
  - Modify BuckConverter.mo.

WHEN TO RE-RUN
--------------
Re-run gen_fmu.py whenever:
  • BuckConverter.mo gains a new parameter, input, or output variable.
  • A variable is renamed or removed in the .mo file.
  • You want to reset BuckConverterBlock.py to a clean generated state
    (useful if BuckConverterBlock.py was manually edited by mistake —
    the "DO NOT EDIT" notice in its header is there for this reason).

HOW TO RUN
----------
    cd C:\EmbedSimProject
    python buck_converter\gen_fmu.py

Expected console output:
    Parsing: C:\EmbedSimProject\buck_converter\modelica\BuckConverter.mo
    ────────────────────────────────────────────────────────────
      Model  : BuckConverter
      Desc   : Simple buck converter plant model ...
    ────────────────────────────────────────────────────────────
      PARAMETERS:
          L                    [H]          = 0.0001  ← Inductance [H]
          C                    [F]          = 0.0001  ← Capacitance [F]
          R_load               [Ω]          = 10.0    ← Load resistance [Ω]
          V_in                 [V]          = 24.0    ← Input voltage [V]
          f_sw                 [Hz]         = 100000.0 ← Switching frequency [Hz]
      INPUTS:
          duty                              ← PWM duty cycle [0-1]
      OUTPUTS:
          V_out                [V]          ← Output voltage [V]
          I_L                  [A]          ← Inductor current [A]
          I_load               [A]          ← Load current [A]
    ────────────────────────────────────────────────────────────
      ✓  Generated: C:\EmbedSimProject\buck_converter\BuckConverterBlock.py
    Generated: C:\EmbedSimProject\buck_converter\BuckConverterBlock.py

CORRECTNESS: CORRECT.
One portability note on the sys.path line — see comment below.
"""

import sys
from pathlib import Path

# ── Add utility_functions/ to sys.path ────────────────────────────────────────
#
# mo_to_fmu_client.py lives at:
#   C:\EmbedSimProject\utility_functions\mo_to_fmu_client.py
#
# It is NOT a Python package (no __init__.py in that folder), so we cannot
# import it with a plain 'import mo_to_fmu_client' unless its parent directory
# is on sys.path. sys.path is the list of directories Python searches when
# resolving imports. sys.path.append() adds our folder to the END of that list.
#
# PORTABILITY NOTE:
# The path 'C:\EmbedSimProject\utility_functions' is hardcoded.
# This works on your machine but will break if the project is moved to a
# different drive or folder. A portable alternative using _path_utils:
#
#     sys.path.insert(0, str(Path(__file__).parent))  # find _path_utils.py
#     from _path_utils import get_project_root
#     sys.path.append(str(get_project_root() / "utility_functions"))
#
# The hardcoded version is acceptable for a personal generator script that
# never runs in a CI/CD pipeline or on a different machine.
utility_path = Path(__file__).parent.parent.parent / 'utility_functions'
sys.path.append(str(utility_path))
from mo_to_fmu_client import generate_fmu_block

# Now mo_to_fmu_client is importable from the path we just added.
# generate_fmu_block() is the single API function we need:
#   parse .mo  →  build ModelInfo  →  generate Python class  →  write .py file
from mo_to_fmu_client import generate_fmu_block


# ── Paths ──────────────────────────────────────────────────────────────────────

# Source: the Modelica file describing the buck converter plant.
# gen_fmu.py reads this file; it never modifies it.
mo_path = str(Path(__file__).parent / 'PMSM_Motor.mo')

# Destination directory for the generated Python file.
# generate_fmu_block() constructs the filename as {ModelName}Block.py,
# so the output will be:
#   C:\EmbedSimProject\buck_converter\BuckConverterBlock.py
# The directory is created automatically if it does not exist.
output_dir =  str(Path(__file__).parent.parent)


# ── Run the generator ──────────────────────────────────────────────────────────
#
# verbose=True prints a summary table to the console so you can verify
# that the parser correctly identified all inputs, outputs, and parameters
# from BuckConverter.mo before committing the generated file.
#
# Internal call chain:
#   generate_fmu_block(mo_path, output_dir, verbose=True)
#     → MoParser(mo_path).parse()          ← reads & regex-parses the .mo file
#     → print_model_summary(model)         ← prints the table to console
#     → ClientGenerator(model).write(path) ← emits BuckConverterBlock.py
#     → returns the absolute output path
#
output_file = generate_fmu_block(mo_path, output_dir, verbose=True)

# Echo the result. Redundant with the "✓ Generated:" line from generate_fmu_block,
# but unambiguous when this script is called from a build pipeline or bat file.
print(f"Generated: {output_file}")
