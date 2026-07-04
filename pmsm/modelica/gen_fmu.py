r"""
gen_fmu.py
==========

PURPOSE
-------
One-shot utility script — run this manually, once, whenever PMSM_Motor.mo
changes. It does exactly two things:

    1. Parses PMSM_Motor.mo with MoParser
    2. Generates PMSM_MotorBlock.py with ClientGenerator
    3. Prints the path to the generated file

THIS SCRIPT DOES NOT:
  - Compile the FMU binary (.fmu / .dll)
  - Run any simulation
  - Modify PMSM_Motor.mo

HOW TO RUN
----------
    cd /home/epl05/EMProject/fs_electrical_machines
    python modelica/gen_fmu.py
"""

import sys
from pathlib import Path

# ── Add utility_functions/ to sys.path ─────────────────────────────
# This allows importing mo_to_fmu_client.py which is not a package.
utility_path = Path(__file__).parent.parent.parent / 'utility_functions'
sys.path.insert(0, str(utility_path))  # safer than append, ensures our module is found first

# Import the parser and generator classes
from mo_to_fmu_client import MoParser, ClientGenerator

# ── Define paths ────────────────────────────────────────────────────

# Source Modelica file
mo_path = Path(__file__).parent / 'PMSM_Motor.mo'
if not mo_path.exists():
    raise FileNotFoundError(f"Modelica file not found: {mo_path}")

# Destination directory for generated Python file
output_dir = Path(__file__).parent.parent
output_dir.mkdir(parents=True, exist_ok=True)  # ensure it exists

# ── Parse Modelica model ────────────────────────────────────────────

# Read the Modelica file text
with open(mo_path, 'r') as f:
    mo_text = f.read()

# Create a parser instance and parse the model
parser = MoParser(mo_text)
model = parser.parse()

# Print model summary (educational/debug)
print("────────────────────────────────────────────")
print(f"  Model  : {model.name}")
print(f"  Desc   : {getattr(model, 'description', 'No description')}")
print("────────────────────────────────────────────")

print("  PARAMETERS:")
for p in model.parameters:
    print(f"    {p.name:<20} = {p.default}  ← {getattr(p, 'unit', '')}")

print("  INPUTS:")
for i in model.inputs:
    print(f"    {i.name}")

print("  OUTPUTS:")
for o in model.outputs:
    print(f"    {o.name}")
print("────────────────────────────────────────────")

# ── Generate Python client ───────────────────────────────────────────

# Create a generator instance and generate the Python code
generator = ClientGenerator(model)
generated_code = generator.generate()

# Determine output file path
output_file = output_dir / f"{model.name}Block.py"

# Write the generated code to the file (overwrite if exists)
output_file.write_text(generated_code)

# Print confirmation (educational)
print(f"✓ Generated: {output_file}")
print(f"Generated: {output_file}")