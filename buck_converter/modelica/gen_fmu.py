r"""
gen_fmu.py
==========

PURPOSE
-------
One-shot utility script — run this manually, once, whenever BuckConverter.mo
changes. It does exactly two things:

    1. Parses BuckConverter.mo with MoParser
    2. Generates BuckConverterBlock.py with ClientGenerator
    3. Prints the path to the generated file

THIS SCRIPT DOES NOT:
  - Compile the FMU binary (.fmu / .dll)
  - Run any simulation
  - Modify BuckConverter.mo

HOW TO RUN
----------
    cd C:\EmbedSimProject\embed_sim_project
    python buck_converter\modelica\gen_fmu.py
"""

import sys
from pathlib import Path

# ── Add utility_functions/ to sys.path ─────────────────────────────
# utility_functions/ sits three levels up from this file:
#   buck_converter/modelica/gen_fmu.py
#   buck_converter/modelica/          ← .parent
#   buck_converter/                   ← .parent.parent
#   embed_sim_project/                ← .parent.parent.parent
#   embed_sim_project/utility_functions/
utility_path = Path(__file__).resolve().parent.parent.parent / 'utility_functions'
sys.path.insert(0, str(utility_path))

from mo_to_fmu_client import MoParser, ClientGenerator

# ── Define paths ────────────────────────────────────────────────────

# Source: BuckConverter.mo sits next to this script
mo_path = Path(__file__).resolve().parent / 'BuckConverter.mo'
if not mo_path.exists():
    raise FileNotFoundError(f"Modelica file not found: {mo_path}")

# Destination: BuckConverterBlock.py goes into buck_converter/
output_dir = Path(__file__).resolve().parent.parent
output_dir.mkdir(parents=True, exist_ok=True)

# ── Parse ───────────────────────────────────────────────────────────

with open(mo_path, 'r') as f:
    mo_text = f.read()

parser = MoParser(mo_text)
model = parser.parse()

print(f"Parsing: {mo_path}")
print("────────────────────────────────────────────────────────────")
print(f"  Model  : {model.name}")
print("────────────────────────────────────────────────────────────")

print("  PARAMETERS:")
for p in model.parameters:
    print(f"    {p.name:<20} = {p.default}")

print("  INPUTS:")
for i in model.inputs:
    print(f"    {i.name}")

print("  OUTPUTS:")
for o in model.outputs:
    print(f"    {o.name}")

print("────────────────────────────────────────────────────────────")

# ── Generate ─────────────────────────────────────────────────────────

generator = ClientGenerator(model)
generated_code = generator.generate()

output_file = output_dir / f"{model.name}Block.py"
output_file.write_text(generated_code)

print(f"  ✓  Generated: {output_file}")
print(f"Generated: {output_file}")
