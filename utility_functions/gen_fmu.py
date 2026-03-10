import sys
from pathlib import Path
sys.path.append(str(Path(r'C:\EmbedSimProject\utility_functions')))
from mo_to_fmu_client import generate_fmu_block

mo_path = r'C:\EmbedSimProject\buck_converter\modelica\BuckConverter.mo'
output_dir = r'C:\EmbedSimProject\buck_converter'

output_file = generate_fmu_block(mo_path, output_dir, verbose=True)
print(f"Generated: {output_file}")