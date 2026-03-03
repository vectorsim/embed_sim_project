from generated_code.three_phase_processor_simblock import ThreePhaseProcessorSimBlock

# Use C backend
processor = ThreePhaseProcessorSimBlock("proc", use_c_backend=True)

# Use Python backend
processor_py = ThreePhaseProcessorSimBlock("proc", use_c_backend=False)