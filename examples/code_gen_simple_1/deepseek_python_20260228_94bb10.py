#!/usr/bin/env python3
"""
Setup script for compiling generated C code
"""

import os
import sys
import subprocess


def compile_generated_code(project_dir):
    """Compile generated C code"""
    original_dir = os.getcwd()
    
    try:
        os.chdir(project_dir)
        
        print(f"\nCompiling in {project_dir}...")
        
        # Run setup script
        result = subprocess.run(
            [sys.executable, "setup_three_phase_processor.py", "build_ext", "--inplace"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Compilation successful!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("✗ Compilation failed:")
            print(result.stderr)
            
    finally:
        os.chdir(original_dir)


def test_c_version():
    """Test the compiled C version"""
    print("\n" + "="*70)
    print("TESTING C VERSION")
    print("="*70)
    
    try:
        # Try to import the generated module
        sys.path.insert(0, "./generated_code")
        from three_phase_processor_simblock import ThreePhaseProcessorSimBlock
        
        # Create instance with C backend
        block = ThreePhaseProcessorSimBlock("test", use_c_backend=True)
        print(f"✓ Created block: {block}")
        print("✓ C module loaded successfully!")
        
    except ImportError as e:
        print(f"✗ Failed to load C module: {e}")
        print("\nMake sure you've compiled the code first:")
        print("  cd generated_code")
        print("  python setup_three_phase_processor.py build_ext --inplace")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compile":
        compile_generated_code("./generated_code")
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test_c_version()
    else:
        print("Usage:")
        print("  python setup_codegen.py compile  - Compile generated code")
        print("  python setup_codegen.py test     - Test compiled code")