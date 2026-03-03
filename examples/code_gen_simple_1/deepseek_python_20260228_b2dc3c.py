#!/usr/bin/env python3
"""
Environment setup and verification script
"""

import sys
import subprocess
import importlib

def check_python_version():
    print(f"Python version: {sys.version}")
    return sys.version_info >= (3, 8)

def check_module(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} installed")
        return True
    except ImportError:
        print(f"✗ {module_name} not installed")
        return False

def check_compiler():
    try:
        if sys.platform == 'win32':
            result = subprocess.run(['cl', '/?'], capture_output=True, text=True)
        else:
            result = subprocess.run(['gcc', '--version'], capture_output=True, text=True)
        print(f"✓ Compiler found: {result.stdout[:50]}...")
        return True
    except FileNotFoundError:
        print("✗ Compiler not found")
        return False

def main():
    print("="*60)
    print("EMBEDSIM CODE GENERATION DEMO - ENVIRONMENT CHECK")
    print("="*60)
    
    print("\n1. Checking Python...")
    python_ok = check_python_version()
    
    print("\n2. Checking required modules...")
    modules = ['numpy', 'matplotlib', 'Cython', 'embedsim']
    modules_ok = all([check_module(m) for m in modules])
    
    print("\n3. Checking compiler...")
    compiler_ok = check_compiler()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Python:      {'✓' if python_ok else '✗'}")
    print(f"Modules:     {'✓' if modules_ok else '✗'