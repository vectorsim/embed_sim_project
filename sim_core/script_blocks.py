"""
script_blocks.py
================

Script-based blocks that allow users to write custom Python code for signal processing.
Supports both direct Python execution and C code generation via AST analysis.

This module provides blocks where users can write Python expressions/scripts that:
1. Execute directly during simulation (development/prototyping)
2. Generate optimized C code from AST for production deployment
3. Automatically compile and link C code for performance

Classes:
    ScriptBlock: Executes user-defined Python script with optional C code generation
    
Author: Vector Simulation Framework
Version: 1.0.0
"""

import ast
import numpy as np
import textwrap
import subprocess
import ctypes
import os
import tempfile
from typing import List, Optional, Dict, Any, Set
from pathlib import Path

# Import from the framework
import sys
import os

# Try direct import first
try:
    from .core_blocks import VectorBlock, VectorSignal, validate_inputs_exist
except ImportError:
    # Fall back to loading from uploads directory
    import importlib.util
    spec = importlib.util.spec_from_file_location("core_blocks", "/mnt/user-data/uploads/core_blocks.py")
    core_blocks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core_blocks)
    VectorBlock = core_blocks.VectorBlock
    VectorSignal = core_blocks.VectorSignal
    validate_inputs_exist = core_blocks.validate_inputs_exist


# =========================
# Script Block with AST-based Code Generation
# =========================

class ScriptBlock(VectorBlock):
    """
    A block that executes user-defined Python code for signal processing.
    
    ScriptBlock allows users to write custom Python expressions/scripts that process
    input signals and produce output signals. The same Python code can be:
    1. Executed directly in Python (fast prototyping, easy debugging)
    2. Analyzed via AST and translated to C code (optimized production deployment)
    
    The script has access to:
    - Input signals via 'u' (list of numpy arrays) or 'u0', 'u1', etc.
    - Time via 't' variable
    - Time step via 'dt' variable
    - Numpy as 'np'
    - User-defined parameters as variables
    - Must assign result to 'output' variable
    
    Execution Modes:
    - 'python': Direct Python execution (default, good for development)
    - 'c': Compiled C code execution (faster, good for production)
    
    Attributes:
        script (str): User-defined Python code
        parameters (Dict[str, Any]): User parameters accessible in script
        mode (str): Execution mode ('python' or 'c')
        output_dim (int): Expected dimension of output vector
        
    Example:
        >>> # Simple gain and offset
        >>> script = '''
        ... # Apply gain and add offset
        ... y = u[0] * gain + offset
        ... output = y
        ... '''
        >>> block = ScriptBlock("custom", script, 
        ...                     parameters={'gain': 2.0, 'offset': 1.0},
        ...                     output_dim=3)
        >>> 
        >>> # More complex processing
        >>> script2 = '''
        ... # Clarke transformation (ABC to αβ)
        ... alpha = (2*u0[0] - u0[1] - u0[2]) / 3
        ... beta = (u0[1] - u0[2]) / np.sqrt(3)
        ... output = np.array([alpha, beta, 0.0])
        ... '''
        >>> block2 = ScriptBlock("clarke", script2, output_dim=3)
    """
    
    def __init__(self, name: str, script: str, 
                 parameters: Optional[Dict[str, Any]] = None,
                 output_dim: int = 3,
                 mode: str = 'python') -> None:
        """
        Initialize a ScriptBlock.
        
        Args:
            name: Unique identifier for this block
            script: Python code to execute. Must assign result to 'output'.
                   Can access: u (list of inputs), u0/u1/... (individual inputs),
                   t (time), dt (time step), np (numpy), and parameters.
            parameters: Dictionary of user-defined parameters accessible in script.
                       Example: {'gain': 2.0, 'cutoff_freq': 50.0}
            output_dim: Expected dimension of output vector (for validation)
            mode: Execution mode - 'python' (direct) or 'c' (compiled)
        
        Raises:
            ValueError: If script is invalid or mode is unknown
            
        Example:
            >>> # PID controller
            >>> pid_script = '''
            ... error = u[0]  # setpoint - measurement
            ... integral = integral + error * dt
            ... derivative = (error - prev_error) / dt
            ... output = Kp * error + Ki * integral + Kd * derivative
            ... prev_error = error
            ... '''
            >>> pid = ScriptBlock("pid", pid_script,
            ...                   parameters={'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.01},
            ...                   output_dim=3)
        """
        super().__init__(name)
        self.script: str = textwrap.dedent(script).strip()
        self.parameters: Dict[str, Any] = parameters or {}
        self.output_dim: int = output_dim
        self.mode: str = mode
        
        # For stateful scripts (variables that persist between calls)
        self.script_locals: Dict[str, Any] = {}
        
        # Parse and validate the script
        self._ast_tree: Optional[ast.Module] = None
        self._validate_script()
        
        # C code generation artifacts
        self._c_code: Optional[str] = None
        self._compiled_function = None
        self._lib_path: Optional[str] = None
        
        # If C mode requested, generate and compile immediately
        if self.mode == 'c':
            self.compile_to_c()
    
    def _validate_script(self) -> None:
        """
        Validate the user script by parsing it into an AST.
        
        Checks for:
        - Valid Python syntax
        - Assignment to 'output' variable
        - No dangerous operations (imports, file I/O, etc.)
        
        Raises:
            SyntaxError: If script has invalid Python syntax
            ValueError: If script doesn't assign to 'output' or uses forbidden operations
        """
        try:
            self._ast_tree = ast.parse(self.script)
        except SyntaxError as e:
            raise SyntaxError(f"Script syntax error in block '{self.name}': {e}")
        
        # Check for output assignment
        has_output = False
        for node in ast.walk(self._ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'output':
                        has_output = True
                        break
        
        if not has_output:
            raise ValueError(
                f"Script in block '{self.name}' must assign result to 'output' variable"
            )
        
        # Check for forbidden operations (security)
        forbidden = self._check_forbidden_operations()
        if forbidden:
            raise ValueError(
                f"Script in block '{self.name}' contains forbidden operations: {forbidden}"
            )
    
    def _check_forbidden_operations(self) -> List[str]:
        """
        Check AST for potentially dangerous operations.
        
        Returns:
            List of forbidden operation names found, empty if safe
        """
        forbidden_found = []
        
        for node in ast.walk(self._ast_tree):
            # Check for imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                forbidden_found.append("import statement")
            
            # Check for file operations
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['open', 'eval', 'exec', 'compile', '__import__']:
                        forbidden_found.append(f"forbidden function: {node.func.id}")
        
        return forbidden_found
    
    def compute(self, t: float, dt: float, 
                input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Execute the script to compute output signal.
        
        Args:
            t: Current simulation time (seconds)
            dt: Time step duration (seconds)
            input_values: List of input signals from connected blocks
        
        Returns:
            VectorSignal: Computed output from script
            
        Raises:
            RuntimeError: If script execution fails
            ValueError: If output dimension doesn't match expected
        """
        if self.mode == 'python':
            return self._compute_python(t, dt, input_values)
        elif self.mode == 'c':
            return self._compute_c(t, dt, input_values)
        else:
            raise ValueError(f"Unknown execution mode: {self.mode}")
    
    def _compute_python(self, t: float, dt: float,
                       input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Execute script in Python mode (direct execution).
        
        Args:
            t: Current simulation time
            dt: Time step duration
            input_values: Input signals
            
        Returns:
            VectorSignal: Output from script execution
        """
        # Prepare execution context
        context = {
            'np': np,
            't': t,
            'dt': dt,
            'output': None,  # Must be assigned by script
        }
        
        # Add user parameters
        context.update(self.parameters)
        
        # Add persistent local variables from previous executions
        context.update(self.script_locals)
        
        # Add input signals
        if input_values:
            # Provide as list: u[0], u[1], ...
            context['u'] = [sig.value for sig in input_values]
            
            # Also provide as individual variables: u0, u1, u2, ...
            for i, sig in enumerate(input_values):
                context[f'u{i}'] = sig.value
        else:
            context['u'] = []
        
        # Execute the script
        try:
            exec(self.script, context)
        except Exception as e:
            raise RuntimeError(
                f"Script execution error in block '{self.name}': {e}"
            )
        
        # Extract output
        output_value = context.get('output')
        if output_value is None:
            raise RuntimeError(
                f"Script in block '{self.name}' did not assign 'output' variable"
            )
        
        # Convert output to numpy array if needed
        if not isinstance(output_value, np.ndarray):
            output_value = np.array(output_value, dtype=float)
        
        # Validate output dimension
        if len(output_value) != self.output_dim:
            raise ValueError(
                f"Script in block '{self.name}' produced output dimension "
                f"{len(output_value)}, expected {self.output_dim}"
            )
        
        # Save local variables for next execution (for stateful scripts)
        # Exclude built-ins, inputs, and output
        exclude_keys = {'np', 't', 'dt', 'u', 'output', '__builtins__'}
        exclude_keys.update(f'u{i}' for i in range(10))  # u0..u9
        exclude_keys.update(self.parameters.keys())
        
        for key, value in context.items():
            if key not in exclude_keys and not key.startswith('_'):
                self.script_locals[key] = value
        
        self.output = VectorSignal(output_value, self.name)
        return self.output
    
    def _compute_c(self, t: float, dt: float,
                   input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Execute script in C mode (compiled code execution).
        
        Args:
            t: Current simulation time
            dt: Time step duration  
            input_values: Input signals
            
        Returns:
            VectorSignal: Output from compiled C function
        """
        if self._compiled_function is None:
            raise RuntimeError(
                f"Block '{self.name}': C code not compiled. Call compile_to_c() first."
            )
        
        # Prepare input array
        num_inputs = len(input_values) if input_values else 0
        if num_inputs > 0:
            # Flatten all input vectors into a single array
            input_sizes = [len(sig.value) for sig in input_values]
            total_input_size = sum(input_sizes)
            input_array = np.zeros(total_input_size, dtype=np.float64)
            
            offset = 0
            for sig in input_values:
                size = len(sig.value)
                input_array[offset:offset+size] = sig.value
                offset += size
        else:
            input_array = np.array([], dtype=np.float64)
            total_input_size = 0
        
        # Prepare output array
        output_array = np.zeros(self.output_dim, dtype=np.float64)
        
        # Prepare parameters array (convert dict to array in consistent order)
        param_names = sorted(self.parameters.keys())
        param_array = np.array([self.parameters[k] for k in param_names], dtype=np.float64)
        
        # Call compiled C function
        # Signature: void compute(double* input, int n_input, double* output, int n_output,
        #                         double* params, int n_params, double t, double dt)
        self._compiled_function(
            input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(total_input_size),
            output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self.output_dim),
            param_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(len(param_array)),
            ctypes.c_double(t),
            ctypes.c_double(dt)
        )
        
        self.output = VectorSignal(output_array, self.name)
        return self.output
    
    def generate_c_code(self) -> str:
        """
        Generate C code from the Python script AST.
        
        This method analyzes the Python AST and translates supported operations
        to equivalent C code. Supports:
        - Arithmetic operations (+, -, *, /, **)
        - Numpy functions (sin, cos, sqrt, etc.)
        - Array indexing and operations
        - Variable assignments
        
        Returns:
            str: Generated C code
            
        Raises:
            NotImplementedError: If script contains unsupported Python operations
            
        Example:
            >>> script = "output = u[0] * 2.0 + np.sin(t)"
            >>> block = ScriptBlock("test", script, output_dim=3)
            >>> c_code = block.generate_c_code()
            >>> print(c_code)  # Shows generated C function
        """
        if self._c_code is not None:
            return self._c_code
        
        # Initialize C code components
        c_lines = []
        
        # Generate function signature
        c_lines.append("void compute(double* input, int n_input, double* output, int n_output,")
        c_lines.append("             double* params, int n_params, double t, double dt) {")
        
        # Add local variable declarations
        declared_vars = set()
        for node in ast.walk(self._ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id != 'output':
                        if target.id not in declared_vars:
                            c_lines.append(f"    double {target.id};")
                            declared_vars.add(target.id)
        
        # Add parameter unpacking
        param_names = sorted(self.parameters.keys())
        for i, param_name in enumerate(param_names):
            c_lines.append(f"    double {param_name} = params[{i}];")
        
        c_lines.append("")
        
        # Translate Python AST to C
        c_body = self._translate_ast_to_c(self._ast_tree)
        c_lines.extend(["    " + line for line in c_body])
        
        c_lines.append("}")
        
        self._c_code = "\n".join(c_lines)
        return self._c_code
    
    def _translate_ast_to_c(self, tree: ast.Module) -> List[str]:
        """
        Translate Python AST nodes to C code lines.
        
        Args:
            tree: Python AST module
            
        Returns:
            List of C code lines
        """
        c_lines = []
        
        for node in tree.body:
            if isinstance(node, ast.Assign):
                c_line = self._translate_assign(node)
                c_lines.append(c_line)
            elif isinstance(node, ast.Expr):
                # Expression statement (rare in this context)
                pass
            else:
                raise NotImplementedError(
                    f"AST node type {type(node).__name__} not supported for C generation"
                )
        
        return c_lines
    
    def _translate_assign(self, node: ast.Assign) -> str:
        """
        Translate an assignment node to C code.
        
        Args:
            node: AST Assign node
            
        Returns:
            C code string for the assignment
        """
        # Get target variable name
        if len(node.targets) != 1:
            raise NotImplementedError("Multiple assignment targets not supported")
        
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise NotImplementedError("Only simple variable assignment supported")
        
        target_name = target.id
        
        # Translate the value expression
        value_c = self._translate_expr(node.value)
        
        # Special handling for output array assignment
        if target_name == 'output':
            # Assume output is a numpy array - need to copy elements
            return f"memcpy(output, {value_c}, n_output * sizeof(double));"
        else:
            return f"{target_name} = {value_c};"
    
    def _translate_expr(self, node: ast.expr) -> str:
        """
        Translate an expression node to C code.
        
        Args:
            node: AST expression node
            
        Returns:
            C code string for the expression
        """
        if isinstance(node, ast.BinOp):
            return self._translate_binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self._translate_unaryop(node)
        elif isinstance(node, ast.Call):
            return self._translate_call(node)
        elif isinstance(node, ast.Subscript):
            return self._translate_subscript(node)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Num):  # For older Python versions
            return str(node.n)
        else:
            raise NotImplementedError(
                f"Expression type {type(node).__name__} not supported for C generation"
            )
    
    def _translate_binop(self, node: ast.BinOp) -> str:
        """Translate binary operation to C."""
        left = self._translate_expr(node.left)
        right = self._translate_expr(node.right)
        
        op_map = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Mod: '%',
        }
        
        if type(node.op) in op_map:
            return f"({left} {op_map[type(node.op)]} {right})"
        elif isinstance(node.op, ast.Pow):
            return f"pow({left}, {right})"
        else:
            raise NotImplementedError(f"Binary operator {type(node.op).__name__} not supported")
    
    def _translate_unaryop(self, node: ast.UnaryOp) -> str:
        """Translate unary operation to C."""
        operand = self._translate_expr(node.operand)
        
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        elif isinstance(node.op, ast.UAdd):
            return f"(+{operand})"
        else:
            raise NotImplementedError(f"Unary operator {type(node.op).__name__} not supported")
    
    def _translate_call(self, node: ast.Call) -> str:
        """Translate function call to C."""
        # Handle numpy functions
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'np':
                func_name = node.func.attr
                args = [self._translate_expr(arg) for arg in node.args]
                
                # Map numpy functions to C math.h functions
                np_to_c = {
                    'sin': 'sin',
                    'cos': 'cos',
                    'tan': 'tan',
                    'sqrt': 'sqrt',
                    'exp': 'exp',
                    'log': 'log',
                    'abs': 'fabs',
                    'array': 'ARRAY',  # Special handling needed
                }
                
                if func_name in np_to_c:
                    c_func = np_to_c[func_name]
                    if c_func == 'ARRAY':
                        # Array constructor - simplified (assumes small literal arrays)
                        return f"{{{', '.join(args)}}}"
                    return f"{c_func}({', '.join(args)})"
        
        raise NotImplementedError(f"Function call not supported for C generation: {ast.dump(node)}")
    
    def _translate_subscript(self, node: ast.Subscript) -> str:
        """Translate array subscript to C."""
        # Handle u[0], u[1], etc.
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            
            if isinstance(node.slice, ast.Constant):
                index = node.slice.value
            elif isinstance(node.slice, ast.Index):  # Older Python
                index = node.slice.value.n
            else:
                raise NotImplementedError("Complex array indexing not supported")
            
            if var_name == 'u':
                # This is input array access - need to map to flat input array
                return f"input[{index}]"  # Simplified - assumes single input vector
            else:
                return f"{var_name}[{index}]"
        
        raise NotImplementedError(f"Subscript not supported: {ast.dump(node)}")
    
    def compile_to_c(self, compiler: str = 'gcc', 
                     optimization: str = '-O2') -> None:
        """
        Compile the generated C code to a shared library and load it.
        
        Args:
            compiler: C compiler to use ('gcc' or 'clang'). Default: 'gcc'
            optimization: Optimization flag. Default: '-O2'
                         Options: '-O0' (none), '-O1', '-O2', '-O3' (aggressive)
        
        Raises:
            RuntimeError: If compilation fails
            FileNotFoundError: If compiler not found
            
        Example:
            >>> block = ScriptBlock("test", "output = u[0] * 2.0", output_dim=3)
            >>> block.compile_to_c()  # Generates, compiles, and loads C code
            >>> block.mode = 'c'  # Switch to C execution mode
        """
        # Generate C code if not already done
        if self._c_code is None:
            self.generate_c_code()
        
        # Create temporary directory for compilation
        temp_dir = tempfile.mkdtemp(prefix=f'script_block_{self.name}_')
        
        # Write C source file
        c_file = Path(temp_dir) / f'{self.name}.c'
        with open(c_file, 'w') as f:
            f.write("#include <math.h>\n")
            f.write("#include <string.h>\n\n")
            f.write(self._c_code)
        
        # Compile to shared library
        lib_file = Path(temp_dir) / f'{self.name}.so'
        
        compile_cmd = [
            compiler,
            optimization,
            '-shared',
            '-fPIC',
            '-o', str(lib_file),
            str(c_file),
            '-lm'  # Link math library
        ]
        
        try:
            result = subprocess.run(compile_cmd, 
                                  capture_output=True, 
                                  text=True,
                                  check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"C compilation failed for block '{self.name}':\n"
                f"Command: {' '.join(compile_cmd)}\n"
                f"Error: {e.stderr}"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Compiler '{compiler}' not found. Install gcc or clang."
            )
        
        # Load the shared library
        lib = ctypes.CDLL(str(lib_file))
        
        # Set function signature
        lib.compute.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # input
            ctypes.c_int,                      # n_input
            ctypes.POINTER(ctypes.c_double),  # output
            ctypes.c_int,                      # n_output
            ctypes.POINTER(ctypes.c_double),  # params
            ctypes.c_int,                      # n_params
            ctypes.c_double,                   # t
            ctypes.c_double,                   # dt
        ]
        lib.compute.restype = None
        
        self._compiled_function = lib.compute
        self._lib_path = str(lib_file)
        
        print(f"✓ Block '{self.name}' compiled successfully")
        print(f"  Library: {lib_file}")
    
    def show_c_code(self) -> None:
        """
        Print the generated C code to console.
        
        Useful for debugging and understanding what C code will be generated.
        
        Example:
            >>> block = ScriptBlock("test", "output = u[0] * 2.0 + 1.0", output_dim=3)
            >>> block.show_c_code()
        """
        if self._c_code is None:
            self.generate_c_code()
        
        print(f"\n{'='*60}")
        print(f"Generated C Code for Block: {self.name}")
        print(f"{'='*60}")
        print(self._c_code)
        print(f"{'='*60}\n")
    
    def reset(self) -> None:
        """
        Reset the block state.
        
        Clears output signals and script local variables (for stateful scripts).
        """
        super().reset()
        self.script_locals.clear()
    
    def switch_mode(self, mode: str) -> None:
        """
        Switch execution mode between Python and C.
        
        Args:
            mode: 'python' for direct execution, 'c' for compiled execution
            
        Raises:
            ValueError: If mode is invalid or C code not compiled
            
        Example:
            >>> block = ScriptBlock("test", "output = u[0] * 2.0", output_dim=3)
            >>> block.switch_mode('c')  # Compile and switch to C mode
        """
        if mode not in ['python', 'c']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'python' or 'c'")
        
        if mode == 'c' and self._compiled_function is None:
            print(f"Compiling block '{self.name}' to C...")
            self.compile_to_c()
        
        self.mode = mode
        print(f"✓ Block '{self.name}' switched to {mode} mode")


# =========================
# Module Metadata
# =========================

__all__ = [
    'ScriptBlock',
]

__version__ = '1.0.0'
__author__ = 'Vector Simulation Framework'
