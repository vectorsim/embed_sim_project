# code_generator.py
# ==================
# EmbedSim code generation support:
#   SimBlockBase      - base class with Python/C backend switch
#   CodeGenStart      - marks region input boundary, introspects input signals
#   CodeGenEnd        - marks region output boundary, generates .h / .pyx / SimBlockBase stub
#   MCUTarget         - target metadata constants
#
# Key workflow:
#   1. Build your simulation in Python (use_c_backend=False)
#   2. Run once so signals are computed and sizes are known
#   3. Call cg_end.generate_pyx_stub(cg_start, "my_block") to get:
#        my_block.h                 - C header the C developer implements
#        my_block_wrapper.pyx       - Cython wrapper (compile once)
#        my_block_simblock.py       - SimBlockBase subclass, use_c_backend=True ready

from typing import List, Optional, Union
from pathlib import Path
import numpy as np
from .core_blocks import VectorBlock, VectorSignal


# =============================================================================
# SimBlockBase
# =============================================================================

# =============================================================================
# SimBlockBase
# =============================================================================
# SimBlockBase is now a thin alias for VectorBlock.
#
# The Python/C dual-backend (use_c_backend, compute_py, compute_c,
# _load_wrapper, switch_backend) is fully implemented in VectorBlock itself,
# so SimBlockBase adds nothing new — it exists only for backward compatibility
# with existing generated SimBlock subclasses.
#
# New blocks should subclass VectorBlock directly.
# =============================================================================

class SimBlockBase(VectorBlock):
    """
    Backward-compatible alias for VectorBlock.

    The Python/C dual-backend is now built into VectorBlock:
        compute()     → routes to compute_py() or compute_c()
        compute_py()  → pure Python path (override in subclass)
        compute_c()   → C/Cython path (override in generated stub)
        _load_wrapper() → loads compiled .pyx (override in generated stub)
        switch_backend() → flip at runtime

    Generated stubs from CodeGenEnd.generate_pyx_stub() inherit from this
    class and therefore from VectorBlock automatically.
    """

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None):
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)


# =============================================================================
# Internal helpers
# =============================================================================

def _sanitize(name: str) -> str:
    """Make a C-safe identifier from a block name."""
    return name.replace(" ", "_").replace("-", "_").replace(".", "_")


def _iter_signals(block_list, label_prefix="signal"):
    """
    Yield (sanitized_name, size) for each block in block_list.
    Size is taken from block.output if available, else vector_size, else 1.
    """
    for i, blk in enumerate(block_list):
        name = _sanitize(blk.name) if blk.name else f"{label_prefix}_{i}"
        if blk.output is not None:
            size = len(blk.output.value)
        else:
            size = getattr(blk, 'vector_size', 1)
        yield name, size


# =============================================================================
# CodeGenStart
# =============================================================================

class CodeGenStart(SimBlockBase):
    """
    Transparent passthrough marking the INPUT boundary of a C code-gen region.

        source >> cg_start >> [your blocks] >> cg_end >> sink

    Name your upstream source blocks meaningfully - those names become
    the C struct field names in the generated header.
    """

    def __init__(self, name: str, use_c_backend: bool = False):
        super().__init__(name, use_c_backend)
        self.is_code_gen_start = True
        self.is_code_gen_end   = False

    def iter_signals(self):
        """Yield (name, size) for every input signal entering the region."""
        yield from _iter_signals(self.inputs, "in")

    def print_signal_info(self):
        if not self.inputs:
            print(f"{self.name}: No inputs connected")
            return
        print(f"\n--- {self.name} Input Signals (entering C region) ---")
        for name, size in self.iter_signals():
            print(f"  {name:<28} double   size={size}")

    def generate_c_struct(self, struct_name: str = "InputSignals") -> str:
        """Generate C typedef struct for signals entering the region."""
        if not self.inputs:
            print(f"{self.name}: No inputs connected")
            return ""
        lines = [f"typedef struct {struct_name} {{"]
        for name, size in self.iter_signals():
            lines.append(f"    double {name};" if size == 1
                         else f"    double {name}[{size}];")
        lines.append(f"}} {struct_name};\n")
        result = "\n".join(lines)
        print(f"\nC struct for '{self.name}':\n{result}")
        return result

    def compute_py(self,
                   t: float,
                   dt: float,
                   input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """Concatenate all upstream outputs into one flat vector."""
        if not input_values:
            self.output = VectorSignal([0.0], self.name)
            return self.output
        combined = np.concatenate([sig.value for sig in input_values])
        self.output      = VectorSignal(combined, self.name)
        self.vector_size = len(combined)
        return self.output


# =============================================================================
# CodeGenEnd
# =============================================================================

class CodeGenEnd(SimBlockBase):
    """
    Transparent passthrough marking the OUTPUT boundary of a C code-gen region.

    Concatenates all upstream outputs into one flat vector so the simulation
    continues unchanged past this boundary marker.

    Key method:
        generate_pyx_stub(cg_start, block_name, output_dir=".")

    Generates three files:
        <block_name>.h                - C header (hand to C developer)
        <block_name>_wrapper.pyx      - Cython wrapper (compile once)
        <block_name>_simblock.py      - SimBlockBase subclass, use_c_backend ready
    """

    def __init__(self, name: str, use_c_backend: bool = False):
        super().__init__(name, use_c_backend)
        self.is_code_gen_start = False
        self.is_code_gen_end   = True

    # -- Transparent passthrough -----------------------------------------------

    def compute_py(self,
                   t: float,
                   dt: float,
                   input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """Concatenate all upstream outputs into one flat vector."""
        if not input_values:
            self.output = VectorSignal([0.0], self.name)
            return self.output
        combined = np.concatenate([sig.value for sig in input_values])
        self.output      = VectorSignal(combined, self.name)
        self.vector_size = len(combined)
        return self.output

    # -- Introspection ---------------------------------------------------------

    def iter_signals(self):
        """Yield (name, size) for every output signal leaving the region."""
        yield from _iter_signals(self.inputs, "out")

    def print_signal_info(self):
        if not self.inputs:
            print(f"{self.name}: No upstream blocks connected")
            return
        print(f"\n--- {self.name} Output Signals (leaving C region) ---")
        for name, size in self.iter_signals():
            print(f"  {name:<28} double   size={size}")

    def generate_c_struct(self, struct_name: str = "OutputSignals") -> str:
        """Generate C typedef struct for signals leaving the region."""
        if not self.inputs:
            print(f"{self.name}: No upstream blocks connected")
            return ""
        lines = [f"typedef struct {struct_name} {{"]
        for name, size in self.iter_signals():
            lines.append(f"    double {name};" if size == 1
                         else f"    double {name}[{size}];")
        lines.append(f"}} {struct_name};\n")
        result = "\n".join(lines)
        print(f"\nC struct for '{self.name}':\n{result}")
        return result

    # =========================================================================
    # MAIN GENERATOR
    # =========================================================================

    def generate_pyx_stub(self,
                          cg_start: "CodeGenStart",
                          block_name: str,
                          output_dir: Union[str, Path] = ".",
                          write_files: bool = True) -> dict:
        """
        Generate all three files needed to switch use_c_backend=True.

        Call this AFTER running the simulation at least one step so that
        block.output is populated and signal sizes are known.

        Args:
            cg_start:    The CodeGenStart block at the region input boundary.
            block_name:  Base name for all generated files and the C function.
                         e.g. "pid_controller" produces:
                           pid_controller.h
                           pid_controller_wrapper.pyx
                           pid_controller_simblock.py
            output_dir:  Directory to write files into (created if missing).
            write_files: False = return strings only, do not write files.

        Returns:
            dict with keys "header", "pyx", "simblock" - generated file contents.

        Example:
            >>> # Wire up your simulation
            >>> source >> cg_start >> my_python_pid >> cg_end >> plant
            >>> VectorSim(sinks=[plant], T=0.01, dt=0.001).run()   # run once
            >>>
            >>> # Generate all stubs
            >>> cg_end.generate_pyx_stub(cg_start, "my_pid", "./codegen")
            >>>
            >>> # C developer implements my_pid.h, compiles libmy_pid.so
            >>> # You compile: python setup_my_pid.py build_ext --inplace
            >>> # Then switch:
            >>> block = MyPidSimBlock("pid", use_c_backend=True)
        """
        out_dir = Path(output_dir)
        if write_files:
            out_dir.mkdir(parents=True, exist_ok=True)

        in_signals  = list(cg_start.iter_signals())
        out_signals = list(self.iter_signals())

        if not in_signals:
            raise RuntimeError(
                f"CodeGenStart '{cg_start.name}' has no inputs - "
                f"run the simulation at least one step first."
            )
        if not out_signals:
            raise RuntimeError(
                f"CodeGenEnd '{self.name}' has no inputs - "
                f"run the simulation at least one step first."
            )

        safe_name  = _sanitize(block_name)
        class_name = (
            "".join(w.capitalize() for w in safe_name.split("_")) + "SimBlock"
        )

        h     = self._gen_header(safe_name, in_signals, out_signals)
        pyx   = self._gen_pyx(safe_name, in_signals, out_signals)
        py    = self._gen_simblock(safe_name, class_name, in_signals, out_signals)
        setup = self._gen_setup_py(safe_name)

        if write_files:
            (out_dir / f"{safe_name}.h").write_text(h, encoding='utf-8')
            (out_dir / f"{safe_name}_wrapper.pyx").write_text(pyx, encoding='utf-8')
            (out_dir / f"{safe_name}_simblock.py").write_text(py, encoding='utf-8')
            (out_dir / f"setup_{safe_name}.py").write_text(setup, encoding='utf-8')
            print(f"\n[CodeGenEnd] Files written to '{out_dir}/':")
            print(f"  {safe_name}.h")
            print(f"  {safe_name}_wrapper.pyx")
            print(f"  {safe_name}_simblock.py")
            print(f"  setup_{safe_name}.py")
            self._print_compile_hint(safe_name, out_dir)

        return {"header": h, "pyx": pyx, "simblock": py, "setup": setup}

    # =========================================================================
    # Private code generators
    # =========================================================================

    def _gen_header(self, safe_name, in_sigs, out_sigs):
        guard = safe_name.upper() + "_H"
        L = [
            f"/* {'='*65} */",
            f"/* ControlForge auto-generated C header                            */",
            f"/* Block : {safe_name}",
            f"/*",
            f"/* Implement {safe_name}_compute() in {safe_name}.c",
            f"/* Compile  : gcc -O2 -shared -fPIC -o lib{safe_name}.so {safe_name}.c",
            f"/* {'='*65} */",
            f"",
            f"#ifndef {guard}",
            f"#define {guard}",
            f"",
            f"#ifdef __cplusplus",
            f'extern "C" {{',
            f"#endif",
            f"",
            f"/* -- Input struct ------------------------------------------------ */",
            f"typedef struct InputSignals {{",
        ]
        for name, size in in_sigs:
            L.append(f"    double {name};" if size == 1
                     else f"    double {name}[{size}];")
        L += [
            f"}} InputSignals;",
            f"",
            f"/* -- Output struct ----------------------------------------------- */",
            f"typedef struct OutputSignals {{",
        ]
        for name, size in out_sigs:
            L.append(f"    double {name};" if size == 1
                     else f"    double {name}[{size}];")
        L += [
            f"}} OutputSignals;",
            f"",
            f"/* -- Function signature ------------------------------------------ */",
            f"void {safe_name}_compute(const InputSignals* in, OutputSignals* out);",
            f"",
            f"#ifdef __cplusplus",
            f"}}",
            f"#endif",
            f"",
            f"#endif /* {guard} */",
        ]
        return "\n".join(L)

    def _gen_pyx(self, safe_name, in_sigs, out_sigs):
        wrapper_class = (
            "".join(w.capitalize() for w in safe_name.split("_")) + "Wrapper"
        )
        L = [
            f"# {safe_name}_wrapper.pyx",
            f"# {'='*65}",
            f"# Auto-generated Cython wrapper for ControlForge block '{safe_name}'",
            f"# cython: language_level=3",
            f"# cython: boundscheck=False",
            f"# cython: wraparound=False",
            f"# cython: cdivision=True",
            f"",
            f"import numpy as np",
            f"cimport numpy as cnp",
            f"",
            f"# -- C declarations -----------------------------------------------",
            f'cdef extern from "{safe_name}.h":',
            f"",
            f"    ctypedef struct InputSignals:",
        ]
        for name, size in in_sigs:
            L.append(f"        double {name}" if size == 1
                     else f"        double {name}[{size}]")
        L += [
            f"",
            f"    ctypedef struct OutputSignals:",
        ]
        for name, size in out_sigs:
            L.append(f"        double {name}" if size == 1
                     else f"        double {name}[{size}]")
        L += [
            f"",
            f"    void {safe_name}_compute(",
            f"        const InputSignals* inp,",
            f"        OutputSignals* out",
            f"    ) nogil",
            f"",
            f"",
            f"# -- Cython wrapper class -----------------------------------------",
            f"cdef class {wrapper_class}:",
            f'    """',
            f"    Cython wrapper for {safe_name}.",
            f"    Structs live on the C stack - no heap allocation on the hot path.",
            f'    """',
            f"    cdef InputSignals  _in",
            f"    cdef OutputSignals _out",
            f"",
            f"    def __cinit__(self):",
        ]
        # zero-init inputs
        for name, size in in_sigs:
            if size == 1:
                L.append(f"        self._in.{name} = 0.0")
            else:
                for k in range(size):
                    L.append(f"        self._in.{name}[{k}] = 0.0")
        # zero-init outputs
        for name, size in out_sigs:
            if size == 1:
                L.append(f"        self._out.{name} = 0.0")
            else:
                for k in range(size):
                    L.append(f"        self._out.{name}[{k}] = 0.0")
        L.append("")

        # set_inputs - flat double[::1] -> struct fields
        total_in = sum(s for _, s in in_sigs)
        L += [
            f"    cpdef void set_inputs(self, double[::1] u):",
            f'        """Pack flat input array into InputSignals struct."""',
        ]
        offset = 0
        for name, size in in_sigs:
            if size == 1:
                L.append(f"        self._in.{name} = u[{offset}]")
                offset += 1
            else:
                for k in range(size):
                    L.append(f"        self._in.{name}[{k}] = u[{offset + k}]")
                offset += size
        L.append("")

        # compute - call C function, release GIL
        L += [
            f"    cpdef void compute(self):",
            f'        """Call C function - GIL released."""',
            f"        with nogil:",
            f"            {safe_name}_compute(&self._in, &self._out)",
            f"",
        ]

        # get_outputs - struct -> flat numpy array
        total_out = sum(s for _, s in out_sigs)
        L += [
            f"    cpdef cnp.ndarray get_outputs(self):",
            f'        """Return output struct as a flat numpy array."""',
            f"        cdef cnp.ndarray y = np.empty({total_out}, dtype=np.float64)",
        ]
        offset = 0
        for name, size in out_sigs:
            if size == 1:
                L.append(f"        y[{offset}] = self._out.{name}")
                offset += 1
            else:
                for k in range(size):
                    L.append(f"        y[{offset + k}] = self._out.{name}[{k}]")
                offset += size
        L += [
            f"        return y",
            f"",
            f"    # -- Individual output properties (convenience) ----------------",
        ]
        for name, size in out_sigs:
            if size == 1:
                L += [
                    f"    @property",
                    f"    def {name}(self) -> float:",
                    f"        return self._out.{name}",
                    f"",
                ]
            else:
                L += [
                    f"    @property",
                    f"    def {name}(self):",
                    f"        return [self._out.{name}[i] for i in range({size})]",
                    f"",
                ]
        return "\n".join(L)

    def _gen_simblock(self, safe_name, class_name, in_sigs, out_sigs):
        wrapper_class = (
            "".join(w.capitalize() for w in safe_name.split("_")) + "Wrapper"
        )
        total_in  = sum(s for _, s in in_sigs)
        total_out = sum(s for _, s in out_sigs)

        L = [
            f"# {safe_name}_simblock.py",
            f"# {'='*65}",
            f"# Auto-generated SimBlockBase subclass for '{safe_name}'",
            f"#",
            f"# Switch between Python and C with one flag:",
            f"#   block = {class_name}('name', use_c_backend=False)  # Python",
            f"#   block = {class_name}('name', use_c_backend=True)   # C (needs .pyx compiled)",
            f"",
            f"from typing import List, Optional",
            f"import numpy as np",
            f"from .core_blocks import VectorSignal",
            f"from .code_generator import SimBlockBase",
            f"",
            f"",
            f"class {class_name}(SimBlockBase):",
            f'    """',
            f"    ControlForge block: {safe_name}",
            f"",
            f"    Inputs  ({total_in} doubles total):",
        ]
        off = 0
        for name, size in in_sigs:
            idx = f"[{off}]" if size == 1 else f"[{off}..{off+size-1}]"
            L.append(f"      {idx:>12}  {name}  (size={size})")
            off += size
        L.append(f"")
        L.append(f"    Outputs ({total_out} doubles total):")
        off = 0
        for name, size in out_sigs:
            idx = f"[{off}]" if size == 1 else f"[{off}..{off+size-1}]"
            L.append(f"      {idx:>12}  {name}  (size={size})")
            off += size
        L += [
            f'    """',
            f"",
            f"    def __init__(self, name: str, use_c_backend: bool = False):",
            f"        super().__init__(name, use_c_backend)",
            f"        self.vector_size = {total_out}",
            f"        self._wrapper = None",
            f"        if use_c_backend:",
            f"            self._load_wrapper()",
            f"",
            f"    def _load_wrapper(self):",
            f"        try:",
            f"            from {safe_name}_wrapper import {wrapper_class}",
            f"            self._wrapper = {wrapper_class}()",
            f"        except ImportError:",
            f"            raise ImportError(",
            f'                "Cython wrapper \'{safe_name}_wrapper\' not found.\\n"',
            f'                "Compile it:\\n"',
            f'                "  python setup_{safe_name}.py build_ext --inplace"',
            f"            )",
            f"",
            f"    # -- Python implementation -------------------------------------",
            f"    def compute_py(",
            f"        self,",
            f"        t: float,",
            f"        dt: float,",
            f"        input_values: Optional[List[VectorSignal]] = None,",
            f"    ) -> VectorSignal:",
            f'        """TODO: implement your Python algorithm here."""',
            f"        # -- Unpack inputs ------------------------------------------",
        ]
        off = 0
        for name, size in in_sigs:
            if size == 1:
                L.append(
                    f"        # {name} = input_values[0].value[{off}]"
                )
                off += 1
            else:
                L.append(
                    f"        # {name} = input_values[0].value[{off}:{off+size}]"
                )
                off += size
        L += [
            f"        y = np.zeros({total_out}, dtype=np.float64)",
            f"        # TODO: fill y with your computed outputs",
            f"        self.output = VectorSignal(y, self.name)",
            f"        return self.output",
            f"",
            f"    # -- C backend ------------------------------------------------",
            f"    def compute_c(",
            f"        self,",
            f"        t: float,",
            f"        dt: float,",
            f"        input_values: Optional[List[VectorSignal]] = None,",
            f"    ) -> VectorSignal:",
            f'        """Call compiled Cython wrapper - zero Python overhead on hot path."""',
            f"        # -- Pack flat input buffer ---------------------------------",
            f"        u = np.empty({total_in}, dtype=np.float64)",
        ]
        off = 0
        for i, (name, size) in enumerate(in_sigs):
            # Try to intelligently reference the right input block by index
            comment = f"  # {name}"
            if size == 1:
                L.append(
                    f"        u[{off}] = input_values[0].value[{off}] "
                    f"if input_values else 0.0{comment}"
                )
                off += 1
            else:
                L.append(
                    f"        u[{off}:{off+size}] = "
                    f"input_values[0].value[{off}:{off+size}] "
                    f"if input_values else np.zeros({size}){comment}"
                )
                off += size
        L += [
            f"        # -- Call C via Cython --------------------------------------",
            f"        self._wrapper.set_inputs(u)",
            f"        self._wrapper.compute()",
            f"        y = self._wrapper.get_outputs()",
            f"        self.output = VectorSignal(y, self.name)",
            f"        return self.output",
        ]
        return "\n".join(L)

    def _gen_setup_py(self, safe_name):
        """Generate setup.py for compiling the Cython wrapper + C source together.
        Sources list includes both the .pyx and the .c implementation so the
        linker can resolve gain_integrator_compute without a separate .dll/.so step.
        Compiler flags are detected at runtime: /O2 for MSVC, -O3 for GCC/Clang.
        """
        L = [
            f"# setup_{safe_name}.py",
            f"# Auto-generated by ControlForge",
            f"#",
            f"# Usage:",
            f"#   python setup_{safe_name}.py build_ext --inplace",
            f"#",
            f"# Prerequisite: {safe_name}.c must exist in the same directory.",
            f"",
            f"import sys",
            f"from setuptools import setup, Extension",
            f"from Cython.Build import cythonize",
            f"import numpy as np",
            f"",
            f"# Detect compiler: MSVC uses /O2, GCC/Clang use -O3",
            f"if sys.platform == 'win32':",
            f"    compile_args = ['/O2']",
            f"else:",
            f"    compile_args = ['-O3', '-ffast-math']",
            f"",
            f"ext = Extension(",
            f"    name='{safe_name}_wrapper',",
            f"    # Include both the Cython wrapper AND your C implementation.",
            f"    # This compiles everything together so the linker finds",
            f"    # {safe_name}_compute() without a separate .dll/.lib step.",
            f"    sources=[",
            f"        '{safe_name}_wrapper.pyx',",
            f"        '{safe_name}.c',",
            f"    ],",
            f"    include_dirs=[np.get_include()],",
            f"    extra_compile_args=compile_args,",
            f")",
            f"",
            f"setup(",
            f"    name='{safe_name}_wrapper',",
            f"    ext_modules=cythonize(",
            f"        [ext],",
            f"        compiler_directives={{",
            f"            'language_level': '3',",
            f"            'boundscheck':    False,",
            f"            'wraparound':     False,",
            f"            'cdivision':      True,",
            f"        }},",
            f"        annotate=True,",
            f"    ),",
            f")",
        ]
        return "\n".join(L)

    # =========================================================================
    # generate_loop  (feature 05121967)
    # =========================================================================

    def generate_loop(self,
                      cg_start: "CodeGenStart",
                      output_dir: Union[str, Path, None] = None,
                      dt_hz: float = 0.0,
                      write_files: bool = True) -> dict:
        """
        Walk every block between *cg_start* and this CodeGenEnd and emit
        a single C file pair that calls each block's C function in the
        correct topological order.

        Output files (written into ``<output_dir>/embedsim_gen/``)
        -----------------------------------------------------------
        embedsim_loop.c     — one ``embedsim_loop_step()`` function that
                              calls every block in sequence
        embedsim_loop.h     — extern declarations + prototype

        Parameters
        ----------
        cg_start   : CodeGenStart
            The start marker of the code-gen region.
        output_dir : str | Path | None
            Root directory.  ``embedsim_gen/`` is created inside it.
            Defaults to the current working directory.
        dt_hz      : float
            If > 0, emits ``#define EMBEDSIM_DT  <value>f`` in the header.
        write_files: bool
            False → return dict only, no disk writes.

        Returns
        -------
        dict with keys "c" and "h" (file text strings).
        """
        gen = LoopGenerator(cg_start, self)
        return gen.generate(output_dir=output_dir,
                            dt_hz=dt_hz,
                            write_files=write_files)

    def _print_compile_hint(self, safe_name, out_dir):
        print(f"\n  Next steps:")
        print(f"  1. Implement {safe_name}_compute() in {safe_name}.c")
        print(f"  2. Copy {safe_name}.c into: {out_dir}")
        print(f"  3. Compile C library:")
        print(f"       Windows: cl /O2 /LD {safe_name}.c /Fe:lib{safe_name}.dll")
        print(f"       Linux  : gcc -O2 -shared -fPIC -o lib{safe_name}.so {safe_name}.c")
        print(f"  4. Compile Cython wrapper (from {out_dir}):")
        print(f"       python setup_{safe_name}.py build_ext --inplace")
        print(f"  5. Use {safe_name}_simblock.py with use_c_backend=True")


# =============================================================================
# MCU target metadata
# =============================================================================

class MCUTarget:
    AURIX_TRICORE = 'tricore'
    CORTEX_M4     = 'CORTEX_M4'


# =============================================================================
# LoopGenerator  (feature 05121967)
# =============================================================================

class LoopGenerator:
    """
    Walks the block graph between CodeGenStart and CodeGenEnd (inclusive)
    and emits ``embedsim_loop.c`` / ``embedsim_loop.h``.

    The generated ``embedsim_loop_step()`` function calls every block's
    C step function in topological (DFS) order — the same order the Python
    simulation engine uses.

    Each block is emitted via ``_emit_block()``.  Blocks that don't have
    CodeGen metadata (no C_SOURCES / step_func) are emitted as a
    ``/* [pass-through] */`` comment so the file still compiles and the
    developer knows what's there.

    Usage (via CodeGenEnd):
        cg_end.generate_loop(cg_start, output_dir=".", dt_hz=10000.0)

    Or directly:
        gen = LoopGenerator(cg_start, cg_end)
        result = gen.generate(output_dir=".", dt_hz=0.0, write_files=False)
        print(result["c"])
    """

    def __init__(self, cg_start: CodeGenStart, cg_end: CodeGenEnd) -> None:
        self.cg_start = cg_start
        self.cg_end   = cg_end

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def generate(self,
                 output_dir: Union[str, Path, None] = None,
                 dt_hz: float = 0.0,
                 write_files: bool = True) -> dict:
        """
        Build the two files and optionally write them to
        ``<output_dir>/embedsim_gen/``.

        Returns dict with keys "c" and "h".
        """
        blocks = self._collect_region_blocks()

        c_text = self._gen_c(blocks)
        h_text = self._gen_h(blocks, dt_hz)

        if write_files:
            root    = Path(output_dir) if output_dir else Path.cwd()
            gen_dir = root / "embedsim_gen"
            gen_dir.mkdir(parents=True, exist_ok=True)
            (gen_dir / "embedsim_loop.c").write_text(c_text, encoding='utf-8')
            (gen_dir / "embedsim_loop.h").write_text(h_text, encoding='utf-8')
            print(f"\n[LoopGenerator] Files written to '{gen_dir}/':")
            print(f"  embedsim_loop.c")
            print(f"  embedsim_loop.h")
            print(f"  ({len(blocks)} block(s) in region)")

        return {"c": c_text, "h": h_text}

    # ──────────────────────────────────────────────────────────────────────────
    # Block graph traversal
    # ──────────────────────────────────────────────────────────────────────────

    def _collect_region_blocks(self) -> list:
        """
        DFS from cg_end backwards to cg_start, collect all blocks
        in the bounded region (exclusive of the start/end markers themselves).

        Returns blocks in forward execution order (sources → sinks).
        """
        visited:  set  = set()
        ordered:  list = []
        boundary: set  = {id(self.cg_start), id(self.cg_end)}

        def dfs(block):
            if id(block) in visited:
                return
            visited.add(id(block))
            if id(block) in boundary:
                return
            for upstream in block.inputs:
                dfs(upstream)
            ordered.append(block)

        # Walk backwards from cg_end's direct inputs (not cg_end itself)
        for blk in self.cg_end.inputs:
            dfs(blk)

        return ordered   # already in topological order

    # ──────────────────────────────────────────────────────────────────────────
    # Per-block C emission
    # ──────────────────────────────────────────────────────────────────────────

    def _emit_block(self, block) -> str:
        """
        Emit the C call snippet for one block.

        Checks (in order):
          0. Block has IS_UNIT_DELAY=True → emit as a z⁻¹ register read.
          1. Block has a ``step_func`` attribute (set by PYXInspector) → emit
             the real C function call with typed local variables.
          2. Block has C_SOURCES but no step_func → emit a TODO comment.
          3. No CodeGen metadata → emit a pass-through comment.

        The snippet is indented with 4 spaces (fits inside a function body).

        Variable naming convention for local buffers:
            u_<safe_name>[N]   — flat input array
            y_<safe_name>[N]   — flat output array
        """
        sn = _sanitize(block.name or block.__class__.__name__)

        # ── Unit delay (z⁻¹) blocks ──────────────────────────────────────────
        # These are LoopBreaker blocks that output the previous-step value of
        # a shared register (e.g. motor output).  In C they are simply a
        # memcpy from the shared static array — no step function needed.
        if getattr(block, 'IS_UNIT_DELAY', False) or \
                getattr(block.__class__, 'IS_UNIT_DELAY', False):
            n_out    = (getattr(block, 'OUTPUT_SIZE', None) or
                        getattr(block.__class__, 'OUTPUT_SIZE', 0))
            if n_out == 0 and block.output is not None:
                n_out = len(block.output.value)
            reg_name = (getattr(block, 'C_REG_NAME', None) or
                        getattr(block.__class__, 'C_REG_NAME', 'motor_reg'))
            lines = [
                f"    /* --- {sn} (z\u207b\u00b9 unit delay) --- */",
                f"    real32_T y_{sn}[{n_out}];",
                f"    memcpy(y_{sn}, {reg_name}, sizeof(y_{sn}));",
                f"    /* {reg_name}[] is updated by the sensor/motor interface "
                f"before each call to embedsim_loop_step() */",
                "",
            ]
            return "\n".join(lines) + "\n"

        # ── Try to pull metadata set by PYXInspector ─────────────────────────
        step_func   = getattr(block, 'step_func',   None)
        n_inputs    = getattr(block, 'NUM_INPUTS',  0)
        n_outputs   = getattr(block, 'OUTPUT_SIZE', 0)
        c_sources   = getattr(block, 'C_SOURCES',   [])
        init_func   = getattr(block, 'init_func',   None)

        # Fallback: check class-level attrs (set by hand in block definition)
        if not step_func:
            step_func = getattr(block.__class__, 'step_func', None)
        if not n_inputs:
            n_inputs  = getattr(block.__class__, 'NUM_INPUTS', 0)
        if not n_outputs:
            n_outputs = getattr(block.__class__, 'OUTPUT_SIZE', 0)
        if not c_sources:
            c_sources = getattr(block.__class__, 'C_SOURCES', [])

        # ── On-demand PYXInspector: run now if step_func still missing ────────
        # This covers the case where __init_subclass__ could not auto-populate
        # (e.g. relative PYX_FILE path failed) but PYX_FILE is now findable.
        if not step_func:
            pyx_file = getattr(block.__class__, 'PYX_FILE', None)
            if pyx_file:
                try:
                    from .pyx_inspector import PYXInspector
                except ImportError:
                    try:
                        from pyx_inspector import PYXInspector
                    except ImportError:
                        PYXInspector = None
                if PYXInspector is not None:
                    try:
                        meta = PYXInspector().inspect(pyx_file)
                        step_func    = meta.step_func    or step_func
                        init_func    = meta.init_func    or init_func
                        if not n_inputs:  n_inputs  = meta.n_inputs
                        if not n_outputs: n_outputs = meta.n_outputs
                        if not c_sources: c_sources = meta.c_sources
                        # Cache on the class so next block of same type is free
                        if meta.step_func:
                            block.__class__.step_func    = meta.step_func
                        if meta.state_struct:
                            block.__class__.state_struct = meta.state_struct
                    except Exception:
                        pass

        if not step_func and not c_sources:
            # Pure Python block inside a region — no C equivalent known
            return (
                f"    /* [{sn}] Python-only block — no C step function. "
                f"Replace with hand-written C or add C_SOURCES + step_func. */\n"
            )

        if not step_func:
            # Has C sources but no step function signature deduced
            return (
                f"    /* TODO [{sn}] C_SOURCES={c_sources} — "
                f"set step_func or run PYXInspector to auto-detect. */\n"
            )

        # ── Determine input wiring ────────────────────────────────────────────
        # Prefer explicit C_INPUT_MAP if declared on the block:
        #   C_INPUT_MAP = [("source_name", src_index), ...]
        #   one entry per element of u[], matching the C function signature.
        # Fall back to auto-wiring from block.inputs (may be imprecise for
        # multi-port blocks where some ports are LoopBreakers).
        c_input_map = (getattr(block, 'C_INPUT_MAP', None) or
                       getattr(block.__class__, 'C_INPUT_MAP', None))

        total_out = n_outputs if n_outputs > 0 else 1
        lines = []
        lines.append(f"    /* --- {sn} ({block.__class__.__name__}) --- */")

        if c_input_map:
            # Explicit port mapping — precise, handles LoopBreaker inputs
            total_in = len(c_input_map)
            lines.append(f"    real32_T u_{sn}[{total_in}];")
            for k, (src_name, src_idx) in enumerate(c_input_map):
                src_sn = _sanitize(src_name)
                lines.append(f"    u_{sn}[{k}] = y_{src_sn}[{src_idx}];")
        else:
            # Auto-wiring: pack outputs of all input blocks sequentially
            in_vars = []
            for inp_block in block.inputs:
                inp_sn = _sanitize(inp_block.name or inp_block.__class__.__name__)
                inp_n_out = getattr(inp_block, 'OUTPUT_SIZE',
                            getattr(inp_block.__class__, 'OUTPUT_SIZE', 0))
                if inp_n_out == 0 and inp_block.output is not None:
                    inp_n_out = len(inp_block.output.value)
                in_vars.append((inp_sn, inp_n_out))

            total_in = n_inputs if n_inputs > 0 else sum(s for _, s in in_vars)

            if total_in > 0:
                lines.append(f"    real32_T u_{sn}[{total_in}];")
                offset = 0
                for inp_sn, inp_sz in in_vars:
                    if inp_sz == 1:
                        lines.append(f"    u_{sn}[{offset}] = y_{inp_sn}[0];")
                        offset += 1
                    else:
                        for k in range(inp_sz):
                            if offset + k < total_in:
                                lines.append(
                                    f"    u_{sn}[{offset + k}] = y_{inp_sn}[{k}];"
                                )
                        offset += inp_sz

        # Output buffer
        lines.append(f"    real32_T y_{sn}[{total_out}];")

        # State struct pointer (if stateful — PYXInspector sets state_struct)
        state_struct = getattr(block, 'state_struct',
                       getattr(block.__class__, 'state_struct', ''))
        if state_struct:
            lines.append(f"    /* State: {state_struct} {sn}_state; "
                         f"(declare as file-scope static) */")

        # The actual C call
        if state_struct:
            # Stateful: signature typically (Block*, Input*, dt, Output*)
            lines.append(
                f"    {step_func}(&{sn}_state, u_{sn}, dt, y_{sn});"
            )
        else:
            # Stateless: (Input*, Output*)
            lines.append(
                f"    {step_func}(u_{sn}, y_{sn});"
            )

        lines.append("")   # blank line between blocks
        return "\n".join(lines) + "\n"

    # ──────────────────────────────────────────────────────────────────────────
    # File assembly
    # ──────────────────────────────────────────────────────────────────────────

    def _gather_headers(self, blocks: list) -> list:
        """Collect unique header files from all blocks in the region."""
        seen:    set  = set()
        headers: list = []
        for blk in blocks:
            for src in (getattr(blk, 'C_HEADERS', None) or
                        getattr(blk.__class__, 'C_HEADERS', [])):
                if src not in seen:
                    seen.add(src)
                    headers.append(src)
        return headers

    def _gen_c(self, blocks: list) -> str:
        """Build embedsim_loop.c text."""
        headers = self._gather_headers(blocks)

        L = [
            "/* embedsim_loop.c",
            " * Auto-generated by EmbedSim LoopGenerator (feature 05121967)",
            " * DO NOT EDIT — re-generate with cg_end.generate_loop()",
            " */",
            "",
            '#include <string.h>   /* memcpy */',
            '#include "embedsim_loop.h"',
        ]
        for h in headers:
            L.append(f'#include "{h}"')

        L += [
            "",
            "",
            "/* ================================================================",
            " * embedsim_loop_step",
            " *",
            " * Execute one control loop step.  Call at fixed sample rate.",
            " *",
            " * Parameters",
            " * ----------",
            " * dt  : sample period in seconds  (e.g. 1.0f/10000.0f for 10 kHz)",
            " * ================================================================",
            " */",
            "void embedsim_loop_step(real32_T dt)",
            "{",
        ]

        for blk in blocks:
            L.append(self._emit_block(blk))

        L += [
            "}",
            "",
        ]
        return "\n".join(L)

    def _gen_h(self, blocks: list, dt_hz: float) -> str:
        """Build embedsim_loop.h text."""
        headers = self._gather_headers(blocks)

        L = [
            "/* embedsim_loop.h",
            " * Auto-generated by EmbedSim LoopGenerator (feature 05121967)",
            " * DO NOT EDIT — re-generate with cg_end.generate_loop()",
            " */",
            "",
            "#ifndef EMBEDSIM_LOOP_H",
            "#define EMBEDSIM_LOOP_H",
            "",
            '#include "Sys_Types.h"   /* real32_T */',
            "",
        ]

        if dt_hz > 0.0:
            dt_val = 1.0 / dt_hz
            L += [
                f"/* Sample period for a {dt_hz:.0f} Hz control loop */",
                f"#define EMBEDSIM_DT  ({dt_val:.10f}f)",
                "",
            ]

        if headers:
            L.append("/* Block headers */")
            for h in headers:
                L.append(f'#include "{h}"')
            L.append("")

        # Forward-declare static state structs for stateful blocks
        statics = []
        for blk in blocks:
            state_struct = (getattr(blk, 'state_struct', None) or
                            getattr(blk.__class__, 'state_struct', ''))
            if state_struct:
                sn = _sanitize(blk.name or blk.__class__.__name__)
                statics.append(f"static {state_struct} {sn}_state;")

        if statics:
            L.append("/* Persistent state (file-scope statics — one per stateful block) */")
            L += statics
            L.append("")

        # Collect unique motor_reg declarations for unit-delay blocks
        reg_names: set = set()
        block_names = {_sanitize(blk.name or blk.__class__.__name__) for blk in blocks}

        for blk in blocks:
            if getattr(blk, 'IS_UNIT_DELAY', False) or \
                    getattr(blk.__class__, 'IS_UNIT_DELAY', False):
                n_out    = (getattr(blk, 'OUTPUT_SIZE', None) or
                            getattr(blk.__class__, 'OUTPUT_SIZE', 0))
                reg_name = (getattr(blk, 'C_REG_NAME', None) or
                            getattr(blk.__class__, 'C_REG_NAME', 'motor_reg'))
                reg_names.add((reg_name, n_out))

            # Also collect extern regs referenced in C_INPUT_MAP
            c_input_map = (getattr(blk, 'C_INPUT_MAP', None) or
                           getattr(blk.__class__, 'C_INPUT_MAP', None))
            if c_input_map:
                for src_name, src_idx in c_input_map:
                    src_sn = _sanitize(src_name)
                    if src_sn not in block_names:
                        # Not a block — must be an extern register
                        # Infer size as max index + 1
                        current = next((n for nm, n in reg_names if nm == src_name), 0)
                        needed  = src_idx + 1
                        reg_names.discard((src_name, current))
                        reg_names.add((src_name, max(current, needed)))

        if reg_names:
            L.append("/* Shared sensor/motor register — filled by integration layer */")
            L.append("/* before each call to embedsim_loop_step().                  */")
            for reg_name, n_out in sorted(reg_names):
                L.append(f"extern real32_T {reg_name}[{n_out}];")
            L.append("")

        L += [
            "/* ── Public API ─────────────────────────────────────────────── */",
            "",
            "/**",
            " * embedsim_loop_init",
            " * Call once at startup to zero all block states.",
            " */",
            "void embedsim_loop_init(void);",
            "",
            "/**",
            " * embedsim_loop_step",
            " * Call every sample period.  dt = sample period [seconds].",
            " */",
            "void embedsim_loop_step(real32_T dt);",
            "",
            "#endif /* EMBEDSIM_LOOP_H */",
            "",
        ]
        return "\n".join(L)


# =============================================================================
# MCU target metadata
# =============================================================================

class MCUTarget:
    AURIX_TRICORE = 'tricore'
    CORTEX_M4     = 'CORTEX_M4'


__all__ = ['SimBlockBase', 'CodeGenStart', 'CodeGenEnd', 'MCUTarget',
           'LoopGenerator']
__version__ = '1.6.0'
__author__ = 'EmbedSim'