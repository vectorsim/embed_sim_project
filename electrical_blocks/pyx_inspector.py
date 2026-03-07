"""
pyx_inspector.py
================

PYXInspector — feature 05121967
Parses Cython wrapper (.pyx) files to extract EmbedSim block metadata.

Lives in:  electrical_blocks/pyx_inspector.py

The inspector reads a .pyx file and returns a BlockMeta dataclass describing:
  - The C header file the wrapper imports from
  - The wrapper class name
  - Input field count  (total scalars entering the C function)
  - Output field count (total scalars leaving the C function)
  - Whether the block is stateful (has a cdef class with a state struct)
  - The C init / step / reset function names
  - The state struct type name (if stateful)

This metadata is used by two consumers:

  1. VectorBlock.__init_subclass__  (core_blocks.py)
     Auto-populates NUM_INPUTS / OUTPUT_SIZE / C_SOURCES / C_HEADERS
     on any subclass that declares  PYX_FILE = "path/to/wrapper.pyx"

  2. CodeGenerator.generate_loop()  (code_generator.py)
     Uses the metadata to emit the correct embedsim_loop.c / .h calls
     for every block inside a CodeGenStart / CodeGenEnd region.

Parsing strategy
----------------
Cython .pyx files are *not* Python — they contain `cdef`, `cimport`,
ctypedef, `nogil` etc. that Python's ast module cannot handle.

We use regex-based line-by-line parsing instead of AST.  The rules are
conservative: we look for well-known structural markers that every
EmbedSim wrapper follows by convention.

Convention (derived from smc_wrapper.pyx, speed_pi_wrapper.pyx, etc.)
-----------------------------------------------------------------------
  Line:   cdef extern from "some_header.h":
          → header file name

  Block:  ctypedef struct <Name>_Input_T:  (or InputSignals:)
          → count the scalar fields inside (arrays count by their size)

  Block:  ctypedef struct <Name>_Output_T:  (or OutputSignals:)
          → count the scalar fields inside

  Line:   cdef class <Name>Wrapper:
          → wrapper_class_name

  Line:   void <Name>_Init(...)  or  void <Name>_Step(...)
          → init_func / step_func

  Presence of a `cdef <StructType> _block` field inside the wrapper class
          → block is stateful, state_struct_type = StructType

Author : EmbedSim Framework
Feature: 05121967
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# =============================================================================
# Data model
# =============================================================================

@dataclass
class BlockMeta:
    """
    Metadata extracted from a single .pyx wrapper file.

    Attributes
    ----------
    pyx_path        : Path to the .pyx source (absolute)
    header_file     : C header declared in ``cdef extern from "...":``
    wrapper_class   : Name of the ``cdef class`` in the wrapper
    n_inputs        : Total scalar inputs (fields of the Input struct)
    n_outputs       : Total scalar outputs (fields of the Output struct)
    stateful        : True if the wrapper owns a persistent C state struct
    state_struct    : Name of the state struct type (e.g. "SMC_Block_T")
    init_func       : C init function name (e.g. "SMC_Init"), or ""
    step_func       : C step/compute function name (e.g. "SMC_Compute"), or ""
    reset_func      : C reset function name (e.g. "SMC_ResetState"), or ""
    c_sources       : C source files inferred from header name (best-effort)
    """
    pyx_path:      Path
    header_file:   str           = ""
    wrapper_class: str           = ""
    n_inputs:      int           = 0
    n_outputs:     int           = 0
    stateful:      bool          = False
    state_struct:  str           = ""
    init_func:     str           = ""
    step_func:     str           = ""
    reset_func:    str           = ""
    c_sources:     list          = field(default_factory=list)

    def __repr__(self) -> str:   # pragma: no cover
        return (
            f"BlockMeta("
            f"header='{self.header_file}', "
            f"class='{self.wrapper_class}', "
            f"n_in={self.n_inputs}, n_out={self.n_outputs}, "
            f"stateful={self.stateful}, "
            f"init='{self.init_func}', step='{self.step_func}')"
        )


# =============================================================================
# Compiled regex patterns  (module-level → compiled once)
# =============================================================================

# cdef extern from "some_header.h":
_RE_EXTERN  = re.compile(r'cdef\s+extern\s+from\s+"([^"]+\.h)"')

# ctypedef struct Foo_Input_T:  or  ctypedef struct InputSignals:
_RE_STRUCT_INPUT  = re.compile(
    r'^\s*ctypedef\s+struct\s+(\w*[Ii]nput\w*)\s*:'
)
_RE_STRUCT_OUTPUT = re.compile(
    r'^\s*ctypedef\s+struct\s+(\w*[Oo]utput\w*)\s*:'
)

# cdef class SMCWrapper:
_RE_CDEF_CLASS = re.compile(r'^\s*cdef\s+class\s+(\w+)\s*[:(]?')

# Inside a struct block:  real32_T foo        → 1 scalar
#                         real32_T bar[N]      → N scalars
_RE_FIELD_SCALAR = re.compile(r'^\s+\w[\w\s\*]*\s+(\w+)\s*$')
_RE_FIELD_ARRAY  = re.compile(r'^\s+\w[\w\s\*]*\s+(\w+)\s*\[(\d+)\]')

# void SMC_Init(SMC_Block_T* ...)
_RE_VOID_FUNC = re.compile(r'void\s+(\w+)\s*\(')

# Inside a cdef class: cdef SMC_Block_T  _block
_RE_CDEF_STATE = re.compile(r'^\s+cdef\s+(\w+)\s+_block\b')

# ctypedef float real32_T  — these are type aliases, not fields → skip
_RE_TYPEDEF_ALIAS = re.compile(r'^\s*ctypedef\s+\w+\s+\w+\s*$')

# cimport / import lines — skip
_RE_IMPORT = re.compile(r'^\s*(c?import)\b')

# Blank / comment lines
_RE_BLANK   = re.compile(r'^\s*(#.*)?$')


# =============================================================================
# PYXInspector
# =============================================================================

class PYXInspector:
    """
    Parse one or more .pyx files and return BlockMeta objects.

    Usage
    -----
        insp = PYXInspector()
        meta = insp.inspect("electrical_blocks/c_src/smc_wrapper.pyx")
        print(meta.n_inputs, meta.n_outputs)   # → 4  2

    Or inspect a whole directory:
        metas = PYXInspector.inspect_dir("electrical_blocks/c_src/")
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def inspect(self, pyx_path: str | Path) -> BlockMeta:
        """
        Parse *pyx_path* and return a BlockMeta.

        Parameters
        ----------
        pyx_path : str or Path
            Path to the .pyx file to parse.

        Returns
        -------
        BlockMeta
            Populated from the parsed content.

        Raises
        ------
        FileNotFoundError
            If pyx_path does not exist.
        """
        pyx_path = Path(pyx_path).resolve()
        if not pyx_path.exists():
            raise FileNotFoundError(f"PYXInspector: file not found: {pyx_path}")

        text = pyx_path.read_text(encoding='utf-8', errors='replace')
        return self._extract_funcs(text, pyx_path)

    @classmethod
    def inspect_dir(cls, directory: str | Path,
                    pattern: str = "*_wrapper.pyx") -> dict[str, BlockMeta]:
        """
        Inspect all .pyx files matching *pattern* under *directory*.

        Returns
        -------
        dict mapping stem name (e.g. "smc_wrapper") to BlockMeta.
        """
        insp  = cls()
        result: dict[str, BlockMeta] = {}
        for p in sorted(Path(directory).rglob(pattern)):
            try:
                result[p.stem] = insp.inspect(p)
            except Exception as exc:   # pragma: no cover
                import warnings
                warnings.warn(f"PYXInspector: skipping {p.name}: {exc}")
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Core parser  (private)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_funcs(self, text: str, pyx_path: Path) -> BlockMeta:
        """
        Parse *text* (full .pyx file contents) and populate a BlockMeta.

        This is the heart of the feature — called by inspect() and by tests.

        The parser is a simple state machine:
          STATE_TOP         : scanning for extern / cdef class / void funcs
          STATE_INPUT_STRUCT: counting fields of the input struct
          STATE_OUTPUT_STRUCT: counting fields of the output struct
          STATE_CDEF_CLASS  : scanning for cdef state field inside wrapper
        """
        meta = BlockMeta(pyx_path=pyx_path)

        STATE_TOP           = 0
        STATE_INPUT_STRUCT  = 1
        STATE_OUTPUT_STRUCT = 2
        STATE_CDEF_CLASS    = 3

        state           = STATE_TOP
        in_extern_block = False   # True after "cdef extern from ..."
        struct_indent   = None    # indentation level of struct body

        lines = text.splitlines()

        # First pass: collect all void function names (for init/step/reset)
        void_funcs: list[str] = []
        for line in lines:
            m = _RE_VOID_FUNC.search(line)
            if m:
                void_funcs.append(m.group(1))

        self._classify_funcs(void_funcs, meta)

        # Second pass: struct field counting + class detection
        for i, line in enumerate(lines):

            # ── extern from → grab header ────────────────────────────────────
            m = _RE_EXTERN.search(line)
            if m:
                meta.header_file  = m.group(1)
                in_extern_block   = True
                # Infer C source file name from header name
                src = m.group(1).replace('.h', '.c')
                if src not in meta.c_sources:
                    meta.c_sources.append(src)
                continue

            # ── input struct ─────────────────────────────────────────────────
            m = _RE_STRUCT_INPUT.match(line)
            if m and in_extern_block:
                state         = STATE_INPUT_STRUCT
                struct_indent = self._indent(line)
                continue

            # ── output struct ────────────────────────────────────────────────
            m = _RE_STRUCT_OUTPUT.match(line)
            if m and in_extern_block:
                state         = STATE_OUTPUT_STRUCT
                struct_indent = self._indent(line)
                continue

            # ── cdef class ───────────────────────────────────────────────────
            m = _RE_CDEF_CLASS.match(line)
            if m:
                # Only pick it up if it looks like a Wrapper class
                cls_name = m.group(1)
                if cls_name.endswith('Wrapper') or 'Wrapper' in cls_name:
                    meta.wrapper_class = cls_name
                state  = STATE_CDEF_CLASS
                in_extern_block = False   # we've left the extern block
                struct_indent   = self._indent(line)
                continue

            # ── field counting (inside struct) ────────────────────────────────
            if state in (STATE_INPUT_STRUCT, STATE_OUTPUT_STRUCT):
                if _RE_BLANK.match(line):
                    continue
                cur_indent = self._indent(line)
                # End of struct: dedented back to or past struct_indent
                if cur_indent <= struct_indent and not _RE_FIELD_SCALAR.match(line) \
                        and not _RE_FIELD_ARRAY.match(line):
                    state = STATE_TOP
                    # Re-process this line at STATE_TOP
                    # (it may be the start of the next struct)
                    m_in  = _RE_STRUCT_INPUT.match(line)
                    m_out = _RE_STRUCT_OUTPUT.match(line)
                    if m_in and in_extern_block:
                        state = STATE_INPUT_STRUCT
                        struct_indent = self._indent(line)
                    elif m_out and in_extern_block:
                        state = STATE_OUTPUT_STRUCT
                        struct_indent = self._indent(line)
                    continue

                # Skip typedef aliases (ctypedef float real32_T)
                if _RE_TYPEDEF_ALIAS.match(line):
                    continue
                if _RE_IMPORT.match(line):
                    continue

                n = self._count_field(line)
                if n > 0:
                    if state == STATE_INPUT_STRUCT:
                        meta.n_inputs  += n
                    else:
                        meta.n_outputs += n
                continue

            # ── inside cdef class: look for _block (stateful) ─────────────────
            if state == STATE_CDEF_CLASS:
                m = _RE_CDEF_STATE.match(line)
                if m:
                    meta.stateful     = True
                    meta.state_struct = m.group(1)

        return meta

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _indent(line: str) -> int:
        """Return the number of leading spaces in *line*."""
        return len(line) - len(line.lstrip(' '))

    @staticmethod
    def _count_field(line: str) -> int:
        """
        Return the scalar count contributed by a single struct field line.

          real32_T foo       → 1
          real32_T bar[3]    → 3
          int      baz[2]    → 2
          (blank / comment)  → 0
        """
        if _RE_BLANK.match(line):
            return 0
        if _RE_TYPEDEF_ALIAS.match(line):
            return 0
        if _RE_IMPORT.match(line):
            return 0
        # Skip lines that start a nested struct or other cdef
        stripped = line.strip()
        if stripped.startswith(('cdef', 'ctypedef', '#', 'pass')):
            return 0

        m = _RE_FIELD_ARRAY.search(line)
        if m:
            return int(m.group(2))

        m = _RE_FIELD_SCALAR.search(line)
        if m:
            return 1

        return 0

    @staticmethod
    def _classify_funcs(func_names: list[str], meta: BlockMeta) -> None:
        """
        Heuristically assign void functions to init / step / reset roles.

        Rules (in priority order):
          - Name contains "Init"  → init_func   (first match wins)
          - Name contains "Reset" → reset_func
          - Name contains any of "Compute", "Step", "Update" → step_func
        """
        for fn in func_names:
            fn_upper = fn.upper()
            if 'INIT' in fn_upper and not meta.init_func:
                meta.init_func = fn
            elif 'RESET' in fn_upper and not meta.reset_func:
                meta.reset_func = fn
            elif any(kw in fn_upper for kw in ('COMPUTE', 'STEP', 'UPDATE')) \
                    and not meta.step_func:
                meta.step_func = fn


# =============================================================================
# __init_subclass__ hook  (called by core_blocks.VectorBlock)
# =============================================================================

def auto_populate_from_pyx(cls, pyx_file: str | Path) -> None:
    """
    Auto-populate CodeGen class attributes on *cls* from its .pyx file.

    Called by VectorBlock.__init_subclass__ when the subclass declares:

        class MyBlock(VectorBlock):
            PYX_FILE = "path/to/my_block_wrapper.pyx"

    Populates (only if not already set by the class author):
        NUM_INPUTS  : int
        OUTPUT_SIZE : int
        C_SOURCES   : list[str]
        C_HEADERS   : list[str]

    If the .pyx file cannot be found or parsed, this is a no-op with a
    warning — it must never crash the import of the block module.
    """
    import warnings

    try:
        insp = PYXInspector()
        meta = insp.inspect(pyx_file)
    except FileNotFoundError:
        warnings.warn(
            f"PYXInspector: PYX_FILE '{pyx_file}' not found for {cls.__name__}. "
            f"Auto-population skipped.",
            stacklevel=3,
        )
        return
    except Exception as exc:
        warnings.warn(
            f"PYXInspector: failed to parse '{pyx_file}' for {cls.__name__}: {exc}. "
            f"Auto-population skipped.",
            stacklevel=3,
        )
        return

    # Only set if the class has not already declared the attribute itself
    if not hasattr(cls, 'NUM_INPUTS') or cls.NUM_INPUTS == 0:
        cls.NUM_INPUTS  = meta.n_inputs
    if not hasattr(cls, 'OUTPUT_SIZE') or cls.OUTPUT_SIZE == 0:
        cls.OUTPUT_SIZE = meta.n_outputs
    if not hasattr(cls, 'C_SOURCES') or not cls.C_SOURCES:
        cls.C_SOURCES   = list(meta.c_sources)
    if not hasattr(cls, 'C_HEADERS') or not cls.C_HEADERS:
        cls.C_HEADERS   = ([meta.header_file] if meta.header_file else [])


# =============================================================================
# Module exports
# =============================================================================

__all__ = ['BlockMeta', 'PYXInspector', 'auto_populate_from_pyx']
__version__ = '1.0.0'
__feature__  = '05121967'
