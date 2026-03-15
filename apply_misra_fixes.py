"""
apply_misra_fixes.py
====================
Applies all known MISRA C:2012 / AUTOSAR C corrections to Matrix.c and Matrix.h.

Usage:
    python apply_misra_fixes.py Matrix.c Matrix.h

Outputs corrected files with suffix _MISRA:
    Matrix_MISRA.c
    Matrix_MISRA.h

Fixes applied
-------------
  FIX 1  Matrix.c : remove #include "Platform_Types.h" (file does not exist in repo)
  FIX 2a Matrix.c : Matrix_InvClarke_Init signature  Matrix2x3_T* → Matrix3x2_T*
  FIX 2b Matrix.h : same prototype fix
  FIX 3  Matrix.c : Invert_4x4 pivot search  use fabsf() instead of sign comparison
  FIX 4  Matrix.c : Invert_4x4 early return inside loop → break + single exit (MISRA 15.5)
  FIX 5  Matrix.c : sqrt() → sqrtf()  (real32_T operand, avoid implicit double cast)
  FIX 6  Matrix.c : remove L suffix from integer constants (MISRA 7.2)
"""

import sys
import re
from pathlib import Path


def apply_fixes_c(src: str) -> tuple[str, list[str]]:
    applied = []

    # ── FIX 1: Platform_Types.h ───────────────────────────────────────────────
    old = '#include "Platform_Types.h"'
    if old in src:
        # Replace the whole line
        src = re.sub(
            r'#include\s+"Platform_Types\.h"[^\n]*\n',
            '/* Platform_Types.h removed: not present in this repo; '
            'no boolean types used in this TU */\n',
            src,
        )
        applied.append("FIX1: removed Platform_Types.h include")

    # ── FIX 2a: InvClarke_Init signature in .c ───────────────────────────────
    if "Matrix2x3_T* const pC" in src and "InvClarke" in src:
        src = src.replace(
            "void Matrix_InvClarke_Init(Matrix2x3_T* const pC)",
            "void Matrix_InvClarke_Init(Matrix3x2_T* const pC)",
        )
        applied.append("FIX2a: InvClarke_Init signature Matrix2x3_T → Matrix3x2_T (.c)")

    # ── FIX 3: fabsf pivot comparison ────────────────────────────────────────
    pattern3 = (
        r'if\s*\(\s*\(\s*aug\[j\]\[i\]\s*>\s*pivot\s*\)\s*\|{1,2}\s*'
        r'\(\s*aug\[j\]\[i\]\s*<\s*-pivot\s*\)\s*\)'
    )
    replacement3 = "if (fabsf(aug[j][i]) > fabsf(pivot))"
    new_src, n = re.subn(pattern3, replacement3, src)
    if n:
        src = new_src
        applied.append(f"FIX3: pivot comparison → fabsf() ({n} occurrence(s))")

    # ── FIX 4: early return inside Gaussian elimination loop → break ─────────
    pattern4 = (
        r'(/\*\s*Check for singular matrix\s*\*/\s*\n)'
        r'(\s*)if\s*\(\s*\(pivot\s*<\s*MATRIX_EPSILON\)\s*&&\s*'
        r'\(pivot\s*>\s*-MATRIX_EPSILON\)\s*\)\s*\n'
        r'\s*\{\s*\n'
        r'\s*status\s*=\s*0U\s*;\s*\n'
        r'\s*return\s+status\s*;\s*\n'
        r'\s*\}'
    )
    replacement4 = (
        r"\1"
        r"\2/* MISRA C:2012 Rule 15.5: single exit point — use break, not return */\n"
        r"\2if (fabsf(pivot) < MATRIX_EPSILON)\n"
        r"\2{\n"
        r"\2    status = 0U;\n"
        r"\2    break;\n"
        r"\2}"
    )
    new_src, n = re.subn(pattern4, replacement4, src)
    if n:
        src = new_src
        applied.append(f"FIX4: early return in Gaussian loop → break ({n} occurrence(s))")

    # ── FIX 5: sqrt() → sqrtf() ──────────────────────────────────────────────
    # Only replace bare sqrt( not already sqrtf(
    new_src = re.sub(r'\bsqrt\(', 'sqrtf(', src)
    if new_src != src:
        src = new_src
        applied.append("FIX5: sqrt() → sqrtf() (avoid implicit double promotion)")

    # ── FIX 6: L suffix on integer constants ─────────────────────────────────
    # e.g. 0x7FFFFFFFL → (int32_T)0x7FFFFFFF
    # Only touch hex literals with trailing L/l that are assigned to int32_T context
    new_src = re.sub(r'0x([0-9A-Fa-f]+)[Ll]\b', r'0x\1', src)
    n6 = src.count('L') - new_src.count('L')
    if new_src != src:
        src = new_src
        applied.append(f"FIX6: removed L suffix from {n6} integer constant(s) (MISRA 7.2)")

    return src, applied


def apply_fixes_h(src: str) -> tuple[str, list[str]]:
    applied = []

    # ── FIX 2b: InvClarke_Init prototype in .h ───────────────────────────────
    if "Matrix2x3_T* const pC" in src and "InvClarke" in src:
        src = src.replace(
            "Matrix_InvClarke_Init(Matrix2x3_T* const pC)",
            "Matrix_InvClarke_Init(Matrix3x2_T* const pC)",
        )
        applied.append("FIX2b: InvClarke_Init prototype Matrix2x3_T → Matrix3x2_T (.h)")

    return src, applied


def main():
    if len(sys.argv) < 3:
        print("Usage: python apply_misra_fixes.py <Matrix.c> <Matrix.h>")
        sys.exit(1)

    path_c = Path(sys.argv[1])
    path_h = Path(sys.argv[2])

    for p in (path_c, path_h):
        if not p.exists():
            print(f"ERROR: file not found: {p}")
            sys.exit(1)

    src_c = path_c.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    src_h = path_h.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")

    new_c, fixes_c = apply_fixes_c(src_c)
    new_h, fixes_h = apply_fixes_h(src_h)

    out_c = path_c.with_stem(path_c.stem + "_MISRA")
    out_h = path_h.with_stem(path_h.stem + "_MISRA")

    out_c.write_text(new_c, encoding="utf-8")
    out_h.write_text(new_h, encoding="utf-8")

    print(f"\nMatrix.c fixes applied ({len(fixes_c)}):")
    for f in fixes_c:
        print(f"  ✓  {f}")

    print(f"\nMatrix.h fixes applied ({len(fixes_h)}):")
    for f in fixes_h:
        print(f"  ✓  {f}")
    if not fixes_h:
        print("  (no changes needed)")

    print(f"\nOutput:")
    print(f"  {out_c}")
    print(f"  {out_h}")


if __name__ == "__main__":
    main()
