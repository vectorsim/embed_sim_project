"""
_path_utils.py
==============

PURPOSE
-------
This module solves a common problem in Python projects: "where is the project
root, and how do I import from it?"

PROBLEM IT SOLVES
-----------------
Imagine you run a script from three different locations:

    cd C:\EmbedSimProject
    python examples\pi_buck_converter\pi_buck_example.py      # works

    cd C:\EmbedSimProject\examples\pi_buck_converter
    python pi_buck_example.py                                  # might fail!

    cd C:\SomeOtherFolder
    python C:\EmbedSimProject\examples\pi_buck_converter\pi_buck_example.py

In case 2 and 3, Python's sys.path does not contain
C:\EmbedSimProject, so "import embedsim" raises ModuleNotFoundError.
_path_utils.py fixes this: every module calls get_embedsim_import_path()
and adds the result to sys.path, so imports work from any working directory.

HOW THE ROOT IS FOUND
---------------------
The project root is identified by a MARKER FILE: .project_root_marker
This file sits at C:\EmbedSimProject\.project_root_marker (empty file, just
a flag). _path_utils.py walks UP the directory tree from its own location
until it finds a folder containing that marker file.

WHY A MARKER FILE? (not just "go up N levels")
-----------------------------------------------
Going up a fixed number of levels (e.g., parent.parent) is fragile —
if the file is moved one folder deeper, the count breaks.
The marker file is robust: the project can be installed at any depth or
moved to any drive and the path discovery still works.

USAGE IN OTHER MODULES
----------------------
    from _path_utils import get_embedsim_import_path, get_project_root
    import sys
    sys.path.insert(0, get_embedsim_import_path())   # then: import embedsim

NOTE ON THE DOCSTRING BUG (minor)
----------------------------------
The docstring says "Shared path utility for the pmsm_blocks library."
This is a copy-paste leftover — the module is used by buck_converter/
and electrical_blocks/ as well. Functionally harmless.

CORRECTNESS: ALL FUNCTIONS ARE CORRECT. No bugs found.
"""

import sys
from pathlib import Path

# ==============================================================================
#  Why "from pathlib import Path"?
# ==============================================================================
# pathlib.Path is Python's modern, cross-platform path library (Python 3.4+).
# It replaces the older os.path module. Key advantages:
#   - Works identically on Windows (\) and Linux (/)
#   - Supports the / operator for joining:  root / "subdir" / "file.txt"
#   - Has useful methods: .exists(), .resolve(), .parents, .parent, .stem
#
# ".resolve()" converts a relative path to an ABSOLUTE path by following
# any symlinks and resolving ".." segments.
# Example:  Path("../foo").resolve() → Path("C:/EmbedSimProject/foo")


def get_project_root() -> Path:
    """
    Find and return the absolute path to the EmbedSim project root.

    Strategy:
        1. Start from the location of THIS file (_path_utils.py).
        2. Walk up the directory tree one level at a time.
        3. At each level, check if a file named '.project_root_marker' exists.
        4. If found, that directory IS the project root — return it.
        5. If we reach the filesystem root without finding the marker,
           fall back to "two levels up from this file" (see note below).

    Returns:
        Path: absolute path to C:\\EmbedSimProject (or wherever the root is)

    Example directory walk for buck_converter/_path_utils.py:
        Start: C:\\EmbedSimProject\\buck_converter\\_path_utils.py
        Check: C:\\EmbedSimProject\\buck_converter\\.project_root_marker → NOT FOUND
        Check: C:\\EmbedSimProject\\.project_root_marker                  → FOUND ✓
        Return: C:\\EmbedSimProject
    """

    # Path(__file__)          → path to this .py file (may be relative)
    # .resolve()              → convert to absolute, canonical path
    # Result: C:\\EmbedSimProject\\buck_converter\\_path_utils.py
    current_path = Path(__file__).resolve()

    # .parents is a sequence of all ancestor directories, from nearest to root:
    #   current_path.parents[0] = C:\\EmbedSimProject\\buck_converter
    #   current_path.parents[1] = C:\\EmbedSimProject
    #   current_path.parents[2] = C:\\
    #   etc.
    # We iterate through them in order (nearest ancestor first).
    for parent in current_path.parents:
        # Check if this directory contains our marker file.
        # (parent / ".project_root_marker") builds the path using the / operator.
        # .exists() returns True if the file is on disk.
        if (parent / ".project_root_marker").exists():
            return parent   # ← found the project root

    # ── FALLBACK ──────────────────────────────────────────────────────────────
    # If .project_root_marker was never found (e.g., marker file was deleted,
    # or the project was cloned without it), we make a best-effort guess:
    # go two levels above this file.
    #
    # For buck_converter/_path_utils.py:
    #   .parent       = C:\\EmbedSimProject\\buck_converter
    #   .parent.parent = C:\\EmbedSimProject   (the project root)
    #
    # This is correct for the current file layout. It would break if
    # _path_utils.py were ever moved to a deeper subdirectory, which is
    # why the marker-file strategy is preferred.
    return current_path.parent.parent


def get_embedsim_import_path() -> str:
    """
    Return the directory that must be on sys.path for 'import embedsim' to work.

    Since the embedsim/ package sits directly under the project root:
        C:\\EmbedSimProject\\
            embedsim\\          ← package directory
                __init__.py
                simulation_engine.py
                ...

    Adding the PROJECT ROOT to sys.path is what makes:
        import embedsim.simulation_engine
    resolve correctly.

    Returns:
        str: the project root as a string (sys.path expects strings)

    TYPICAL USAGE in every block module:
        from _path_utils import get_embedsim_import_path
        import sys
        sys.path.insert(0, get_embedsim_import_path())
        from embedsim.simulation_engine import EmbedSim   # now works anywhere
    """
    # get_project_root() returns a Path object; str() converts it for sys.path.
    return str(get_project_root())


def get_modelica_path(name: str) -> str:
    """
    Build the full path to a Modelica (.mo) or FMU file.

    NOTE: This function is hardcoded to examples/rlc_fmu/modelica/.
    It was written for the RLC FMU example and is NOT used by the
    buck_converter package. The buck converter example builds its own
    FMU path inline in pi_buck_example.py using:
        project_root / "buck_converter" / "modelica" / "BuckConverter.fmu"

    If you need a similar helper for a different examples folder, copy
    this function and change the path segments.

    Args:
        name: filename, e.g. "RLC_Sine_DigitalTwin_OM.fmu"

    Returns:
        str: absolute path like
             "C:\\EmbedSimProject\\examples\\rlc_fmu\\modelica\\RLC...fmu"
    """
    root = get_project_root()
    # The / operator on Path joins path segments — cross-platform, no slashes needed.
    return str(root / "examples" / "rlc_fmu" / "modelica" / name)


def get_current_parent() -> Path:
    """
    Return the directory that CONTAINS this _path_utils.py file.

    For buck_converter/_path_utils.py this returns:
        C:\\EmbedSimProject\\buck_converter

    For electrical_blocks/_path_utils.py this returns:
        C:\\EmbedSimProject\\electrical_blocks

    USE CASE: When a block needs to find a sibling file (e.g., a .pyx or
    .fmu in the same folder), it can do:
        here = get_current_parent()
        pyx_path = here / "c_src" / "pi_buck_wrapper.pyx"

    Returns:
        Path: the folder containing this file
    """
    current_path = Path(__file__).resolve()
    # .parent of the file path gives the containing directory.
    return current_path.parent