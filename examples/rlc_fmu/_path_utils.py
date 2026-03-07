"""
_path_utils.py
==============

Shared path utility for the pmsm_blocks library.

Every module in this package calls ``setup_embedsim_path()`` at import time
so the embedsim package is always locatable regardless of how the script
is launched or where the working directory is.

The project root is identified by the presence of a ``.project_root_marker``
file in one of the ancestor directories of this file.
"""

import sys
from pathlib import Path


def get_project_root() -> Path:
    """
    Walk up the directory tree from this file's location until a directory
    containing ``.project_root_marker`` is found.

    Returns
    -------
    Path
        Absolute path to the project root directory.

    Notes
    -----
    Falls back to two levels above this file if the marker is never found.
    This handles the case where the framework is used without the marker
    (e.g. a plain checkout without the marker placed yet).
    """
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if (parent / ".project_root_marker").exists():
            return parent

    # Fallback: two levels up from this file's directory
    return current_path.parent.parent


def get_embedsim_import_path() -> str:
    """
    Return the directory that must be prepended to ``sys.path`` so that
    ``import embedsim`` resolves correctly.

    Returns
    -------
    str
        Absolute path string of the project root.

    Usage
    -----
    Typical call at the top of any script that needs embedsim::

        import sys
        from _path_utils import get_embedsim_import_path
        sys.path.insert(0, get_embedsim_import_path())
        import embedsim
    """
    return str(get_project_root())


def get_modelica_path(name: str) -> str:
    """
    Build the full path to a Modelica / FMU artefact stored in the
    canonical ``examples/rlc_fmu/modelica/`` folder.

    Parameters
    ----------
    name : str
        Filename (with extension) of the Modelica file, e.g.
        ``"RLC_Sine_DigitalTwin_OM.fmu"``.

    Returns
    -------
    str
        Absolute path string to the requested file.
    """
    root = get_project_root()
    return str(root / "examples" / "rlc_fmu" / "modelica" / name)


def get_current_parent() -> str:
    """
    Return the absolute path of the directory that contains this file.

    Returns
    -------
    str
        Absolute path string of the parent directory of ``_path_utils.py``.

    Notes
    -----
    Useful when a module needs to reference sibling files without knowing
    the working directory from which the interpreter was launched.
    """
    return str(Path(__file__).resolve().parent)
