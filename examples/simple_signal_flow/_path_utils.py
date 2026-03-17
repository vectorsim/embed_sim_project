"""
_path_utils.py
==============

Shared path utility for the pmsm_blocks library.

Every module in this package calls `setup_embedsim_path()` at import time
so the embedsim package is always locatable regardless of how the script
is launched or where the working directory is.

The project root is identified by the presence of a `.project_root_marker`
file in one of the parent directories.  Place that empty file at the top
of your repository::

    touch /path/to/your/project/.project_root_marker

Directory layout assumed::

    <project_root>/
    ├── .project_root_marker
    ├── embedsim/          ← the EmbedSim framework package
    │   ├── __init__.py
    │   └── ...
    └── pmsm_blocks/       ← this library
        ├── __init__.py
        └── ...

Typical usage (one line at the top of any script or module)::

    from pmsm_blocks._path_utils import setup_embedsim_path
    setup_embedsim_path()

    from embedsim import EmbedSim, ODESolver   # now always works
"""

import sys
from pathlib import Path


def get_project_root() -> Path:
    """
    Walk up the directory tree looking for a ``.project_root_marker`` file.

    Returns
    -------
    Path
        The first ancestor directory that contains ``.project_root_marker``.
        Falls back to two levels above this file if no marker is found
        (i.e. ``pmsm_blocks/_path_utils.py`` -> ``pmsm_blocks/`` -> project root).
    """
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if (parent / ".project_root_marker").exists():
            return parent

    # Fallback: pmsm_blocks/_path_utils.py  ->  pmsm_blocks/  ->  project root
    return current_path.parent.parent


def get_embedsim_import_path() -> str:
    """
    Return the project root as a string suitable for ``sys.path`` insertion,
    so that ``from embedsim import ...`` always resolves correctly.

    Does NOT modify ``sys.path`` -- call :func:`setup_embedsim_path` for that.

    Returns
    -------
    str
        The resolved project root directory (parent of ``embedsim/``).
    """
    return str(get_project_root())


def setup_embedsim_path() -> Path:
    """
    Insert the project root onto ``sys.path`` so that
    ``from embedsim import ...`` always resolves correctly.

    Safe to call multiple times -- inserts only once (guards with a
    membership check before modifying ``sys.path``).

    Returns
    -------
    Path
        The resolved project root that was (or already was) on ``sys.path``.

    Example
    -------
    ::

        from pmsm_blocks._path_utils import setup_embedsim_path
        setup_embedsim_path()

        from embedsim import EmbedSim, ODESolver   # now always works
    """
    root = get_project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
