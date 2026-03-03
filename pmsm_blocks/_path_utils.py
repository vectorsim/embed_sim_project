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
"""

import sys
from pathlib import Path


def get_project_root() -> Path:
    """
    Get project root by looking for a .project_root_marker file.

    Returns:
        Path object representing the project root.
        Falls back to the parent of *this* file if no marker is found.
    """
    current_path = Path(__file__).resolve()

    # Walk up the directory tree looking for the marker file
    for parent in current_path.parents:
        if (parent / ".project_root_marker").exists():
            return parent

    # Fallback: assume project root is two levels up from this file
    # (pmsm_blocks/_path_utils.py  →  pmsm_blocks/  →  project root)
    return current_path.parent.parent


def setup_embedsim_path() -> Path:
    """
    Insert <project_root>/embedsim onto sys.path so that
    ``from embedsim import ...`` always works.

    Safe to call multiple times — inserts only once.

    Returns:
        The resolved project root Path.
    """
    root = get_project_root()
    embedsim_dir = str(root / "embedsim")

    if embedsim_dir not in sys.path:
        sys.path.insert(0, embedsim_dir)

    return root
