"""
_path_utils.py  —  fs_electrical_machines
==========================================
Marker-file walk: finds the EmbedSim project root by walking upward until
a directory containing 'embedsim/' is found.

Identical contract to:
    electrical_blocks/_path_utils.py
    buck_converter/_path_utils.py
"""

from pathlib import Path


def _find_project_root(start: Path, max_levels: int = 6) -> Path:
    """Walk upward from *start* until we find a dir that contains 'embedsim/'."""
    current = start.resolve()
    for _ in range(max_levels):
        if (current / "embedsim").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(
        f"Could not locate EmbedSim project root from {start!r}. "
        "Expected to find an 'embedsim/' subdirectory somewhere up the tree."
    )


def get_embedsim_import_path() -> str:
    """Return the project root as a str suitable for sys.path.insert(0, ...)."""
    here = Path(__file__).resolve().parent          # fs_electrical_machines/
    root = _find_project_root(here)
    return str(root)


def get_current_parent() -> Path:
    """Return the absolute path of the fs_electrical_machines/ package folder."""
    return Path(__file__).resolve().parent
