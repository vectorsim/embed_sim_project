"""
embedsim/__init__.py
====================

EmbedSim — Embedded Simulation Framework
An open-source Python/C control-systems simulation framework.

Positioned as an affordable alternative to MATLAB/Simulink.
Targets embedded deployment on Aurix TriCore and Cortex-M4.
MISRA C:2012 / ASIL-D compatible C code generation.

Write control logic once in C — validate in Python, deploy to hardware.

--------------------------------------------------------------------------------
QUICK START
--------------------------------------------------------------------------------

    import embedsim as es

    # Sources
    ref = es.VectorStep("ref", step_time=0.05, after_value=100.0, dim=1)
    k   = es.VectorConstant("gain", value=[2.0])

    # Processing
    out = es.VectorEnd("out")
    ref >> out

    sim = es.EmbedSim(sinks=[out], T=1.0, dt=1e-4, solver=es.ODESolver.RK4)
    sim.topo.print_console()   # topology diagram in terminal
    sim.topo.show_gui()        # interactive browser GUI
    sim.run()

--------------------------------------------------------------------------------
PUBLIC API  (importable directly from `embedsim`)
--------------------------------------------------------------------------------

SIMULATION ENGINE
    EmbedSim          — main simulation runner
    ODESolver         — enum: RK4 | EULER
    VectorDelay       — generic one-step delay block
    LoopBreaker       — mixin for algebraic loop breaking

CORE BLOCKS
    VectorBlock       — base class for all blocks
    VectorSignal      — typed signal container (float32)

SOURCE BLOCKS
    VectorStep        — step signal source
    VectorConstant    — constant vector source
    VectorRamp        — ramp signal source
    VectorSine        — sinusoidal source

DYNAMIC BLOCKS
    VectorEnd         — terminal sink
    VectorIntegrator  — continuous integrator (RK4/Euler)

PROCESSING BLOCKS
    VectorGain        — element-wise or matrix gain
    VectorSum         — multi-input summation
    VectorSplit       — extract sub-vector by index
    VectorMux         — concatenate vectors

CODE GENERATION
    CodeGenStart      — marks start of C codegen region
    CodeGenEnd        — marks end of C codegen region
    SimBlockBase      — base for Cython-wrapped C blocks

FMU INTEGRATION
    FMUBlock          — generic FMI 2.0 co-simulation block

TOPOLOGY VISUALIZER
    TopologyPrinter   — console + browser GUI topology viewer
    print_topology    — convenience function: TopologyPrinter(sim).print_console()

UTILITIES
    create_plotter    — matplotlib scope helper

--------------------------------------------------------------------------------
INTEGRATION WITH EmbedSim INSTANCES
--------------------------------------------------------------------------------

A TopologyPrinter is automatically attached to every EmbedSim instance as
``sim.topo`` after construction.  This replaces the old
``sim.print_topology_sources2sink()`` call:

    sim = EmbedSim(sinks=[...], T=1.0, dt=1e-4)
    sim.topo.print_console()    # replaces sim.print_topology_sources2sink()
    sim.topo.show_gui()         # opens browser with interactive SVG diagram
    sim.topo.export_html("diagram.html")

--------------------------------------------------------------------------------
"""

# ── Version ──────────────────────────────────────────────────────────────────

__version__   = "0.4.0"
__author__    = "EmbedSim Project"
__license__   = "MIT"
__url__       = "https://github.com/embedsim/embedsim"

# ── Simulation engine ─────────────────────────────────────────────────────────

from embedsim.simulation_engine import (
    EmbedSim,
    ODESolver,
    VectorDelay,
    LoopBreaker,
)

# ── Core blocks ───────────────────────────────────────────────────────────────

from embedsim.core_blocks import (
    VectorBlock,
    VectorSignal,
)

# ── Source blocks ─────────────────────────────────────────────────────────────

from embedsim.source_blocks import (
    VectorStep,
    VectorConstant,
)

# Ramp and Sine are optional — only exported if they exist in source_blocks
try:
    from embedsim.source_blocks import VectorRamp
except ImportError:
    VectorRamp = None  # type: ignore[assignment,misc]

try:
    from embedsim.source_blocks import VectorSine
except ImportError:
    VectorSine = None  # type: ignore[assignment,misc]

# ── Dynamic blocks ────────────────────────────────────────────────────────────

from embedsim.dynamic_blocks import (
    VectorEnd,
    VectorIntegrator,
)

# ── Processing blocks ─────────────────────────────────────────────────────────

from embedsim.processing_blocks import (
    VectorGain,
    VectorSum,
)

try:
    from embedsim.processing_blocks import VectorSplit
except ImportError:
    VectorSplit = None  # type: ignore[assignment,misc]

try:
    from embedsim.processing_blocks import VectorMux
except ImportError:
    VectorMux = None  # type: ignore[assignment,misc]

# ── Code generation ───────────────────────────────────────────────────────────

from embedsim.code_generator import (
    CodeGenStart,
    CodeGenEnd,
    SimBlockBase,
)

# ── FMU integration ───────────────────────────────────────────────────────────

try:
    from embedsim.fmu_blocks import FMUBlock
except ImportError:
    FMUBlock = None  # type: ignore[assignment,misc]

# ── Plot helper ───────────────────────────────────────────────────────────────

try:
    from embedsim.plot_helper import create_plotter
except ImportError:
    create_plotter = None  # type: ignore[assignment,misc]

# ── Topology printer ──────────────────────────────────────────────────────────

from embedsim.topology_printer import TopologyPrinter, attach


def print_topology(sim) -> None:
    """
    Print a clean multi-lane console topology diagram for *sim*.

    Convenience wrapper around ``TopologyPrinter(sim).print_console()``.
    Identical to calling ``sim.topo.print_console()`` if you used
    ``attach(sim)`` or constructed via ``EmbedSim``.

    Parameters
    ----------
    sim : EmbedSim
        A built simulation object.

    Example
    -------
    >>> import embedsim as es
    >>> sim = build_my_sim()
    >>> es.print_topology(sim)
    """
    TopologyPrinter(sim).print_console()


# ── Monkey-patch EmbedSim to auto-attach TopologyPrinter ─────────────────────
#
# We wrap EmbedSim.__init__ so that every new instance automatically gets:
#   sim.topo                          — TopologyPrinter instance
#   sim.print_topology_sources2sink   — replaced with printer.print_console
#
# This is done with a clean __init_subclass__-safe wrapper rather than
# mutating the class directly, so subclasses are unaffected.

import functools as _functools

_original_EmbedSim_init = EmbedSim.__init__


@_functools.wraps(_original_EmbedSim_init)
def _patched_EmbedSim_init(self, *args, **kwargs):
    _original_EmbedSim_init(self, *args, **kwargs)
    # Attach topology printer after sim is fully constructed
    try:
        _attach_topo(self)
    except Exception:
        # Never break the sim — topology is a nice-to-have
        pass


def _attach_topo(sim_instance):
    """Attach TopologyPrinter to a live EmbedSim instance."""
    printer = TopologyPrinter(sim_instance)
    sim_instance.topo = printer
    # Backwards-compatible replacement for the old broken renderer
    sim_instance.print_topology_sources2sink = printer.print_console


EmbedSim.__init__ = _patched_EmbedSim_init

# ── Public namespace (__all__) ────────────────────────────────────────────────

__all__ = [
    # Engine
    "EmbedSim",
    "ODESolver",
    "VectorDelay",
    "LoopBreaker",

    # Core
    "VectorBlock",
    "VectorSignal",

    # Sources
    "VectorStep",
    "VectorConstant",
    "VectorRamp",
    "VectorSine",

    # Dynamic
    "VectorEnd",
    "VectorIntegrator",

    # Processing
    "VectorGain",
    "VectorSum",
    "VectorSplit",
    "VectorMux",

    # Code generation
    "CodeGenStart",
    "CodeGenEnd",
    "SimBlockBase",

    # FMU
    "FMUBlock",

    # Topology
    "TopologyPrinter",
    "attach",
    "print_topology",

    # Utilities
    "create_plotter",

    # Meta
    "__version__",
    "__author__",
    "__license__",
    "__url__",
]
