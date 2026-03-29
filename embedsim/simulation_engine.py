"""
Enhanced Simulation Engine with Feedback Loop Support
======================================================

Overview:
---------
This simulation engine extends the original EmbedSim Framework to
handle systems with **feedback loops** safely and efficiently. It provides
a framework for simulating interconnected vector-based blocks, where some
blocks have internal states (dynamic) and others are stateless (static).

Key Concepts:
-------------
1. **Blocks**:
   - Each block represents a computational unit (e.g., sum, gain, delay).
   - Blocks have inputs and outputs represented as VectorSignal objects.
   - Blocks can be **dynamic** (with internal state, integrated over time)
     or **static** (instantaneous computation).

2. **Loop Breakers**:
   - Feedback loops can create algebraic cycles that prevent topological
     ordering of computations.
   - Blocks implementing the `LoopBreaker` interface (e.g., `VectorDelay`)
     provide a previous output instead of current input for loop-breaking.
   - This ensures simulation proceeds without algebraic loops while retaining
     causal behavior.
   - **Implemented in:** `LoopBreaker` class & `VectorDelay` class.

3. **Dependency Graph Traversal**:
   - Builds a **topologically sorted execution order** from sinks to sources.
   - DFS traversal is used, but loop breakers are not recursively traversed.
   - Detects true algebraic loops and raises errors if no loop breaker exists.
   - **Implemented in:** `traverse_blocks_from_sinks_with_loops()` function.

How It Works:
-------------
Step-by-step flow of the simulation:

1. **Block Creation and Connection**
   - Create blocks (sources, processing blocks, sinks).
   - Connect outputs to inputs forming the system graph.
   - Some blocks may be dynamic (with internal states), others static.

2. **Loop Breaker Identification**
   - Engine scans for blocks implementing `LoopBreaker`.
   - Loop breakers supply previous outputs for feedback paths to prevent cycles.

3. **Execution Order Computation**
   - Depth-First Search (DFS) is used to traverse blocks from sinks backward.
   - Loop breakers are added but their inputs are not traversed recursively.
   - Produces a **topological order** for safe block computation.

4. **Simulation Loop**
   - For each timestep `t` from 0 to `T`:
       a. **Compute All Blocks**: compute outputs in topological order.
          - **Algorithm:** regular forward computation.
          - **Implemented in:** `_compute_all_blocks()` method.
       b. **Record Signals**: store outputs in `VectorScope` for later analysis.
          - **Implemented in:** `VectorScope.record()`.
       c. **Integrate Dynamics**: dynamic blocks updated using chosen ODE solver.
          - **Euler**: first-order, fast (`_integrate_dynamics_euler()`).
          - **RK4**: fourth-order accurate (`_integrate_dynamics_rk4()`).
          - **Heun**: second-order compromise (can be implemented similarly).
       d. **Progress Display**: optional progress bar.

5. **Finalization**
   - Final computation at last timestep.
   - Final recording of signals.
   - Statistics available in `SimulationStats`:
     total steps, compute time, loop breakers, feedback loops.

Visual Representation:
---------------------
Simplified ASCII diagram:

Sources ──► [Block1] ──► [VectorDelay] ──► Sinks
              ↑               │
              │_______________│
          (feedback path, broken by VectorDelay)

Legend:
- `[Block]`: computational block
- `VectorDelay`: loop breaker that provides previous output to break feedback
- Arrows: signal flow direction

Algorithms Summary:
-------------------
1. **Topological Sorting with Loop Breakers**
   - Algorithm: Depth-First Search (DFS)
   - Implemented in: `traverse_blocks_from_sinks_with_loops()`

2. **Loop-Breaking Mechanism**
   - Provides previous output for blocks in feedback paths
   - Implemented in: `LoopBreaker.get_loop_breaking_output()`, `VectorDelay`

3. **Dynamic Block Integration**
   - Algorithms: Euler, RK4 (Runge-Kutta 4), Heun (2nd order)
   - Implemented in: `_integrate_dynamics_euler()`, `_integrate_dynamics_rk4()`

4. **Signal Recording**
   - Stores scalar or vector signals for plotting and analysis
   - Implemented in: `VectorScope.record()`

Typical Workflow:
-----------------
1. Create and connect blocks.
2. Instantiate `VectorSim` with sinks, T, dt, solver.
3. Add blocks to `scope` for recording.
4. Call `sim.run()` to simulate.
5. Use `print_topology*()` or `plot()` to visualize results.

Author: EmbedSim Framework - Enhanced Edition
Version: 3.1.0

Changelog v3.1.0:
  - TopologyPrinter auto-attached to every EmbedSim instance as sim.topo
  - sim.topo.print_console()  — clean multi-lane ASCII diagram (replaces broken renderer)
  - sim.topo.show_gui()       — opens interactive browser SVG diagram
  - sim.topo.export_html(p)  — save standalone HTML topology file
  - sim.print_topology()          — concise block count + execution order table
  - Fixed plot() referencing self.data instead of self.scope.data
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass
import time
import logging
import os

from .core_blocks import (VectorBlock, VectorSignal, DEFAULT_DTYPE)

# ---------------------------------------------------------------------------
# Topology printer — imported lazily to avoid circular imports.
# The actual attach happens at the end of EmbedSim.__init__.
# ---------------------------------------------------------------------------
def _get_topology_printer():
    """Lazy import of TopologyPrinter to avoid circular dependency."""
    try:
        from .topology_printer import TopologyPrinter
        return TopologyPrinter
    except ImportError:
        return None


# =============================================================================
# SimulationLogger
# =============================================================================
#
# Why a dedicated logger instead of print()?
# -------------------------------------------
# Python's print() has no concept of severity, no timestamps, and writes
# to no persistent record.  Python's stdlib `logging` module solves all of
# that.  SimulationLogger wraps it with EmbedSim-specific conveniences:
#
#   • Anchored log directory — log files land in <project_root>/embedsim_log/
#     regardless of which directory Python is launched from.  The anchor is
#     __file__ (this source file), so the path is always deterministic.
#
#   • Per-block parse logging  — log_parse(block) is called by the graph
#     traversal so you can see exactly which blocks were discovered and in
#     what order during the DFS.
#
#   • Per-block execute logging — log_execute(step, t, block) is called
#     inside _compute_all_blocks() for every block on every timestep.
#     This is DEBUG level (silent at INFO) but produces a full audit trail
#     in the log file.
#
#   • Severity levels:
#       DEBUG    (10) — per-step, per-block detail
#       INFO     (20) — run start/end, block counts, topology summary
#       WARNING  (30) — unexpected but recoverable (zero-fallback input)
#       ERROR    (40) — serious failures (algebraic loop detected)
#
# Log file location:
#   <this_file>/../embedsim_log/<sink_name>_YYYYMMDD_HHMMSS.log
#   e.g.  C:\EmbedSimProject\embed_sim_project\embedsim_log\pmsm_foc_20260315_143200.log

# Resolve the project root from this file's location so the log directory
# is always <project_root>/embedsim_log/ — never a mystery CWD location.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_PROJECT_ROOT, "embedsim_log")


class SimulationLogger:
    """
    Structured logger for EmbedSim — records graph parsing, block execution,
    and simulation lifecycle events to both console and a timestamped file.

    Log file is always written to ``<project_root>/embedsim_log/``.

    Attributes:
        log_file (str | None): Absolute path of the active .log file.
        logger (logging.Logger): The underlying stdlib Logger object.

    Usage::

        # Accessible as sim.logger after EmbedSim.__init__:
        sim.logger.info("Custom message")
        sim.logger.log_parse(block)          # during graph traversal
        sim.logger.log_execute(step, t, block)  # during compute pass
    """

    def __init__(
        self,
        name: str = "EmbedSim",
        console_level: int = logging.INFO,
        log_to_file: bool = True,
    ) -> None:
        self.name = name
        self.log_file: Optional[str] = None

        # Named logger — one per EmbedSim instance, identified by sink name.
        # Using a unique name prevents handler accumulation across multiple
        # EmbedSim objects in the same Python process (e.g. Jupyter notebook
        # cells that run repeatedly without restarting the kernel).
        self.logger = logging.getLogger(f"EmbedSim.{name}")
        self.logger.setLevel(logging.DEBUG)  # accept everything; handlers filter

        if self.logger.handlers:
            # Already initialised (e.g. second call in same process) — clear
            # stale handlers so we don't get duplicate log lines.
            self.logger.handlers.clear()

        # ── Console handler ──────────────────────────────────────────────────
        # Always active.  Level is configurable; default INFO keeps the
        # console clean during normal runs.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(levelname)-7s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        ))
        self.logger.addHandler(console_handler)

        # ── File handler ─────────────────────────────────────────────────────
        # Always DEBUG level so the file contains the full record (including
        # per-block parse and execute lines) even when the console shows INFO.
        # The directory is anchored to the project root via _LOG_DIR so it
        # is always predictable regardless of the launch CWD.
        if log_to_file:
            try:
                os.makedirs(_LOG_DIR, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self.log_file = os.path.join(_LOG_DIR, f"{name}_{timestamp}.log")
                file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(logging.Formatter(
                    fmt="%(asctime)s [%(levelname)-7s] %(name)s — %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                ))
                self.logger.addHandler(file_handler)
            except OSError as exc:
                self.logger.warning(f"Could not open log file in {_LOG_DIR}: {exc}")

    # ── Severity helpers ──────────────────────────────────────────────────────

    def debug(self, msg: str) -> None:
        """Emit DEBUG — verbose detail, per-step internals."""
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        """Emit INFO — normal lifecycle events (run start, block counts)."""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Emit WARNING — unexpected but recoverable conditions."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Emit ERROR — serious failure; results may be invalid."""
        self.logger.error(msg)

    # ── EmbedSim-specific structured log helpers ──────────────────────────────

    def log_parse(self, block) -> None:
        """
        Record that a block was discovered during graph traversal (DFS parse).

        Called by traverse_blocks_from_sinks_with_loops() each time a block
        is added to the execution order.  The log line includes the block's
        sequential position, name, type, and whether it is dynamic or a loop
        breaker — giving a complete picture of the graph as it is built.

        Example log line (DEBUG level, visible in file, silent on console)::

            [DEBUG] EmbedSim.pmsm — PARSE  #003  inv_park        (InvParkBlock      ) dynamic=False  loop_breaker=False
        """
        is_lb = getattr(block, 'is_loop_breaker', False)
        is_dyn = getattr(block, 'is_dynamic', False)
        self.logger.debug(
            f"PARSE  {block.name:<30s}  type={type(block).__name__:<25s}"
            f"  dynamic={is_dyn}  loop_breaker={is_lb}"
        )

    def log_execute(self, step: int, t: float, block) -> None:
        """
        Record that a block is about to be computed at a given timestep.

        Called inside _compute_all_blocks() for every block on every step.
        This is DEBUG level — it produces one line per block per timestep,
        which can be thousands of lines per second, so it only appears in
        the log file (not the console at INFO level).

        Use this to answer questions like:
          "Is block 'smc' being executed before 'inv_park' at step 1200?"
          "Which block is producing NaN at t = 0.005 s?"

        Example log line::

            [DEBUG] EmbedSim.pmsm — EXEC   step=00042  t=0.042000  smc             (SMCBlock)
        """
        self.logger.debug(
            f"EXEC   step={step:05d}  t={t:.6f} s  "
            f"{block.name:<30s}  ({type(block).__name__})"
        )

    def log_step(self, step: int, t: float, message: str = "") -> None:
        """
        Emit one summary DEBUG line per timestep (not per block).

        Lighter than log_execute when you only need to confirm the step
        counter is advancing correctly, not trace individual blocks.
        """
        self.logger.debug(f"STEP   step={step:05d}  t={t:.6f} s  {message}")

    def close(self) -> None:
        """Flush and close all handlers — call after sim.run() finishes."""
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)


# =========================
# Loop Breaker Interface
# =========================

class LoopBreaker:
    """
    Mixin interface for blocks that break algebraic feedback loops.

    In a block diagram with feedback, a direct cycle (A → B → A) creates
    a circular dependency that prevents topological sorting.  A LoopBreaker
    resolves this by supplying its *previous* timestep output instead of
    requiring its current input to be computed first.

    Any block that participates in a feedback path should inherit from this
    class alongside VectorBlock.  The simulation engine detects LoopBreaker
    instances during graph traversal and treats their edges as "already
    resolved", effectively cutting the cycle.

    Concrete examples: VectorDelay, UnitDelay, DiscreteIntegrator.

    Attributes:
        is_loop_breaker (bool): Class-level flag; always True.  Used by the
            engine to identify loop-breaking blocks without isinstance checks
            on every graph edge.

    Usage:
        class MyDelay(VectorBlock, LoopBreaker):
            def get_loop_breaking_output(self):
                return self.last_output   # value from previous timestep
    """

    is_loop_breaker: bool = True

    def get_loop_breaking_output(self) -> Optional[VectorSignal]:
        """
        Return the signal value used to break the feedback cycle.

        Called by the simulation engine *before* the main compute pass so
        that downstream blocks can use this block's output even though its
        own inputs have not been computed yet for the current timestep.

        Implementations should return the output stored from the previous
        timestep (or a user-supplied initial value at t = 0).

        Returns:
            VectorSignal: The previously stored output signal, or None if
                no previous value is available (engine will use zero).

        Raises:
            NotImplementedError: If a subclass forgets to implement this.
        """
        raise NotImplementedError("Loop breakers must implement get_loop_breaking_output()")


# =========================
# VectorDelay
# =========================

class VectorDelay(VectorBlock, LoopBreaker):
    """
    One-timestep vector delay block that safely breaks feedback loops.

    VectorDelay outputs the signal it received in the *previous* timestep,
    making it causal and suitable for placement in any feedback path.  It
    implements the LoopBreaker interface so the simulation engine can
    topologically sort the graph without treating the feedback edge as a
    circular dependency.

    Mathematical behaviour:
        y(t) = u(t - dt)      for t > 0
        y(0) = initial        (user-supplied, or zero if omitted)

    Typical use — closing a PID control loop:
        error  = VectorSum("e",  [setpoint, delay], signs=[1, -1])
        pid    = PIDBlock("pid", [error])
        plant  = PlantModel("plant", [pid])
        delay  = VectorDelay("delay", initial=[0.0])   # feeds back plant output
        plant >> delay
        delay >> error   # delay sits in the feedback path, breaking the cycle

    Attributes:
        last_output (Optional[VectorSignal]): The signal stored from the
            previous timestep.  None until the first compute() call when
            no initial value was provided.
        is_loop_breaker (bool): Always True (inherited from LoopBreaker).
    """

    is_loop_breaker = True

    def __init__(self, name: str, initial: Optional[List[float]] = None) -> None:
        """
        Initialise the delay block.

        Args:
            name:    Unique identifier for this block.
            initial: Optional initial output vector for t = 0.  If None the
                     block outputs zeros on the first timestep.  The length
                     must match the signal dimension used in the rest of the
                     diagram.

        Example:
            >>> delay = VectorDelay("fb_delay", initial=[0.0, 0.0, 0.0])
        """
        super().__init__(name)
        if initial is not None:
            self.last_output = VectorSignal(np.array(initial, dtype=DEFAULT_DTYPE))
        else:
            self.last_output = None

    def get_loop_breaking_output(self) -> Optional[VectorSignal]:
        """
        Return the stored output from the previous timestep.

        Called by the engine *before* the main compute pass so that
        downstream blocks can read this block's output even before its own
        inputs are resolved for the current step.

        Returns:
            VectorSignal: Previous timestep output, or None if no value has
                been stored yet (engine treats None as zero).
        """
        return self.last_output

    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Emit the previous timestep's input and buffer the current input.

        On each call the block:
          1. Reads ``last_output`` (the value stored in the previous step).
          2. Stores the current input as ``last_output`` for the next step.
          3. Returns the old value as ``self.output``.

        This one-step delay breaks the circular dependency in feedback loops
        while keeping the signal chain causal.

        Args:
            t:            Current simulation time in seconds (not used directly).
            dt:           Simulation timestep in seconds (not used directly).
            input_values: List with exactly one VectorSignal — the signal
                          arriving at this block's input port.

        Returns:
            VectorSignal: The delayed signal (previous timestep's input).

        Raises:
            ValueError: If ``input_values`` is empty or None.

        Example:
            >>> delay = VectorDelay("d", initial=[0.0])
            >>> sig_t0 = VectorSignal([5.0])
            >>> out_t0 = delay.compute(0.0, 0.01, [sig_t0])
            >>> print(out_t0.value)   # [0.0]  — the initial value
            >>> sig_t1 = VectorSignal([7.0])
            >>> out_t1 = delay.compute(0.01, 0.01, [sig_t1])
            >>> print(out_t1.value)   # [5.0]  — previous input
        """
        if not input_values:
            raise ValueError(f"{self.name}: No input provided")

        # Output is previous input (or initial value)
        if self.last_output is not None:
            val = self.last_output.value.copy()
        else:
            val = np.zeros_like(input_values[0].value)

        # Store current input for next timestep
        self.last_output = VectorSignal(input_values[0].value.copy())

        self.output = VectorSignal(val, self.name)
        return self.output


# =========================
# Enhanced Dependency Graph Traversal
# =========================

def traverse_blocks_from_sinks_with_loops(
    sinks: List[VectorBlock],
    logger: Optional["SimulationLogger"] = None,
) -> List[VectorBlock]:
    """
    Build a topologically sorted execution order for all blocks reachable
    from the given sink blocks, correctly handling feedback loops.

    Algorithm — two-pass DFS
    ------------------------
    Pass 1  (find_loop_breakers):
        Walk the full graph recursively from every sink.  Any block that
        implements the LoopBreaker interface is recorded in ``loop_breakers``.

    Pass 2  (dfs):
        Walk the graph again with a standard post-order DFS.  When an input
        edge points to a LoopBreaker the edge is *not* followed recursively;
        instead the loop-breaker block is inserted into the order immediately
        (it will supply its previous-timestep output at runtime).  This cuts
        feedback cycles so the DFS terminates without infinite recursion.

        A ``visiting`` set detects *true* algebraic loops — cycles that have
        no LoopBreaker to cut them — and raises ValueError with a descriptive
        message.

    Args:
        sinks:  List of terminal (sink) VectorBlock objects.  Traversal starts
                here and walks backwards through the ``inputs`` graph.
        logger: Optional SimulationLogger.  When provided, every block that
                is added to the execution order is logged via log_parse() at
                DEBUG level, giving a full trace of graph discovery order.

    Returns:
        List[VectorBlock]: All reachable blocks ordered so that every block
        appears *after* all of its non-loop-breaking dependencies.  This list
        is the safe sequential execution order for the simulation engine.

    Raises:
        ValueError: If a true algebraic loop is detected (a cycle exists with
            no LoopBreaker block to break it).  The error message names the
            block where the cycle was detected.

    Example:
        >>> blocks = traverse_blocks_from_sinks_with_loops([output_sink])
        >>> for b in blocks:
        ...     b.compute(t, dt, [inp.output for inp in b.inputs])
    """

    blocks_order = []
    blocks_set = set()
    visiting = set()
    loop_breakers = set()

    def _log_parse(block):
        """Emit a PARSE log line if a logger is attached."""
        if logger is not None:
            logger.log_parse(block)

    # First pass: identify all loop breakers
    def find_loop_breakers(block: VectorBlock):
        if block in blocks_set or block in visiting:
            return
        visiting.add(block)

        if isinstance(block, LoopBreaker):
            loop_breakers.add(block)

        for inp in block.inputs:
            find_loop_breakers(inp)

        visiting.remove(block)
        blocks_set.add(block)

    for sink in sinks:
        find_loop_breakers(sink)

    # Reset for second pass
    blocks_set.clear()
    visiting.clear()

    # Second pass: build execution order
    def dfs(block: VectorBlock) -> None:
        if block in blocks_set:
            return

        if block in visiting:
            raise ValueError(
                f"Algebraic loop detected at block '{block.name}'. "
                f"Add a VectorDelayEnhanced in the feedback path to break the loop."
            )

        visiting.add(block)

        # Process inputs
        for inp in block.inputs:
            if inp not in loop_breakers:
                # Normal dependency - recurse
                dfs(inp)
            else:
                # Loop breaker - add it but don't follow its inputs
                if inp not in blocks_set:
                    blocks_set.add(inp)
                    blocks_order.append(inp)
                    _log_parse(inp)   # ← log loop-breaker discovery

        visiting.remove(block)
        blocks_set.add(block)
        blocks_order.append(block)
        _log_parse(block)             # ← log normal block discovery

    for sink in sinks:
        dfs(sink)

    # Second pass: any block that feeds a LoopBreaker from the feed-forward
    # direction may have been skipped by the DFS (because LoopBreaker inputs
    # are not followed).  Find all such blocks and insert them into the order
    # immediately before the LoopBreaker that depends on them.
    # We iterate until stable to handle chains of missing blocks.
    changed = True
    while changed:
        changed = False
        for lb in list(loop_breakers):
            if lb not in blocks_set:
                continue
            lb_idx = blocks_order.index(lb)
            for inp in lb.inputs:
                if inp not in blocks_set:
                    # inp was never visited — run a mini-DFS to add it and
                    # its own dependencies, then insert before lb.
                    mini_order = []
                    mini_visited = set()

                    def mini_dfs(b):
                        if b in blocks_set or b in mini_visited:
                            return
                        mini_visited.add(b)
                        for b_inp in b.inputs:
                            if b_inp not in loop_breakers:
                                mini_dfs(b_inp)
                        mini_visited.discard(b)
                        blocks_set.add(b)
                        mini_order.append(b)
                        _log_parse(b)  # ← log late-discovered block

                    mini_dfs(inp)
                    if mini_order:
                        # Find insertion point: just before lb's current position
                        lb_idx = blocks_order.index(lb)
                        for i, b in enumerate(mini_order):
                            blocks_order.insert(lb_idx + i, b)
                        changed = True

    return blocks_order


# =========================
# Enhanced Simulation Engine
# =========================

@dataclass
class SimulationStats:
    """
    Runtime statistics collected during a simulation run.

    Populated by VectorSim.run() and available after the simulation
    completes.  Useful for benchmarking and diagnosing performance.

    Attributes:
        total_steps (int):          Number of discrete timesteps executed,
                                    equal to int(T / dt).
        compute_time (float):       Wall-clock time (seconds) for the full
                                    simulation loop, excluding setup.
        avg_step_time (float):      Mean wall-clock time per timestep
                                    (compute_time / total_steps).
        loop_breakers_count (int):  Number of LoopBreaker blocks found in
                                    the block graph.
        feedback_loops_count (int): Estimated number of feedback loops
                                    (currently equals loop_breakers_count).

    Example:
        >>> sim.run()
        >>> print(f"Ran {sim.stats.total_steps} steps in "
        ...       f"{sim.stats.compute_time:.3f} s")
    """
    total_steps: int = 0
    compute_time: float = 0.0
    avg_step_time: float = 0.0
    loop_breakers_count: int = 0
    feedback_loops_count: int = 0


class VectorScope:
    """
    Signal recorder for post-simulation analysis and plotting.

    VectorScope acts as an oscilloscope: you register blocks of interest
    before the run, and it samples their outputs at every timestep.
    After the run, signals are available as NumPy arrays indexed by label
    and vector component.

    Attributes:
        data (Dict[str, List[float]]):
            Scalar time-series keyed as ``"label[i]"`` where *i* is the
            vector component index.  Populated by record().
        full_signals (Dict[str, List[np.ndarray]]):
            Full vector snapshots keyed by label.  Each entry is a list of
            1-D arrays, one per timestep.  Only stored when record_full=True.
        t (List[float]):
            Simulation time values corresponding to each recorded sample.
        monitored_blocks (Dict[VectorBlock, Dict]):
            Internal registry mapping each monitored block to its recording
            configuration (label, indices, record_full flag).

    Typical usage:
        >>> scope = sim.scope
        >>> scope.add(integrator, indices=[0], label="x_pos")
        >>> scope.add(velocity_block, label="vel")   # records all components
        >>> sim.run()
        >>> x = scope.get_signal("x_pos", index=0)  # → np.ndarray of shape (N,)
    """

    def __init__(self) -> None:
        self.data: Dict[str, List[float]] = {}
        self.full_signals: Dict[str, List[np.ndarray]] = {}
        self.t: List[float] = []
        self.monitored_blocks: Dict[VectorBlock, Dict] = {}

    def add(self, block: VectorBlock, indices: Optional[List[int]] = None,
            label: Optional[str] = None, record_full: bool = True) -> None:
        """
        Register a block for signal recording.

        Must be called before sim.run().  The block's output is sampled at
        every timestep and stored in self.data under the given label.

        Args:
            block:       The VectorBlock whose output should be recorded.
            indices:     List of vector component indices to record as
                         individual scalar channels (e.g. ``[0, 2]``).
                         If None, *all* components are recorded.
            label:       Key used to retrieve the signal via get_signal().
                         Defaults to block.name if not provided.
            record_full: If True (default) also store the complete vector
                         snapshot in full_signals for later numpy access.

        Example:
            >>> sim.scope.add(motor_block, indices=[0], label="speed")
            >>> sim.scope.add(abc_block)            # records all 3 phases
        """
        label = label if label else block.name
        self.monitored_blocks[block] = {
            'indices': indices,
            'label': label,
            'record_full': record_full
        }

    def record(self, t: float) -> None:
        """
        Sample all registered blocks and store their current outputs.

        Called automatically by the simulation engine at every timestep.
        Do not call this manually unless implementing a custom engine.

        Args:
            t: Current simulation time in seconds.  Appended to self.t.

        Note:
            Blocks whose output is still None (not yet computed) are silently
            skipped for that timestep.
        """
        self.t.append(t)

        for block, config in self.monitored_blocks.items():
            if block.output is not None:
                val = block.output.value
                indices = config['indices']
                label = config['label']
                record_full = config.get('record_full', False)

                if record_full:
                    self.full_signals.setdefault(label, []).append(val.copy())

                if indices is None:
                    for i in range(len(val)):
                        key = f"{label}[{i}]"
                        self.data.setdefault(key, []).append(val[i])
                else:
                    for i in indices:
                        if i < len(val):
                            key = f"{label}[{i}]"
                            self.data.setdefault(key, []).append(val[i])

    def get_signal(self, label: str, index: int = 0) -> Optional[np.ndarray]:
        """
        Retrieve a recorded scalar channel as a NumPy array.

        Args:
            label: The label used in scope.add() (or block.name by default).
            index: Vector component index (0-based).  For a scalar signal
                   use the default of 0.

        Returns:
            np.ndarray of shape (N,) containing the time-series values, or
            None if the requested label/index combination was never recorded.

        Example:
            >>> x = sim.scope.get_signal("integrator", index=0)
            >>> plt.plot(sim.scope.t, x)
        """
        key = f"{label}[{index}]"
        return np.array(self.data[key]) if key in self.data else None

    def get_full_signal(self, label: str) -> Optional[np.ndarray]:
        """
        Retrieve all vector components of a recorded signal as a 2-D array.

        Args:
            label: The label used in scope.add().

        Returns:
            np.ndarray of shape (N, dim) where N is the number of timesteps
            and dim is the signal dimension, or None if not found.

        Example:
            >>> abc = sim.scope.get_full_signal("3phase_gen")
            >>> phase_u = abc[:, 0]
            >>> phase_v = abc[:, 1]
        """
        return np.array(self.full_signals[label]) if label in self.full_signals else None


# =========================
# ODE Solver Selection
# =========================

class ODESolver:
    """
    Enumeration of available ODE integration methods.

    Methods:
        EULER: Forward Euler (first-order, fast, less accurate)
        RK4: Runge-Kutta 4 (fourth-order, slower, most accurate)
        HEUN: Heun's method (second-order, good compromise)

    Example:
        >>> sim = EmbedSim(sinks=[sink], T=0.1, dt=0.0001, solver=ODESolver.RK4)
    """
    EULER = 'euler'
    RK4 = 'rk4'
    HEUN = 'heun'


class EmbedSim:
    """
    Main simulation engine for the EmbedSim Framework.

    EmbedSim orchestrates the complete simulation lifecycle:
      1. Builds the topologically sorted execution order from the block graph.
      2. Categorises blocks into dynamic (stateful) and static groups.
      3. Identifies LoopBreaker blocks for feedback handling.
      4. Steps through time, computing blocks, recording signals, and
         integrating dynamic states using the selected ODE solver.

    Supported ODE solvers (via ODESolver constants):
        - ``ODESolver.EULER``  — Forward Euler, first-order, fastest.
        - ``ODESolver.RK4``    — Runge-Kutta 4, fourth-order accurate.
        - ``ODESolver.HEUN``   — Heun's method, second-order (planned).

    Attributes:
        sinks (List[VectorBlock]):          Sink blocks passed at construction.
        T (float):                          Total simulation duration (seconds).
        dt (float):                         Fixed timestep size (seconds).
        solver (str):                       Active ODE solver identifier.
        scope (VectorScope):                Signal recorder; add blocks here
                                            before calling run().
        stats (SimulationStats):            Performance metrics; populated
                                            after run() completes.
        blocks (List[VectorBlock]):         All blocks in execution order.
        dynamic_blocks (List[VectorBlock]): Blocks with internal state.
        static_blocks (List[VectorBlock]):  Stateless (combinatorial) blocks.
        loop_breakers (List[VectorBlock]):  Blocks that cut feedback cycles.

    Example:
        >>> sim = EmbedSim(sinks=[output], T=1.0, dt=0.001,
        ...                 solver=ODESolver.RK4)
        >>> sim.scope.add(my_block, indices=[0], label="signal")
        >>> sim.run()
        >>> data = sim.scope.get_signal("signal")
    """

    def __init__(self, sinks: List[VectorBlock], T: float, dt: float, solver: str = 'rk4') -> None:
        """
        Initialise and configure the simulation engine.

        Immediately traverses the block graph to build the execution order.
        Raises an error if the graph contains an unbreakable algebraic loop.

        Args:
            sinks:  One or more terminal (sink) VectorBlock objects.  The
                    graph traversal starts here and walks backwards through
                    the ``inputs`` links to discover all connected blocks.
            T:      Total simulation time in seconds.
            dt:     Fixed integration timestep in seconds.  Smaller values
                    give more accurate results but increase runtime.
            solver: ODE integration method.  Use ODESolver constants:
                    ``ODESolver.EULER``, ``ODESolver.RK4``, or
                    ``ODESolver.HEUN``.  Default is ``'rk4'``.

        Raises:
            ValueError: If the block graph contains a true algebraic loop
                (circular dependency with no LoopBreaker to cut it).

        Example:
            >>> sim = EmbedSim(sinks=[sink_block], T=5.0, dt=0.01)
        """
        self.sinks = sinks
        self.T = T
        self.dt = dt
        self.solver = solver
        self.scope = VectorScope()
        self.stats = SimulationStats()

        # ── Logger — created first so all subsequent init steps can emit logs ─
        # Uses the first sink's name as the logger/file identity so each
        # simulation instance produces a distinct log file in embedsim_log/.
        # The log directory is anchored to the project root (see _LOG_DIR),
        # not the launch CWD, so the file is always in a predictable location.
        sink_name = sinks[0].name if sinks else "unnamed"
        self.logger = SimulationLogger(name=sink_name, log_to_file=True)
        self.logger.info(
            f"EmbedSim init  T={T} s  dt={dt} s  solver={solver.upper()}  "
            f"log={self.logger.log_file}"
        )

        # ── Build execution order with loop support ─────────────────────────
        # Pass the logger so traverse_blocks_from_sinks_with_loops() can call
        # logger.log_parse(block) for every block it adds to the order — giving
        # a full audit trail of graph discovery in the log file.
        try:
            self.blocks = traverse_blocks_from_sinks_with_loops(
                self.sinks, logger=self.logger
            )
        except ValueError as e:
            self.logger.error(f"Block diagram error: {e}")
            raise ValueError(f"Block diagram error: {e}")

        # Categorize blocks
        self.dynamic_blocks = [b for b in self.blocks if b.is_dynamic]
        self.loop_breakers  = [b for b in self.blocks if isinstance(b, LoopBreaker)]
        self.static_blocks  = [b for b in self.blocks
                                if not b.is_dynamic and not isinstance(b, LoopBreaker)]

        self.stats.loop_breakers_count = len(self.loop_breakers)

        self.logger.info(
            f"Graph resolved: {len(self.blocks)} blocks  "
            f"({len(self.dynamic_blocks)} dynamic, "
            f"{len(self.static_blocks)} static, "
            f"{len(self.loop_breakers)} loop breakers)"
        )

        # Detect feedback loops
        self._detect_feedback_loops()

        # ── Attach TopologyPrinter ──────────────────────────────────────────
        # sim.topo gives access to both console and GUI topology rendering.
        # print_topology_sources2sink() wired as alias to topo.print_console()
        # but now delegates to the clean printer instead of the old renderer.
        TopologyPrinter = _get_topology_printer()
        if TopologyPrinter is not None:
            self.topo = TopologyPrinter(self)
            self.print_topology_sources2sink = self.topo.print_console
        else:
            # Fallback: keep the legacy method as-is if module not found
            self.topo = None

    def _detect_feedback_loops(self) -> None:
        """
        Count feedback loops present in the block diagram.

        Currently uses a simple heuristic: the number of feedback loops
        equals the number of LoopBreaker blocks, since each LoopBreaker
        cuts exactly one cycle.  The result is stored in
        ``self.stats.feedback_loops_count``.
        """
        # Simple heuristic: count loop breaker blocks
        self.stats.feedback_loops_count = len(self.loop_breakers)

    @property
    def execution_order(self) -> list:
        """
        The DFS-topologically-sorted list of blocks that the simulation
        engine executes at every time step (sources first, sinks last).

        This is an alias for ``self.blocks`` exposed under the name used
        in all EmbedSim example scripts.
        """
        return self.blocks

    def _compute_all_blocks(self, t: float) -> None:
        """
        Execute one forward pass: compute every block's output in topological order.

        Steps:
          1. Pre-initialise any LoopBreaker blocks that have not yet produced
             an output (e.g. at t = 0) by calling their
             ``get_loop_breaking_output()`` so downstream blocks see a valid
             signal rather than None.
          2. Iterate through ``self.blocks`` in order and call each block's
             ``compute(t, dt, input_values)`` method.  Input values are
             gathered from the ``output`` attribute of connected upstream
             blocks.  A zero signal is substituted as a fallback if an
             upstream block's output is still None.

        Args:
            t: Current simulation time in seconds passed to every block's
               compute() call.
        """
        # First, initialize loop breakers with their breaking output
        for block in self.blocks:
            if isinstance(block, LoopBreaker) and block.output is None:
                # Initialize loop breaker output
                breaking_output = block.get_loop_breaking_output()
                if breaking_output is not None:
                    block.output = breaking_output

        # Now compute all blocks — log_execute emits one DEBUG line per block.
        # This is silent at INFO level (console) but fully recorded in the
        # log file, giving a complete per-step execution trace for debugging.
        for block in self.blocks:
            self.logger.log_execute(self._step_counter, t, block)

            # Get inputs from connected blocks
            if len(block.inputs) > 0:
                input_values = []
                for inp in block.inputs:
                    if inp.output is not None:
                        input_values.append(inp.output)
                    else:
                        # Use zero signal as fallback
                        self.logger.warning(
                            f"  zero-fallback input: block '{block.name}' "
                            f"← upstream '{inp.name}' has no output yet at t={t:.6f}"
                        )
                        input_values.append(VectorSignal([0.0]))
            else:
                input_values = None

            block.compute(t, self.dt, input_values)

    def _integrate_dynamics_euler(self, t: float) -> None:
        """
        Advance all dynamic block states by one timestep using Forward Euler.

        For each dynamic block:
          1. Gather current input signals from connected upstream blocks.
          2. Compute the state derivative  dx/dt = f(x, u, t)  via
             ``block.get_derivative()``.
          3. Update the state: x(t + dt) = x(t) + dx/dt · dt.

        This is a first-order method — fast but accumulates O(dt) error
        per step.  Use RK4 for stiff systems or when accuracy matters.

        Args:
            t: Current simulation time in seconds.
        """
        for b in self.dynamic_blocks:
            input_values = [inp.output for inp in b.inputs] if b.inputs else None
            b.derivative = b.get_derivative(t, input_values)
            b.integrate_state(self.dt, solver='euler')

    def _integrate_dynamics_rk4(self, t: float) -> None:
        """
        Advance all dynamic block states by one timestep using Runge-Kutta 4.

        RK4 evaluates the derivative at four points within the interval
        [t, t + dt] and combines them with weights (1, 2, 2, 1)/6 to
        achieve fourth-order accuracy (O(dt⁴) local error).

        Stages:
          k1 — derivative at t using current state.
          k2 — derivative at t + dt/2 using state advanced by k1·dt/2.
          k3 — derivative at t + dt/2 using state advanced by k2·dt/2.
          k4 — derivative at t + dt   using state advanced by k3·dt.

        Final update:
          x(t + dt) = x(t) + (dt/6)·(k1 + 2·k2 + 2·k3 + k4)

        At each intermediate stage the full block graph is re-evaluated
        (via ``_compute_all_blocks()``) so that input signals reflect the
        perturbed states.

        Args:
            t: Current simulation time in seconds.

        Note:
            RK4 calls ``_compute_all_blocks()`` three extra times per
            timestep compared with Euler.  For large block graphs this
            increases wall-clock time significantly; choose the solver that
            matches your accuracy vs. speed requirement.
        """
        # Save initial states
        initial_states = {}
        for b in self.dynamic_blocks:
            initial_states[b] = b.state.copy()

        # K1
        for b in self.dynamic_blocks:
            input_values = [inp.output for inp in b.inputs] if b.inputs else None
            b.k1 = b.get_derivative(t, input_values)

        # K2
        for b in self.dynamic_blocks:
            b.state = initial_states[b] + 0.5 * self.dt * b.k1
        self._compute_all_blocks(t + 0.5 * self.dt)
        for b in self.dynamic_blocks:
            input_values = [inp.output for inp in b.inputs] if b.inputs else None
            b.k2 = b.get_derivative(t + 0.5 * self.dt, input_values)

        # K3
        for b in self.dynamic_blocks:
            b.state = initial_states[b] + 0.5 * self.dt * b.k2
        self._compute_all_blocks(t + 0.5 * self.dt)
        for b in self.dynamic_blocks:
            input_values = [inp.output for inp in b.inputs] if b.inputs else None
            b.k3 = b.get_derivative(t + 0.5 * self.dt, input_values)

        # K4
        for b in self.dynamic_blocks:
            b.state = initial_states[b] + self.dt * b.k3
        self._compute_all_blocks(t + self.dt)
        for b in self.dynamic_blocks:
            input_values = [inp.output for inp in b.inputs] if b.inputs else None
            b.k4 = b.get_derivative(t + self.dt, input_values)

        # Final update
        for b in self.dynamic_blocks:
            b.state = initial_states[b] + (self.dt / 6.0) * (b.k1 + 2 * b.k2 + 2 * b.k3 + b.k4)

    def run(self, verbose: bool = True, progress_bar: bool = True) -> None:
        """
        Execute the simulation from t = 0 to t = T.

        Main simulation loop:
          For each timestep t in [0, T) with step dt:
            1. Compute all block outputs in topological order.
            2. Record monitored signals via scope.record(t).
            3. Integrate dynamic block states (Euler or RK4).
          After the loop, one final compute + record at t = T ensures the
          last sample is always captured.

        Args:
            verbose:      If True, print a configuration summary before the
                          run and a completion message afterwards.
            progress_bar: If True, print a ``Progress: xx.x%`` line that
                          updates in place every 5 % of the simulation.

        Side effects:
            - All blocks are reset (``block.reset()``) before the loop
              starts, clearing any outputs from a previous run.
            - ``self.stats`` is populated with timing and step-count data.
            - ``self.scope.t`` and ``self.scope.data`` accumulate samples.

        Example:
            >>> sim.run(verbose=False, progress_bar=True)
            >>> print(f"Done in {sim.stats.compute_time:.2f} s")
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"EmbedSim v3.1.0")
            print(f"{'=' * 70}")
            print(f"  Duration:       {self.T} s")
            print(f"  Time step:      {self.dt} s  ({1.0/self.dt:.0f} Hz)")
            print(f"  Solver:         {self.solver.upper()}")
            print(f"  Total blocks:   {len(self.blocks)}")
            print(f"  Dynamic:        {len(self.dynamic_blocks)}")
            print(f"  Static:         {len(self.static_blocks)}")
            print(f"  Loop breakers:  {len(self.loop_breakers)}")
            print(f"  Feedback loops: {self.stats.feedback_loops_count}")
            if self.topo is not None:
                print(f"  Topology:       sim.topo.print_console() | sim.topo.show_gui()")
            print(f"{'=' * 70}\n")

        steps = int(self.T / self.dt)
        self.logger.info(
            f"run() start  T={self.T} s  dt={self.dt} s  "
            f"steps={steps}  solver={self.solver.upper()}"
        )

        # Reset all blocks
        for b in self.blocks:
            b.reset()

        # _step_counter is read by _compute_all_blocks → log_execute so that
        # every EXEC log line carries the correct step index.  It is kept as
        # an instance attribute (not a local) so RK4 sub-steps that call
        # _compute_all_blocks() internally don't advance it independently.
        self._step_counter = 0

        t = 0.0
        start_time = time.time()

        if verbose:
            print("Simulation started...")

        for step in range(steps):
            self._step_counter = step

            # Compute all blocks
            self._compute_all_blocks(t)

            # Record signals
            self.scope.record(t)

            # Integrate dynamics
            if self.solver == ODESolver.EULER:
                self._integrate_dynamics_euler(t)
            elif self.solver == ODESolver.RK4:
                self._integrate_dynamics_rk4(t)

            # Progress
            if progress_bar and step % max(1, steps // 20) == 0:
                progress = (step / steps) * 100
                print(f"  Progress: {progress:5.1f}%", end='\r')

            t += self.dt

        # Final computation and recording
        self._step_counter = steps
        self._compute_all_blocks(t)
        self.scope.record(t)

        end_time = time.time()

        self.stats.total_steps = steps
        self.stats.compute_time = end_time - start_time
        self.stats.avg_step_time = self.stats.compute_time / max(steps, 1)

        if progress_bar:
            print(f"  Progress: 100.0%")
        if verbose:
            print(f"\n✓ Simulation complete\n")

        self.logger.info(
            f"run() complete  {steps} steps  "
            f"wall={self.stats.compute_time:.3f} s  "
            f"avg={self.stats.avg_step_time*1e6:.1f} µs/step"
        )
        # Flush and close file handler — ensures the .log is fully written
        # to disk before the caller moves on to plotting / post-processing.
        self.logger.close()

    def print_topology(self) -> None:
        """
        Print a concise summary of the block diagram and its execution order.

        Output sections:
          - **Block counts** — total, dynamic, static, loop breakers, and
            detected feedback loops.
          - **Execution order table** — one row per block showing its
            sequential position, dynamic / loop-breaker marks, name, type,
            and number of inputs.

        Marks used in the table:
          ⚡  Dynamic block (has an internal integrated state).
          🔄  Loop-breaker block (cuts a feedback cycle).

        Example output:
            ======================================================================
            BLOCK DIAGRAM TOPOLOGY (with Feedback Loop Support)
            ======================================================================
            Total Blocks: 5
              Dynamic:        1
              Static:         4
              Loop Breakers:  0
              Feedback Loops: 0
            ...
              1.     sin_source   (SinusoidalGenerator ) <- 0 input(s)
              2.     gain         (VectorGain          ) <- 1 input(s)
              3. ⚡   integrator   (VectorIntegrator    ) <- 1 input(s)
        """
        print("\n" + "=" * 70)
        print("BLOCK DIAGRAM TOPOLOGY (with Feedback Loop Support)")
        print("=" * 70)

        print(f"\nTotal Blocks: {len(self.blocks)}")
        print(f"  Dynamic:        {len(self.dynamic_blocks)}")
        print(f"  Static:         {len(self.static_blocks)}")
        print(f"  Loop breakers:  {len(self.loop_breakers)}")
        print(f"  Feedback loops: {self.stats.feedback_loops_count}")

        print("\n" + "=" * 70)
        print("EXECUTION ORDER")
        print("=" * 70)

        for i, block in enumerate(self.blocks, 1):
            dynamic_mark = "⚡" if block.is_dynamic else "  "
            loop_mark = "🔄" if isinstance(block, LoopBreaker) else "  "
            num_inputs = len(block.inputs)
            block_type = type(block).__name__

            print(f"{i:3d}. {dynamic_mark}{loop_mark} {block.name:25s} ({block_type:20s}) <- {num_inputs} input(s)")

        print("\n" + "=" * 70)
        print("Legend:")
        print("  ⚡ = Dynamic block (has internal state)")
        print("  🔄 = Loop breaker (breaks feedback loops)")
        print("=" * 70 + "\n")

    def plot(self, title: str = "Simulation Results", figsize: tuple = (12, 6),
             signals: Optional[List[str]] = None,
             time_range: Optional[Tuple[float, float]] = None) -> None:
        """
        Plot all (or selected) recorded signals against simulation time.

        Calls matplotlib to display a line chart.  Each recorded scalar
        channel (``"label[i]"``) becomes one line.

        Args:
            title:      Figure title displayed above the plot.
            figsize:    Matplotlib figure size as (width, height) in inches.
            signals:    List of channel keys to plot, e.g.
                        ``['speed[0]', 'torque[0]']``.  If None (default)
                        all recorded channels are plotted.
            time_range: Optional (t_start, t_end) tuple to restrict the
                        horizontal axis.  Values must be within [0, T].

        Raises:
            (no exception): If no signals have been recorded, prints a
            warning and returns without creating a figure.

        Example:
            >>> sim.plot(title="Motor Response", signals=["speed[0]"],
            ...          time_range=(0.5, 2.0))
        """
        if not self.scope.data:
            print("⚠ No signals to plot. Use scope.add(block) to add signals.")
            return

        # Determine time range
        t_arr = self.scope.t
        if time_range:
            t0, t1 = time_range
            indices = [i for i, tv in enumerate(t_arr) if t0 <= tv <= t1]
            start_idx = indices[0] if indices else 0
            end_idx   = indices[-1] + 1 if indices else len(t_arr)
        else:
            start_idx, end_idx = 0, len(t_arr)

        t_plot = t_arr[start_idx:end_idx]

        # Determine which signals to plot
        if signals is None:
            plot_signals = list(self.scope.data.items())
        else:
            plot_signals = [(name, self.scope.data[name])
                            for name in signals if name in self.scope.data]

        plt.figure(figsize=figsize)

        for name, values in plot_signals:
            plt.plot(t_plot, values[start_idx:end_idx], label=name, linewidth=1.5)

        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Export
__all__ = [
    'LoopBreaker',
    'VectorDelay',
    'VectorScope',
    'SimulationStats',
    'SimulationLogger',
    'EmbedSim',
    'traverse_blocks_from_sinks_with_loops',
    'ODESolver',
]