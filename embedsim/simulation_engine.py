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
Version: 3.0.0
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass
import time

from .core_blocks import (VectorBlock, VectorSignal, DEFAULT_DTYPE)


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

def traverse_blocks_from_sinks_with_loops(sinks: List[VectorBlock]) -> List[VectorBlock]:
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
        sinks: List of terminal (sink) VectorBlock objects.  Traversal starts
               here and walks backwards through the ``inputs`` graph.

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

        visiting.remove(block)
        blocks_set.add(block)
        blocks_order.append(block)

    for sink in sinks:
        dfs(sink)

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

        # Build execution order with loop support
        try:
            self.blocks = traverse_blocks_from_sinks_with_loops(self.sinks)
        except ValueError as e:
            raise ValueError(f"Block diagram error: {e}")

        # Categorize blocks
        self.dynamic_blocks = [b for b in self.blocks if b.is_dynamic]
        self.static_blocks = [b for b in self.blocks if not b.is_dynamic]
        self.loop_breakers = [b for b in self.blocks if isinstance(b, LoopBreaker)]

        self.stats.loop_breakers_count = len(self.loop_breakers)

        # Detect feedback loops
        self._detect_feedback_loops()

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

        # Now compute all blocks
        for block in self.blocks:
            # Get inputs from connected blocks
            if len(block.inputs) > 0:
                input_values = []
                for inp in block.inputs:
                    if inp.output is not None:
                        input_values.append(inp.output)
                    else:
                        # Use zero signal as fallback
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
            print(f"EmbedSim")
            print(f"{'=' * 70}")
            print(f"  Duration:       {self.T} s")
            print(f"  Time step:      {self.dt} s")
            print(f"  Solver:         {self.solver}")
            print(f"  Total blocks:   {len(self.blocks)}")
            print(f"  Dynamic:        {len(self.dynamic_blocks)}")
            print(f"  Loop breakers:  {len(self.loop_breakers)}")
            print(f"  Feedback loops: {self.stats.feedback_loops_count}")
            print(f"{'=' * 70}\n")

        # Reset all blocks
        for b in self.blocks:
            b.reset()

        t = 0.0
        steps = int(self.T / self.dt)
        start_time = time.time()

        if verbose:
            print("Simulation started...")

        for step in range(steps):
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
        print(f"  Loop Breakers:  {len(self.loop_breakers)}")
        print(f"  Feedback Loops: {self.stats.feedback_loops_count}")

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
        if not self.data:
            print("⚠ No signals to plot. Use scope.add(block) to add signals.")
            return

        # Determine time range
        if time_range:
            start_idx, end_idx = self.get_time_range(time_range[0], time_range[1])
            t_plot = self.t[start_idx:end_idx]
        else:
            start_idx, end_idx = 0, len(self.t)
            t_plot = self.t

        # Determine which signals to plot
        if signals is None:
            plot_signals = self.data.items()
        else:
            plot_signals = [(name, self.data[name]) for name in signals if name in self.data]

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

    def print_topology_tree(self) -> None:
        """
        Print the block diagram as a reverse dependency tree (Sinks → Sources).

        Walks backwards from each sink through the ``inputs`` graph and
        renders the hierarchy with box-drawing characters (``└──``, ``├──``,
        ``│``).  Blocks that appear in multiple branches are printed in full
        the first time and shown as ``(see above)`` thereafter to avoid
        duplicating subtrees.

        After the tree, an **Execution Order** table lists every block with
        its sequential index, type, and input count — the same order used by
        the simulation loop.

        Marks used:
          ⚡  Dynamic block (has integrated state).
          ○   Static / combinatorial block.

        Example output (3-source fan-in system):
            └── ○ output (VectorEnd)
                └── ⚡ integrator (VectorIntegrator)
                    └── ○ gain (VectorGain)
                        └── ○ source_sum (VectorSum)
                            ├── ○ cosine_source (SinusoidalGenerator)
                            ├── ○ sin_source (SinusoidalGenerator)
                            └── ○ const_1 (VectorConstant)
        """
        print("\n" + "=" * 70)
        print("BLOCK DIAGRAM TOPOLOGY")
        print("=" * 70)

        # Summary
        print(f"\nTotal Blocks: {len(self.blocks)}")
        print(f"  Dynamic (with state): {len(self.dynamic_blocks)}")
        print(f"  Static (no state): {len(self.static_blocks)}")
        print(f"  Sink blocks: {len(self.sinks)}")

        # Build dependency tree (from sinks backwards to sources)
        print("\n" + "=" * 70)
        print("DEPENDENCY TREE (Sinks → Sources)")
        print("=" * 70)
        print()

        # Track which blocks have been printed to avoid duplicates
        printed = set()

        def print_block_tree(block, prefix="", is_last=True):
            """Recursively print block and its dependencies."""
            if id(block) in printed:
                # Show reference to already printed block
                branch = "└── " if is_last else "├── "
                dynamic_mark = "⚡" if block.is_dynamic else "○"
                print(f"{prefix}{branch}{dynamic_mark} {block.name} (see above)")
                return

            printed.add(id(block))

            # Create the tree branch characters
            branch = "└── " if is_last else "├── "

            # Format block info
            dynamic_mark = "⚡" if block.is_dynamic else "○"
            block_type = type(block).__name__

            print(f"{prefix}{branch}{dynamic_mark} {block.name} ({block_type})")

            # Get input blocks (dependencies)
            inputs = block.inputs if hasattr(block, 'inputs') else []

            # Prepare prefix for children
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension

            # Print each input dependency
            for i, input_block in enumerate(inputs):
                is_last_input = (i == len(inputs) - 1)
                print_block_tree(input_block, new_prefix, is_last_input)

        # Print tree starting from each sink block
        for i, sink in enumerate(self.sinks):
            is_last_sink = (i == len(self.sinks) - 1)
            print_block_tree(sink, "", is_last_sink)

        # Handle any orphaned blocks (not reachable from sinks)
        orphaned = [b for b in self.blocks if id(b) not in printed]
        if orphaned:
            print("\n⚠ Unreachable Blocks (not connected to any sink):")
            for block in orphaned:
                dynamic_mark = "⚡" if block.is_dynamic else "○"
                block_type = type(block).__name__
                print(f"  {dynamic_mark} {block.name} ({block_type})")
                if hasattr(block, 'inputs') and len(block.inputs) > 0:
                    print(f"    ← Inputs: {', '.join(b.name for b in block.inputs)}")

        # Execution order
        print("\n" + "=" * 70)
        print("EXECUTION ORDER")
        print("=" * 70)
        for i, block in enumerate(self.blocks, 1):
            dynamic_mark = "⚡" if block.is_dynamic else "○"
            num_inputs = len(block.inputs) if hasattr(block, 'inputs') else 0
            block_type = type(block).__name__
            print(f"{i:3d}. {dynamic_mark} {block.name:30s} ({block_type:25s}) ← {num_inputs} input(s)")

        print("\n" + "=" * 70)
        print("Legend:")
        print("  ⚡ = Dynamic block (has internal state)")
        print("  ○ = Static block (no state)")
        print("=" * 70 + "\n")

    def print_topology_sources2sink(self) -> None:
        """
        Render a horizontal ASCII flow diagram: Sources → Sinks.

        Every block box shows instance name and class type::

            [gain (VectorGain)]
            [⚡integrator (VectorIntegrator)]

        Every fan-in source gets its own ──► arrow into the merge bar.

        Feedback loops are drawn as return arcs BELOW the main diagram.
        The arc shows the complete path — driver, loop-breaker, receiver —
        so the signal flow is unambiguous:

            [sin] ──► [sum] ──► [gain] ──► [output]
                               │          │
                               ◄──────────┘
                               delay (VectorDelay)

        The ◄ tip marks exactly where the delayed signal re-enters.
        If the lb label fits in the dash span it is embedded inline;
        otherwise it is printed on a third row below the arc.

        Example outputs
        ---------------
        Pure chain::

            [3phase_gen (ThreePhaseGenerator)] ──► [output (VectorEnd)]

        3-source fan-in::

            [cosine_source (SinusoidalGenerator)] ──►┐[source_sum (VectorSum)] ──► ...
                                                     │
            [sin_source (SinusoidalGenerator)] ─────►┤
                                                     │
            [const_1 (VectorConstant)] ─────────────►┘

        Self-feedback (sum drives delay, delay feeds back into sum)::

            [sin] ──► [sum] ──► [gain] ──► [output]
                      │       │
                      ◄───────┘
                      delay (VectorDelay)

        Cross-feedback (controller drives delay, delay feeds into plant)::

            [source] ──► [controller] ──► [plant] ──► [sink]
                                      │   │
                                      └───◄
                                      delay (VectorDelay)
        """
        import unicodedata

        loop_breaker_set = set(id(b) for b in self.loop_breakers)
        ARROW = " ──► "

        def vlen(s: str) -> int:
            """Visual terminal width accounting for wide/emoji Unicode glyphs."""
            w = 0
            for ch in s:
                ea = unicodedata.east_asian_width(ch)
                w += 2 if ea in ("W", "F") else 1
            return w

        def block_label(b: VectorBlock) -> str:
            """⚡name (ClassName) — dynamic mark + instance name + class type."""
            return ("⚡" if b.is_dynamic else "") + f"{b.name} ({type(b).__name__})"

        def box(b: VectorBlock) -> str:
            return f"[{block_label(b)}]"

        def box_vlen(b: VectorBlock) -> int:
            return vlen(box(b))

        # Only non-loop-breaker blocks occupy the main canvas.
        visible = [b for b in self.blocks if id(b) not in loop_breaker_set]

        # ── 1. Lane assignment ────────────────────────────────────────────────
        # A block shares a lane with its primary (first normal) input UNLESS
        # that driver already has another consumer ahead of it on the same
        # lane, which would place this block BEHIND its driver.  In that case
        # a new lane is allocated so the block can sit beside (not behind) its
        # driver, and a fan-out arrow will be drawn in Pass C.
        lane_of = {}
        next_lane = [0]
        lane_primary = {}  # lane → id of the block that "owns" that lane slot

        def alloc_lane() -> int:
            l = next_lane[0];
            next_lane[0] += 1;
            return l

        for b in visible:
            normal_inps = [i for i in getattr(b, "inputs", [])
                           if id(i) not in loop_breaker_set and id(i) in lane_of]
            if not normal_inps:
                lane_of[id(b)] = alloc_lane()
            else:
                driver = normal_inps[0]
                drv_lane = lane_of[id(driver)]
                # Check if this lane already has a block that comes AFTER
                # the driver in execution order — if so, this is a secondary
                # fan-out consumer and needs its own lane.
                drv_idx = visible.index(driver)
                already_used = any(
                    lane_of[id(v)] == drv_lane and visible.index(v) > drv_idx
                    for v in visible if id(v) in lane_of and v is not b
                )
                if already_used:
                    lane_of[id(b)] = alloc_lane()
                else:
                    lane_of[id(b)] = drv_lane

        n_lanes = next_lane[0]

        # ── 2. Column assignment ──────────────────────────────────────────────
        # col_end sized after lane assignment so all lanes are covered.
        col_of = {}
        col_end = [0] * n_lanes

        for b in visible:
            lane = lane_of[id(b)]
            normal_inps = [i for i in getattr(b, "inputs", [])
                           if id(i) not in loop_breaker_set and id(i) in col_of]
            if normal_inps:
                after = max(col_of[id(i)] + box_vlen(i) + len(ARROW) for i in normal_inps)
                col = max(after, col_end[lane])
            else:
                # Fan-out secondary: align with column just after its driver
                driver_candidates = [i for i in getattr(b, "inputs", [])
                                     if id(i) not in loop_breaker_set and id(i) in col_of]
                if driver_candidates:
                    drv = driver_candidates[0]
                    after = col_of[id(drv)] + box_vlen(drv) + len(ARROW)
                    col = max(after, col_end[lane])
                else:
                    col = col_end[lane]
            col_of[id(b)] = col
            col_end[lane] = col + box_vlen(b) + len(ARROW)

        total_width = max(col_end) + 8

        # ── 3. Build canvas ───────────────────────────────────────────────────
        n_rows = max(1, n_lanes * 2 - 1)
        rows = [list(" " * total_width) for _ in range(n_rows)]

        def row_of(lane: int) -> int:
            return lane * 2

        def put(r: int, c: int, text: str) -> None:
            for ch in text:
                if 0 <= r < n_rows and 0 <= c < total_width:
                    rows[r][c] = ch
                c += 1

        # Pass A — boxes
        for b in visible:
            put(row_of(lane_of[id(b)]), col_of[id(b)], box(b))

        # Pass B — arrows + fan-in merge bars
        for b in visible:
            b_col = col_of[id(b)]
            normal_inps = [i for i in getattr(b, "inputs", [])
                           if id(i) not in loop_breaker_set
                           and id(i) in lane_of and id(i) in col_of]
            if not normal_inps:
                continue

            primary, side_inps = normal_inps[0], normal_inps[1:]
            p_row = row_of(lane_of[id(primary)])
            p_end = col_of[id(primary)] + box_vlen(primary)

            if not side_inps:
                put(p_row, p_end, ARROW)
            else:
                # Fan-in: vertical bar at (b_col - 1); every source gets ──►
                merge_col = b_col - 1

                run = merge_col - p_end
                put(p_row, p_end,
                    " " + "─" * max(0, run - 4) + "──►" if run >= 4 else ARROW[:run])

                for s in side_inps:
                    if id(s) not in lane_of or id(s) not in col_of:
                        continue
                    s_row = row_of(lane_of[id(s)])
                    s_end = col_of[id(s)] + box_vlen(s)
                    run_s = merge_col - s_end
                    if run_s >= 4:
                        put(s_row, s_end, " " + "─" * max(0, run_s - 4) + "──►")
                    elif run_s > 0:
                        put(s_row, s_end, "─" * run_s)

                involved = sorted(set(
                    [p_row] +
                    [row_of(lane_of[id(s)]) for s in side_inps if id(s) in lane_of]
                ))
                top_r, bot_r = involved[0], involved[-1]
                for r in range(top_r, bot_r + 1):
                    if r == top_r == bot_r:
                        ch = "│"
                    elif r == top_r:
                        ch = "┐"
                    elif r == bot_r:
                        ch = "┘"
                    elif r == p_row:
                        ch = "┼"
                    elif r in involved:
                        ch = "┤"
                    else:
                        ch = "│"
                    put(r, merge_col, ch)

        # Pass C — fan-out split arrows
        # When block b feeds multiple consumers on the SAME lane, the column
        # assignment places them sequentially. Pass B only draws the arrow to
        # the FIRST consumer (the one immediately to the right of b).  Every
        # additional same-lane consumer needs a branch drawn from the tap point
        # (right edge of b) forward to that consumer's box.
        #
        # Strategy: for each visible block b, collect all consumers.
        # The "primary" consumer is the one whose left edge is closest to
        # b's right edge (already handled by Pass B).  All others get an
        # explicit branch: a tap ┬ on the main arrow line, a vertical drop
        # to a spare row (or same row), then horizontal run ──► to the box.
        for b in visible:
            b_row = row_of(lane_of[id(b)])
            b_end = col_of[id(b)] + box_vlen(b)  # column just after ]

            # All consumers of b that are visible and not loop breakers
            consumers = [c for c in visible
                         if b in getattr(c, "inputs", [])
                         and id(b) not in loop_breaker_set]
            if len(consumers) <= 1:
                continue  # single consumer: Pass B already drew the arrow

            # Sort consumers by column so the leftmost (nearest) is primary
            consumers.sort(key=lambda c: col_of.get(id(c), 0))
            secondary = consumers[1:]  # skip primary (nearest) consumer

            for c in secondary:
                if id(c) not in col_of:
                    continue
                c_col = col_of[id(c)]
                c_row = row_of(lane_of[id(c)])

                if c_row != b_row:
                    # Different lane: draw vertical + horizontal branch
                    v_col = b_end - 1
                    run_h = c_col - v_col - 2  # -2 so ► lands at c_col-1, not on [ of target box
                    if b_row < c_row:
                        put(b_row, v_col, "┬")
                        for r in range(b_row + 1, c_row):
                            put(r, v_col, "│")
                        put(c_row, v_col, "└")
                    else:
                        put(b_row, v_col, "┴")
                        for r in range(c_row + 1, b_row):
                            put(r, v_col, "│")
                        put(c_row, v_col, "┌")
                    if run_h > 3:
                        put(c_row, v_col + 1, " " + "─" * (run_h - 3) + "──►")
                    elif run_h > 0:
                        put(c_row, v_col + 1, "─" * run_h)
                else:
                    # Same lane/row: tap off the arrow between b and primary
                    # Find a spare inter-lane row below for the branch line
                    branch_row = b_row + 1 if b_row + 1 < n_rows else b_row - 1
                    tap_col = b_end + len(ARROW) // 2  # mid-arrow tap point
                    run_h = c_col - tap_col - 2  # -2 so ► lands at c_col-1, not on [ of target box
                    # Tap down
                    put(b_row, tap_col, "┬")
                    put(branch_row, tap_col, "└")
                    # Horizontal run on branch row
                    if run_h > 3:
                        put(branch_row, tap_col + 1, " " + "─" * (run_h - 3) + "──►")
                    elif run_h > 0:
                        put(branch_row, tap_col + 1, "─" * run_h)
                    # Vertical leg back up (if branch_row != c_row-1)
                    for r in range(branch_row + 1, c_row):
                        put(r, c_col - 1, "│")
                    if branch_row < c_row - 1:
                        put(c_row - 1, c_col - 1, "└")

        # ── 4. Print main canvas ──────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("BLOCK DIAGRAM  (Sources → Sinks)")
        print("=" * 70 + "\n")

        for row in rows:
            line = "".join(row).rstrip()
            if line.strip():
                print("  " + line)

        # ── 5. Feedback return arcs ───────────────────────────────────────────
        # For each (driver, loop-breaker) pair we collect ALL receivers so
        # that a loop-breaker which fans out to multiple forward-path blocks
        # is drawn as ONE arc with multiple ◄ re-entry points, rather than
        # one separate arc per receiver.
        #
        # Arc geometry (drawn below the main diagram):
        #   drop_col  = right edge of driver box    → vertical leg / └ here
        #   rise_cols = left  edges of all receivers → ◄ re-entry arrows here
        #
        #   Row 1: │ at drop_col and at every rise_col
        #   Row 2: └──── lb_label ────◄  with ◄ at every rise_col
        #   Row 3: lb label below arc (only when it does not fit inline)

        # Build grouped arcs: key=(id(driver),id(lb)), value=(d, lb, [receivers])
        arc_map = {}
        for lb in self.loop_breakers:
            drivers = [i for i in getattr(lb, "inputs", [])
                       if id(i) not in loop_breaker_set]
            receivers = [b for b in visible
                         if id(lb) in [id(i) for i in getattr(b, "inputs", [])]]
            for d in drivers:
                key = (id(d), id(lb))
                if key not in arc_map:
                    arc_map[key] = (d, lb, [])
                arc_map[key][2].extend(receivers)

        for (d, lb, receivers) in arc_map.values():
            if id(d) not in col_of:
                continue
            valid_rcv = [r for r in receivers if id(r) in col_of]
            if not valid_rcv:
                continue

            drop_col = col_of[id(d)] + box_vlen(d)
            rise_cols = sorted(set(col_of[id(r)] for r in valid_rcv))
            lb_label = f" {lb.name} ({type(lb).__name__}) "

            all_cols = [drop_col] + rise_cols
            left = min(all_cols)
            right = max(all_cols)
            span = right - left - 1

            # Row 1: vertical legs
            r1 = list(" " * total_width)
            for c in all_cols:
                if 0 <= c < total_width:
                    r1[c] = "│"
            print("  " + "".join(r1).rstrip())

            # Row 2: horizontal arc
            r2 = list(" " * total_width)

            if left == right:
                # Degenerate: all columns coincide
                if 0 <= drop_col < total_width:
                    r2[drop_col] = "└"
                for i, ch in enumerate(lb_label.strip()):
                    if drop_col + 1 + i < total_width:
                        r2[drop_col + 1 + i] = ch
                print("  " + "".join(r2).rstrip())
                continue

            # Dash span
            for c in range(left + 1, right):
                r2[c] = "─"

            # Drop corner
            r2[drop_col] = "└" if drop_col == left else "┘"

            # Re-entry arrows at every receiver column
            for rc in rise_cols:
                if 0 <= rc < total_width:
                    r2[rc] = "◄"

            # Embed lb label centred between the leftmost and rightmost ◄.
            # If it would overwrite a ◄ or doesn't fit inline, use row 3.
            # Centre point is between the two outermost re-entry arrows.
            inner_left = rise_cols[0] + 1  # just right of leftmost  ◄
            inner_right = rise_cols[-1] - 1  # just left  of rightmost ◄
            inner_span = inner_right - inner_left + 1

            need_row3 = True
            if inner_span >= len(lb_label):
                s = inner_left + (inner_span - len(lb_label)) // 2
                safe = all(0 <= s + i < total_width and r2[s + i] != "◄"
                           for i in range(len(lb_label)))
                if safe:
                    for i, ch in enumerate(lb_label):
                        r2[s + i] = ch
                    need_row3 = False
            print("  " + "".join(r2).rstrip())
            if need_row3:
                r3 = list(" " * total_width)
                # Centre the label in the full arc span on row 3
                s = left + max(0, (span + 1 - len(lb_label.strip())) // 2)
                for i, ch in enumerate(lb_label.strip()):
                    if 0 <= s + i < total_width:
                        r3[s + i] = ch
                print("  " + "".join(r3).rstrip())

        print()
        print("  Legend:  ⚡ dynamic block    ┘ signal departs forward path    ◄ signal re-enters forward path")
        print("\n" + "=" * 70 + "\n")


# Export
__all__ = [
    'LoopBreaker',
    'VectorDelay',
    'VectorScope',
    'EmbedSim',
    'traverse_blocks_from_sinks_with_loops',
    'ODESolver'
]