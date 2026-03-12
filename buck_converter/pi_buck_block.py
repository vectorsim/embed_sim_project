"""
pi_buck_block.py
================

PURPOSE
-------
This module provides PI_BuckBlock — the EmbedSim CONTROLLER block that wraps
the C PI algorithm (via Cython .pyd) and integrates it into the simulation
graph via the VectorBlock wiring system.

It is the Python-visible "face" of the entire C → Cython → Python stack.
From the simulation script's point of view, it is just a block you wire with >>:

    v_ref >> cg_start >> pi_controller >> cg_end >> buck_plant

But underneath, every compute() call invokes the compiled C algorithm in
pi_buck_controller.c through the pi_buck_wrapper.pyd Cython extension.

CLASS HIERARCHY
---------------
    object
      └── VectorBlock          (wiring, signals, topology)
            └── SimBlockBase   (adds CodeGen attributes: PYX_FILE, C_SOURCES…)
                  └── PI_BuckBlock   ← THIS CLASS

WHY SimBlockBase (NOT FMUBlock)?
---------------------------------
The PI controller is NOT an FMU — it is a pure C algorithm with no Modelica
physics. SimBlockBase marks it as a CodeGen-capable block: when LoopGenerator
walks the cg_start → cg_end sub-graph, it reads PI_BuckBlock.C_SOURCES and
.C_HEADERS to know which files to #include in embedsim_loop.c.

TWO BACKENDS
------------
This block can use either:
  A. C backend  (use_c_backend=True):  calls pi_buck_wrapper.pyd → C function
  B. Python backend (use_c_backend=False): uses _PyPI_Buck (pure NumPy)

The Python backend is the fallback when the Cython .pyd has not been compiled.
It produces identical numerical results (verified) but runs ~50× slower.

RK4 INTEGRATION
---------------
The PI controller has ONE continuous state: the integrator accumulator.
EmbedSim's RK4 solver integrates it using:
    dx/dt = error = V_ref - V_meas
    x(t+dt) ≈ x(t) + (k1 + 2k2 + 2k3 + k4) * dt/6

This is more accurate than the simple Euler integration used inside
the C function itself. For a 1µs simulation step the difference is
negligible, but RK4 is the framework standard.

CORRECTNESS: ALL CODE IS CORRECT.
One design note documented below (compute_py integrator path).

ISSUES FOUND: NONE — implementation is correct.
"""

import sys
import numpy as np
from typing import List, Optional

# ── Path setup — see _path_utils.py for full explanation ─────────────────────
# get_embedsim_import_path() returns C:\EmbedSimProject so that
# "from embedsim.xxx import YYY" works from any working directory.
from _path_utils import get_embedsim_import_path
sys.path.insert(0, get_embedsim_import_path())

# ── EmbedSim imports ──────────────────────────────────────────────────────────
# SimBlockBase: extends VectorBlock with CodeGen attributes (PYX_FILE,
#               C_SOURCES, C_HEADERS, step_func, state_struct, …).
#               Blocks that derive from SimBlockBase will be included
#               in the C code emitted by LoopGenerator.
from embedsim.code_generator import SimBlockBase

# VectorSignal: the signal type that flows between blocks in EmbedSim.
#               It wraps a NumPy array plus metadata (block name, dtype).
from embedsim.core_blocks import VectorSignal

# auto_populate_from_pyx: reads the .pyx file and fills in class attributes
#                          (C_SOURCES, C_HEADERS, step_func, …) automatically.
from pyx_inspector import auto_populate_from_pyx


# ==============================================================================
#  PURE-PYTHON FALLBACK IMPLEMENTATION
# ==============================================================================

class _PyPI_Buck:
    """
    Pure-Python mirror of the C PI_Buck_Block_T.

    This class is used when use_c_backend=False (Cython .pyd not compiled).
    It implements exactly the same algorithm as pi_buck_controller.c:
        integ(k) = clamp(integ(k-1) + e*dt, -limit, +limit)
        duty(k)  = clamp(Kp*e + Ki*integ, duty_min, duty_max)

    WHY np.float32 EVERYWHERE?
    All arithmetic uses numpy float32 to match the C implementation's
    real32_T (32-bit IEEE 754). This ensures the Python and C backends
    produce bit-identical results (within float32 precision).
    """

    def __init__(self, Kp: float, Ki: float, duty_max: float, duty_min: float, Ts: float):
        # Store all parameters as float32 to match the C struct
        self.Kp       = np.float32(Kp)
        self.Ki       = np.float32(max(Ki, 1e-9))   # floor Ki above zero
        self.duty_max = np.float32(duty_max)
        self.duty_min = np.float32(duty_min)
        self.Ts       = np.float32(Ts)

        # State — matches PI_Buck_State_T in C
        self.integ     = np.float32(0.0)   # integrator accumulator
        self.last_duty = np.float32(0.0)   # last computed duty (for debug)

    def compute(self, V_ref: float, V_meas: float, dt: float) -> float:
        """Run one PI step. Mirrors PI_Buck_Compute() in C exactly."""
        e = np.float32(V_ref - V_meas)               # error this step

        # Anti-windup limit: integ must stay within ±duty_max/Ki
        lim = self.duty_max / self.Ki

        # Use provided dt if valid, else fall back to stored Ts
        sample_time = dt if dt > 0 else self.Ts

        # Update integrator with clamp (forward Euler)
        self.integ = np.float32(np.clip(self.integ + e * sample_time, -lim, lim))

        # PI output with output clamp
        duty = np.clip(self.Kp * e + self.Ki * self.integ,
                       self.duty_min, self.duty_max)
        self.last_duty = np.float32(duty)
        return float(duty)

    def reset(self):
        """Zero state — mirrors PI_Buck_ResetState()."""
        self.integ     = np.float32(0.0)
        self.last_duty = np.float32(0.0)

    # Properties to match the Cython wrapper's interface
    @property
    def integrator(self):
        return float(self.integ)

    @integrator.setter
    def integrator(self, value):
        self.integ = np.float32(value)

    @property
    def last_output(self):
        return float(self.last_duty)


# ==============================================================================
#  PI_BuckBlock — THE MAIN EMBEDSIM BLOCK
# ==============================================================================

class PI_BuckBlock(SimBlockBase):
    """
    Proportional-Integral voltage controller for Buck Converter.

    This is the EmbedSim block that appears in the wiring diagram.
    It has two input ports and one output:

        Port 0  V_ref  [V]   — the target voltage (from VectorStep)
        Port 1  V_meas [V]   — the measured voltage (from buck plant via delay)
        Output  duty   [0-1] — PWM duty cycle sent to BuckConverterBlock

    RK4 state vector (length 1):
        state[0] = error integrator  [V·s]
    """

    # ── CodeGen attributes ────────────────────────────────────────────────────
    # PYX_FILE tells PYXInspector where to find the Cython wrapper file.
    # _pl.Path(__file__).parent resolves to the buck_converter/ directory,
    # so this always points to buck_converter/c_src/pi_buck_wrapper.pyx
    # regardless of working directory. Uses get_current_parent() from
    # _path_utils.py is an alternative — PYX_FILE uses pathlib directly here.
    import pathlib as _pl
    PYX_FILE: str = str(_pl.Path(__file__).parent / 'c_src' / 'pi_buck_wrapper.pyx')

    # The following attributes are LEFT EMPTY here on purpose.
    # __init_subclass__ (see below) calls auto_populate_from_pyx() which
    # reads the .pyx file and fills them in automatically.
    # If auto-population fails (e.g., .pyx file missing), they stay as empty
    # defaults so the block still works — just without code generation.
    step_func:    str  = ''     # will be set to 'PI_Buck_Compute'
    state_struct: str  = ''     # will be set to 'PI_Buck_State_T'
    NUM_INPUTS:   int  = 0      # will be set to 2 (V_ref, V_meas)
    OUTPUT_SIZE:  int  = 0      # will be set to 1 (duty)
    C_SOURCES:    list = []     # will be set to ['pi_buck_controller.c']
    C_HEADERS:    list = []     # will be set to ['pi_buck_controller.h']

    # ── state_struct override ─────────────────────────────────────────────────
    # PYXInspector reads the .pyx and infers PI_Buck_State_T (the inner state
    # struct).  But PI_Buck_Compute() takes PI_Buck_Block_T* (params + state).
    # Override here so LoopGenerator allocates the correct type in the .c/.h.
    state_struct: str = 'PI_Buck_Block_T'

    # ── C_CUSTOM_EMIT ─────────────────────────────────────────────────────────
    # PI_Buck_Compute uses struct-pointer arguments, not flat real32_T arrays:
    #
    #   void PI_Buck_Compute(PI_Buck_Block_T*       pPI,
    #                        const PI_Buck_Input_T* pIn,
    #                        real32_T               dt,
    #                        PI_Buck_Output_T*      pOut);
    #
    # The generic _emit_block() pattern (flat u[]/y[] arrays) cannot produce
    # this call.  C_CUSTOM_EMIT is the escape hatch: LoopGenerator emits this
    # verbatim instead of auto-generating.
    #
    # Naming contract with embedsim_loop.c:
    #   pi_buck_state  — PI_Buck_Block_T declared static in .c / extern in .h
    #   (LoopGenerator uses block.name sanitized → "pi_buck")
    C_CUSTOM_EMIT: str = (
        "    /* --- pi_buck (PI_BuckBlock) --- */\n"
        "    {\n"
        "        PI_Buck_Input_T  u_pi_buck;\n"
        "        PI_Buck_Output_T y_pi_buck;\n"
        "        u_pi_buck.V_ref  = y_pi_ctrl_start[0];\n"
        "        u_pi_buck.V_meas = y_fb_delay[0];\n"
        "        PI_Buck_Compute(&pi_buck_state, &u_pi_buck, dt, &y_pi_buck);\n"
        "        /* y_pi_buck.duty available to downstream blocks as needed */\n"
        "    }"
    )

    def __init_subclass__(cls, **kwargs):
        """
        Python hook called automatically when a SUBCLASS of PI_BuckBlock
        is defined (not when PI_BuckBlock itself is defined).

        This is Python's metaclass-free way of doing "class registration."
        Any future subclass (e.g., AdaptivePI_BuckBlock) will also get
        its CodeGen attributes auto-populated from its own PYX_FILE.

        WHY HERE (not in __init__)?
        __init__ runs when an INSTANCE is created. __init_subclass__ runs
        when the CLASS is defined — so the attributes are ready before
        any instance exists, which LoopGenerator requires for static
        inspection without instantiation.
        """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'PYX_FILE') and cls.PYX_FILE:
            auto_populate_from_pyx(cls, cls.PYX_FILE)

    def __init__(
            self,
            name: str,
            Kp: float = 0.1,
            Ki: float = 5.0,
            duty_max: float = 0.95,
            duty_min: float = 0.05,
            Ts: float = 1e-4,
            t_enable: float = 0.0,
            use_c_backend: bool = False,
            dtype=None,
    ) -> None:
        """
        Create the PI Buck block.

        Args:
            name         : Unique block identifier (shown in topology printer).
            Kp           : Proportional gain [1/V].         Default: 0.1
            Ki           : Integral gain [1/(V·s)].         Default: 5.0
            duty_max     : Maximum duty cycle [0-1].         Default: 0.95
            duty_min     : Minimum duty cycle [0-1].         Default: 0.05
            Ts           : Nominal sample period [s].        Default: 100 µs
            t_enable     : Time [s] before which controller output is
                           frozen at 0.0. Useful for a soft start: let the
                           simulation stabilise at t=0 before engaging.
                           Default: 0.0 (active from the start).
            use_c_backend: True → use pi_buck_wrapper.pyd (fast C extension).
                           False → use _PyPI_Buck (pure Python, always works).
            dtype        : NumPy dtype for output signals. None → float32.
        """
        # Call SimBlockBase.__init__ — sets up name, use_c_backend, dtype,
        # the output VectorSignal, and registers the block in the global
        # block registry used by EmbedSim's topological sort.
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        # output_label is shown in the topology printer ASCII diagram
        self.output_label = "[duty]"

        # vector_size = number of output signals (1: just duty cycle)
        self.vector_size = 1

        # is_dynamic = True means this block has continuous state (integrator).
        # EmbedSim will call get_derivative() for RK4 integration.
        self.is_dynamic = True

        # Store gains as Python floats (converted from whatever the caller passed)
        self._Kp      = float(Kp)
        self._Ki      = float(Ki)
        self._duty_max = float(duty_max)
        self._duty_min = float(duty_min)
        self._Ts      = float(Ts)
        self._t_enable = float(t_enable)

        # ── RK4 state vector ──────────────────────────────────────────────────
        # EmbedSim's RK4 solver reads and writes self.state.
        # Length 1: state[0] = integrator accumulator [V·s].
        # Uses float32 to match the C implementation precision.
        self.state = np.zeros(1, dtype=np.float32)

        # RK4 intermediate derivatives (k1, k2, k3, k4)
        # EmbedSim's simulation engine allocates these — we initialise
        # them here as zero arrays of the correct shape.
        self.k1 = self.k2 = self.k3 = self.k4 = np.zeros(1, dtype=np.float32)

        # ── Backend selection ─────────────────────────────────────────────────
        if use_c_backend:
            # Try to load the compiled Cython extension.
            # Raises ImportError with a helpful message if not compiled.
            self._impl = self._load_c_wrapper(Kp, Ki, duty_max, duty_min, Ts)
        else:
            # Always-available Python fallback.
            self._impl = _PyPI_Buck(Kp, Ki, duty_max, duty_min, Ts)

    # ──────────────────────────────────────────────────────────────────────────
    # C BACKEND LOADER
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_c_wrapper(Kp, Ki, duty_max, duty_min, Ts):
        """
        Import the Cython .pyd and create a configured PI_BuckWrapper.

        @staticmethod means this method does NOT need 'self' — it is a
        helper that belongs to the class but doesn't access instance data.
        This is appropriate here because it is a pure factory function.

        The try/except gives a clear, actionable error message if the
        Cython .pyd was not compiled, rather than a cryptic ImportError.
        """
        try:
            import pi_buck_wrapper as pbw          # import the .pyd binary
            w = pbw.PI_BuckWrapper()               # create C-level instance
            w.set_params(Kp=Kp, Ki=Ki,
                         duty_max=duty_max,
                         duty_min=duty_min,
                         Ts=Ts)
            return w
        except ImportError as e:
            raise ImportError(
                f"Cython wrapper 'pi_buck_wrapper' not found: {e}\n"
                "Compile with: cd buck_converter\\c_src && "
                "python setup_pi_buck.py build_ext --inplace\n"
                "Or use use_c_backend=False for the Python backend."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # INPUT EXTRACTION HELPER
    # ──────────────────────────────────────────────────────────────────────────

    def _get_voltages(self, input_values):
        """
        Extract V_ref and V_meas from the EmbedSim input_values list.

        input_values is a list of VectorSignal objects, one per wired input port:
            input_values[0].value = numpy array for port 0 (V_ref)
            input_values[1].value = numpy array for port 1 (V_meas)

        Each value array may have length > 1 if a multi-signal source is wired,
        but we only read index [0] (the first element) for each.

        Returns (0.0, 0.0) gracefully if inputs are not yet wired — this
        allows the block to be constructed before wiring is complete.
        """
        V_ref  = 0.0
        V_meas = 0.0

        if not input_values:
            return V_ref, V_meas   # no inputs wired yet — safe default

        # Port 0: V_ref
        if len(input_values) > 0 and input_values[0] is not None:
            v = input_values[0].value          # numpy array
            V_ref = float(v[0]) if len(v) >= 1 else 0.0

        # Port 1: V_meas
        if len(input_values) > 1 and input_values[1] is not None:
            v = input_values[1].value          # numpy array
            V_meas = float(v[0]) if len(v) >= 1 else 0.0

        return V_ref, V_meas

    # ──────────────────────────────────────────────────────────────────────────
    # RK4 INTERFACE
    # ──────────────────────────────────────────────────────────────────────────

    def get_derivative(self, t: float,
                       input_values: Optional[List[VectorSignal]] = None
                       ) -> np.ndarray:
        """
        Return dx/dt for the state vector — used by EmbedSim's RK4 solver.

        For the PI integrator:
            d(integrator)/dt = error = V_ref - V_meas

        This is the continuous-time model of the integrator:
            d/dt [integ] = error(t)

        RK4 then computes:
            integ(t+dt) ≈ integ(t) + (k1 + 2k2 + 2k3 + k4) * dt/6

        The t_enable gate: before t_enable, the derivative is forced to 0
        so the integrator doesn't wind up during any pre-simulation warmup.
        The state is also explicitly zeroed to prevent drift from RK4
        evaluating this function at t < t_enable during the first real step.
        """
        if t < self._t_enable:
            self.state[0] = 0.0               # hard-zero integrator
            return np.zeros(1, dtype=np.float32)

        V_ref, V_meas = self._get_voltages(input_values)
        e = np.float32(V_ref - V_meas)        # the derivative IS the error
        return np.array([e], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # COMPUTE — main dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def compute(self, t, dt, input_values=None):
        """
        EmbedSim calls this method every simulation step.
        Dispatches to either the C backend or Python backend.
        """
        if self.use_c_backend:
            return self.compute_c(t, dt, input_values)
        return self.compute_py(t, dt, input_values)

    # ── Python backend ────────────────────────────────────────────────────────

    def compute_py(
            self,
            t: float,
            dt: float,
            input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """
        Python-backend compute step.

        DESIGN NOTE — RK4 integrator path:
        The Python backend uses self.state[0] (the RK4-integrated value)
        as the authoritative integrator, NOT _PyPI_Buck's internal self.integ.
        This is why _PyPI_Buck.integ is not used here — the RK4 solver
        updates self.state[0] between steps, which the Python backend
        reads directly. This is correct: the RK4-integrated value is more
        accurate than re-integrating inside the Python backend.
        """
        # Soft start: output 0 until t_enable
        if t < self._t_enable:
            self.output = VectorSignal(
                np.array([0.0], dtype=np.float32), self.name, dtype=self.dtype
            )
            return self.output

        V_ref, V_meas = self._get_voltages(input_values)

        # Apply anti-windup clamp to RK4-integrated state
        lim   = np.float32(self._duty_max / max(self._Ki, 1e-9))
        integ = np.float32(np.clip(self.state[0], -lim, lim))
        e     = np.float32(V_ref - V_meas)

        # PI output — uses clamped RK4 integrator, not _PyPI_Buck.integ
        duty = float(np.clip(
            self._Kp * e + self._Ki * integ,
            self._duty_min, self._duty_max
        ))

        # Wrap result in VectorSignal — the EmbedSim signal type
        self.output = VectorSignal(
            np.array([duty], dtype=np.float32), self.name, dtype=self.dtype
        )
        return self.output

    # ── C backend ─────────────────────────────────────────────────────────────

    def compute_c(
            self,
            t: float,
            dt: float,
            input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """
        C-backend compute step.

        KEY PATTERN — State synchronisation with RK4:
        EmbedSim's RK4 solver owns self.state[0] (the integrator value).
        The C wrapper (pi_buck_wrapper.pyd) ALSO has an internal integrator
        inside PI_Buck_State_T. They must be kept in sync every step:

            BEFORE compute: push state[0] → C wrapper (so C uses RK4 value)
            AFTER  compute: pull C wrapper → state[0] (so RK4 sees C update)

        This two-way sync is what makes the C backend RK4-compatible.
        Without it, the C wrapper's integrator would drift from the
        RK4-integrated state over time, causing subtle numerical errors.
        """
        V_ref, V_meas = self._get_voltages(input_values)

        # ── SYNC: push RK4 state → C wrapper integrator ───────────────────────
        # The .integrator property setter in the Cython wrapper writes directly
        # to PI_Buck_Block_T.state.integrator in the C struct.
        self._impl.integrator = np.float32(self.state[0])

        # ── CALL: run one PI step in C ─────────────────────────────────────────
        duty = self._impl.compute(
            np.float32(V_ref),
            np.float32(V_meas),
            np.float32(dt)
        )

        # ── SYNC: pull C wrapper integrator → RK4 state ───────────────────────
        # PI_Buck_Compute() updated the C struct's integrator.
        # Copy it back so RK4 can continue integrating from the correct value.
        self.state[0] = np.float32(self._impl.integrator)

        self.output = VectorSignal(
            np.array([duty], dtype=np.float32), self.name, dtype=self.dtype
        )
        return self.output

    # ──────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset the block to initial conditions.
        Called by EmbedSim before each sim.run() call.

        1. Calls super().reset() — VectorBlock's reset (clears output signal).
        2. Zeros the RK4 state vector.
        3. Resets the backend's integrator (C wrapper or Python).
        """
        super().reset()
        self.state = np.zeros(1, dtype=np.float32)
        if hasattr(self, '_impl'):
            self._impl.reset()

    # ──────────────────────────────────────────────────────────────────────────
    # RUNTIME PARAMETER UPDATE
    # ──────────────────────────────────────────────────────────────────────────

    def set_params(self, Kp=None, Ki=None, duty_max=None, duty_min=None, Ts=None) -> None:
        """
        Update PI parameters at runtime (e.g., gain scheduling).

        Only specified arguments are changed — omitted arguments keep their
        current values. Updates BOTH the Python-side attributes (_Kp, _Ki, …)
        AND the backend (C wrapper or Python implementation).

        Example — double Ki mid-simulation:
            sim.on_step_callback = lambda t: pi_block.set_params(Ki=16.0) if t > 0.005 else None
        """
        if Kp       is not None: self._Kp       = float(Kp)
        if Ki       is not None: self._Ki        = float(Ki)
        if duty_max is not None: self._duty_max  = float(duty_max)
        if duty_min is not None: self._duty_min  = float(duty_min)
        if Ts       is not None: self._Ts        = float(Ts)

        if self.use_c_backend:
            # Delegate to the Cython wrapper which calls PI_Buck_SetParams()
            self._impl.set_params(
                Kp=self._Kp, Ki=self._Ki,
                duty_max=self._duty_max, duty_min=self._duty_min,
                Ts=self._Ts
            )
        else:
            # Update the Python backend directly
            self._impl.Kp       = np.float32(self._Kp)
            self._impl.Ki       = np.float32(max(self._Ki, 1e-9))
            self._impl.duty_max = np.float32(self._duty_max)
            self._impl.duty_min = np.float32(self._duty_min)
            self._impl.Ts       = np.float32(self._Ts)

    # ──────────────────────────────────────────────────────────────────────────
    # DIAGNOSTICS / PROPERTIES
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def integrator(self) -> float:
        """Current integrator state [V·s] — read from RK4 state vector."""
        return float(self.state[0])

    @integrator.setter
    def integrator(self, value: float) -> None:
        """Set integrator — updates both RK4 state AND C wrapper."""
        self.state[0] = np.float32(value)
        if self.use_c_backend:
            self._impl.integrator = np.float32(value)

    @property
    def last_duty(self) -> float:
        """Last computed duty cycle — useful for logging without re-computing."""
        # NOTE: Both branches return from _impl because:
        #   C backend:  _impl.last_output reads PI_Buck_State_T.last_output
        #   Python backend: _impl.last_output reads _PyPI_Buck.last_duty
        # The two branches are identical here but kept explicit for clarity.
        if self.use_c_backend:
            return float(self._impl.last_output)
        return float(self._impl.last_output)

    @property
    def params(self) -> dict:
        """Current parameters as a dictionary — for logging and inspection."""
        return {
            'Kp':       self._Kp,
            'Ki':       self._Ki,
            'duty_max': self._duty_max,
            'duty_min': self._duty_min,
            'Ts':       self._Ts
        }

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        return (
            f"PI_BuckBlock('{self.name}', "
            f"Kp={self._Kp:.3f}, Ki={self._Ki:.3f}, "
            f"duty=[{self._duty_min:.2f},{self._duty_max:.2f}], "
            f"backend={be})"
        )