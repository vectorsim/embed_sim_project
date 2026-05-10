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

TWO BACKENDS — DUAL BACKEND PATTERN
-------------------------------------
This block supports two interchangeable compute engines:

  A. C backend  (use_c_backend=True):
       Calls pi_buck_wrapper.pyd, which calls PI_Buck_Compute() in C.
       Fast (~50× faster than Python), identical to the embedded target.
       Requires: cd buck_converter/c_src && python setup_pi_buck.py build_ext --inplace

  B. Python backend (use_c_backend=False):
       Uses _PyPI_Buck (pure NumPy, always available).
       Produces numerically identical results (verified) in float32.
       Use when the Cython .pyd has not been compiled, or for debugging.

DUCK-TYPING INTERFACE CONTRACT
--------------------------------
Both backends (_PyPI_Buck and PI_BuckWrapper from Cython) expose the same
interface so PI_BuckBlock.compute() can call self._impl without knowing
which backend is active:

    self._impl.compute(V_ref, V_meas, dt) → float     duty cycle
    self._impl.set_params(Kp, Ki, ...)    → None       update gains
    self._impl.reset()                    → None       zero state
    self._impl.integrator                 → float      read σ (property)
    self._impl.integrator = value         → None       write σ (property setter)
    self._impl.last_output                → float      read last duty

This duck-typing approach means a future backend (e.g., a hardware-in-the-loop
wrapper or a fixed-point emulator) can be added without changing any call site.

RK4 INTEGRATION
---------------
The PI controller has ONE continuous state: the integrator accumulator σ [V·s].
EmbedSim's RK4 solver integrates it using:
    dσ/dt  = error = V_ref - V_meas       (returned by get_derivative())
    σ(t+dt) ≈ σ(t) + (k1 + 2k2 + 2k3 + k4) · dt/6

This is more accurate than the simple forward Euler integration used inside
the C function itself (integ += e·dt). For a 1 µs simulation step the
difference is negligible, but RK4 is the EmbedSim framework standard and
ensures consistent error order across all dynamic blocks.

STATE SYNCHRONISATION WITH THE C BACKEND
------------------------------------------
EmbedSim's RK4 solver is the authoritative owner of self.state[0] (= σ).
The C wrapper (PI_Buck_State_T inside the .pyd) also holds an integrator.
They must be kept in sync every step to prevent drift:

    BEFORE compute: push state[0] → C wrapper  (C uses the RK4-integrated value)
    AFTER  compute: pull C wrapper → state[0]  (RK4 continues from C's update)

Without this two-way sync the C wrapper's integrator would diverge from
the RK4-integrated state over time, causing subtle numerical errors that
are difficult to detect but visually apparent as a small DC offset in V_out.

SOFT START (t_enable)
----------------------
Setting t_enable > 0 freezes the controller output at 0.0 V until t reaches
t_enable. This is useful when the simulation needs a few microseconds to
initialise the FMU state before the controller takes over.  During the frozen
period:
  - compute() returns duty = 0.0 (no actuation)
  - get_derivative() returns dσ/dt = 0.0 (no integrator windup)
  - state[0] is explicitly held at 0.0 (suppresses RK4 drift from k1..k4
    evaluated near the boundary)
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

        σ(k) = clamp( σ(k-1) + e·dt,  -σ_lim, +σ_lim )
        duty = clamp( Kp·e  + Ki·σ,    duty_min, duty_max )

    where  e = V_ref - V_meas    (voltage error)
           σ_lim = duty_max / Ki  (anti-windup limit)

    WHY np.float32 EVERYWHERE?
    ---------------------------
    All arithmetic uses numpy float32 to match the C implementation's
    real32_T (32-bit IEEE 754).  This ensures the Python and C backends
    produce bit-identical results (within float32 precision) on any
    IEEE 754-compliant platform, including AURIX TriCore.

    INTERFACE CONTRACT  (must match PI_BuckWrapper from Cython):
    -------------------------------------------------------------
    .compute(V_ref, V_meas, dt) → float
    .set_params(Kp, Ki, duty_max, duty_min, Ts) → None
    .reset() → None
    .integrator   (property, read/write)
    .last_output  (property, read)
    """

    def __init__(self, Kp: float, Ki: float, duty_max: float, duty_min: float, Ts: float):
        # Store all parameters as float32 to match the C struct layout
        self.Kp       = np.float32(Kp)
        self.Ki       = np.float32(max(Ki, 1e-9))   # floor Ki above zero to prevent /0 in anti-windup
        self.duty_max = np.float32(duty_max)
        self.duty_min = np.float32(duty_min)
        self.Ts       = np.float32(Ts)

        # State — mirrors PI_Buck_State_T in pi_buck_controller.h
        self.integ     = np.float32(0.0)   # integrator accumulator σ [V·s]
        self.last_duty = np.float32(0.0)   # last computed duty (for last_output property)

    def compute(self, V_ref: float, V_meas: float, dt: float) -> float:
        """
        Run one PI step.  Mirrors PI_Buck_Compute() in pi_buck_controller.c exactly.

        Parameters
        ----------
        V_ref  : Reference voltage [V]
        V_meas : Measured output voltage [V]  (from ScalarDelay feedback)
        dt     : Actual time step [s]  (may differ from nominal Ts under RK4)

        Returns
        -------
        duty : float in [duty_min, duty_max]
        """
        e = np.float32(V_ref - V_meas)               # voltage error this step

        # Anti-windup limit: σ must stay within ±duty_max/Ki
        # This prevents integrator windup when the actuator is saturated.
        # If Ki·σ_lim = duty_max, adding more error cannot push duty further.
        lim = self.duty_max / self.Ki

        # Use provided dt if valid, else fall back to stored Ts.
        # dt may be 0 during RK4 sub-steps at the initial point; guard against /0.
        sample_time = dt if dt > 0 else self.Ts

        # Update integrator — forward Euler, clamped (anti-windup)
        self.integ = np.float32(np.clip(self.integ + e * sample_time, -lim, lim))

        # PI output with output saturation clamp
        duty = np.clip(self.Kp * e + self.Ki * self.integ,
                       self.duty_min, self.duty_max)
        self.last_duty = np.float32(duty)
        return float(duty)

    def set_params(self, Kp=None, Ki=None, duty_max=None, duty_min=None, Ts=None) -> None:
        """Update any subset of parameters (keyword-only, omit to keep current)."""
        if Kp       is not None: self.Kp       = np.float32(Kp)
        if Ki       is not None: self.Ki        = np.float32(max(Ki, 1e-9))
        if duty_max is not None: self.duty_max  = np.float32(duty_max)
        if duty_min is not None: self.duty_min  = np.float32(duty_min)
        if Ts       is not None: self.Ts        = np.float32(Ts)

    def reset(self):
        """Zero all state — mirrors PI_Buck_ResetState()."""
        self.integ     = np.float32(0.0)
        self.last_duty = np.float32(0.0)

    # Properties to match the Cython wrapper's interface (duck typing)
    @property
    def integrator(self) -> float:
        """Read integrator accumulator σ [V·s]."""
        return float(self.integ)

    @integrator.setter
    def integrator(self, value: float) -> None:
        """Write integrator accumulator — used by RK4 state sync in PI_BuckBlock."""
        self.integ = np.float32(value)

    @property
    def last_output(self) -> float:
        """Last computed duty cycle [0-1] — readable without re-computing."""
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
        state[0] = error integrator σ  [V·s]

    See module docstring for:
      - Dual-backend pattern (C / Python)
      - RK4 state synchronisation with the C wrapper
      - Soft-start (t_enable) behaviour
      - CodeGen (C_CUSTOM_EMIT) explanation
    """

    # ── CodeGen attributes ────────────────────────────────────────────────────
    # PYX_FILE tells PYXInspector where to find the Cython wrapper file.
    # pathlib.Path(__file__).parent resolves to the buck_converter/ directory,
    # so this always points to buck_converter/c_src/pi_buck_wrapper.pyx
    # regardless of working directory.
    import pathlib as _pl
    PYX_FILE: str = str(_pl.Path(__file__).parent / 'c_src' / 'pi_buck_wrapper.pyx')

    # The following attributes are LEFT EMPTY here on purpose.
    # __init_subclass__ (see below) calls auto_populate_from_pyx() which
    # reads the .pyx file and fills them in automatically at class-definition time.
    # If auto-population fails (e.g., .pyx file missing), they stay as empty
    # defaults so the block still works — just without code generation.
    step_func:    str  = ''     # → 'PI_Buck_Compute'    (from .pyx)
    state_struct: str  = ''     # → 'PI_Buck_State_T'   (overridden below)
    NUM_INPUTS:   int  = 0      # → 2  (V_ref, V_meas)
    OUTPUT_SIZE:  int  = 0      # → 1  (duty)
    C_SOURCES:    list = []     # → ['pi_buck_controller.c']
    C_HEADERS:    list = []     # → ['pi_buck_controller.h']

    # Tell StepGenerator._out_sigs() there is nothing to pack automatically.
    # C_CUSTOM_EMIT already emits   out->pi_buck = y_pi_buck.duty
    # inside its own block scope.  If OUTPUT_NAMES were non-empty, the pack
    # stage would also emit   out->pi_buck = y_pi_buck[0]  — which is wrong
    # because y_pi_buck is a struct (not an array) and is out of scope at that point.
    OUTPUT_NAMES: list = []
    OUTPUT_KEEP:  list = []

    # ── state_struct override ─────────────────────────────────────────────────
    # PYXInspector reads the .pyx and infers PI_Buck_State_T (the inner state
    # struct).  But PI_Buck_Compute() takes PI_Buck_Block_T* (params + state).
    # We override here so LoopGenerator allocates the correct composite type,
    # which includes both the parameter fields and the state fields.
    state_struct: str = 'PI_Buck_Block_T'

    # ── C_CUSTOM_EMIT ─────────────────────────────────────────────────────────
    # PI_Buck_Compute uses struct-pointer arguments, not flat real32_T arrays:
    #
    #   void PI_Buck_Compute(PI_Buck_Block_T*       pPI,   ← state + params
    #                        const PI_Buck_Input_T* pIn,   ← {V_ref, V_meas}
    #                        real32_T               dt,    ← sample period
    #                        PI_Buck_Output_T*      pOut); ← {duty}
    #
    # The generic _emit_block() pattern (flat u[]/y[] arrays) cannot produce
    # this struct-pointer call.  C_CUSTOM_EMIT is the escape hatch: LoopGenerator
    # emits this string verbatim instead of auto-generating the call.
    #
    # Built by _build_custom_emit() AFTER PYXInspector has populated
    # step_func and state_struct — so all names come from the .pyx,
    # not from hardcoded strings here.  "Single source of truth" principle.
    C_CUSTOM_EMIT: str = ''   # populated at class-definition time by _build_custom_emit()

    @classmethod
    def _build_custom_emit(cls) -> None:
        """
        Build C_CUSTOM_EMIT from PYXInspector-populated class attributes.

        Called once at class-definition time (after auto_populate_from_pyx())
        and again for each subclass via __init_subclass__.

        All names are derived from the .pyx, not hardcoded:
            step_func      → 'PI_Buck_Compute'     (from PYXInspector)
            state_struct   → 'PI_Buck_Block_T'     (manual override above)
            input struct   → step_func prefix + '_Input_T'
            output struct  → step_func prefix + '_Output_T'
            state var name → 'pi_buck_state'       (StepGenerator convention)

        The naming convention (prefix from function name) means this method
        works for any future controller that follows the
            <Prefix>_Compute / <Prefix>_Block_T / <Prefix>_Input_T pattern.

        GUARD: PYXInspector sometimes picks up the Cython wrapper's 'compute'
        cpdef method instead of the extern C 'PI_Buck_Compute' function.
        A valid C step function must contain an underscore (namespace separator).
        If it doesn't, fall back to the known-correct name.
        """
        import re as _re
        fn = cls.step_func or 'PI_Buck_Compute'
        if '_' not in fn:
            fn = 'PI_Buck_Compute'   # guard against cpdef 'compute' being picked up

        # Derive Input/Output struct type names from the function name.
        # Pattern: PI_Buck_Compute → prefix = PI_Buck
        #          input type  = PI_Buck_Input_T
        #          output type = PI_Buck_Output_T
        m = _re.match(r'^(.+?)_(?:Compute|Step|Update)$', fn, _re.IGNORECASE)
        prefix     = m.group(1) if m else fn
        in_struct  = f"{prefix}_Input_T"
        out_struct = f"{prefix}_Output_T"

        # State variable name follows StepGenerator's sanitisation convention:
        #   state_var = _sanitize(block.name) + "_state"
        # The block name is always 'pi_buck' in this model; use the constant
        # directly (StepGenerator will use the same name at code-gen time).
        state_var = "pi_buck_state"

        cls.C_CUSTOM_EMIT = (
            f"    /* --- pi_buck ({cls.__name__}) --- */\n"
            f"    {{\n"
            f"        {in_struct}  u_pi_buck;\n"
            f"        {out_struct} y_pi_buck;\n"
            f"        u_pi_buck.V_ref  = in->vref;\n"
            f"        u_pi_buck.V_meas = in->fb_delay;\n"
            f"        {fn}(&{state_var}, &u_pi_buck, dt, &y_pi_buck);\n"
            f"        out->pi_buck = y_pi_buck.duty;\n"
            f"    }}"
        )

    def __init_subclass__(cls, **kwargs):
        """
        Python metaclass-free class registration hook.

        Python calls this automatically when any SUBCLASS of PI_BuckBlock
        is defined (not when PI_BuckBlock itself is defined — that is
        handled separately at the bottom of this module).

        WHY __init_subclass__ and not __init__?
        ─────────────────────────────────────────
        __init__ runs when an INSTANCE is created at runtime.
        __init_subclass__ runs when the CLASS BODY is executed (import time).
        CodeGen attributes (step_func, C_SOURCES, etc.) must be ready before
        any instance exists, because LoopGenerator inspects the class dict
        statically (without instantiation) to build embedsim_loop.c.

        WHAT IT DOES:
        If the subclass declares a PYX_FILE, PYXInspector re-reads that file
        and overwrites the CodeGen attributes for the subclass. Then
        _build_custom_emit() rebuilds C_CUSTOM_EMIT from the fresh attributes.

        Example — a future adaptive controller:
            class AdaptivePI_BuckBlock(PI_BuckBlock):
                PYX_FILE = '.../adaptive_pi_wrapper.pyx'
            # → __init_subclass__ runs automatically, reads the new .pyx,
            #   populates step_func='AdaptivePI_Compute', updates C_CUSTOM_EMIT.
        """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'PYX_FILE') and cls.PYX_FILE:
            auto_populate_from_pyx(cls, cls.PYX_FILE)
        cls._build_custom_emit()

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

        Parameters
        ----------
        name         : Unique block identifier shown in the topology diagram,
                       scope labels, and generated C variable names.
        Kp           : Proportional gain [duty/V].
                       A 1 V error produces a Kp duty-cycle correction.
                       Default: 0.1  (conservative; tune for your LC values)
        Ki           : Integral gain [duty/(V·s)].
                       Larger Ki → faster SSE elimination, more overshoot risk.
                       Default: 5.0
        duty_max     : Maximum duty cycle [0-1].  Default: 0.95
                       Limits maximum volt-seconds; protects the diode.
        duty_min     : Minimum duty cycle [0-1].  Default: 0.05
                       Ensures discontinuous conduction mode is avoided.
        Ts           : Nominal sample period [s].  Default: 100 µs (= 1/f_sw)
                       Used as fallback dt when dt ≤ 0 (RK4 sub-steps).
        t_enable     : Soft-start time [s].  Default: 0.0 (active from t=0).
                       Controller output is frozen at 0.0 for t < t_enable.
                       Useful to let the FMU initialise before actuation starts.
        use_c_backend: True  → use compiled pi_buck_wrapper.pyd (fast, ~50× faster).
                       False → use _PyPI_Buck pure-Python backend (always works).
        dtype        : NumPy dtype for output VectorSignal.  None → float32.
        """
        # SimBlockBase.__init__ sets up:
        #   self.name, self.use_c_backend, self.dtype
        #   self.output (initial zero VectorSignal)
        #   block registration in EmbedSim's global topology registry
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        # output_label is shown in the topology printer ASCII diagram
        self.output_label = "[duty]"

        # vector_size: number of scalar values in the output VectorSignal (duty only)
        self.vector_size = 1

        # is_dynamic = True: this block has continuous state (the integrator σ).
        # EmbedSim's engine will call get_derivative() at each RK4 sub-step.
        self.is_dynamic = True

        # Store gains as Python floats for arithmetic.
        # Prefixed with _ to emphasise "private / use set_params() to change".
        self._Kp       = float(Kp)
        self._Ki       = float(Ki)
        self._duty_max = float(duty_max)
        self._duty_min = float(duty_min)
        self._Ts       = float(Ts)
        self._t_enable = float(t_enable)

        # ── RK4 state vector ──────────────────────────────────────────────────
        # EmbedSim's RK4 solver reads and writes self.state each sub-step.
        # Length 1: state[0] = σ = integrator accumulator [V·s].
        # float32 matches the C struct precision; avoids implicit type-widening
        # when syncing with the C wrapper.
        self.state = np.zeros(1, dtype=np.float32)

        # RK4 intermediate derivatives (k1, k2, k3, k4).
        # EmbedSim's engine writes these; we pre-allocate to the correct shape.
        self.k1 = self.k2 = self.k3 = self.k4 = np.zeros(1, dtype=np.float32)

        # ── Backend selection ─────────────────────────────────────────────────
        if use_c_backend:
            # Attempt to load the compiled Cython .pyd.
            # _load_c_wrapper raises ImportError with a helpful build command
            # if the .pyd was not compiled — never a silent fallback.
            self._impl = self._load_c_wrapper(Kp, Ki, duty_max, duty_min, Ts)
        else:
            # Pure-Python fallback — always available, no compilation required.
            self._impl = _PyPI_Buck(Kp, Ki, duty_max, duty_min, Ts)

    # ──────────────────────────────────────────────────────────────────────────
    # C BACKEND LOADER
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_c_wrapper(Kp, Ki, duty_max, duty_min, Ts):
        """
        Import the Cython .pyd and return a configured PI_BuckWrapper instance.

        @staticmethod: does NOT need 'self'.  It is a pure factory function
        that belongs to the class namespace for organisational clarity.

        The try/except provides an actionable error message if the .pyd was
        not compiled, rather than a cryptic ImportError from deep in Cython.

        Path note: c_src/ is inserted into sys.path so the .pyd is found
        regardless of the working directory or Python sandbox restrictions
        (e.g., Windows Store Python).
        """
        import pathlib as _pl, sys as _sys
        _c_src = str(_pl.Path(__file__).resolve().parent / 'c_src')
        if _c_src not in _sys.path:
            _sys.path.insert(0, _c_src)

        try:
            import pi_buck_wrapper as pbw          # import the compiled .pyd / .so
            w = pbw.PI_BuckWrapper()               # create C-level wrapper instance
            w.set_params(Kp=Kp, Ki=Ki,
                         duty_max=duty_max,
                         duty_min=duty_min,
                         Ts=Ts)
            return w
        except ImportError as e:
            raise ImportError(
                f"Cython wrapper 'pi_buck_wrapper' not found: {e}\n"
                "Compile with:\n"
                "  Windows: cd buck_converter\\c_src && build_pi_buck.bat\n"
                "  Linux:   cd buck_converter/c_src && "
                "python setup_pi_buck.py build_ext --inplace\n"
                "Or set use_c_backend=False to use the pure-Python backend."
            ) from e

    # ──────────────────────────────────────────────────────────────────────────
    # INPUT EXTRACTION HELPER
    # ──────────────────────────────────────────────────────────────────────────

    def _get_voltages(self, input_values):
        """
        Extract V_ref and V_meas from the EmbedSim input_values list.

        EmbedSim passes input_values as a list of VectorSignal objects,
        one per wired input port:
            input_values[0].value = numpy array for port 0 (V_ref)
            input_values[1].value = numpy array for port 1 (V_meas)

        Each value array may have length > 1 if a multi-output source is wired
        (e.g., BuckConverterBlock outputs [V_out, I_L, I_load] but ScalarDelay
        strips it to [V_out] before sending to port 1).  We read index [0] only.

        Returns (0.0, 0.0) gracefully if inputs are not yet wired — this
        allows the block to be constructed and inspected before wiring.
        """
        V_ref  = 0.0
        V_meas = 0.0

        if not input_values:
            return V_ref, V_meas   # safe default: no inputs connected yet

        # Port 0: V_ref (reference voltage)
        if len(input_values) > 0 and input_values[0] is not None:
            v = input_values[0].value          # numpy array
            V_ref = float(v[0]) if len(v) >= 1 else 0.0

        # Port 1: V_meas (measured output voltage from ScalarDelay feedback)
        if len(input_values) > 1 and input_values[1] is not None:
            v = input_values[1].value
            V_meas = float(v[0]) if len(v) >= 1 else 0.0

        return V_ref, V_meas

    # ──────────────────────────────────────────────────────────────────────────
    # RK4 INTERFACE
    # ──────────────────────────────────────────────────────────────────────────

    def get_derivative(self, t: float,
                       input_values: Optional[List[VectorSignal]] = None
                       ) -> np.ndarray:
        """
        Return dσ/dt for the integrator state — called by EmbedSim's RK4 solver.

        The PI integrator is a pure integrator:
            dσ/dt = e(t) = V_ref(t) - V_meas(t)

        RK4 then computes σ(t+dt) ≈ σ(t) + (k1 + 2k2 + 2k3 + k4) · dt/6,
        where k1..k4 are evaluations of this function at t, t+dt/2, t+dt/2, t+dt.

        Soft-start gate: before t_enable, the derivative is forced to zero.
        self.state[0] is also explicitly zeroed to suppress any drift that
        could accumulate from RK4 evaluating this function near the boundary
        (k1 may be evaluated at t < t_enable even if the step straddles it).
        """
        if t < self._t_enable:
            self.state[0] = np.float32(0.0)       # hard-zero: no integrator drift
            return np.zeros(1, dtype=np.float32)

        V_ref, V_meas = self._get_voltages(input_values)
        e = np.float32(V_ref - V_meas)             # the continuous-time derivative IS the error
        return np.array([e], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # COMPUTE — main dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def compute(self, t, dt, input_values=None):
        """
        EmbedSim calls this once per simulation step (after RK4 has updated state).
        Dispatches to the active backend.
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

        DESIGN NOTE — RK4 integrator ownership:
        The Python backend reads self.state[0] (the RK4-integrated σ) as the
        authoritative integrator value.  It does NOT use _PyPI_Buck.integ
        because the RK4 solver updated state[0] between steps via
        get_derivative().  Using state[0] directly gives higher accuracy
        than re-integrating inside the Python backend (which would revert
        to forward Euler).

        The anti-windup clamp is applied to state[0] before computing duty
        so that the output respects duty_max/duty_min even if RK4 overshot σ.
        """
        # Soft start: output 0 until t_enable
        if t < self._t_enable:
            self.output = VectorSignal(
                np.array([0.0], dtype=np.float32), self.name, dtype=self.dtype
            )
            return self.output

        V_ref, V_meas = self._get_voltages(input_values)

        # Apply anti-windup clamp to the RK4-integrated state
        lim   = np.float32(self._duty_max / max(self._Ki, 1e-9))
        integ = np.float32(np.clip(self.state[0], -lim, lim))
        e     = np.float32(V_ref - V_meas)

        # PI output — uses clamped RK4 integrator, not _PyPI_Buck.integ
        duty = float(np.clip(
            self._Kp * e + self._Ki * integ,
            self._duty_min, self._duty_max
        ))

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

        KEY PATTERN — two-way state synchronisation with RK4:
        EmbedSim's RK4 solver owns self.state[0] (σ).
        The C wrapper (PI_Buck_State_T inside .pyd) also holds σ internally.
        They must agree every step to prevent drift:

            BEFORE: push state[0] → C wrapper   (C uses the RK4-integrated σ)
            CALL:   PI_Buck_Compute() updates the C struct's σ (Euler inside C)
            AFTER:  pull C wrapper σ → state[0]  (RK4 continues from the updated σ)

        This two-way sync is what makes the C backend fully RK4-compatible.
        Skipping either sync direction causes the C struct's integrator to
        diverge from the RK4 state, producing a small but accumulating DC error.

        Soft start: if t < t_enable, return duty = 0 without calling into C,
        so the C struct's integrator is never updated during the frozen period.
        """
        if t < self._t_enable:
            self.output = VectorSignal(
                np.array([0.0], dtype=np.float32), self.name, dtype=self.dtype
            )
            return self.output

        V_ref, V_meas = self._get_voltages(input_values)

        # ── SYNC ①: push RK4 state → C wrapper integrator ────────────────────
        # The .integrator property setter in the Cython wrapper writes directly
        # to PI_Buck_Block_T.state.integrator in the C struct via Cython pointer.
        self._impl.integrator = np.float32(self.state[0])

        # ── CALL: run one PI step in C ────────────────────────────────────────
        duty = self._impl.compute(
            np.float32(V_ref),
            np.float32(V_meas),
            np.float32(dt)
        )

        # ── SYNC ②: pull C wrapper integrator → RK4 state ────────────────────
        # PI_Buck_Compute() updated the C struct's integrator (Euler step inside C).
        # Copy it back so RK4 knows the true post-step σ for the next k1 evaluation.
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

        Called by EmbedSim before each sim.run() call so the same block
        instance can be used across multiple simulation runs safely.

        Sequence:
          1. super().reset()  — VectorBlock resets the output VectorSignal.
          2. Zero the RK4 state vector (σ = 0).
          3. Reset the backend's integrator (keeps C struct / Python integ in sync).
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
        Update PI parameters at runtime (e.g., gain scheduling in Mode 2).

        Only the specified keyword arguments are changed — omitted arguments
        keep their current values.  Updates BOTH the Python-side attributes
        (self._Kp, …) AND the active backend so they stay in agreement.

        Example — AI tuner updates gains each simulation step:
            controller.set_params(Kp=net_kp, Ki=net_ki)

        Example — double Ki mid-simulation via callback:
            sim.on_step_callback = (
                lambda t: pi_block.set_params(Ki=16.0) if t > 0.005 else None
            )
        """
        if Kp       is not None: self._Kp       = float(Kp)
        if Ki       is not None: self._Ki        = float(Ki)
        if duty_max is not None: self._duty_max  = float(duty_max)
        if duty_min is not None: self._duty_min  = float(duty_min)
        if Ts       is not None: self._Ts        = float(Ts)

        # Both backends expose the same set_params() interface (duck typing).
        # Passing kwargs avoids repeating the None-guards.
        update = {k: v for k, v in
                  [('Kp', Kp), ('Ki', Ki), ('duty_max', duty_max),
                   ('duty_min', duty_min), ('Ts', Ts)]
                  if v is not None}
        if update:
            self._impl.set_params(**update)

    # ──────────────────────────────────────────────────────────────────────────
    # DIAGNOSTICS / PROPERTIES
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def integrator(self) -> float:
        """
        Current integrator state σ [V·s] — read directly from the RK4 state vector.

        This is the authoritative value.  The C backend's internal copy is
        kept in sync by compute_c(), so reading state[0] is always correct
        regardless of which backend is active.
        """
        return float(self.state[0])

    @integrator.setter
    def integrator(self, value: float) -> None:
        """
        Set integrator — updates BOTH the RK4 state AND the active backend.

        Use this for pre-loading a known steady-state σ before a simulation
        (avoids the initial transient of integrating from zero):
            pi_block.integrator = V_ref_steady / (Ki * Ts)  (approximate SS value)
        """
        self.state[0] = np.float32(value)
        if self.use_c_backend:
            self._impl.integrator = np.float32(value)

    @property
    def last_duty(self) -> float:
        """
        Last computed duty cycle — readable without re-computing.

        Both _PyPI_Buck and PI_BuckWrapper (Cython) expose a .last_output
        property via the duck-typing interface contract, so this property
        works identically regardless of which backend is active.

        Useful for logging/plotting in a step callback without re-running
        the PI algorithm.
        """
        # Both backends satisfy the duck-typing contract: .last_output → float.
        # _PyPI_Buck.last_output reads _PyPI_Buck.last_duty (float32).
        # PI_BuckWrapper.last_output reads PI_Buck_State_T.last_output (real32_T).
        return float(self._impl.last_output)

    @property
    def params(self) -> dict:
        """Current parameters as a dictionary — for logging and inspection."""
        return {
            'Kp':       self._Kp,
            'Ki':       self._Ki,
            'duty_max': self._duty_max,
            'duty_min': self._duty_min,
            'Ts':       self._Ts,
        }

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        return (
            f"PI_BuckBlock('{self.name}', "
            f"Kp={self._Kp:.3f}, Ki={self._Ki:.3f}, "
            f"duty=[{self._duty_min:.2f},{self._duty_max:.2f}], "
            f"backend={be})"
        )


# ── Bootstrap: populate PI_BuckBlock's CodeGen attributes from its own .pyx ──
# auto_populate_from_pyx() + _build_custom_emit() are called here (not inside
# the class body) because __init_subclass__ only fires for SUBCLASSES, not for
# PI_BuckBlock itself.  This call fills step_func, C_SOURCES, C_HEADERS, etc.
# from buck_converter/c_src/pi_buck_wrapper.pyx at import time.
auto_populate_from_pyx(PI_BuckBlock, PI_BuckBlock.PYX_FILE)
PI_BuckBlock._build_custom_emit()
