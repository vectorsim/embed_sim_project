# pi_buck_wrapper.pyx
# ====================
#
# PURPOSE
# -------
# This is a CYTHON file (.pyx). Cython is a language that is a superset of
# Python: you can write normal Python code in it, but you can also write
# C-style typed declarations. Cython then compiles this file into a .c file,
# which is compiled into a .pyd (on Windows) or .so (on Linux) — a binary
# Python extension module that can be imported like any Python module.
#
# WHY DO WE NEED THIS?
# --------------------
# Our PI controller is written in C (pi_buck_controller.c / .h).
# Python cannot directly call C functions — it does not know what a
# "PI_Buck_Block_T*" is, what sizeof(PI_Buck_Input_T) is, or how to
# marshal Python floats into C float arguments.
#
# Cython acts as the BRIDGE:
#   Python code            →  Cython (.pyx)  →  C extension (.pyd)
#   ctrl.compute(12, 0, dt)   fills structs      calls PI_Buck_Compute()
#
# THE KEY BENEFIT: ZERO OVERHEAD AT RUNTIME
# ------------------------------------------
# Because the structs (_block, _in, _out) are declared "cdef" (C-level),
# they live INSIDE the Python object on the C heap — not as Python
# dictionaries or lists. When you call ctrl.compute(), Cython writes
# directly into those C structs and calls the C function. No Python
# objects are created during the hot path. This is essentially as fast
# as calling the C function directly from another C file.
#
# HOW TO BUILD THIS FILE
# ----------------------
#   cd buck_converter\c_src
#   python setup_pi_buck.py build_ext --inplace
#   # or just run: build_pi_buck.bat
#
# The build pipeline:
#   pi_buck_wrapper.pyx  →(Cython)→  pi_buck_wrapper.c
#                                              ↓
#   pi_buck_controller.c  ──────────(MSVC/GCC)→  pi_buck_wrapper.pyd
#
# IMPORTING IN PYTHON
# --------------------
#   import pi_buck_wrapper
#   ctrl = pi_buck_wrapper.PI_BuckWrapper()
#   ctrl.set_params(Kp=0.15, Ki=8.0, duty_max=0.9, duty_min=0.1, Ts=1e-4)
#   duty = ctrl.compute(V_ref=12.0, V_meas=10.5, dt=1e-4)
#
# ==============================================================================


# ==============================================================================
#  SECTION 1 — TELL CYTHON ABOUT THE C API
# ==============================================================================
#
# "cdef extern from" is a Cython directive that says:
#   "The types and functions declared inside this block are defined in the
#    named C header file. When you compile, include that header."
#
# We are essentially re-declaring the C structs and function prototypes here
# in Cython's syntax so that Cython knows their shape and can generate the
# correct C code to interact with them.
#
# IMPORTANT: These declarations do NOT re-define the structs — they just
# tell Cython what already exists in the C header. If the C header changes
# (e.g., a new field is added), you must update both here AND in the .pyx.
# ==============================================================================

cdef extern from "pi_buck_controller.h":

    # ── Struct: PI_Buck_Params_T ──────────────────────────────────────────────
    # "ctypedef struct" mirrors the C typedef struct exactly.
    # Each field must have the correct C type. We use "float" here (not
    # real32_T) because Cython maps "float" directly to C's float type,
    # and real32_T is defined as "typedef float real32_T" in Sys_Types.h.
    # Using "float" avoids needing to also declare real32_T in Cython.
    ctypedef struct PI_Buck_Params_T:
        float Kp        # Proportional gain [1/V]
        float Ki        # Integral gain [1/(V·s)]
        float duty_max  # Max duty cycle [0-1]
        float duty_min  # Min duty cycle [0-1]
        float Ts        # Default sample period [s]

    # ── Struct: PI_Buck_State_T ───────────────────────────────────────────────
    # The controller's "memory" between steps.
    ctypedef struct PI_Buck_State_T:
        float integrator    # Accumulated integral of voltage error
        float prev_error    # Previous step error (for debug only)
        float last_output   # Last duty cycle produced (for debug only)

    # ── Struct: PI_Buck_Block_T ───────────────────────────────────────────────
    # The complete controller "object" — contains params and state nested.
    # In C: struct { PI_Buck_Params_T params; PI_Buck_State_T state; }
    ctypedef struct PI_Buck_Block_T:
        PI_Buck_Params_T params    # tuning constants
        PI_Buck_State_T  state     # run-time memory

    # ── Struct: PI_Buck_Input_T ───────────────────────────────────────────────
    # One-step input bundle passed into PI_Buck_Compute().
    ctypedef struct PI_Buck_Input_T:
        float V_ref    # Desired voltage [V]
        float V_meas   # Measured voltage [V]

    # ── Struct: PI_Buck_Output_T ──────────────────────────────────────────────
    # One-step output from PI_Buck_Compute().
    ctypedef struct PI_Buck_Output_T:
        float duty     # PWM duty cycle [0-1]

    # ── C function prototypes ─────────────────────────────────────────────────
    # These declarations tell Cython that these four functions exist in C.
    # Cython will generate code that calls them directly — no overhead.
    #
    # Note: the "const" qualifier on pIn in PI_Buck_Compute is declared here
    # and enforced at C compile time. It prevents any accidental modification
    # of the input struct inside the C function.

    void PI_Buck_Init(PI_Buck_Block_T* pPI)
    # Initialise block with safe defaults. Must be called before Compute().

    void PI_Buck_SetParams(PI_Buck_Block_T* pPI,
                           float Kp, float Ki,
                           float duty_max, float duty_min,
                           float Ts)
    # Override default parameters with tuned values.

    void PI_Buck_ResetState(PI_Buck_Block_T* pPI)
    # Zero integrator and state memory (called on fault clear or re-enable).

    void PI_Buck_Compute(PI_Buck_Block_T*       pPI,
                         const PI_Buck_Input_T* pIn,
                         float                   dt,
                         PI_Buck_Output_T*      pOut)
    # Run ONE step of the PI algorithm. Called every control period.


# ==============================================================================
#  SECTION 2 — THE PYTHON-CALLABLE WRAPPER CLASS
# ==============================================================================
#
# "cdef class" is a Cython extension type — it compiles to a C struct
# that IS a Python object (a subtype of PyObject), but whose attributes
# can be C-level (cdef) for zero-overhead access.
#
# From Python you use it exactly like a normal class:
#   ctrl = PI_BuckWrapper()
#   duty = ctrl.compute(12.0, 10.0, 1e-4)
#
# But internally there are no Python dictionaries involved — the three
# cdef attributes below are stored as raw C structs inside the object.
# ==============================================================================

cdef class PI_BuckWrapper:
    """
    Python wrapper for the PI Buck Controller C implementation.

    This class provides a Pythonic interface to the C PI controller
    while maintaining the zero-overhead performance of direct C calls.

    Usage example:
        ctrl = PI_BuckWrapper()
        ctrl.set_params(Kp=0.15, Ki=8.0, duty_max=0.9, duty_min=0.1, Ts=1e-4)
        for each_step:
            duty = ctrl.compute(V_ref, V_meas, dt)
            # use duty to set PWM...
    """

    # ── C-level attributes (NOT Python objects) ───────────────────────────────
    # These three lines declare that the wrapper object CONTAINS (by value,
    # not by pointer) the three C structs needed by the algorithm.
    #
    # "By value" means the actual bytes of the struct are allocated inside
    # the Python object — no separate malloc(), no extra pointer dereference.
    # When the Python object is garbage-collected, the structs are freed too.
    #
    # "cdef" attributes are NOT accessible from Python code directly
    # (you can't do "ctrl._block.state.integrator" from Python).
    # Access is provided through Python properties defined below.

    cdef PI_Buck_Block_T   _block  # complete PI controller (params + state)
    cdef PI_Buck_Input_T   _in     # input scratchpad (reused every compute call)
    cdef PI_Buck_Output_T  _out    # output scratchpad (reused every compute call)


    # ── __cinit__: C-level constructor ────────────────────────────────────────
    # "__cinit__" (C init) is called by Cython BEFORE __init__ and even before
    # the object is fully created. This is the right place to initialise C-level
    # attributes because by the time __cinit__ runs, the memory for _block,
    # _in, and _out has been allocated.
    #
    # WHY NOT __init__?
    # Cython's C-level structs (cdef attributes) must be initialised in
    # __cinit__, not __init__, to guarantee they are valid even if a subclass
    # raises an exception before __init__ completes.
    def __cinit__(self):
        # Call the C Init function to set default params and zero the state.
        # "&self._block" is the C address-of operator — we pass a pointer
        # to the C struct so the C function can modify it.
        PI_Buck_Init(&self._block)

        # Zero the input and output scratchpads.
        # These will be overwritten on every compute() call anyway,
        # but it is good practice to initialise them to a known state.
        self._in.V_ref  = 0.0
        self._in.V_meas = 0.0
        self._out.duty  = 0.0


    # ── set_params() ──────────────────────────────────────────────────────────
    # Public Python method to set PI gains and limits.
    #
    # The "float" type annotation in the argument list is a Cython directive:
    # Python floats (64-bit) are automatically narrowed to 32-bit C floats
    # before being passed to the C function. This is intentional — our
    # controller uses real32_T (float) throughout.
    #
    # If the caller passes a value outside the valid range (e.g., Ki=-1),
    # PI_Buck_SetParams() in C will silently guard it (floor Ki at 1e-6).
    def set_params(self, float Kp, float Ki, float duty_max, float duty_min, float Ts):
        """Set PI gains and duty cycle limits.

        Args:
            Kp:       Proportional gain [1/V]. Typical: 0.1 – 0.3
            Ki:       Integral gain [1/(V·s)]. Typical: 5.0 – 20.0. Must be > 0.
            duty_max: Maximum duty cycle [0-1]. Typical: 0.9
            duty_min: Minimum duty cycle [0-1]. Typical: 0.05 – 0.1
            Ts:       Default sample period [s]. Typical: 1e-4 (100 µs)
        """
        PI_Buck_SetParams(&self._block, Kp, Ki, duty_max, duty_min, Ts)


    # ── reset() ───────────────────────────────────────────────────────────────
    # Zeros the integrator and all state memory.
    # Call this before re-enabling the controller after a pause.
    def reset(self):
        """Zero the integrator and all state memory."""
        PI_Buck_ResetState(&self._block)


    # ── compute() — THE HOT PATH ──────────────────────────────────────────────
    # This is called every simulation step (every 1 µs in the EmbedSim RK4
    # simulation = 10,000 calls per 10 ms run).
    #
    # THE CRITICAL PERFORMANCE PATH:
    #   1. Python call overhead (minimal — Cython wraps it efficiently)
    #   2. Two C float assignments (write V_ref, V_meas into _in struct)
    #   3. ONE C function call: PI_Buck_Compute()
    #      → inside: 5 float operations + 2 clamp calls = ~20 FP instructions
    #   4. Return _out.duty as a Python float
    #
    # No Python objects are created during steps 2-4. This is the same
    # performance as calling the C function from another C file.
    def compute(self, float V_ref, float V_meas, float dt):
        """Run one PI control step.

        Args:
            V_ref:  Desired output voltage [V]
            V_meas: Measured output voltage [V]
            dt:     Time step for this call [s]. If 0, uses stored Ts.

        Returns:
            float: PWM duty cycle [0.0 – 1.0]
        """
        # Fill the input struct with this step's values.
        # Cython generates: self->_in.V_ref = (float)V_ref;
        self._in.V_ref  = V_ref
        self._in.V_meas = V_meas

        # Call the C function. This is a direct C call — no GIL manipulation,
        # no Python object creation. Cython generates:
        #   PI_Buck_Compute(&self->_block, &self->_in, dt, &self->_out);
        PI_Buck_Compute(&self._block, &self._in, dt, &self._out)

        # Return the duty cycle result. Cython converts the C float to a
        # Python float automatically.
        return self._out.duty


    # ── integrator property ───────────────────────────────────────────────────
    # Python @property lets you read ctrl.integrator as if it were a normal
    # attribute, but it actually calls into the C struct.
    #
    # WHY EXPOSE THE INTEGRATOR?
    # EmbedSim's RK4 solver needs to be able to READ and WRITE the controller
    # state when computing the intermediate k1, k2, k3, k4 stages. Without
    # this property, state synchronisation between RK4 stages would be impossible.
    @property
    def integrator(self):
        """Current integrator value [V·s]. Read/write.

        Used by EmbedSim's RK4 solver for state synchronisation between
        integration stages. Also useful for oscilloscope / debugging.
        """
        return self._block.state.integrator

    @integrator.setter
    def integrator(self, float value):
        """Set integrator directly (used by RK4 state synchronisation)."""
        self._block.state.integrator = value


    # ── last_output property ──────────────────────────────────────────────────
    # Read-only property exposing the last computed duty cycle.
    # Useful for logging and scope without calling compute() again.
    @property
    def last_output(self):
        """Last computed duty cycle [0-1]. Read-only."""
        return self._block.state.last_output


    # ── Backward compatibility methods ────────────────────────────────────────
    # These methods duplicate the property functionality as regular methods.
    # They exist because older versions of pi_buck_block.py called
    # ctrl.get_integrator() instead of ctrl.integrator.
    # Keep them to avoid breaking any existing code that uses the old API.
    #
    # In new code, prefer the property syntax (ctrl.integrator = x)
    # over the method syntax (ctrl.set_integrator(x)).

    def get_integrator(self):
        """Backward compatibility. Prefer: ctrl.integrator"""
        return self._block.state.integrator

    def set_integrator(self, float value):
        """Backward compatibility. Prefer: ctrl.integrator = value"""
        self._block.state.integrator = value

    def get_last_output(self):
        """Backward compatibility. Prefer: ctrl.last_output"""
        return self._block.state.last_output

# ── End of pi_buck_wrapper.pyx ────────────────────────────────────────────────
