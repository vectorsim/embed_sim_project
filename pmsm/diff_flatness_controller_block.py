"""
diff_flatness_controller_block.py  —  pmsm/
============================================
Differential Flatness FOC Controller block — NANOTEC DB42S02 (sensorless, v4).

EmbedSim VectorBlock wrapper for the C implementation in:
    pmsm/c_src/embed_sim_dfc_controller.c / .h

There is exactly ONE implementation: the compiled AURIX C code, reached
through the Cython extension dfc_controller_wrapper.  The former Python
mirror (SpeedFusion / SlidingModeObserver / voltage law in Python) has been
removed — what runs in simulation is bit-for-bit the code that runs on the
TC38x.  compute_py() simply delegates to compute_c().

ARCHITECTURE (all inside DFC_Step() in C)
=========================================
    Clarke        phase currents -> αβ
    SMO           back-EMF observer on the z^-1 commanded voltage
    Mode SM       ALIGN (0.3 s rotor pre-position)
                  -> OPENLOOP (I-f ramp to DFC_OL_OMEGA_HANDOVER_E)
                  -> CLOSEDLOOP (flatness control on the SMO angle)
    Ref shaper    2nd-order critically damped trajectory filter
                  (OmegaRefF, analytic AlphaRefF — no numerical d/dt)
    Flatness law  IqFf = (J*AlphaRefF + B*OmegaRefF)/KT  + Kp_speed feedback
                  Vd   = R*IdRef - We*Lq*IqRef + Kp_id*e_id + IdIntegral
                  Vq   = R*IqRef + Lq*dIq/dt + We*(Ld*IdRef + LambdaPm) + Kp_iq*e_iq
    SVPWM         integrated — DFC_Step() emits Ta/Tb/Tc directly

SIGNAL BUS (input_values[0], 4 elements — DFC_Input_T field order)
==================================================================
    u[0]  speed_ref_rpm  [RPM]   Mechanical speed reference
    u[1]  ia             [A]     Phase-A current
    u[2]  ib             [A]     Phase-B current
    u[3]  ic             [A]     Phase-C current

    Sensorless: there is NO theta_m input.  The rotor angle comes from
    the internal SMO; startup is handled by the ALIGN / I-f state machine.

OUTPUT (3 elements — feeds the plant / GTM compare stage directly)
==================================================================
    y[0]  ta   [0.0 - 1.0]   Phase-A duty cycle
    y[1]  tb   [0.0 - 1.0]   Phase-B duty cycle
    y[2]  tc   [0.0 - 1.0]   Phase-C duty cycle

C ALIGNMENT
===========
All motor parameters and default gains are compile-time constants in
embed_sim_dfc_controller.h / embed_sim_dfc_gains.h.  Gains passed to the
constructor are applied at runtime via DFC_GainSet_Apply() — identical to
an AURIX overlay write during HIL commissioning.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE

# ---------------------------------------------------------------------------
# Path helpers — make the compiled wrapper importable from pmsm/ or pmsm/c_src
# ---------------------------------------------------------------------------
_HERE  = Path(__file__).resolve().parent          # pmsm/
_C_SRC = _HERE / "c_src"

for _p in (str(_HERE), str(_C_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ===========================================================================
# DFControllerBlock
# ===========================================================================

class DFControllerBlock(VectorBlock):
    """
    EmbedSim VectorBlock wrapping the sensorless Differential Flatness FOC
    Controller (C v4, integrated SVPWM).

    Accepts the 4-element input bus [speed_ref_rpm, ia, ib, ic] and emits
    the 3-element duty-cycle bus [ta, tb, tc].

    The block is C-backend only.  compute_py() delegates to compute_c(),
    so the block behaves identically regardless of how the simulation
    engine dispatches it.
    """

    # ---- EmbedSim CodeGen interface ----------------------------------------
    NUM_INPUTS  = 1
    OUTPUT_SIZE = 9

    # Input bus layout — matches DFC_Input_T signal order
    INPUT_NAMES = ["speed_ref_rpm", "ia", "ib", "ic"]
    INPUT_KEEP  = [0, 1, 2, 3]

    # Output bus layout — the full DFC_Output_T, exposed as named scalar fields
    # in the generated EmbedSim<Prefix>_Output_T struct (not an opaque array).
    #   [0] ta, [1] tb, [2] tc          duty cycles      -> GTM ATOM compare
    #   [3] omega_mech                  speed estimate   [rad/s mech]
    #   [4] theta_mech                  rotor position   [rad, 0..2pi)
    #   [5] ia_echo, [6] ib_echo, [7] ic_echo  phase currents [A] (W = -U-V)
    #   [8] mode                        0 ALIGN / 1 OPENLOOP / 2 CLOSEDLOOP
    OUTPUT_NAMES = ["ta", "tb", "tc",
                    "omega_mech", "theta_mech",
                    "ia_echo", "ib_echo", "ic_echo", "mode"]
    OUTPUT_KEEP  = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # C struct field comments emitted into the generated Input_T / Output_T.
    # Input and output use distinct field names so each gets its own comment.
    C_FIELD_COMMENTS = {
        # --- inputs (DFC_Input_T) ---
        "speed_ref_rpm": "IN : Mechanical speed reference [RPM]; range [0, 3000]",
        "ia":            "IN : Phase-A current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
        "ib":            "IN : Phase-B current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
        "ic":            "IN : Phase-C current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
        # --- outputs (DFC_Output_T) ---
        "ta":            "OUT: Phase-A duty cycle [0.0..1.0] -> GTM ATOM compare",
        "tb":            "OUT: Phase-B duty cycle [0.0..1.0] -> GTM ATOM compare",
        "tc":            "OUT: Phase-C duty cycle [0.0..1.0] -> GTM ATOM compare",
        "omega_mech":    "OUT: SMO mechanical speed estimate [rad/s]",
        "theta_mech":    "OUT: Integrated mechanical rotor position [rad, 0..2pi)",
        "ia_echo":       "OUT: Phase-A current echo [A] (passthrough of input U)",
        "ib_echo":       "OUT: Phase-B current echo [A] (passthrough of input V)",
        "ic_echo":       "OUT: Phase-C current echo [A] (star-point W = -U-V)",
        "mode":          "OUT: Controller mode (0 ALIGN / 1 OPENLOOP / 2 CLOSEDLOOP)",
    }

    # ---- C code generation linkage -----------------------------------------
    # Matches the v4 public API in embed_sim_dfc_controller.h exactly.
    step_func    = "DFC_Step"        # C: DFC_Step()
    state_struct = "DFC_State_T"     # C: DFC_State_T
    init_func    = "DFC_Init"        # C: DFC_Init()
    reset_func   = "DFC_Reset"       # C: DFC_Reset()
    C_INIT_ARGS  = []                # DFC_Init takes only the state pointer
    C_SOURCES    = [
        "embed_sim_dfc_controller.c",
        "embed_sim_coordinate_transform.c",
        "embed_sim_sv_pwm.c",
        "embed_sim_matrix.c",
    ]
    C_HEADERS    = ["embed_sim_dfc_controller.h"]

    # Cython wrapper source (consumed by PYXInspector / feature 05121967)
    PYX_FILE = str(_C_SRC / "dfc_controller_wrapper.pyx")

    # Custom C snippet emitted verbatim into the generated step function by
    # StepGenerator.  It runs inside EmbedSim<Prefix>_Step(), where the region
    # input arrives as the flat array `in-><upstream_block>[...]`.  Here the
    # upstream block is dfc_packer, so the 4 signals are in->dfc_packer[0..3]
    # in DFC_Input_T order: [speed_ref_rpm, ia, ib, ic].
    #
    # v4: SVPWM runs inside DFC_Step().  The full DFC_Output_T is unpacked into
    # the 9-wide region output y_dfc[0..8]:
    #   [0..2] Ta/Tb/Tc  [3] AngularVelocity  [4] RotorPosition
    #   [5..7] phase-current echo (W = -U-V)   [8] Mode
    # NOTE: Transform_Init() and SVM_Init() must be called once by the
    # application (cdd_app.c on the AURIX) before the first step — put them in
    # EmbedSim<Prefix>_Init() or your AppInit().
    C_CUSTOM_EMIT = """\
        /* --- dfc (DFControllerBlock, v4 sensorless) --- */
        /* DFC_Step() runs Clarke -> SMO -> mode SM -> flatness law -> SVPWM  */
        /* and outputs Ta/Tb/Tc duties plus speed/position/current estimates. */
        /* Region input : named fields speed_ref_rpm, ia, ib, ic.             */
        /* Prerequisite (once, at startup): Transform_Init(); SVM_Init();     */
        DFC_Input_T   u_dfc;
        DFC_Output_T  y_dfc_out;
        real32_T      y_dfc[9];

        u_dfc.SpeedRefRpm     = in->speed_ref_rpm;
        u_dfc.PhaseCurrents.U = in->ia;
        u_dfc.PhaseCurrents.V = in->ib;
        u_dfc.PhaseCurrents.W = in->ic;
        u_dfc.LoopOption      = DFC_LOOP_CLOSEDLOOP;  /* v4.3.0: MUST be set —
                                 u_dfc is a stack local; an unset field is
                                 garbage.  Overridden per block config, see
                                 __init__(loop_option=...).                  */

        (void)DFC_Step(&dfc_state, &u_dfc, dt, &y_dfc_out);

        y_dfc[0] = y_dfc_out.Ta;                    /* duty A               */
        y_dfc[1] = y_dfc_out.Tb;                    /* duty B               */
        y_dfc[2] = y_dfc_out.Tc;                    /* duty C               */
        y_dfc[3] = y_dfc_out.AngularVelocity;       /* speed est [rad/s]    */
        y_dfc[4] = y_dfc_out.RotorPosition;         /* position  [rad]      */
        y_dfc[5] = y_dfc_out.PhaseCurrents.U;       /* ia echo   [A]        */
        y_dfc[6] = y_dfc_out.PhaseCurrents.V;       /* ib echo   [A]        */
        y_dfc[7] = y_dfc_out.PhaseCurrents.W;       /* ic echo   [A] (=-U-V)*/
        y_dfc[8] = (real32_T)y_dfc_out.Mode;        /* mode 0/1/2           */"""

    # ---- Class-level constants ---------------------------------------------
    # Controller mode codes (mirror DFC_Mode_T)
    MODE_ALIGN:      int = 0
    MODE_OPENLOOP:   int = 1
    MODE_CLOSEDLOOP: int = 2
    MODE_NAMES = {0: "ALIGN", 1: "OPENLOOP", 2: "CLOSEDLOOP"}

    # Loop option codes (mirror DFC_LoopOption_T, v4.3.0)
    #   Option A — "open"  : ALIGN + I-f ramp, hold at handover speed (~477 RPM);
    #              SMO observable, never in control.  Commissioning path.
    #              NOTE: I-f pull-out limit KT*I_BOOST = 12.6 mN.m.
    #   Option B — "closed": full ALIGN -> I-f -> CLOSEDLOOP handover.
    LOOP_OPTIONS = {"open": 0, "closed": 1}

    # Diagnostic log decimation: log every N steps (20 -> 1 kHz at 20 kHz ISR).
    # Set DFC_DBG=1 in the environment for per-step logging.
    DIAG_DECIM: int = 1 if os.environ.get("DFC_DBG") == "1" else 20

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------
    def __init__(
        self,
        name: str  = "dfc",
        dt_s: float = 50e-6,        # [s]  nominal ISR period (20 kHz)
        # ---- Runtime gain overrides (None -> compile-time C defaults) ------
        kp_speed: Optional[float] = None,   # [A/(rad/s)]  C: DFC_KP_SPEED = 0.10
        kp_id:    Optional[float] = None,   # [V/A]        C: DFC_KP_ID    = 0.15
        kp_iq:    Optional[float] = None,   # [V/A]        C: DFC_KP_IQ    = 2.5
        ki_id:    Optional[float] = None,   # [V/(A*s)]    C: DFC_KI_ID    = 0.045
        ref_wn:   Optional[float] = None,   # [rad/s]      C: DFC_REF_WN   = 40.0
        ref_zeta: Optional[float] = None,   # [-]          C: DFC_REF_ZETA = 1.0
        loop_option: str = "closed",        # v4.3.0: "closed" (Option B, default)
                                            #         "open"   (Option A, I-f hold)
        dtype = None,
    ) -> None:
        # C backend only — the dispatcher in VectorBlock.compute() will call
        # compute_c(); compute_py() delegates there as well.
        super().__init__(name, use_c_backend=True, dtype=dtype)

        self.dt_s = float(dt_s)     # [s]

        # ---- Loop option (v4.3.0) -------------------------------------------
        if loop_option not in self.LOOP_OPTIONS:
            raise ValueError(
                f"loop_option must be one of {sorted(self.LOOP_OPTIONS)}, "
                f"got {loop_option!r}"
            )
        self.loop_option: str = loop_option
        self._loop_option_code: int = self.LOOP_OPTIONS[loop_option]

        # Option A: override the codegen emit so generated C matches the
        # configured SiL behaviour (instance attribute shadows the class one).
        if self._loop_option_code == 0:
            self.C_CUSTOM_EMIT = self.C_CUSTOM_EMIT.replace(
                "DFC_LOOP_CLOSEDLOOP", "DFC_LOOP_OPENLOOP")

        # ---- VectorBlock metadata ------------------------------------------
        self.vector_size  = 9
        self.output_label = "[ta,tb,tc,w_m,th_m,ia,ib,ic,mode]"
        self.is_dynamic   = False

        # ---- C backend wrapper ---------------------------------------------
        self._wrapper = self._load_wrapper()

        # ---- Loop option -> wrapper (v4.3.0) --------------------------------
        # FAIL LOUDLY if the compiled extension predates the LoopOption field:
        # an old wrapper leaves DFC_Input_T.LoopOption = 0 (Option A) no matter
        # what was requested — the silent open-loop failure this guard exists
        # to prevent.
        set_loop = getattr(self._wrapper, "set_loop_option", None)
        if set_loop is None:
            raise RuntimeError(
                "dfc_controller_wrapper predates DFC v4.3.0 (no "
                "set_loop_option): the LoopOption field cannot be driven and "
                "the controller would silently hold Option A (I-f open loop).\n"
                "Rebuild the extension against the v4.3.0 sources:\n"
                "  cd pmsm/c_src && python setup_dfc_controller.py "
                "build_ext --inplace"
            )
        set_loop(self._loop_option_code)

        # ---- Gains: fill unspecified entries from the C defaults -----------
        defaults = self._wrapper.get_default_gains()
        self.kp_speed = float(kp_speed) if kp_speed is not None else defaults[0]
        self.kp_id    = float(kp_id)    if kp_id    is not None else defaults[1]
        self.kp_iq    = float(kp_iq)    if kp_iq    is not None else defaults[2]
        self.ki_id    = float(ki_id)    if ki_id    is not None else defaults[3]
        self.ref_wn   = float(ref_wn)   if ref_wn   is not None else defaults[4]
        self.ref_zeta = float(ref_zeta) if ref_zeta is not None else defaults[5]

        if any(v is not None for v in
               (kp_speed, kp_id, kp_iq, ki_id, ref_wn, ref_zeta)):
            self._wrapper.apply_gains(self.kp_speed, self.kp_id, self.kp_iq,
                                      self.ki_id, self.ref_wn, self.ref_zeta)

        # ---- Diagnostic log (from DFC_GetDiagnostics(), decimated) ---------
        self.log_data: dict = self._empty_log()
        self._step_count: int = 0

        # ---- Startup diagnostics -------------------------------------------
        print(f"[DFC] Differential Flatness Controller '{name}' initialised "
              f"(C backend, sensorless v4, dt={dt_s*1e6:.0f} us)")
        print(f"[DFC]   Loop  : Option {'B — CLOSED (SMO handover enabled)' if self._loop_option_code == 1 else 'A — OPEN (I-f hold, no handover)'}")
        print(f"[DFC]   Gains : Kp_speed={self.kp_speed:.3f} A/(rad/s), "
              f"Kp_id={self.kp_id:.3f} V/A, Ki_id={self.ki_id:.4f} V/(A*s), "
              f"Kp_iq={self.kp_iq:.2f} V/A")
        print(f"[DFC]   Shaper: wn={self.ref_wn:.1f} rad/s, "
              f"zeta={self.ref_zeta:.2f}")
        print(f"[DFC]   Output: [ta, tb, tc] duty cycles "
              f"(SVPWM integrated in DFC_Step)")

    # -----------------------------------------------------------------------
    @staticmethod
    def _empty_log() -> dict:
        return {
            "t":           [],   # [s]           simulation time
            "speed_ref":   [],   # [RPM]         commanded speed
            "omega_smo":   [],   # [rad/s mech]  SMO speed estimate
            "omega_ref_f": [],   # [rad/s mech]  shaped speed reference
            "iq_ref":      [],   # [A]           q-axis current reference
            "id":          [],   # [A]           measured d-axis current
            "iq":          [],   # [A]           measured q-axis current
            "vd":          [],   # [V]           d-axis voltage command
            "vq":          [],   # [V]           q-axis voltage command
            "mode":        [],   # [-]           0 ALIGN / 1 OPENLOOP / 2 CLOSEDLOOP
        }

    # -----------------------------------------------------------------------
    # _load_wrapper  —  C backend initialisation
    # -----------------------------------------------------------------------
    def _load_wrapper(self):
        """
        Load the Cython extension wrapping the C DFC controller.

        Raises
        ------
        ImportError
            If the .so / .pyd extension has not been built.
        RuntimeError
            If the wrapper object cannot be instantiated.
        """
        try:
            from dfc_controller_wrapper import DFCControllerWrapper
        except ImportError as exc:
            raise ImportError(
                "dfc_controller_wrapper not found. Build with:\n"
                "  cd pmsm/c_src\n"
                "  build_dfc_controller.bat        (Windows)\n"
                "  ./build_dfc_controller.sh       (Linux)\n"
                "or:\n"
                "  python setup_dfc_controller.py build_ext --inplace"
            ) from exc

        try:
            return DFCControllerWrapper(self.dt_s)
        except Exception as exc:
            raise RuntimeError(
                f"DFCControllerWrapper instantiation failed: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # compute_c  —  the one and only implementation
    # -----------------------------------------------------------------------
    def compute_c(
        self,
        t:            float,
        dt:           float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """
        Execute one DFC step via the compiled DFC_Step().

        All state lives inside the C DFC_State_T struct managed by the
        wrapper.

        Returns
        -------
        VectorSignal
            value = [ta, tb, tc, omega_mech, theta_mech, ia, ib, ic, mode]
            (duties [0..1], speed [rad/s], position [rad], phase currents [A],
             mode 0/1/2) — the full DFC_Output_T.
        """
        safe = np.zeros(9, dtype=np.float32)   # duties default to SVPWM midpoint
        safe[0] = safe[1] = safe[2] = 0.5

        if not input_values or input_values[0] is None:
            self.output = VectorSignal(safe.copy(), self.name)
            return self.output

        u = input_values[0].value
        if len(u) < 4:
            self.output = VectorSignal(safe.copy(), self.name)
            return self.output

        ia_in, ib_in = float(u[1]), float(u[2])

        # Pack input bus — DFC_Input_T signal order
        self._wrapper.set_inputs_individual(
            float(u[0]),   # speed_ref_rpm [RPM]
            ia_in,         # ia            [A]
            ib_in,         # ib            [A]
            float(u[3]),   # ic            [A]
        )
        self._wrapper.compute(float(dt))

        # Decimated diagnostic log
        if (self._step_count % self.DIAG_DECIM) == 0:
            d = self._wrapper.get_diagnostics()
            ld = self.log_data
            ld["t"].append(float(t))
            ld["speed_ref"].append(float(d[4]))
            ld["omega_smo"].append(float(d[0]))
            ld["omega_ref_f"].append(float(d[6]))
            ld["iq_ref"].append(float(d[1]))
            ld["iq"].append(float(d[2]))
            ld["id"].append(float(d[3]))
            ld["vd"].append(float(d[7]))
            ld["vq"].append(float(d[8]))
            ld["mode"].append(float(d[9]))
        self._step_count += 1

        # Full DFC_Output_T bus.  Phase-current echo mirrors the C:
        # U,V pass through, W is the star-point reconstruction -U-V.
        self.output = VectorSignal(
            np.array([
                self._wrapper.ta,               # [0] duty A
                self._wrapper.tb,               # [1] duty B
                self._wrapper.tc,               # [2] duty C
                self._wrapper.speed_est,        # [3] omega_mech [rad/s]
                self._wrapper.rotor_position,   # [4] theta_mech [rad]
                ia_in,                          # [5] ia echo [A]
                ib_in,                          # [6] ib echo [A]
                -(ia_in + ib_in),               # [7] ic echo [A] (W = -U-V)
                float(self._wrapper.mode),      # [8] mode 0/1/2
            ], dtype=DEFAULT_DTYPE),
            self.name)
        return self.output

    # -----------------------------------------------------------------------
    # compute_py  —  delegates to compute_c (no Python dual implementation)
    # -----------------------------------------------------------------------
    def compute_py(
        self,
        t:            float,
        dt:           float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """C-only block: the Python entry point delegates to compute_c()."""
        return self.compute_c(t, dt, input_values)

    # -----------------------------------------------------------------------
    # apply_gains  —  runtime retuning (C: DFC_GainSet_Apply())
    # -----------------------------------------------------------------------
    def apply_gains(
        self,
        kp_speed: Optional[float] = None,
        kp_id:    Optional[float] = None,
        kp_iq:    Optional[float] = None,
        ki_id:    Optional[float] = None,
        ref_wn:   Optional[float] = None,
        ref_zeta: Optional[float] = None,
    ) -> None:
        """
        Retune at runtime without a rebuild — e.g. from a tuner loop.
        Unspecified arguments keep their current values.
        """
        if kp_speed is not None: self.kp_speed = float(kp_speed)
        if kp_id    is not None: self.kp_id    = float(kp_id)
        if kp_iq    is not None: self.kp_iq    = float(kp_iq)
        if ki_id    is not None: self.ki_id    = float(ki_id)
        if ref_wn   is not None: self.ref_wn   = float(ref_wn)
        if ref_zeta is not None: self.ref_zeta = float(ref_zeta)

        self._wrapper.apply_gains(self.kp_speed, self.kp_id, self.kp_iq,
                                  self.ki_id, self.ref_wn, self.ref_zeta)

    # -----------------------------------------------------------------------
    # reset  —  C: DFC_Reset()
    # -----------------------------------------------------------------------
    def reset(self) -> None:
        """
        Reset all controller state.  Mirrors DFC_Reset() -> DFC_Init() in C:
        a single canonical zeroing path.  Applied gains are re-applied after
        the reset (DFC_Init restores the compile-time defaults).
        """
        super().reset()
        self._wrapper.reset()
        self._wrapper.apply_gains(self.kp_speed, self.kp_id, self.kp_iq,
                                  self.ki_id, self.ref_wn, self.ref_zeta)
        self.log_data = self._empty_log()
        self._step_count = 0

    # -----------------------------------------------------------------------
    # get_diagnostics  —  C: DFC_GetDiagnostics()
    # -----------------------------------------------------------------------
    def get_diagnostics(self) -> dict:
        """
        Return the current diagnostic snapshot from the C state.

        Returns
        -------
        dict
            speed_est_rad_s : SMO mechanical speed estimate [rad/s]
            omega_ref_f     : shaped speed reference [rad/s mech]
            iq_ref          : q-axis current reference [A]
            id_meas, iq_meas: measured dq currents [A]
            speed_ref_rpm   : commanded speed [RPM]
            vd, vq          : dq voltage commands [V]
            mode            : "ALIGN" / "OPENLOOP" / "CLOSEDLOOP"
        """
        d = self._wrapper.get_diagnostics()
        return {
            "speed_est_rad_s": float(d[0]),
            "omega_ref_f":     float(d[6]),
            "iq_ref":          float(d[1]),
            "iq_meas":         float(d[2]),
            "id_meas":         float(d[3]),
            "speed_ref_rpm":   float(d[4]),
            "vd":              float(d[7]),
            "vq":              float(d[8]),
            "mode":            self.MODE_NAMES.get(int(d[9]), "?"),
        }

    # -----------------------------------------------------------------------
    # Diagnostic properties (thin views onto the C state)
    # -----------------------------------------------------------------------

    @property
    def mode(self) -> int:
        """Controller mode: 0 ALIGN / 1 OPENLOOP / 2 CLOSEDLOOP."""
        return int(self._wrapper.mode)

    @property
    def speed_est(self) -> float:
        """SMO mechanical speed estimate [rad/s]."""
        return float(self._wrapper.speed_est)

    @property
    def rotor_position(self) -> float:
        """Integrated mechanical rotor angle [rad, 0 - 2pi)."""
        return float(self._wrapper.rotor_position)

    @property
    def smo_theta_e(self) -> float:
        """SMO estimated electrical angle [rad]."""
        return float(self._wrapper.get_theta_e())

    @property
    def smo_omega_e(self) -> float:
        """SMO electrical speed estimate [rad/s elec]."""
        return float(self._wrapper.omega_e)

    @property
    def iq_ref(self) -> float:
        """q-axis current reference [A]."""
        return float(self._wrapper.iq_ref)

    def __repr__(self) -> str:
        return (
            f"DFControllerBlock('{self.name}', backend=C, "
            f"mode={self.MODE_NAMES.get(self.mode, '?')}, "
            f"Kp_speed={self.kp_speed} A/(rad/s), "
            f"Kp_id={self.kp_id} V/A, Ki_id={self.ki_id:.4f} V/(A*s), "
            f"Kp_iq={self.kp_iq} V/A)"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = ["DFControllerBlock"]
