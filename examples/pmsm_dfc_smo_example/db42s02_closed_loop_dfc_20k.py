"""
db42s02_closed_loop_dfc_20k.py  —  pmsm/
=========================================
Closed-loop sensorless DFC (Differential Flatness Controller) simulation of
the NANOTEC DB42S02 PMSM at a 20 kHz ISR rate — C backend only.

WHAT CHANGED vs THE fs_electrical_machines VERSION
==================================================
The v4 C controller (pmsm/c_src/embed_sim_dfc_controller.c) is fully
sensorless and has SVPWM integrated into DFC_Step():

    * NO encoder input      — theta_m is gone from the bus; the rotor angle
                              comes from the internal SMO after an
                              ALIGN -> I-f open-loop -> closed-loop startup.
    * NO external SVPWM     — DFC_Step() emits Ta/Tb/Tc duty cycles directly
                              (SVPWMPackBlock / SVPWMBlock removed).
    * NO Python controller  — the DFControllerBlock runs the compiled AURIX
                              C code through the Cython wrapper.  What is
                              simulated is what flashes onto the TC38x.

SIGNAL CHAIN
============
    speed_ref [RPM] ──┐
                      ├─> dfc_packer ─> (cg_start) ─> dfc ─> (cg_end)
    motor_delay ──────┘    [rpm,ia,ib,ic]                 [ta,tb,tc]
         ^                                                     │
         │                                              load_adapter
         │                                          [ta,tb,tc,Vdc,Tload]
         │                                                     │
         └───────────────────────  motor  <───────────────────┘
                     [rpm,ia,ib,ic,theta_m,Tem,id,iq]

    motor        : PMSM_Plant_FMUBlock (Modelica FMU, pmsm/modelica/PMSM_Plant_FMU.fmu)
    dfc_packer   : merges the RPM command with the (optionally noisy)
                   measured phase currents into the DFC_Input_T bus order
    load_adapter : appends the DC-bus voltage (with optional 100 Hz ripple)
                   and the timed load-torque schedule to the duty bus
    motor_delay  : z^-1 loop breaker on the plant output

NOISE MODEL (--noise, default on)
=================================
    AURIX EVADC 12-bit path on the phase currents:
        Gaussian thermal noise -> LSB quantisation -> rail clamp
        + sporadic PWM coupling spikes
    DC bus: 100 Hz rectifier ripple.
    (The encoder noise chain of the old example is gone — sensorless.)

LOAD SCHEDULE
=============
    t <  0.5 s   : 0 N.m       (startup: align + I-f ramp + handover)
    0.5 - 1.6 s  : 5 mN.m      (light load)
    t >= 1.6 s   : 20 mN.m     (heavy load)

CONSOLE MONITOR (v4.2.0)
========================
    While the simulation runs, a [mon] line is printed every 0.25 s of
    simulated time with the controller mode, plant speed, SMO estimate,
    iq_ref and the load-torque observer estimate (if the wrapper exposes
    t_load_est).  Mode transitions and load-schedule steps are printed the
    step they occur.  A WARNING is printed if the C wrapper ever reports a
    nonzero status (DFC_Step forces safe 0.5 duties on such steps — without
    the check those steps would fail silently).

USAGE
=====
    python db42s02_closed_loop_dfc_20k.py                 # 5 s, 2000 RPM, noise
    python db42s02_closed_loop_dfc_20k.py --rpm 1500 --t 3
    python db42s02_closed_loop_dfc_20k.py --no-noise
    python db42s02_closed_loop_dfc_20k.py --codegen       # emit CodeGen stubs

Prerequisite: build the Cython extension once —
    cd pmsm/c_src && build_dfc_controller.bat   (Windows)
    cd pmsm/c_src && ./build_dfc_controller.sh  (Linux)
"""

from __future__ import annotations

import sys
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE = get_current_parent()          # this example's folder (examples/pmsm_dfc_smo_example/)
_ROOT = get_project_root()            # project root (holds .project_root_marker)
_PMSM = _ROOT / "pmsm"                # the DFC package: plant, controller block, c_src/

# sys.path: project root (embedsim package), the pmsm/ package, pmsm/c_src (the
# compiled dfc_controller_wrapper).  Everything this example needs lives under
# pmsm/ — the plant, the controller block, and the C transforms surfaced by the
# wrapper — so there is no dependency on fs_electrical_machines/.
for _p in (get_embedsim_import_path(),
           str(_PMSM),
           str(_PMSM / "c_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep
from embedsim.simulation_engine import VectorDelay
from embedsim.code_generator import CodeGenStart, CodeGenEnd

from PMSM_Plant_FMUBlock import PMSM_Plant_FMUBlock
from diff_flatness_controller_block import DFControllerBlock


# =============================================================================
# Simulation constants — DB42S02, matching the C #defines exactly
# =============================================================================

P_POLES     = 4          # [-]    C: DFC_P_POLES
R_S         = 0.19       # [Ohm]  C: DFC_R_S
L_D         = 0.125e-3   # [H]    C: DFC_L_D
L_Q         = 0.125e-3   # [H]    C: DFC_L_Q
LAMBDA_PM   = 0.0014     # [Wb]   C: DFC_LAMBDA_PM
J_ROTOR     = 2.4e-6     # [kg.m2] C: DFC_J_ROTOR
B_FRIC      = 1.0e-6     # [N.m.s/rad] C: DFC_B_FRIC
I_MAX       = 3.57       # [A]    C: DFC_I_MAX
V_DC        = 17.0       # [V]    C: DFC_V_DC

TARGET_RPM  = 2000.0
T_SIM       = 5.0
DT          = 50e-6      # 20 kHz ISR period
ENABLE_NOISE = True      # AURIX ADC / bus-ripple noise chain on the current path

T_LOAD_T1   = 0.5        # [s]   light-load step (after startup completes)
T_LOAD_T2   = 1.6        # [s]   heavy-load step (v4.2.0: was 1.2 — gives the
                         #       light-load plateau a real 0.5+ s window and
                         #       clears the load-observer hold-off, 0.5 s
                         #       after the ~0.8 s closed-loop handover)
T_LOAD_ZERO  = 0.000     # [N.m]
T_LOAD_LIGHT = 0.005     # [N.m]
T_LOAD_HEAVY = 0.020     # [N.m]

_MOTOR_OUT_SIZE = 8      # [rpm, ia, ib, ic, theta_m, Tem, id, iq]

# Modelica FMU plant (pmsm/modelica/).  DEFAULT_PARAMS inside the wrapper
# (R=0.19, L=125 uH, lambda_pm=1.4 mWb, J=2.4e-6, B=1e-6, p=4) already match
# the C #defines above, so no parameter override is needed.
FMU_PATH = _PMSM / "modelica" / "PMSM_Plant_FMU.fmu"


# =============================================================================
# Active gains — single source of truth.  None -> compile-time C defaults
# from embed_sim_dfc_gains.h (DFC_KP_SPEED = 0.10, DFC_KP_ID = 0.15,
# DFC_KP_IQ = 2.5, DFC_KI_ID = 0.045, DFC_REF_WN = 40, DFC_REF_ZETA = 1.0).
# =============================================================================

_ACTIVE_GAINS: dict = {
    "kp_speed": None,    # [A/(rad/s)]
    "kp_id":    None,    # [V/A]
    "kp_iq":    None,    # [V/A]
    "ki_id":    None,    # [V/(A*s)]
    "ref_wn":   None,    # [rad/s]
    "ref_zeta": None,    # [-]
}


# =============================================================================
# AURIX hardware-specific noise parameters (current path + DC bus only —
# the controller is sensorless, so the encoder noise chain is gone)
# =============================================================================

_ADC_BITS        = 12
_ADC_I_FS        = I_MAX * 2.0
_ADC_LSB_A       = _ADC_I_FS / (2 ** _ADC_BITS)   # ~1.74 mA per LSB
_ADC_NOISE_SIGMA = 1.5 * _ADC_LSB_A
_ADC_SAT_LIMIT   = I_MAX * 1.26                    # ~4.5 A rail clamp

_SPIKE_PROB      = 0.05
_SPIKE_AMP       = 0.50                            # A

_BUS_RIPPLE_AMP  = 0.50                            # V
_BUS_RIPPLE_HZ   = 100.0                           # Hz

_RNG             = np.random.default_rng(seed=20240101)


def _adc_noise(current_A: float) -> float:
    """AURIX EVADC 12-bit: Gaussian thermal noise -> quantise -> rail clamp."""
    noisy = current_A + _RNG.normal(0.0, _ADC_NOISE_SIGMA)
    quant = round(noisy / _ADC_LSB_A) * _ADC_LSB_A
    return float(max(-_ADC_SAT_LIMIT, min(_ADC_SAT_LIMIT, quant)))


def _pwm_spike(ia: float) -> float:
    """Sporadic PWM-coupling spike on the current measurement."""
    if _RNG.random() < _SPIKE_PROB:
        return ia + float(_RNG.choice([-1.0, 1.0])) * _SPIKE_AMP * _RNG.random()
    return ia


def _vdc_ripple(t: float) -> float:
    """DC bus with 100 Hz rectifier ripple."""
    return V_DC + _BUS_RIPPLE_AMP * math.sin(2.0 * math.pi * _BUS_RIPPLE_HZ * t)


# =============================================================================
# Console monitor — prints between simulation steps (v4.2.0)
# =============================================================================

_RADPS_TO_RPM_M = 60.0 / (2.0 * math.pi)
_MODE_NAMES_M   = {0: "ALIGN", 1: "OPENLOOP", 2: "CLOSEDLOOP"}


class ConsoleMonitor:
    """
    Lightweight run-time console reporter.  Driven from DfcBusPacker (which
    executes once per ISR tick and sees the plant bus), reads controller
    diagnostics defensively from the DFControllerBlock so it degrades
    gracefully if a wrapper field is absent.
    """

    def __init__(self, dfc, period_s: float = 0.25) -> None:
        self._dfc      = dfc
        self._period   = float(period_s)
        self._next_t   = 0.0
        self._mode     = -1
        self._load     = None
        self._warned   = False

    def _diag(self, key, default=0.0):
        try:
            ld = self._dfc.log_data
            if ld and len(ld.get(key, ())) > 0:
                return float(ld[key][-1])
        except Exception:
            pass
        return default

    def _tload_mnm(self):
        # t_load_est requires the v4.4 wrapper property; fall back silently.
        v = getattr(self._dfc, "t_load_est", None)
        try:
            return float(v) * 1e3 if v is not None else None
        except Exception:
            return None

    def tick(self, t: float, rpm_plant: float) -> None:
        # wrapper status guard — a silent nonzero status means safe-duty steps
        st = getattr(self._dfc, "status", 0) or getattr(
            getattr(self._dfc, "_wrapper", None), "status", 0)
        if st and not self._warned:
            print(f"[WARN] t={t:.4f}s  DFC wrapper status={st} — "
                  f"safe 0.5 duties forced; check inputs/build")
            self._warned = True

        mode = int(self._diag("mode", -1.0))
        if mode != self._mode:
            print(f"[mode] t={t:8.4f}s  {_MODE_NAMES_M.get(self._mode, '--'):>10} "
                  f"-> {_MODE_NAMES_M.get(mode, '?')}")
            self._mode = mode

        load = LoadAdapterBlock._load_schedule(t)
        if load != self._load:
            if self._load is not None:
                print(f"[load] t={t:8.4f}s  T_load -> {load*1e3:.0f} mN.m")
            self._load = load

        if t >= self._next_t:
            self._next_t += self._period
            smo   = self._diag("omega_smo") * _RADPS_TO_RPM_M
            iqr   = self._diag("iq_ref")
            tl    = self._tload_mnm()
            tl_s  = f"  T^={tl:5.1f} mN.m" if tl is not None else ""
            print(f"[mon]  t={t:8.4f}s  {_MODE_NAMES_M.get(mode, '?'):<10} "
                  f"plant={rpm_plant:7.1f} RPM  smo={smo:7.1f} RPM  "
                  f"iq_ref={iqr:+5.2f} A{tl_s}")


# =============================================================================
# DfcBusPacker — merges the RPM command with the measured phase currents
# into the DFC_Input_T bus order: [speed_ref_rpm, ia, ib, ic]
# =============================================================================

class DfcBusPacker(VectorBlock):
    """
    Two-port packer feeding DFControllerBlock.

    Inputs (any connection order — identified by width):
        1-element bus : speed reference [RPM]           (from VectorStep)
        8-element bus : delayed plant output            (from VectorDelay)
                        [rpm, ia, ib, ic, theta_m, Tem, id, iq]

    Output (4 elements — DFC_Input_T signal order):
        [speed_ref_rpm, ia, ib, ic]

    Owns the ADC noise model on the current path when with_noise=True.
    """

    TOPO_CATEGORY     = "utility"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[rpm_ref,ia,ib,ic]"

    # Names/comments for the generated EmbedSimDfc_Input_T.  This block feeds
    # cg_start, so the StepGenerator reads these to emit named, commented scalar
    # fields (speed_ref_rpm, ia, ib, ic) instead of an opaque dfc_packer[4].
    INPUT_NAMES = ["speed_ref_rpm", "ia", "ib", "ic"]
    INPUT_KEEP  = [0, 1, 2, 3]
    C_FIELD_COMMENTS = {
        "speed_ref_rpm": "IN : Mechanical speed reference [RPM]; range [0, 3000]",
        "ia":            "IN : Phase-A current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
        "ib":            "IN : Phase-B current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
        "ic":            "IN : Phase-C current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
    }

    def __init__(self, name: str = "dfc_packer",
                 with_noise: bool = True,
                 monitor: "ConsoleMonitor | None" = None) -> None:
        super().__init__(name)
        self.with_noise  = bool(with_noise)
        self.vector_size = 4
        self.monitor     = monitor

    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        rpm_ref = 0.0
        ia = ib = ic = 0.0

        if input_values:
            for sig in input_values:
                if sig is None:
                    continue
                v = np.atleast_1d(sig.value)
                if len(v) >= _MOTOR_OUT_SIZE:          # plant bus
                    ia, ib, ic = float(v[1]), float(v[2]), float(v[3])
                elif len(v) >= 1:                       # speed reference
                    rpm_ref = float(v[0])

        if self.with_noise:
            ia = _adc_noise(_pwm_spike(ia))
            ib = _adc_noise(_pwm_spike(ib))
            ic = _adc_noise(_pwm_spike(ic))

        if self.monitor is not None:
            rpm_plant = 0.0
            if input_values:
                for sig in input_values:
                    if sig is None:
                        continue
                    v = np.atleast_1d(sig.value)
                    if len(v) >= _MOTOR_OUT_SIZE:
                        rpm_plant = float(v[0])
            self.monitor.tick(t, rpm_plant)

        self.output = VectorSignal(
            np.array([rpm_ref, ia, ib, ic], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# LoadAdapterBlock — duty bus -> plant input bus
# =============================================================================

class LoadAdapterBlock(VectorBlock):
    """
    Sits between DFControllerBlock and the PMSM plant (FMU).

    Input  : [ta, tb, tc]                        (from dfc / cg_end)
    Output : [ta, tb, tc, v_dc, T_load]          (plant input bus)

    Owns the timed load-torque schedule and the DC-bus ripple model.
    """

    TOPO_CATEGORY     = "utility"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[ta,tb,tc,Vdc,Tload]"

    def __init__(self, name: str = "load_adapter",
                 with_noise: bool = True) -> None:
        super().__init__(name)
        self.with_noise  = bool(with_noise)
        self.vector_size = 5
        self.t_load      = 0.0

    @staticmethod
    def _load_schedule(t: float) -> float:
        if t >= T_LOAD_T2:
            return T_LOAD_HEAVY
        if t >= T_LOAD_T1:
            return T_LOAD_LIGHT
        return T_LOAD_ZERO

    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        ta = tb = tc = 0.5
        if input_values and input_values[0] is not None:
            v = np.atleast_1d(input_values[0].value)
            if len(v) >= 3:
                ta, tb, tc = float(v[0]), float(v[1]), float(v[2])

        self.t_load = self._load_schedule(t)
        v_dc = _vdc_ripple(t) if self.with_noise else V_DC

        self.output = VectorSignal(
            np.array([ta, tb, tc, v_dc, self.t_load], dtype=DEFAULT_DTYPE),
            self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# Wire labels (topology printer)
# =============================================================================

_WIRE_LABELS = {
    ("speed_ref",    "dfc_packer"):   "rpm_ref [RPM]",
    ("motor_delay",  "dfc_packer"):   "[rpm,ia,ib,ic,th_m,Tem,id,iq] z-1",
    ("dfc_packer",   "cg_start"):     "[rpm_ref,ia,ib,ic]",
    ("dfc_packer",   "dfc"):          "[rpm_ref,ia,ib,ic]",
    ("cg_start",     "dfc"):          "[rpm_ref,ia,ib,ic]",
    ("dfc",          "cg_end"):       "[ta,tb,tc]",
    ("dfc",          "load_adapter"): "[ta,tb,tc]",
    ("cg_end",       "load_adapter"): "[ta,tb,tc]",
    ("load_adapter", "motor"):        "[ta,tb,tc,Vdc,Tload]",
    ("motor",        "motor_delay"):  "[rpm,ia,ib,ic,th_m,Tem,id,iq]",
    ("motor",        "sink"):         "[rpm,ia,ib,ic,th_m,Tem,id,iq]",
}


# =============================================================================
# Core simulation runner
# =============================================================================

def _run_sim(*,
             target_rpm:         float = TARGET_RPM,
             t_sim:              float = T_SIM,
             with_noise:         bool  = True,
             with_codegen_hooks: bool  = False) -> dict | None:
    """
    Build and run one closed-loop sensorless DFC simulation.

    Returns a result dict (signals interpolated onto the scope timebase)
    or None on failure.
    """
    try:
        # ---- Controller (C backend only) --------------------------------
        dfc = DFControllerBlock(
            "dfc",
            dt_s     = DT,
            kp_speed = _ACTIVE_GAINS["kp_speed"],
            kp_id    = _ACTIVE_GAINS["kp_id"],
            kp_iq    = _ACTIVE_GAINS["kp_iq"],
            ki_id    = _ACTIVE_GAINS["ki_id"],
            ref_wn   = _ACTIVE_GAINS["ref_wn"],
            ref_zeta = _ACTIVE_GAINS["ref_zeta"],
        )

        # ---- Plant (Modelica FMU: PMSM_Plant_FMU) ------------------------
        # Inputs  [duty_a, duty_b, duty_c, v_dc, T_load]  <- load_adapter bus
        # Outputs [rpm, ia, ib, ic, theta_m, T_em, id, iq] — same order as
        # the old Python plant, so packer / delay / scope stay unchanged.
        if not FMU_PATH.is_file():
            raise FileNotFoundError(
                f"FMU not found: {FMU_PATH}\n"
                f"Regenerate it with pmsm/modelica/gen_fmu.py")
        motor = PMSM_Plant_FMUBlock("motor", fmu_path=str(FMU_PATH))

        # ---- Signal chain blocks -----------------------------------------
        speed_ref    = VectorStep("speed_ref", step_time=0.0,
                                  before_value=target_rpm,
                                  after_value=target_rpm, dim=1)
        motor_delay  = VectorDelay("motor_delay",
                                   initial=[0.0] * _MOTOR_OUT_SIZE)
        monitor      = ConsoleMonitor(dfc, period_s=0.25)
        dfc_packer   = DfcBusPacker("dfc_packer", with_noise=with_noise,
                                    monitor=monitor)
        load_adapter = LoadAdapterBlock("load_adapter", with_noise=with_noise)
        sink         = VectorEnd("sink")

        # ---- CodeGen boundary (optional): cg_start >> dfc >> cg_end -----
        if with_codegen_hooks:
            cg_start = CodeGenStart("cg_start")
            cg_end   = CodeGenEnd("cg_end")
            dfc_packer >> cg_start >> dfc >> cg_end >> load_adapter
        else:
            cg_start = cg_end = None
            dfc_packer >> dfc >> load_adapter

        # ---- Wiring common to both modes ---------------------------------
        speed_ref    >> dfc_packer
        motor_delay  >> dfc_packer
        load_adapter >> motor
        motor        >> motor_delay
        motor        >> sink

        # ---- Scope --------------------------------------------------------
        sim = EmbedSim(sinks=[sink], T=t_sim, dt=DT, solver=ODESolver.EULER)
        sim.scope.add(dfc,   indices=[0, 1, 2],
                      label="Duties")
        sim.scope.add(motor, indices=[0, 1, 2, 3, 5, 6, 7],
                      label="Motor")

        sim.run()

    except Exception as exc:
        import traceback
        print(f"  [sim error] {exc}")
        traceback.print_exc()
        return None

    sc = sim.scope
    t  = np.array(sc.t, dtype=np.float32)
    if len(t) < 100:
        return None

    def _s(label, pos):
        sig = sc.get_signal(label, pos)
        return sig if sig is not None else np.zeros(len(t), np.float32)

    # DFC diagnostic log (decimated inside the block) -> scope timebase
    ld = dfc.log_data
    _log_ok = len(ld["t"]) > 1

    def _i(key):
        if _log_ok:
            return np.interp(t, ld["t"], ld[key]).astype(np.float32)
        return np.zeros(len(t), np.float32)

    _RADPS_TO_RPM = 60.0 / (2.0 * math.pi)

    return {
        "t":             t,
        # Plant truth — Motor bus: [rpm, ia, ib, ic, Tem, id, iq]
        "speed_rpm":     _s("Motor", 0),
        "ia":            _s("Motor", 1),
        "ib":            _s("Motor", 2),
        "ic":            _s("Motor", 3),
        "torque":        _s("Motor", 4),
        "id":            _s("Motor", 5),
        "iq":            _s("Motor", 6),
        # Duty cycles out of DFC_Step (SVPWM integrated)
        "ta":            _s("Duties", 0),
        "tb":            _s("Duties", 1),
        "tc":            _s("Duties", 2),
        # Controller diagnostics (C state via DFC_GetDiagnostics)
        "omega_ref_rpm": np.full(len(t), target_rpm, dtype=np.float32),
        "omega_ref_f":   _i("omega_ref_f") * _RADPS_TO_RPM,   # shaped ref [RPM]
        "rpm_smo":       _i("omega_smo") * _RADPS_TO_RPM,     # SMO estimate [RPM]
        "iq_ref":        _i("iq_ref"),
        "id_ctrl":       _i("id"),
        "iq_ctrl":       _i("iq"),
        "vd":            _i("vd"),
        "vq":            _i("vq"),
        "mode":          _i("mode"),
        "_cg_start":     cg_start,
        "_cg_end":       cg_end,
        "_sim":          sim,
        "_dfc":          dfc,
    }


# =============================================================================
# CodeGen (optional, --codegen)
# =============================================================================

def _run_codegen(d: dict) -> None:
    """
    Emit the AURIX-ready step function for the DFC region.

    Produces embedsim_gen/EmbedSimDfc_step.{h,c} at the project root:
      * EmbedSimDfc_Input_T   — [speed_ref_rpm, ia, ib, ic]  (dfc_packer bus)
      * EmbedSimDfc_Output_T  — [ta, tb, tc]                 (duty cycles)
      * EmbedSimDfc_Init()    — calls DFC_Init()
      * EmbedSimDfc_Step()    — calls DFC_Step() (Clarke->SMO->SM->law->SVPWM)

    Compile alongside embed_sim_dfc_controller.c, embed_sim_coordinate_transform.c,
    embed_sim_sv_pwm.c and embed_sim_matrix.c.  Call Transform_Init() and
    SVM_Init() once (e.g. in your AppInit) before the first EmbedSimDfc_Step().
    """
    cg_start = d.get("_cg_start")
    cg_end   = d.get("_cg_end")
    sim      = d.get("_sim")
    if cg_start is None or cg_end is None or sim is None:
        print("[codegen] hooks not present — rerun with --codegen")
        return

    # Topology (console + interactive HTML next to this example)
    try:
        print("\n[Topology]")
        sim.topo.print_console()
        sim.topo.export_html(str(_HERE / "db42s02_dfc_topology.html"),
                             wire_labels=_WIRE_LABELS)
    except Exception as exc:
        print(f"[codegen] topology export skipped: {exc}")

    # AURIX step function -> <root>/embedsim_gen/
    try:
        print("\n[CodeGen] Generating AURIX C step function ...")
        cg_end.generate_step(
            cg_start    = cg_start,
            output_dir  = _ROOT,
            dt_hz       = 1.0 / DT,
            prefix      = "EmbedSimDfc",
            write_files = True,
        )
        gen = _ROOT / "embedsim_gen"
        print(f"  {gen / 'embedsimdfc_step.h'}")
        print(f"  {gen / 'embedsimdfc_step.c'}")
    except Exception as exc:
        import traceback
        print(f"[codegen] generation failed: {exc}")
        traceback.print_exc()


# =============================================================================
# Plotting
# =============================================================================

def plot_results(d: dict, filename: str = "db42s02_dfc_foc_20k_results.png") -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
    fig.suptitle(
        "DB42S02 — Sensorless Differential Flatness FOC @ 20 kHz "
        "(C backend, integrated SVPWM)",
        fontsize=13, color="white")

    t = d["t"]

    def _vl(a):
        for tv, c in ((T_LOAD_T1, "#888"), (T_LOAD_T2, "#aaa")):
            a.axvline(tv, color=c, lw=0.8, ls=":", alpha=0.8)

    def _leg(a):
        a.legend(fontsize=7, facecolor="#222", labelcolor="white",
                 edgecolor="#444", loc="upper right")

    def _fmt(a, y, title):
        a.set_ylabel(y, fontsize=9)
        a.set_title(title, fontsize=9, color="#ccc", loc="left")
        a.grid(alpha=0.25)

    # 1 — speed
    ax[0].plot(t, d["omega_ref_rpm"], color="white", lw=1.0, ls="--",
               alpha=0.6, label="command")
    ax[0].plot(t, d["omega_ref_f"], color="#ffb347", lw=1.0,
               alpha=0.9, label="shaped ref (OmegaRefF)")
    ax[0].plot(t, d["speed_rpm"], color="#00e5ff", lw=1.0, label="plant")
    ax[0].plot(t, d["rpm_smo"], color="#ff6ec7", lw=0.8, alpha=0.8,
               label="SMO estimate")
    _vl(ax[0]); _leg(ax[0]); _fmt(ax[0], "RPM", "Speed — command / shaped reference / plant / SMO")

    # 2 — dq currents
    ax[1].plot(t, d["iq_ref"], color="white", lw=0.9, ls="--", alpha=0.7,
               label="iq_ref (DFC)")
    ax[1].plot(t, d["iq"], color="#7CFC00", lw=0.9, label="iq (plant)")
    ax[1].plot(t, d["id"], color="#ff8c00", lw=0.9, label="id (plant)")
    _vl(ax[1]); _leg(ax[1]); _fmt(ax[1], "A", "dq currents — reference vs plant truth")

    # 3 — phase currents
    for k, c in (("ia", "#00e5ff"), ("ib", "#ffb347"), ("ic", "#ff6ec7")):
        ax[2].plot(t, d[k], lw=0.5, color=c, alpha=0.85, label=k)
    _vl(ax[2]); _leg(ax[2]); _fmt(ax[2], "A", "Phase currents (plant)")

    # 4 — duty cycles
    for k, c in (("ta", "#00e5ff"), ("tb", "#ffb347"), ("tc", "#ff6ec7")):
        ax[3].plot(t, d[k], lw=0.5, color=c, alpha=0.85, label=k)
    _vl(ax[3]); _leg(ax[3]); _fmt(ax[3], "duty [0-1]",
                                  "SVPWM duty cycles (integrated in DFC_Step)")

    # 5 — dq voltages + mode
    ax[4].plot(t, d["vd"], color="#ff8c00", lw=0.9, label="vd")
    ax[4].plot(t, d["vq"], color="#7CFC00", lw=0.9, label="vq")
    ax4b = ax[4].twinx()
    ax4b.plot(t, d["mode"], color="#888", lw=1.0, ls=":", label="mode")
    ax4b.set_ylim(-0.2, 2.4)
    ax4b.set_yticks([0, 1, 2])
    ax4b.set_yticklabels(["ALIGN", "OL", "CL"], fontsize=7, color="#aaa")
    _vl(ax[4]); _leg(ax[4]); _fmt(ax[4], "V",
                                  "dq voltage commands + controller mode")
    ax[4].set_xlabel("t [s]", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = _HERE / filename
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[plot] saved {out}")


# =============================================================================
# Summary
# =============================================================================

def print_summary(d: dict, target_rpm: float) -> None:
    t   = d["t"]
    rpm = d["speed_rpm"]

    def _win(t0, t1):
        m = (t >= t0) & (t < t1)
        return rpm[m] if np.any(m) else np.array([0.0])

    # closed-loop entry: first sample where mode reaches 2
    cl = np.argmax(d["mode"] >= 2.0) if np.any(d["mode"] >= 2.0) else -1
    t_cl = float(t[cl]) if cl >= 0 else float("nan")

    # No-load steady-state window: from 0.3 s after closed-loop handover up to
    # the first load step.  If the light-load step lands before handover
    # settles, fall back to the light-load plateau (T_LOAD_T1..T_LOAD_T2).
    t_settle = t_cl + 0.3 if cl >= 0 else 0.0
    if t_settle < T_LOAD_T1:
        w_light_lbl = "no load"
        w_light = _win(t_settle, T_LOAD_T1)
    else:
        w_light_lbl = "light load (5 mN.m)"
        w_light = _win(max(t_settle, T_LOAD_T1), T_LOAD_T2)
    w_heavy = _win(T_LOAD_T2 + 0.5, float(t[-1]))

    print()
    print("=" * 64)
    print(" DB42S02 sensorless DFC — run summary")
    print("=" * 64)
    print(f"  target speed              : {target_rpm:8.1f} RPM")
    print(f"  closed-loop handover at   : {t_cl:8.3f} s   (align 0.30 s + I-f ramp)")
    print(f"  mean speed, {w_light_lbl:<16}: {np.mean(w_light):8.1f} RPM "
          f"(err {np.mean(w_light) - target_rpm:+.1f})")
    print(f"  mean speed, 20 mN.m load  : {np.mean(w_heavy):8.1f} RPM "
          f"(err {np.mean(w_heavy) - target_rpm:+.1f})")
    print(f"  speed ripple (heavy, 1σ)  : {np.std(w_heavy):8.2f} RPM")
    print(f"  iq peak                   : {np.max(np.abs(d['iq'])):8.3f} A "
          f"(I_MAX {I_MAX} A)")
    dfc = d.get("_dfc")
    tl  = getattr(dfc, "t_load_est", None) if dfc is not None else None
    if tl is not None:
        print(f"  load-torque observer T^   : {float(tl)*1e3:8.1f} mN.m "
              f"(true 20.0)")
    print(f"  final mode                : "
          f"{DFControllerBlock.MODE_NAMES.get(int(d['mode'][-1]), '?')}")
    print("=" * 64)


# =============================================================================
# Main
# =============================================================================

def _ask_codegen() -> bool:
    """
    Ask, after the simulation, whether to generate the AURIX C step function.
    Accepts only Y or N (re-asks on anything else).  If stdin is closed
    (truly non-interactive), input() raises EOFError and we skip cleanly.
    """
    while True:
        try:
            ans = input("Generate code? [Y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if ans in ("y", "n"):
            return ans == "y"


def main() -> int:
    print(f"[run] target={TARGET_RPM:.0f} RPM  T={T_SIM}s  dt={DT*1e6:.0f}us  "
          f"noise={'on' if ENABLE_NOISE else 'off'}")

    # CodeGen boundary hooks are transparent passthroughs (identical results),
    # so code generation is available after the run.
    d = _run_sim(target_rpm=TARGET_RPM,
                 t_sim=T_SIM,
                 with_noise=ENABLE_NOISE,
                 with_codegen_hooks=True)
    if d is None:
        print("[run] simulation failed")
        return 1

    print_summary(d, TARGET_RPM)
    plot_results(d)

    # After the simulation: generate AURIX C code?  Yes or No.
    if _ask_codegen():
        _run_codegen(d)

    return 0


if __name__ == "__main__":
    sys.exit(main())
