# svpwm_block.py
# =============================================================================
# EmbedSim VectorBlock wrapper for the SVPWM C implementation.
# Location: fs_electrical_machines/svpwm_block.py
#
# Pipeline position
# -----------------
#   InvPark → SVPWMPackBlock → SVPWMBlock → cg_end / plant
#              [v_α, v_β]    [Vref,α_rad]  [ta,tb,tc,sector]
#
# INPUTS  (3 scalars) — pre-computed polar form from SVPWMPackBlock
# -------
#   [0] Vref       = sqrt(v_α²+v_β²)   magnitude, clipped to 0.95
#   [1] angle_rad  = atan2(v_β, v_α)   wrapped to [0, 2π)
#   [2] Vdc        DC bus voltage [V]  (passed through, not used by SVM)
#
#   SVPWMPackBlock performs both the clip and the wrap before passing here.
#   These are exactly the two scalar arguments to SVM_CalculateDutyCycle():
#       status = SVM_CalculateDutyCycle(Vref, angle_rad, &svm_duty)
#   which mirrors SpaceVectorModulation1() in the AURIX application layer.
#
# OUTPUTS (4 scalars)
# -------
#   [0] ta     — Phase A duty cycle [0, 1]
#   [1] tb     — Phase B duty cycle [0, 1]
#   [2] tc     — Phase C duty cycle [0, 1]
#   [3] sector — Active sector 0..5
#
# C API (svpwm.c):
#   MatrixStatus_Type SVM_CalculateDutyCycle(
#       const MatrixFloat modulation_index,
#       const MatrixFloat angle_rad,
#       SVM_DutyCycle_Type * const duty);
#
# CodeGen
# -------
#   C_CUSTOM_EMIT is required: SVM_CalculateDutyCycle has a non-standard
#   ABI (two separate scalar inputs + pointer-to-struct output).
#   StepGenerator emits the C_CUSTOM_EMIT block verbatim, reading
#   in->magnitude and in->angle_rad from EmbedSim_Input_T directly.
#   The integration layer (AURIX app) pre-computes sqrtf/atan2f/clip/wrap.
# =============================================================================

import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from embedsim.core_blocks import VectorBlock, VectorSignal

_HERE  = Path(__file__).resolve().parent
_C_SRC = _HERE / "c_src"


class SVPWMBlock(VectorBlock):
    """
    Space Vector PWM duty-cycle calculator.

    Receives pre-computed polar form from SVPWMPackBlock and calls
    SVM_CalculateDutyCycle() to produce three normalised PWM duty
    cycles and the active sector number.

    Inputs
    ------
    [0] Vref      : float  Modulation index (magnitude), clipped to 0.95
    [1] angle_rad : float  Voltage angle [rad], wrapped to [0, 2π)
    [2] Vdc       : float  DC bus voltage [V]  (pass-through)

    Outputs
    -------
    [0] ta     : float  Phase A duty cycle [0, 1]
    [1] tb     : float  Phase B duty cycle [0, 1]
    [2] tc     : float  Phase C duty cycle [0, 1]
    [3] sector : float  Sector index 0..5
    """

    # ── CodeGen ──────────────────────────────────────────────────────────────
    PYX_FILE    : str = str(_C_SRC / "svpwm_wrapper.pyx")
    C_SOURCES        = ["SV_PWM"
                        ".c"]
    C_HEADERS        = ["SV_PWM.h"]
    NUM_INPUTS       = 1     # one upstream port: SVPWMPackBlock [Vref, angle, Vdc]
    OUTPUT_SIZE      = 4     # [ta, tb, tc, sector]

    # Expose only ta, tb, tc at the CodeGen boundary.
    # Sector is internal to SVM — it must not appear in EmbedSim_Output_T.
    # StepGenerator reads OUTPUT_NAMES / OUTPUT_KEEP to build the output
    # struct fields and the pack-outputs section.
    # C_OUTPUT_TYPES overrides the default real32_T for sector -> uint8_T.
    OUTPUT_NAMES   = ["ta", "tb", "tc", "sector"]
    OUTPUT_KEEP    = [0, 1, 2, 3]
    C_OUTPUT_TYPES = {"sector": "uint8_T"}

    # Non-standard ABI — reads magnitude/angle_rad from EmbedSim_Input_T.
    # Integration layer (AURIX app) is responsible for:
    #   in->magnitude = sqrtf(v_alpha^2+v_beta^2)  clipped to 0.95
    #   in->angle_rad = atan2f(v_beta, v_alpha)     wrapped to [0, 2pi)
    C_CUSTOM_EMIT = """\
    /* --- svpwm (SVPWMBlock) — SVM_CalculateDutyCycle --- */
    real32_T y_svpwm[4];
    {
        SVM_DutyCycle_Type svm_duty;
        MatrixStatus_Type  svm_status;
        svm_status = SVM_CalculateDutyCycle(
                         in->magnitude,   /* modulation index — from integration layer */
                         in->angle_rad,   /* angle [rad]      — from integration layer */
                         &svm_duty);
        if (svm_status == MATRIX_SUCCESS)
        {
            y_svpwm[0] = (real32_T)svm_duty.ta     / (real32_T)Q31_ONE;
            y_svpwm[1] = (real32_T)svm_duty.tb     / (real32_T)Q31_ONE;
            y_svpwm[2] = (real32_T)svm_duty.tc     / (real32_T)Q31_ONE;
            y_svpwm[3] = (real32_T)svm_duty.sector;
        }
        else
        {
            y_svpwm[0] = 0.5f;
            y_svpwm[1] = 0.5f;
            y_svpwm[2] = 0.5f;
            y_svpwm[3] = 0.0f;
        }
    }"""

    # ── Constants — mirror svpwm.h ────────────────────────────────────────────
    _SQRT3_OVER_2: float = 0.86602540378
    _PI_OVER_6:    float = 0.5235987756
    _PI_OVER_3:    float = 1.0471975512
    _2PI_OVER_3:   float = 2.0943951024
    _PI:           float = 3.14159265359
    _4PI_OVER_3:   float = 4.18879020479
    _5PI_OVER_3:   float = 5.23598775598
    _2PI:          float = 6.28318530718

    def __init__(
            self,
            name: str = "svpwm",
            use_c_backend: bool = False,
            dtype=np.float32,
    ) -> None:
        self.use_c_backend: bool = use_c_backend
        super().__init__(name, dtype=dtype)
        self.vector_size  = 4
        self.output_label = "[ta,tb,tc,sector]"
        self._wrapper     = None
        if use_c_backend:
            self._load_wrapper()

    def _load_wrapper(self) -> None:
        try:
            from svpwm_wrapper import EmbedSimSVPWM
            self._wrapper = EmbedSimSVPWM()
        except ImportError as exc:
            raise ImportError(
                "svpwm_wrapper.pyd not found. Build with:\n"
                "  cd fs_electrical_machines/c_src\n"
                "  python setup_svpwm.py build_ext --inplace"
            ) from exc

    @staticmethod
    def _get_sector(angle: float) -> int:
        """Map angle in [0, 2π) to sector 0..5."""
        if   angle < SVPWMBlock._PI_OVER_3:   return 0
        elif angle < SVPWMBlock._2PI_OVER_3:  return 1
        elif angle < SVPWMBlock._PI:           return 2
        elif angle < SVPWMBlock._4PI_OVER_3:  return 3
        elif angle < SVPWMBlock._5PI_OVER_3:  return 4
        else:                                  return 5

    def compute_py(
            self,
            t: float,
            dt: float,
            input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """
        Compute SVPWM duty cycles — mirrors SVM_CalculateDutyCycle() in svpwm.c.

        Input contract (from SVPWMPackBlock):
          vals[0] = Vref      — magnitude, already clipped to 0.95
          vals[1] = angle_rad — already wrapped to [0, 2π)
        """
        zero = np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float32)
        if not input_values or not input_values[0]:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        vals = input_values[0].value
        if len(vals) < 2:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        m     = float(vals[0])   # Vref — already clipped by SVPWMPackBlock
        angle = float(vals[1])   # angle_rad — already wrapped by SVPWMPackBlock

        sector = self._get_sector(angle)

        scale = self._SQRT3_OVER_2 * m
        _PI6  = self._PI_OVER_6

        if   sector == 0:
            t1 = scale * math.cos(angle + _PI6)
            t2 = scale * math.cos(angle - math.pi / 2.0)
        elif sector == 1:
            t1 = scale * math.cos(angle - _PI6)
            t2 = scale * math.cos(angle - 5.0 * _PI6)
        elif sector == 2:
            t1 = scale * math.cos(angle - math.pi / 2.0)
            t2 = scale * math.cos(angle - 7.0 * _PI6)
        elif sector == 3:
            t1 = scale * math.cos(angle - 5.0 * _PI6)
            t2 = scale * math.cos(angle - 3.0 * math.pi / 2.0)
        elif sector == 4:
            t1 = scale * math.cos(angle - 7.0 * _PI6)
            t2 = scale * math.cos(angle - 11.0 * _PI6)
        else:
            t1 = scale * math.cos(angle - 3.0 * math.pi / 2.0)
            t2 = scale * math.cos(angle - _PI6)

        t1 = max(0.0, t1)
        t2 = max(0.0, t2)

        if (t1 + t2) > 1.0:
            sf = 1.0 / (t1 + t2)
            t1 *= sf
            t2 *= sf

        t0 = max(0.0, (1.0 - t1 - t2) * 0.5)

        if   sector == 0:   ta = t1+t2+t0;  tb = t2+t0;     tc = t0
        elif sector == 1:   ta = t1+t0;     tb = t1+t2+t0;  tc = t0
        elif sector == 2:   ta = t0;        tb = t1+t2+t0;  tc = t2+t0
        elif sector == 3:   ta = t0;        tb = t1+t0;     tc = t1+t2+t0
        elif sector == 4:   ta = t2+t0;     tb = t0;        tc = t1+t2+t0
        else:               ta = t1+t2+t0;  tb = t0;        tc = t1+t0

        ta = max(0.0, min(1.0, ta))
        tb = max(0.0, min(1.0, tb))
        tc = max(0.0, min(1.0, tc))

        self.output = VectorSignal(
            np.array([ta, tb, tc, float(sector)], dtype=np.float32),
            self.name)
        return self.output

    def compute_c(
            self,
            t: float,
            dt: float,
            input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """Call SVM_CalculateDutyCycle via Cython wrapper with Vref and angle_rad."""
        zero = np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float32)
        if not input_values or not input_values[0]:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output
        vals      = input_values[0].value
        vref      = float(vals[0]) if len(vals) > 0 else 0.0
        angle_rad = float(vals[1]) if len(vals) > 1 else 0.0
        self._wrapper.calculate(vref, angle_rad)
        self.output = VectorSignal(
            np.array([self._wrapper.ta, self._wrapper.tb,
                      self._wrapper.tc, float(self._wrapper.sector)],
                     dtype=np.float32),
            self.name)
        return self.output

    def compute(
            self,
            t: float,
            dt: float,
            input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        if self.use_c_backend:
            return self.compute_c(t, dt, input_values)
        return self.compute_py(t, dt, input_values)

    def reset(self) -> None:
        super().reset()

    def __repr__(self) -> str:
        return f"SVPWMBlock('{self.name}', backend={'C' if self.use_c_backend else 'Python'})"
