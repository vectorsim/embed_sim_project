# three_phase_processor_simblock.py
# =================================================================
# Auto-generated SimBlockBase subclass for 'three_phase_processor'
#
# Switch between Python and C with one flag:
#   block = ThreePhaseProcessorSimBlock('name', use_c_backend=False)  # Python
#   block = ThreePhaseProcessorSimBlock('name', use_c_backend=True)   # C (needs .pyx compiled)

from typing import List, Optional
import numpy as np
from embedsim.core_blocks import VectorSignal
from embedsim.code_generator import SimBlockBase


class ThreePhaseProcessorSimBlock(SimBlockBase):
    """
    ControlForge block: three_phase_processor

    Inputs  (3 doubles total):
            [0..2]  source  (size=3)

    Outputs (3 doubles total):
            [0..2]  gain  (size=3)
    """

    def __init__(self, name: str, use_c_backend: bool = False):
        super().__init__(name, use_c_backend)
        self.vector_size = 3
        self._wrapper = None
        if use_c_backend:
            self._load_wrapper()

    def _load_wrapper(self):
        try:
            from three_phase_processor_wrapper import ThreePhaseProcessorWrapper
            self._wrapper = ThreePhaseProcessorWrapper()
        except ImportError:
            raise ImportError(
                "Cython wrapper 'three_phase_processor_wrapper' not found.\n"
                "Compile it:\n"
                "  python setup_three_phase_processor.py build_ext --inplace"
            )

    # -- Python implementation -------------------------------------
    def compute_py(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """TODO: implement your Python algorithm here."""
        # -- Unpack inputs ------------------------------------------
        # source = input_values[0].value[0:3]
        y = np.zeros(3, dtype=np.float64)
        # TODO: fill y with your computed outputs
        self.output = VectorSignal(y, self.name)
        return self.output

    # -- C backend ------------------------------------------------
    def compute_c(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """Call compiled Cython wrapper - zero Python overhead on hot path."""
        # -- Pack flat input buffer ---------------------------------
        u = np.empty(3, dtype=np.float64)
        u[0:3] = input_values[0].value[0:3] if input_values else np.zeros(3)  # source
        # -- Call C via Cython --------------------------------------
        self._wrapper.set_inputs(u)
        self._wrapper.compute()
        y = self._wrapper.get_outputs()
        self.output = VectorSignal(y, self.name)
        return self.output