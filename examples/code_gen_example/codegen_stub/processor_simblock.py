# processor_simblock.py
# =================================================================
# Auto-generated SimBlockBase subclass for 'processor'
#
# Switch between Python and C with one flag:
#   block = ProcessorSimBlock('name', use_c_backend=False)  # Python
#   block = ProcessorSimBlock('name', use_c_backend=True)   # C (needs .pyx compiled)

from typing import List, Optional
import numpy as np
from .core_blocks import VectorSignal
from .code_generator import SimBlockBase


class ProcessorSimBlock(SimBlockBase):
    """
    ControlForge block: processor

    Inputs  (2 doubles total):
               [0]  sine_a  (size=1)
               [1]  gain_b  (size=1)

    Outputs (1 doubles total):
               [0]  summer  (size=1)
    """

    def __init__(self, name: str, use_c_backend: bool = False):
        super().__init__(name, use_c_backend)
        self.vector_size = 1
        self._wrapper = None
        if use_c_backend:
            self._load_wrapper()

    def _load_wrapper(self):
        try:
            from processor_wrapper import ProcessorWrapper
            self._wrapper = ProcessorWrapper()
        except ImportError:
            raise ImportError(
                "Cython wrapper 'processor_wrapper' not found.\n"
                "Compile it:\n"
                "  python setup_processor.py build_ext --inplace"
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
        # sine_a = input_values[0].value[0]
        # gain_b = input_values[0].value[1]
        y = np.zeros(1, dtype=np.float32)
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
        u = np.empty(2, dtype=np.float32)
        u[0] = input_values[0].value[0] if input_values else 0.0  # sine_a
        u[1] = input_values[0].value[1] if input_values else 0.0  # gain_b
        # -- Call C via Cython --------------------------------------
        self._wrapper.set_inputs(u)
        self._wrapper.compute()
        y = self._wrapper.get_outputs()
        self.output = VectorSignal(y, self.name)
        return self.output