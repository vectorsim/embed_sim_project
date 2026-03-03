"""
utility_blocks.py
=================

Utility blocks for PMSM FOC simulations.

    VectorCombineBlock  — merge N scalar (or vector) inputs into one flat vector
    RecordingSinkBlock  — extended VectorEnd with shape validation,
                          per-channel RMS / statistics, and safe data export

Both blocks use SimBlockBase (→ VectorBlock) and support the dual
Python / C backend, though for utility blocks the C backend is rarely needed.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ._path_utils import setup_embedsim_path
setup_embedsim_path()

from embedsim.code_generator import SimBlockBase
from embedsim.dynamic_blocks  import VectorEnd
from embedsim.core_blocks      import VectorSignal
from embedsim.simulation_engine import LoopBreaker


# ==============================================================================
# ExtractDelay  — combined channel extractor + loop-breaking one-step delay
# ==============================================================================

class ExtractDelay(SimBlockBase, LoopBreaker):
    """
    Channel-extracting loop-breaking one-step delay.

    This solves a fundamental ordering problem in EmbedSim: a VectorDelay
    placed *after* a SignalExtractBlock runs before the extractor in the
    topological order (because it's a LoopBreaker and is added early by DFS).
    When it tries to update last_output it finds the extractor output is None,
    so it stores a scalar fallback and the signal is corrupted.

    ExtractDelay fuses extraction and delay into ONE block that is itself
    the LoopBreaker.  Its input is connected directly to the full upstream
    vector (e.g. the motor 5-vector), and it internally extracts the desired
    channels before storing.  Because the extraction happens inside compute_py
    (called during the main forward pass, after sources/motors), and the
    LoopBreaker output is pre-initialised before that pass, downstream blocks
    always see a valid delayed signal.

    Usage:
        # Replace:
        #     extract_theta = SignalExtractBlock("extract_theta", channels=[4])
        #     theta_delay   = VectorDelay("theta_delay", initial=[0.0])
        #     motor >> extract_theta
        #     extract_theta >> theta_delay
        # With:
        theta_delay = ExtractDelay("theta_delay", channels=[4], initial=[0.0])
        motor >> theta_delay          # motor output is [ia,ib,ic,ω,θ], extracts [θ]
        theta_delay >> inv_park
        theta_delay >> park
    """

    is_loop_breaker = True

    def __init__(
        self,
        name:     str,
        channels: List[int],
        initial:  List[float],
    ) -> None:
        super().__init__(name, use_c_backend=False)
        self.channels    = list(channels)
        self.vector_size = len(channels)
        self.is_dynamic  = False
        self._initial    = list(initial)          # preserved for reset()
        self.last_output = VectorSignal(
            np.array(initial, dtype=np.float64), name
        )

    def reset(self) -> None:
        """Override base reset() so last_output survives the pre-run reset."""
        self.output      = None
        self.last_output = VectorSignal(
            np.array(self._initial, dtype=np.float64), self.name
        )

    # LoopBreaker interface — called by engine BEFORE the main compute pass
    def get_loop_breaking_output(self):
        return self.last_output

    def compute_py(
        self,
        t:            float,
        dt:           float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        # Guard: should never be None after reset() override, but be safe
        if self.last_output is None:
            self.last_output = VectorSignal(
                np.array(self._initial, dtype=np.float64), self.name
            )

        # Output = last step's extracted value (initial value on first step)
        out_val = self.last_output.value.copy()
        self.output = VectorSignal(out_val, self.name, dtype=np.float64)

        # Update stored value from current input — becomes output NEXT step
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) > max(self.channels):
                extracted = np.array([float(v[ch]) for ch in self.channels],
                                     dtype=np.float64)
                self.last_output = VectorSignal(extracted, self.name)
            # else: upstream not ready yet — keep last_output unchanged

        return self.output

    def compute_c(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)

    def _load_wrapper(self):
        pass

    def __repr__(self):
        return f"ExtractDelay('{self.name}', channels={self.channels})"


# ==============================================================================
# SignalExtractBlock
# ==============================================================================

class SignalExtractBlock(SimBlockBase):
    """
    Extract a slice of channels from a vector signal.

    Useful for splitting the motor output [ia, ib, ic, ω, θ] into
    separate signals for the feedback paths.

    Example — split PMSMMotorBlock output:
        extract_abc   = SignalExtractBlock("extract_abc",   channels=[0,1,2])
        extract_theta = SignalExtractBlock("extract_theta", channels=[4])
        extract_speed = SignalExtractBlock("extract_speed", channels=[3])

        motor >> extract_abc
        motor >> extract_theta
        motor >> extract_speed
    """

    def __init__(
        self,
        name:     str,
        channels: List[int],
        use_c_backend: bool = False,
        dtype = None,
    ) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)
        self.channels    = list(channels)
        self.vector_size = len(channels)
        self.is_dynamic  = False

    def compute_py(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        if not input_values:
            self.output = VectorSignal(
                np.zeros(len(self.channels)), self.name, dtype=self.dtype
            )
            return self.output

        v = input_values[0].value
        extracted = np.array([float(v[ch]) for ch in self.channels], dtype=np.float64)
        self.output = VectorSignal(extracted, self.name, dtype=self.dtype)
        return self.output

    def compute_c(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        # For such a trivial operation the Python path is fast enough;
        # C backend falls through to Python.
        return self.compute_py(t, dt, input_values)

    def _load_wrapper(self) -> None:
        pass  # No separate wrapper needed — C path calls compute_py

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        return (
            f"SignalExtractBlock('{self.name}', channels={self.channels}, "
            f"backend={be})"
        )


# ==============================================================================
# VectorCombineBlock
# ==============================================================================

class VectorCombineBlock(SimBlockBase):
    """
    Merge N input signals into one flat output vector.

    By default, takes the first element of each input port and
    concatenates them:
        [port0[0], port1[0], ..., portN[0]]

    Set ``full_vectors=True`` to concatenate the complete value arrays
    of every connected input port.

    Example — combine d and q voltage references:
        id_pi >> dq_combine     # port 0 → [vd]
        iq_pi >> dq_combine     # port 1 → [vq]
        dq_combine >> inv_park  # output  → [vd, vq]

    Attributes
        n_inputs     : expected number of input ports (informational)
        full_vectors : if True, concatenate complete arrays; else take [0] of each
    """

    def __init__(
        self,
        name:         str,
        n_inputs:     int  = 2,
        full_vectors: bool = False,
        use_c_backend: bool = False,
        dtype = None,
    ) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)
        self.n_inputs     = n_inputs
        self.full_vectors = full_vectors
        self.is_dynamic   = False
        self.vector_size  = n_inputs   # updated on first compute

        if use_c_backend:
            self._load_wrapper()

    # -- Python ----------------------------------------------------------------

    def compute_py(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        if not input_values:
            self.output = VectorSignal(np.zeros(self.n_inputs), self.name,
                                       dtype=self.dtype)
            return self.output

        if self.full_vectors:
            parts = [inp.value for inp in input_values]
        else:
            parts = [np.atleast_1d(inp.value)[0:1] for inp in input_values]

        combined = np.concatenate(parts).astype(np.float64)
        self.vector_size = len(combined)
        self.output = VectorSignal(combined, self.name, dtype=self.dtype)
        return self.output

    # -- C backend (rarely needed, provided for completeness) ------------------

    def compute_c(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        u = np.concatenate([inp.value for inp in input_values]).astype(np.float64)
        self._wrapper.set_inputs(u)
        self._wrapper.compute()
        y = self._wrapper.get_outputs()
        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def _load_wrapper(self) -> None:
        try:
            from vector_combine_wrapper import VectorCombineWrapper
            self._wrapper = VectorCombineWrapper()
        except ImportError:
            raise ImportError(
                "Cython wrapper 'vector_combine_wrapper' not found.\n"
                "Generate with CodeGenEnd.generate_pyx_stub() and compile."
            )

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        return (
            f"VectorCombineBlock('{self.name}', backend={be}, "
            f"n_inputs={self.n_inputs}, full_vectors={self.full_vectors})"
        )


# ==============================================================================
# RecordingSinkBlock
# ==============================================================================

class RecordingSinkBlock(VectorEnd):
    """
    Extended sink block with shape validation and analysis helpers.

    Inherits from VectorEnd so EmbedSim treats it as a sink.

    Features
    --------
    • Shape consistency check — warns if the signal shape changes mid-run.
    • get_data()              — returns (data_array, times_array) as float64 ndarrays.
    • get_time_series(ch)     — single-channel (times, values) pair.
    • get_statistics(ch)      — dict with mean, std, max, min, peak-to-peak, RMS.
    • rms(ch)                 — convenience scalar RMS for channel ch.

    The class also exposes ``recorded_data`` and ``recorded_times`` lists for
    direct iteration if needed.

    Example
    -------
        sink = RecordingSinkBlock("motor_out")
        motor >> sink

        data, t = sink.get_data()      # shape (N, 5), (N,)
        t_ch, v = sink.get_time_series(channel=3)   # ω over time
        stats   = sink.get_statistics(channel=0)    # ia statistics
        print(stats['rms'])
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.recorded_data:  List[np.ndarray] = []
        self.recorded_times: List[float]      = []
        self.expected_shape: Optional[tuple]  = None
        self._shape_warn_count: int           = 0
        self._MAX_SHAPE_WARNS:  int           = 3
        # Step counter gate: the engine calls _compute_all_blocks multiple
        # times per timestep (RK4 substeps at t + 0, dt/2, dt/2, dt).
        # We only record on the FIRST call at each FULL step boundary.
        # _dt_guard = dt*0.6.  This value sits between dt/2 (RK4 substep offset)
        # and dt (next main step advance), so substep calls are blocked while
        # every main-step call (including the final t=T recording) passes.
        self._last_recorded_t:  float         = -1.0
        self._dt_guard:         float         = 0.0   # initialised on first compute call

    # -- Compute (sink, no C backend needed) -----------------------------------

    def compute_py(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> Optional[VectorSignal]:
        if not input_values:
            return self.output

        value = input_values[0].value

        # Learn dt on first call and set the guard to 0.4*dt.
        # This threshold sits between 0 (main step) and dt/2 (RK4 substep),
        # so only the first call at each integer multiple of dt is recorded.
        if self._dt_guard == 0.0 and dt > 0.0:
            self._dt_guard = dt * 0.6

        # Suppress RK4 substep calls
        if t - self._last_recorded_t < self._dt_guard:
            self.output = input_values[0]
            return self.output
        self._last_recorded_t = t

        # First sample — record expected shape
        if self.expected_shape is None:
            self.expected_shape = value.shape

        # Shape guard
        if value.shape != self.expected_shape:
            if self._shape_warn_count < self._MAX_SHAPE_WARNS:
                print(
                    f"[RecordingSinkBlock '{self.name}'] shape mismatch: "
                    f"expected {self.expected_shape}, got {value.shape} at t={t:.5f}"
                )
                self._shape_warn_count += 1
            return self.output

        self.recorded_data.append(value.copy())
        self.recorded_times.append(t)
        self.output = input_values[0]
        return self.output

    # -- Data export -----------------------------------------------------------

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (data_array, times_array).

        data_array : shape (N, D) — N time steps, D signal channels
        times_array: shape (N,)

        Returns (empty, empty) if no data recorded yet.
        """
        if not self.recorded_data:
            return np.empty((0,)), np.empty((0,))

        try:
            data  = np.array(self.recorded_data, dtype=np.float64)
            times = np.array(self.recorded_times, dtype=np.float64)
            return data, times
        except (ValueError, TypeError):
            # Fallback: shapes may be inconsistent; try stacking
            shapes = {d.shape for d in self.recorded_data}
            if len(shapes) == 1:
                data  = np.stack(self.recorded_data).astype(np.float64)
                times = np.array(self.recorded_times, dtype=np.float64)
                return data, times
            print(
                f"[RecordingSinkBlock '{self.name}'] cannot stack data: "
                f"multiple shapes found: {shapes}"
            )
            return np.empty((0,)), np.empty((0,))

    def get_time_series(self, channel: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (times, values) for a single output channel.

        Args:
            channel: zero-based channel index

        Returns:
            (times, values) — both shape (N,).
            (empty, empty) if no data or channel out of range.
        """
        data, times = self.get_data()
        if data.ndim < 2 or channel >= data.shape[1]:
            return np.empty((0,)), np.empty((0,))
        return times, data[:, channel]

    def get_statistics(self, channel: Optional[int] = None) -> Dict:
        """
        Compute statistics for one or all channels.

        Args:
            channel: if None, compute for every channel.

        Returns:
            dict keyed by 'channel_N' with sub-dict:
                mean, std, max, min, peak_to_peak, rms
        """
        data, times = self.get_data()
        if data.size == 0:
            return {}

        if data.ndim == 1:
            data = data[:, np.newaxis]

        stats: Dict = {}
        channels = [channel] if channel is not None else range(data.shape[1])

        for ch in channels:
            if ch >= data.shape[1]:
                continue
            v = data[:, ch]
            stats[f"channel_{ch}"] = {
                "mean":         float(np.mean(v)),
                "std":          float(np.std(v)),
                "max":          float(np.max(v)),
                "min":          float(np.min(v)),
                "peak_to_peak": float(np.ptp(v)),
                "rms":          float(np.sqrt(np.mean(v ** 2))),
            }

        return stats

    def rms(self, channel: int = 0) -> float:
        """Convenience: return scalar RMS for one channel."""
        stats = self.get_statistics(channel)
        key   = f"channel_{channel}"
        return stats[key]["rms"] if key in stats else 0.0

    def clear(self) -> None:
        """Reset all recorded data (useful for repeated runs)."""
        self.recorded_data.clear()
        self.recorded_times.clear()
        self.expected_shape    = None
        self._shape_warn_count = 0
        self._last_recorded_t  = -1.0
        self._dt_guard         = 0.0

    def __repr__(self) -> str:
        n = len(self.recorded_data)
        shape = self.expected_shape or "?"
        return f"RecordingSinkBlock('{self.name}', samples={n}, shape={shape})"
