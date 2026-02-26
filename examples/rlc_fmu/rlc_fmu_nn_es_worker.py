"""
rlc_fmu_nn_es_worker.py
-------------
Standalone worker script — runs ONE FMU episode and writes the RMSE result
to a temporary file, then exits.

EDUCATIONAL NOTE — WHY A SEPARATE PROCESS?
-------------------------------------------
The RLC FMU (compiled OpenModelica C code) accumulates internal heap state
across sequential calls.  After ~1000-2000 rollouts in one process, Windows
raises exit code 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN / heap corruption).

No amount of FMU reset() or object recreation fixes this — the corruption is
inside the FMU's DLL memory space, which only Windows can fully reclaim.

Solution: run each FMU episode as a fresh subprocess.
When the subprocess exits, Windows releases ALL its memory unconditionally.
The parent process (main script) is never exposed to the corruption.

USAGE (called automatically by main script — do not run directly):
    python rlc_fmu_nn_es_worker.py <weights_file.npy> <ref_file.npy> <dt> <result_file.npy>

Arguments:
    weights_file : path to .npy file containing flat NN weight array
    ref_file     : path to .npy file containing reference signal array
    dt           : time step as float string
    result_file  : path where worker writes [rmse, y_out...] result array
"""

import sys
import os
import numpy as np

# ---- Resolve paths ----
_WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_WORKER_DIR, '..', '..')))

import torch
import torch.nn as nn
from embedsim import FMUBlock
from embedsim.core_blocks import VectorSignal
from pathlib import Path

# ---- Constants (must match main script) ----
U_LIMIT     = 50.0
VREF_AMPL   = 10.0
FREQ        = 50.0
LAYERS      = [3, 6, 6, 1]
ERR_SCALE   = 1.0 / VREF_AMPL
INTEG_SCALE = (2 * np.pi * FREQ) / VREF_AMPL
REF_SCALE   = 1.0 / VREF_AMPL

FMU_PATH = Path(_WORKER_DIR) / "modelica" / "RLC_Sine_DigitalTwin_OM.fmu"


class NNController(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.append(nn.Tanh())
        self.register_buffer(
            'input_scale',
            torch.tensor([[ERR_SCALE, INTEG_SCALE, REF_SCALE]], dtype=torch.float32)
        )

    def forward(self, x):
        x = x * self.input_scale
        for layer in self.net:
            x = layer(x)
        return x

    def set_flat_weights(self, w):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(torch.tensor(w[idx:idx+n].reshape(p.shape), dtype=torch.float32))
            idx += n


def main():
    weights_file = sys.argv[1]
    ref_file     = sys.argv[2]
    step_dt      = float(sys.argv[3])
    result_file  = sys.argv[4]

    weights = np.load(weights_file)
    ref     = np.load(ref_file)
    steps   = len(ref)

    ctrl = NNController(LAYERS)
    ctrl.set_flat_weights(weights)

    plant = FMUBlock(
        name="RLC_plant",
        fmu_path=FMU_PATH,
        input_names=["Vcontrol_python"],
        output_names=["Vout"],
        parameters={
            "R": 10.0, "L": 10e-3, "C": 100e-6,
            "Vref_ampl": VREF_AMPL, "freq": FREQ,
            "usePythonControl": 1.0,
        },
    )
    plant.reset()

    y_out = np.zeros(steps)
    integ = 0.0

    with torch.no_grad():
        for k in range(steps):
            y_prev = y_out[k-1] if k > 0 else 0.0
            err    = ref[k] - y_prev
            integ += err * step_dt

            nn_in    = torch.tensor([[err, integ, ref[k]]], dtype=torch.float32)
            u        = float(ctrl(nn_in).item())
            u        = float(np.clip(u, -U_LIMIT, U_LIMIT))
            sig_in   = VectorSignal([u], "ctrl")
            y_out[k] = float(plant.compute(k * step_dt, step_dt, [sig_in]).value[0])

    rmse = float(np.sqrt(np.mean((ref - y_out) ** 2)))

    # Write [rmse, y_out[0], y_out[1], ...] as result
    result = np.concatenate([[rmse], y_out])
    np.save(result_file, result)


if __name__ == '__main__':
    main()
