# MPC CodeGen — Status Note
**Date:** 2026-04-16  
**Author:** EmbedSim / Paul Abraham

---

## Status: WORKING ✓

The `MPCControllerBlock` code generation pipeline is fully functional,
producing correct AURIX-ready `embedsim_step.c` / `embedsim_step.h`.
It follows the same pattern as the SMC (`DFControllerBlock`) and
Differential Flatness (`DFCControllerBlock`) blocks.

---

## How the MPC block hooks into StepGenerator

`StepGenerator._emit_block()` reads the following class-level attributes
from `MPCControllerBlock` (defined in `mpc_controller_block.py`, lines 400–442):

| Attribute | Value | Purpose |
|---|---|---|
| `state_struct` | `"MPC_Controller_T"` | Emits `static MPC_Controller_T mpc_state;` |
| `init_func` | `"MPC_Controller_Init"` | Emits `MPC_Controller_Init(&mpc_state, dt);` in `EmbedSim_Init()` |
| `C_INIT_ARGS` | `["dt_s"]` | Instance attribute read for the `dt` init argument value |
| `C_HEADERS` | `["embed_sim_mpc_controller.h"]` | `#include` in both `.h` and `.c` |
| `C_SOURCES` | `["embed_sim_mpc_controller.c"]` | Informational — not directly emitted |
| `C_CUSTOM_EMIT` | (snippet, see below) | **Bypasses** the generic flat-array path entirely |

---

## Why C_CUSTOM_EMIT is needed

The generic `_emit_block()` path produces flat arrays:

```c
real32_T u_mpc[5];
real32_T y_mpc[2];
u_mpc[0] = y_ctrl_packer[0];
...
MPC_Controller_Step(&mpc_state, u_mpc, dt, y_mpc);   // WRONG signature
```

But `MPC_Controller_Step()` takes typed struct pointers, not flat arrays:

```c
void MPC_Controller_Step(MPC_Controller_T*, const MPC_Input_T*, MatrixFloat, MPC_Output_T*);
```

`C_CUSTOM_EMIT` bypasses the generic path and emits the correct typed snippet verbatim:

```c
/* --- mpc (MPCControllerBlock) --- */
MPC_Input_T   u_mpc;
MPC_Output_T  y_mpc_out;
real32_T      y_mpc[2];

u_mpc.omega_ref_mech = in->omega_ref_mech;
u_mpc.theta_m        = in->theta_m;
u_mpc.ia             = in->ia;
u_mpc.ib             = in->ib;
u_mpc.ic             = in->ic;

MPC_Controller_Step(&mpc_state, &u_mpc, dt, &y_mpc_out);

y_mpc[0] = y_mpc_out.v_alpha;
y_mpc[1] = y_mpc_out.v_beta;
```

`y_mpc[2]` is the downstream buffer that `svpwm_pack` reads via `u_svpwm_pack[0..1]`,
preserving the standard `y_<blockname>[N]` naming contract for the rest of the chain.

---

## StepGenerator two-pass rule and C_CUSTOM_EMIT exemption

`_gen_c()` does a two-pass emission for MISRA C:2012 Rule 8.1
(all declarations before statements):

- **Pass 1** (`decls_only=True`) — collects all `real32_T u_[]` / `y_[]` declarations
- **Pass 2** (`stmts_only=True`) — emits all assignments and function calls

`C_CUSTOM_EMIT` blocks are **exempt** from this split: `decls_only` returns `""`,
and `stmts_only` emits the snippet verbatim. This is safe because `C_CUSTOM_EMIT`
snippets use an inner brace block `{ }` (or interleave decls/stmts in a single
self-contained scope), which is valid C99/C11 used on AURIX TriCore TASKING.

---

## Verified output (2026-04-16, DB42S02 MPC FOC 20 kHz simulation)

```
[StepGenerator] Files written to 'embedsim_gen/':
  embedsim_step.h
  embedsim_step.c
  (3 block(s) in region)
  Input_T  : omega_ref_mech, theta_m, ia, ib, ic
  Output_T : ta, tb, tc, sector
```

Three blocks in region: `mpc` → `svpwm_pack` → `svpwm`. All emit correctly.

`EmbedSim_Init()` produces:
```c
MPC_Controller_Init(&mpc_state, 0.00005000f);
SVPWMPack_Init(&svpwm_pack_state, 17.00000000f);
```

---

## If a future MPC variant block fails to emit

Check that the new block class declares **all six** of the attributes in the
table above, particularly `C_CUSTOM_EMIT`. Without it, `_emit_block()` falls
through to the generic flat-array path which generates a wrong function
signature and a C compile error on TASKING.

The `step_func` / `state_struct` attributes are still required even when
`C_CUSTOM_EMIT` is present — `_gen_c()` reads `state_struct` and `init_func`
separately (before `_emit_block()`) to build the static state declarations
and `EmbedSim_Init()` body.
