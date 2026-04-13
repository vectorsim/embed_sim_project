pmsm_fmu_open_loop_example
==========================
EmbedSim — Open-loop V/f control — NANOTEC DB42S02
Library : fs_electrical_machines


PURPOSE
-------
Demonstrates open-loop Volts-per-Hertz (V/f) control of the NANOTEC DB42S02
PMSM using EmbedSim's Python-first block graph with co-simulation via an
OpenModelica FMU plant.

Serves three goals simultaneously:

  1. Verify the full open-loop signal chain from speed ramp through SVPWM
     duty cycles in Python simulation.

  2. Validate the EmbedSim CodeGen boundary: cg_start → SVPWMBlock → cg_end
     produces EmbedSim_step.c / EmbedSim_step.h ready for AURIX TC387.

  3. Provide a visual reference (static plots + animated phasor) for
     inspecting sector transitions, phase current quality, and V/f boost
     behaviour at low speed.


FILE LAYOUT
-----------
  db42s02_openloop_fmu.py        Main script — blocks, wiring, plots, animation
  _path_utils.py                 Project-root resolver (copied from embedsim root)
  read_me/
    read_me.txt                  This file

  Generated on first run:
    db42s02_openloop_results.png  4-panel static plot (speed/currents/duty/sector)
    db42s02_phasor_sectors.png    Static α-β SVPWM hexagon with full trajectory
    db42s02_phasor_anim.gif       6-panel animated phasor + VSI switch states
    db42s02_topology.html         EmbedSim signal-flow topology diagram
    embedsim_gen/EmbedSim_step.c  Generated ISR step function
    embedsim_gen/EmbedSim_step.h  Generated ISR header


DEPENDENCIES
------------
  Python packages  : embedsim, numpy, matplotlib
  fs_electrical_machines blocks:
    motor_utility_blocks.py        SpeedRampBlock, VfAngleBlock, VfDQBlock,
                                   VfThetaBlock, SVPWMPackBlock
    coordinate_transform_blocks.py InvParkTransformBlock  (C backend)
    svpwm_block.py                 SVPWMBlock             (C backend)
    PMSM_Plant_FMUBlock.py         FMUBlock wrapper for PMSM_Plant_FMU.fmu
  FMU file:
    fs_electrical_machines/modelica/PMSM_Plant_FMU.fmu
    (compiled from PMSM_Motor.mo via OpenModelica — star-connected PMSM,
     amplitude-invariant Clarke, duty-cycle input interface)


MOTOR PARAMETERS  (NANOTEC DB42S02)
------------------------------------
  R        = 0.19    Ω
  Ld = Lq  = 0.125   mH     (non-salient: Ld = Lq)
  λ_pm     = 0.0014  Wb
  J        = 2.4e-6  kg·m²
  B_fric   = 1e-6    N·m·s/rad
  p        = 4       pole pairs
  V_dc     = 17.0    V


SIMULATION PARAMETERS
----------------------
  T_sim        = 2.0 s          (0.5 s ramp + 1.5 s steady-state)
  dt           = 100 µs         (10 kHz equivalent; matches AURIX ISR rate)
  Solver       = Euler
  Speed cmd    = 400 RPM
  Ramp time    = 0.5 s
  T_load       = 0.01 N·m      (10% of rated — light bench load)


BLOCK GRAPH  (execution order)
-------------------------------

  [CodeGen region — enclosed by cg_start / cg_end]
  ┌──────────────────────────────────────────────────────────────┐
  │  SpeedRampBlock ──► VfAngleBlock ──► VfDQBlock ──┐          │
  │                               └──► VfThetaBlock ─┤          │
  │                                                   ▼          │
  │                               InvParkTransformBlock          │
  │                                        │                     │
  │                               SVPWMPackBlock                 │
  │                          [indices 0,1 cross cg_start]        │
  │  cg_start ─────────────────────► SVPWMBlock                 │
  │                                        │                     │
  │  cg_end  ◄──────────── [ta, tb, tc, sector]                 │
  └──────────────────────────────────────────────────────────────┘
                                           │
                                   DB42S02PlantBlock
                                   (PMSM_Plant_FMU.fmu)
                                   duty_a/b/c = ta/tb/tc
                                   v_dc = 17 V  (constant)
                                   T_load = 0.01 N·m  (constant)

  EmbedSim_Input_T  : empty  (_reserved byte for C99 compliance)
  EmbedSim_Output_T : { float32 ta; float32 tb; float32 tc; uint8 sector; }


V/f CONTROL LAW
---------------
  VF_RATIO = V_phase_peak / ω_e_rated
           = (V_dc / √3) / (p · ω_m_rated)
           ≈ 9.815 V / (4 · 837.8 rad/s)
           ≈ 2.929e-3  V·s/rad

  V_q = VF_RATIO · ω_e + VF_BOOST
  V_d = 0

  VF_BOOST = R · I_nom = 0.19 · 1.0 = 0.19 V
  (compensates resistive drop at low speed; <2% of V_q at rated speed)

  θ_e  accumulated by integrating ω_e = p · ω_m_ref (open loop).
  InvPark: [v_α, v_β] = InvPark([V_d, V_q], θ_e)
  SVPWMPack: Vref = |v_αβ|, angle = atan2(v_β, v_α)  → SVPWMBlock → [ta, tb, tc]


CODEGEN NOTES
-------------
  SVPWMBlock is the only block with C_CUSTOM_EMIT.
  It emits SVM_CalculateDutyCycle() — scalar ABI, output normalised by V_dc/2.

  On AURIX the ISR writes:
      ATOM0_CH0_CM0 = (uint32_t)(out.ta * GTM_PERIOD)
      ATOM0_CH2_CM0 = (uint32_t)(out.tb * GTM_PERIOD)
      ATOM0_CH4_CM0 = (uint32_t)(out.tc * GTM_PERIOD)

  v_dc and T_load are physical constants injected inside DB42S02PlantBlock
  and are never part of the CodeGen region.

  No DutyPackBlock.  No hand-written C strings in db42s02_openloop_fmu.py.


FMU INTERFACE  (PMSM_Plant_FMUBlock)
--------------------------------------
  INPUT_VARS  : duty_a, duty_b, duty_c, v_dc, T_load
  OUTPUT_VARS : rpm, ia, ib, ic, theta_m, T_em, id_out, iq_out
                [0]  [1] [2] [3]  [4]     [5]   [6]     [7]

  DB42S02PlantBlock.compute_py() reads indices 0..5 and re-packs
  a stable 5-element scope bus:
      [0] speed_rpm  [1] i_a  [2] i_b  [3] i_c  [4] T_em


RUN
---
  cd examples/pmsm_fmu_open_loop_example
  python db42s02_openloop_fmu.py

  Linux:   ./run_db42s02_openloop.sh   (if present)
  Windows: run_db42s02_openloop.bat    (if present)


EXPECTED OUTPUT
---------------
  [Topology] printed to console + db42s02_topology.html
  [Sim] 20 000 steps  (T=2.0 s, dt=100 µs)
  [CodeGen] EmbedSim_step.c / .h written to embedsim_gen/
  [Plot] db42s02_openloop_results.png
  [Plot] db42s02_phasor_sectors.png
  [Anim] 200 frames → db42s02_phasor_anim.gif
  [Done]

  At 400 RPM open-loop the phase currents will show a triangular
  component superimposed on the fundamental sine — this is expected.
  The motor operates below the R/L corner frequency (≈ 3630 RPM) so
  X_L << R and the inductive filtering is minimal.  VF_BOOST ensures
  adequate torque-producing voltage.


KNOWN LIMITATIONS
-----------------
  - Open-loop V/f: no current feedback, no rotor position sensing.
    Susceptible to pull-out if load torque exceeds T_pullout(ω).
  - theta_e is integrated from the reference speed, not measured.
    Any mismatch between reference and actual speed accumulates phase error.
  - FMU uses fixed Euler integration at dt=100 µs.
    Stiff inductance transients (L/R ≈ 0.66 µs) are under-resolved;
    acceptable for system-level V/f validation, not for waveform fidelity.
  - Animation requires pillow (GIF) or ffmpeg (MP4).


RELATED EXAMPLES
----------------
  pmsm_smc_smo_example/    Closed-loop SMC FOC with SMO observer
  pmsm_dfc_smo_example/    Closed-loop DFC FOC with SMO + SpeedFusion
