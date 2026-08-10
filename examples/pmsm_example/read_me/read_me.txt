================================================================================
  SpeedFusion & Differential Flatness FOC — NANOTEC DB42S02 / AURIX TC3xx
  EmbedSim Implementation Reference
================================================================================

OVERVIEW
--------
This document describes the speed estimation and voltage control architecture
implemented in:

    fs_electrical_machines/diff_flatness_controller_block.py   (Python simulation)
    fs_electrical_machines/embed_sim_dfc_controller.c           (AURIX C firmware)
    fs_electrical_machines/embed_sim_dfc_controller.h
    fs_electrical_machines/embed_sim_dfc_gains.h

The controller combines two complementary speed estimation sources (SpeedFusion)
with a model-based voltage law derived from differential flatness theory to
achieve precise Field-Oriented Control (FOC) of the DB42S02 PMSM at 20 kHz
on the Infineon AURIX TC3xx.


================================================================================
PART 1 — SPEEDFUSION: COMPLEMENTARY SPEED ESTIMATION
================================================================================

Two independent speed sources are available, each with complementary
strengths and weaknesses.  SpeedFusion blends them into a single estimate.

Source 1: Encoder finite-difference
    Input:    theta_m [rad]  — GTM TIM quadrature capture (1000 PPR × 4 = 4000 cnt/rev)
    Output:   omega_enc [rad/s mechanical]
    Strength: Ground-truth angle, low latency, accurate at any speed
    Weakness: Finite-difference amplifies quantisation noise at low speed;
              at very low speed (< 50 rad/s) the angular step per ISR period
              is only ~0.08 rad/count, making omega_raw noisy.

Source 2: Sliding Mode Observer (SMO)
    Input:    v_alpha, v_beta [V], i_alpha, i_beta [A]
    Output:   omega_smo [rad/s electrical]
    Strength: Model-based; accurate and low-noise at medium-to-high speed
              once the back-EMF LPF has converged (~20 ms warmup)
    Weakness: Back-EMF amplitude = omega_e * lambda_pm = 920 * 0.0014 = 1.29 V
              at rated speed.  At low speed this signal is too small relative
              to the SMO switching gain (2.0 V) for reliable angle extraction.

Blend equation
--------------
    omega_final = (1 - alpha) * omega_enc_e  +  alpha * omega_smo_gated

    where:
        omega_enc_e  = P_POLES * omega_enc_filt   [rad/s electrical]
        alpha        = blend weight in [0.0, 1.0]

Alpha schedule (linear ramp):
    |omega_mech| <= 50  rad/s  ->  alpha = 0.0   (pure encoder)
    |omega_mech| >= 250 rad/s  ->  alpha = 1.0   (pure SMO)
    50 < |omega_mech| < 250    ->  alpha = linear interpolation

These thresholds correspond to:
    50  rad/s  =  478 RPM   (lower blend threshold)
    250 rad/s  =  2387 RPM  (upper blend threshold)

Adaptive IIR smoothing on encoder speed
-----------------------------------------
    iir_coeff = iir_lo + alpha * (iir_hi - iir_lo)
    omega_enc_filt[k] = (1 - iir_coeff) * omega_enc_filt[k-1]
                      + iir_coeff * omega_raw[k]

    iir_lo = 0.05   (heavy smoothing at low speed, tau_eff ~ 950 us)
    iir_hi = 0.30   (light smoothing at high speed, tau_eff ~ 117 us)

SMO plausibility gate
----------------------
    Before blending, the SMO output is checked against the encoder:
        if |omega_smo_e - omega_enc_e| > 1000 rad/s (electrical):
            omega_smo_gated = omega_enc_e   (encoder fallback)
        else:
            omega_smo_gated = omega_smo     (SMO accepted)

    This catches atan2f phase-wrap artefacts that survive the
    omega_e_hat clamp (3000 rad/s electrical ceiling).

Encoder fallback during SMO warmup
------------------------------------
    For the first 400 ISR steps (20 ms), the SMO back-EMF LPF has not
    yet converged and omega_smo is forced to zero.  If the encoder shows
    motion above the lower threshold, the blend is overridden:
        if |omega_smo| < 1 rad/s  AND  |omega_enc_filt| > 50 rad/s:
            omega_final = omega_enc_e

The value omega_meas_mech = omega_enc_filt [rad/s mechanical] is used
as the speed P-loop feedback signal.  The fused omega_e [rad/s electrical]
is used only inside the flatness voltage law feedforward terms.


================================================================================
PART 2 — DIFFERENTIAL FLATNESS VOLTAGE LAW
================================================================================

The flatness voltage law replaces the two independent PI current controllers
of classical FOC with a model-based precompensation derived from the PMSM
voltage equations in the dq frame.

Classical PI-FOC computes:
    vd = Kp_d * (0 - id)  +  Ki_d * integral(0 - id)
    vq = Kp_q * (iq_ref - iq)  +  Ki_q * integral(iq_ref - iq)
    The cross-coupling (omega_e*L*iq in vd; omega_e*lambda in vq) is left
    as a disturbance for the integrators to reject.

The DFC pre-computes these coupling terms analytically:

D-axis voltage equation
------------------------
    vd = -omega_e * L_q * i_q_ref           [cross-coupling cancellation]
       + Kp_id * (0 - i_d_meas)             [id = 0 MTPA enforcement]

    Units:  [rad/s] * [H] * [A] = [V]  (feedforward term)
            [V/A]   * [A]       = [V]  (feedback term)

    Physical meaning of each term:
      -omega_e * L_q * i_q_ref   : Cancels the voltage the q-axis current
                                    induces in the d-axis winding through
                                    mutual inductance.  Without this the
                                    d-axis would drift from zero whenever
                                    iq changes (speed changes, load steps).
      Kp_id * (0 - i_d_meas)     : Corrects residual id error from parameter
                                    mismatch and ADC noise.
                                    Kp_id = 0.4 V/A
                                    Closed-loop d-axis bandwidth:
                                    Kp_id / L_d = 0.4 / 368e-6 = 1087 rad/s

Q-axis voltage equation
------------------------
    vq = R_S * i_q_ref                      [resistive drop at reference current]
       + L_q * d(i_q_ref)/dt               [inductive drop for current ramp]
       + omega_e * lambda_pm               [back-EMF cancellation]
       + Kp_iq * (i_q_ref - i_q_meas)     [residual error correction]

    Units:  [Ohm] * [A]         = [V]
            [H]   * [A/s]       = [V]
            [rad/s] * [Wb]      = [V]
            [V/A]   * [A]       = [V]

    Physical meaning of each term:
      R_S * i_q_ref               : Voltage to overcome winding resistance
                                    at the commanded torque current.
      L_q * d(i_q_ref)/dt         : Voltage to ramp the current through the
                                    winding inductance.  d(i_q_ref)/dt is
                                    computed as a finite-difference and
                                    smoothed by a 1 kHz LPF (tau = 1 ms).
      omega_e * lambda_pm         : Back-EMF cancellation.  The motor
                                    generates this opposing voltage at speed;
                                    without cancellation it would appear as
                                    a steady-state vq error.
      Kp_iq * (i_q_ref - i_q_meas): Corrects for R mismatch (~20%), lambda_pm
                                    mismatch (~10%), and SMO back-EMF lag.
                                    Kp_iq = 8.0 V/A
                                    Closed-loop q-axis bandwidth:
                                    Kp_iq / L_q = 8.0 / 368e-6 = 21739 rad/s

Why no integrators?
--------------------
    The three flatness feedforward terms (R*iq, L*diq/dt, omega*lambda) supply
    the exact steady-state vq needed to maintain constant speed against winding
    resistance and back-EMF.  There is no persistent tracking error for an
    integrator to remove.  Adding integrators would risk wind-up during the
    20 ms SMO warmup transient.

Voltage saturation
-------------------
    If ||[vd, vq]|| > V_DC / sqrt(3) = 17.0 / 1.732 = 9.81 V, both components
    are scaled proportionally:
        scale = V_MAX / ||[vd, vq]||
        vd *= scale
        vq *= scale
    Proportional scaling preserves the id/iq ratio (MTPA current angle).


================================================================================
PART 3 — FULL SIGNAL CHAIN (one 50 us ISR step)
================================================================================

    Inputs:   omega_ref_mech [rad/s]   speed reference from host
              theta_m        [rad]     encoder angle from GTM TIM
              ia, ib, ic     [A]       phase currents from EVADC

    Step 1.  Clarke transform (abc -> alphabeta)
                 i_alpha, i_beta = Clarke(ia, ib, ic)

    Step 2.  SMO step (uses voltages from PREVIOUS step, z-1 delay)
                 omega_smo_e = SMO(v_alpha_prev, v_beta_prev, i_alpha, i_beta)

    Step 3.  SpeedFusion
                 theta_e, omega_e, omega_meas_mech =
                     SpeedFusion(theta_m, omega_smo_e)

    Step 4.  Speed P-loop
                 i_q_ref = clamp(Kp_speed * (omega_ref - omega_meas_mech),
                                 -I_MAX, +I_MAX)
                 Kp_speed = 0.4 A/(rad/s)
                 I_MAX    = 3.57 A

    Step 5.  Current derivative LPF
                 d(i_q_ref)/dt = LPF( (i_q_ref - i_q_ref_prev) / dt )
                 LPF tau = 1 ms,  clamped to +-3570 A/s

    Step 6.  Park transform (alphabeta -> dq, using theta_e from encoder)
                 i_d_meas, i_q_meas = Park(i_alpha, i_beta, theta_e)

    Step 7.  Flatness voltage law
                 vd, vq = DFC_VoltageLaw(i_q_ref, di_q/dt,
                                         i_d_meas, i_q_meas, omega_e)

    Step 8.  Inverse Park (dq -> alphabeta)
                 v_alpha, v_beta = InvPark(vd, vq, theta_e)

    Step 9.  Latch voltages for next step's SMO (z-1 delay)
                 v_alpha_prev = v_alpha
                 v_beta_prev  = v_beta

    Outputs:  v_alpha, v_beta [V]  ->  SVPWM  ->  GTM PWM duty cycles


================================================================================
PART 4 — MOTOR PARAMETERS (NANOTEC DB42S02)
================================================================================

    P_POLES    = 4                  pole pairs              [-]
    R_S        = 0.285              stator resistance       [Ohm]
    L_D = L_Q  = 0.3675e-3         phase inductance        [H]    (SPMSM: Ld=Lq)
    LAMBDA_PM  = 0.0014             flux linkage            [Wb]
    I_MAX      = 3.57               peak phase current      [A]
    V_DC       = 17.0               DC bus voltage          [V]    (bench supply)
    V_MAX      = 9.81               max phase voltage       [V]    = V_DC/sqrt(3)
    J          = 2.4e-6             rotor inertia           [kg*m^2]
    dt         = 50e-6              ISR period              [s]    (20 kHz)


================================================================================
PART 5 — GAIN SUMMARY
================================================================================

    Kp_speed   = 0.4    [A/(rad/s)]  speed P-loop;  saturates at 8.9 rad/s error
    Kp_id      = 0.4    [V/A]        d-axis P-gain; BW = Kp_id/Ld = 1087 rad/s
    Kp_iq      = 8.0    [V/A]        q-axis P-gain; BW = Kp_iq/Lq = 21739 rad/s
    SMO_K      = 2.0    [V]          switching gain; > omega_e_max*lambda_pm=1.29V
    SMO_TAU_E  = 0.2e-3 [s]          back-EMF LPF;  corner = 796 Hz
    DIQ_TAU    = 1.0e-3 [s]          diq/dt LPF;    corner = 159 Hz


================================================================================
PART 6 — FILES
================================================================================

    embed_sim_dfc_controller.h          Controller API and state struct
    embed_sim_dfc_controller.c          Full C implementation (AURIX firmware)
    embed_sim_dfc_gains.h               Compile-time gain constants
    diff_flatness_controller_block.py   EmbedSim Python simulation block
    db42s02_closed_loop_dfc_20k.py      Closed-loop simulation with AURIX
                                        noise model (ADC, encoder, dead-time,
                                        bus ripple)


================================================================================
PART 7 — RUNNING THE SIMULATION
================================================================================

    cd examples/fs_electrical_example
    python db42s02_closed_loop_dfc_20k.py

    Outputs:
        db42s02_dfc_foc_20k_results.png   -- 3x3 plot (speed, currents, noise)
        db42s02_dfc_topology.html         -- interactive block diagram
        embedsim_gen/embedsim_step.c      -- generated AURIX C loop
        embedsim_gen/embedsim_step.h


================================================================================