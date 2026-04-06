"""
patch_dfc_align.py
==================
Run this script ONCE from any directory.
It finds and patches diff_flatness_controller_block.py in-place.

Usage:
    python patch_dfc_align.py

It will:
  1. Locate the file automatically
  2. Verify the old text is present
  3. Replace it with the fixed version
  4. Print confirmation
"""

import sys
import os
from pathlib import Path

# ── Locate the file ───────────────────────────────────────────────────────────
CANDIDATES = [
    Path(r"C:\EmbedSimProject\embed_sim_project\fs_electrical_machines\c_src\diff_flatness_controller_block.py"),
    Path(r"C:\EmbedSimProject\embed_sim_project\fs_electrical_machines\diff_flatness_controller_block.py"),
]

targets = [p for p in CANDIDATES if p.exists()]
if not targets:
    print("ERROR: Could not find diff_flatness_controller_block.py in expected locations.")
    sys.exit(1)

# ── Patch text ────────────────────────────────────────────────────────────────
OLD_VOLTAGE = '    _ALIGN_VOLTAGE       = 2.0    # V — holds rotor at theta_e=0'
NEW_VOLTAGE  = '    _ALIGN_VOLTAGE       = 0.5    # V — gentle pull: id_ss=0.5/0.285=1.75A'

OLD_ALIGN = '''\
            if self._startup_state == self._STARTUP_ALIGN:
                # ALIGN: full FOC pipeline with theta_e=0 fixed.
                # Mirror omega_ref into omega_meas so speed error=0
                # → iq_ref=0.  Only id flows (MTPA), pulling rotor to
                # alpha axis.  No early return — let FOC compute v_alpha/v_beta.
                theta_e         = 0.0
                omega_e         = 0.0
                omega_meas_mech = omega_ref_mech  # speed error=0 → iq_ref=0
                if self._startup_timer >= self._ALIGN_TIME:
                    # Seed EKF from actual encoder angle — not assumed zero
                    theta_e_init = float(self.P_POLES) * theta_m
                    while theta_e_init >  math.pi: theta_e_init -= 2.0*math.pi
                    while theta_e_init < -math.pi: theta_e_init += 2.0*math.pi
                    self._ekf.x[0] = i_alpha
                    self._ekf.x[1] = i_beta
                    self._ekf.x[2] = 0.0
                    self._ekf.x[3] = theta_e_init
                    self._ekf.P[2, 2] = 1e4
                    self._ekf.P[3, 3] = 1e-4
                    self._ekf.P[0, 0] = 1e-2
                    self._ekf.P[1, 1] = 1e-2
                    self._ekf.theta_e_hat = theta_e_init
                    self._startup_state   = self._STARTUP_CLOSED_LOOP'''

NEW_ALIGN = '''\
            if self._startup_state == self._STARTUP_ALIGN:
                # ALIGN: bypass DFC law entirely.
                # Output fixed v_alpha=V_ALIGN, v_beta=0 directly.
                # No feedforward, no integrator windup.
                # id_ss = 0.5 / 0.285 = 1.75 A — pulls rotor to theta_e=0.
                _v_a = self._ALIGN_VOLTAGE
                _v_b = 0.0
                self._v_alpha_prev = _v_a
                self._v_beta_prev  = _v_b
                self.output = VectorSignal(
                    np.array([_v_a, _v_b], dtype=np.float32), self.name)
                self._log_ekf_step(t, omega_ref_mech, 0.0, 0.0, 0.0,
                                   0.0, omega_e_smo, omega_ekf_mech, _p_omega)
                if self._startup_timer >= self._ALIGN_TIME:
                    # Seed EKF from actual encoder angle
                    theta_e_init = float(self.P_POLES) * theta_m
                    while theta_e_init >  math.pi: theta_e_init -= 2.0*math.pi
                    while theta_e_init < -math.pi: theta_e_init += 2.0*math.pi
                    self._ekf.x[0] = i_alpha
                    self._ekf.x[1] = i_beta
                    self._ekf.x[2] = 0.0
                    self._ekf.x[3] = theta_e_init
                    self._ekf.P[2, 2] = 1e4
                    self._ekf.P[3, 3] = 1e-4
                    self._ekf.P[0, 0] = 1e-2
                    self._ekf.P[1, 1] = 1e-2
                    self._ekf.theta_e_hat = theta_e_init
                    self._startup_state   = self._STARTUP_CLOSED_LOOP
                return self.output'''

# ── Apply patch ───────────────────────────────────────────────────────────────
for target in targets:
    print(f"\nPatching: {target}")
    content = target.read_text(encoding='utf-8')

    already = 'ALIGN: bypass DFC law' in content
    if already:
        print("  Already patched — skipping.")
        continue

    if OLD_ALIGN not in content:
        print("  ERROR: expected ALIGN block not found. File may be a different version.")
        print("  Searching for '_STARTUP_ALIGN':", '_STARTUP_ALIGN' in content)
        continue

    content = content.replace(OLD_VOLTAGE, NEW_VOLTAGE, 1)
    content = content.replace(OLD_ALIGN,   NEW_ALIGN,   1)

    target.write_text(content, encoding='utf-8')
    print("  Patched successfully.")

    # Verify
    check = target.read_text(encoding='utf-8')
    if 'ALIGN: bypass DFC law' in check and '0.5' in check:
        print("  Verification: OK")
    else:
        print("  Verification: FAILED")

print("\nDone. Run the simulation now.")
