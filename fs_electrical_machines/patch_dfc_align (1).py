"""
patch_dfc_align.py  (v2 -- line-based, encoding-safe)
Run once:  python patch_dfc_align.py
"""
import sys
from pathlib import Path

CANDIDATES = [
    Path(r"C:\EmbedSimProject\embed_sim_project\fs_electrical_machines\c_src\diff_flatness_controller_block.py"),
    Path(r"C:\EmbedSimProject\embed_sim_project\fs_electrical_machines\diff_flatness_controller_block.py"),
]

NEW_VOLTAGE_LINE = "    _ALIGN_VOLTAGE       = 0.5    # V -- gentle pull: id_ss=1.75A\n"

NEW_ALIGN_BLOCK = """\
            if self._startup_state == self._STARTUP_ALIGN:
                # ALIGN: bypass DFC law entirely.
                # Fixed v_alpha=0.5V, v_beta=0 -- no feedforward, no windup.
                # id_ss = 0.5/0.285 = 1.75A pulls rotor to theta_e=0.
                _v_a = self._ALIGN_VOLTAGE
                _v_b = 0.0
                self._v_alpha_prev = _v_a
                self._v_beta_prev  = _v_b
                self.output = VectorSignal(
                    np.array([_v_a, _v_b], dtype=np.float32), self.name)
                self._log_ekf_step(t, omega_ref_mech, 0.0, 0.0, 0.0,
                                   0.0, omega_e_smo, omega_ekf_mech, _p_omega)
                if self._startup_timer >= self._ALIGN_TIME:
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
                return self.output
"""

targets = [p for p in CANDIDATES if p.exists()]
if not targets:
    print("ERROR: no target files found"); sys.exit(1)

for target in targets:
    print(f"\nPatching: {target}")
    content = target.read_text(encoding='utf-8')

    if 'ALIGN: bypass DFC law' in content:
        print("  Already patched."); continue

    # Patch 1: voltage constant
    if '= 2.0    # V' in content:
        content = content.replace(
            next(l for l in content.splitlines() if '_ALIGN_VOLTAGE' in l and '= 2.0' in l),
            "    _ALIGN_VOLTAGE       = 0.5    # V -- gentle pull: id_ss=1.75A")
        print("  Patched _ALIGN_VOLTAGE.")

    # Patch 2: find ALIGN block by scanning lines
    lines = content.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines)
                  if 'if self._startup_state == self._STARTUP_ALIGN:' in l
                  and l.startswith('            if')), None)
    if start is None:
        print("  ERROR: ALIGN start not found"); continue

    # Find end: line containing '_startup_state   = self._STARTUP_CLOSED_LOOP'
    end = next((i for i in range(start, len(lines))
                if '_startup_state   = self._STARTUP_CLOSED_LOOP' in lines[i]), None)
    if end is None:
        print("  ERROR: ALIGN end not found"); continue

    end += 1  # include the closing line
    # Skip trailing blank line if present
    if end < len(lines) and lines[end].strip() == '':
        end += 1

    print(f"  Replacing lines {start}-{end}")
    new_block = NEW_ALIGN_BLOCK.splitlines(keepends=True)
    new_block.append('\n')
    lines = lines[:start] + new_block + lines[end:]
    content = ''.join(lines)

    target.write_text(content, encoding='utf-8')
    ok = 'ALIGN: bypass DFC law' in target.read_text(encoding='utf-8')
    print(f"  Verification: {'OK' if ok else 'FAILED'}")

print("\nDone.")
