import os
import shutil

# Directories
dir_a = r"C:\EmbedSimProject\embed_sim_project\fs_electrical_machines\c_src"
dir_b = r"C:\Aurix_EmbedSim\PMSM\EmbedSim"

# List of file names (only names, not full paths)
file_list = [
    "Sys_Types.h",
    "Matrix.h",
    "Matrix.c",
    "motor_utility_blocks.h",
    "motor_utility_blocks.c",
    "coordinate_transform.h",
    "coordinate_transform.c",
    "svpwm.h",
    "svpwm.c",

]

def copy_if_exists(src_dir, dst_dir, filename):
    src = os.path.join(src_dir, filename)
    dst = os.path.join(dst_dir, filename)

    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"Missing: {src}")

for f in file_list:
    # A → B
    copy_if_exists(dir_a, dir_b, f)

    # B → A
   # copy_if_exists(dir_b, dir_a, f)