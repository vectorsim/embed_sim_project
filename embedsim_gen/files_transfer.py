import os
import shutil

# Directories
dir_a = r"C:\EmbedSimProject\embed_sim_project\embedsim_gen"
dir_b = r"C:\Aurix_EmbedSim\AURIX_TC3x_Motor_Control_SDK\AppSw\PmsmFoc\EmbedSim"

# List of file names (only names, not full paths)
file_list = [
    "embedsim_step.h",
    "embedsim_step.c",
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