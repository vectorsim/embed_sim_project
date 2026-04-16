################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../AppSw/PmsmFoc/EmbedSim/embed_sim_coordinate_transform.c" \
"../AppSw/PmsmFoc/EmbedSim/embed_sim_dfc_controller.c" \
"../AppSw/PmsmFoc/EmbedSim/embed_sim_matrix.c" \
"../AppSw/PmsmFoc/EmbedSim/embed_sim_motor_utility_blocks.c" \
"../AppSw/PmsmFoc/EmbedSim/embed_sim_smc_controller.c" \
"../AppSw/PmsmFoc/EmbedSim/embed_sim_sv_pwm.c" \
"../AppSw/PmsmFoc/EmbedSim/embedsim_step.c" 

COMPILED_SRCS += \
"AppSw/PmsmFoc/EmbedSim/embed_sim_coordinate_transform.src" \
"AppSw/PmsmFoc/EmbedSim/embed_sim_dfc_controller.src" \
"AppSw/PmsmFoc/EmbedSim/embed_sim_matrix.src" \
"AppSw/PmsmFoc/EmbedSim/embed_sim_motor_utility_blocks.src" \
"AppSw/PmsmFoc/EmbedSim/embed_sim_smc_controller.src" \
"AppSw/PmsmFoc/EmbedSim/embed_sim_sv_pwm.src" \
"AppSw/PmsmFoc/EmbedSim/embedsim_step.src" 

C_DEPS += \
"./AppSw/PmsmFoc/EmbedSim/embed_sim_coordinate_transform.d" \
"./AppSw/PmsmFoc/EmbedSim/embed_sim_dfc_controller.d" \
"./AppSw/PmsmFoc/EmbedSim/embed_sim_matrix.d" \
"./AppSw/PmsmFoc/EmbedSim/embed_sim_motor_utility_blocks.d" \
"./AppSw/PmsmFoc/EmbedSim/embed_sim_smc_controller.d" \
"./AppSw/PmsmFoc/EmbedSim/embed_sim_sv_pwm.d" \
"./AppSw/PmsmFoc/EmbedSim/embedsim_step.d" 

OBJS += \
"AppSw/PmsmFoc/EmbedSim/embed_sim_coordinate_transform.o" \
"AppSw/PmsmFoc/EmbedSim/embed_sim_dfc_controller.o" \
"AppSw/PmsmFoc/EmbedSim/embed_sim_matrix.o" \
"AppSw/PmsmFoc/EmbedSim/embed_sim_motor_utility_blocks.o" \
"AppSw/PmsmFoc/EmbedSim/embed_sim_smc_controller.o" \
"AppSw/PmsmFoc/EmbedSim/embed_sim_sv_pwm.o" \
"AppSw/PmsmFoc/EmbedSim/embedsim_step.o" 


# Each subdirectory must supply rules for building sources it contributes
"AppSw/PmsmFoc/EmbedSim/embed_sim_coordinate_transform.src":"../AppSw/PmsmFoc/EmbedSim/embed_sim_coordinate_transform.c" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/EmbedSim/embed_sim_coordinate_transform.o":"AppSw/PmsmFoc/EmbedSim/embed_sim_coordinate_transform.src" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/EmbedSim/embed_sim_dfc_controller.src":"../AppSw/PmsmFoc/EmbedSim/embed_sim_dfc_controller.c" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/EmbedSim/embed_sim_dfc_controller.o":"AppSw/PmsmFoc/EmbedSim/embed_sim_dfc_controller.src" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/EmbedSim/embed_sim_matrix.src":"../AppSw/PmsmFoc/EmbedSim/embed_sim_matrix.c" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/EmbedSim/embed_sim_matrix.o":"AppSw/PmsmFoc/EmbedSim/embed_sim_matrix.src" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/EmbedSim/embed_sim_motor_utility_blocks.src":"../AppSw/PmsmFoc/EmbedSim/embed_sim_motor_utility_blocks.c" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/EmbedSim/embed_sim_motor_utility_blocks.o":"AppSw/PmsmFoc/EmbedSim/embed_sim_motor_utility_blocks.src" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/EmbedSim/embed_sim_smc_controller.src":"../AppSw/PmsmFoc/EmbedSim/embed_sim_smc_controller.c" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/EmbedSim/embed_sim_smc_controller.o":"AppSw/PmsmFoc/EmbedSim/embed_sim_smc_controller.src" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/EmbedSim/embed_sim_sv_pwm.src":"../AppSw/PmsmFoc/EmbedSim/embed_sim_sv_pwm.c" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/EmbedSim/embed_sim_sv_pwm.o":"AppSw/PmsmFoc/EmbedSim/embed_sim_sv_pwm.src" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/EmbedSim/embedsim_step.src":"../AppSw/PmsmFoc/EmbedSim/embedsim_step.c" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/EmbedSim/embedsim_step.o":"AppSw/PmsmFoc/EmbedSim/embedsim_step.src" "AppSw/PmsmFoc/EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-AppSw-2f-PmsmFoc-2f-EmbedSim

clean-AppSw-2f-PmsmFoc-2f-EmbedSim:
	-$(RM) ./AppSw/PmsmFoc/EmbedSim/embed_sim_coordinate_transform.d ./AppSw/PmsmFoc/EmbedSim/embed_sim_coordinate_transform.o ./AppSw/PmsmFoc/EmbedSim/embed_sim_coordinate_transform.src ./AppSw/PmsmFoc/EmbedSim/embed_sim_dfc_controller.d ./AppSw/PmsmFoc/EmbedSim/embed_sim_dfc_controller.o ./AppSw/PmsmFoc/EmbedSim/embed_sim_dfc_controller.src ./AppSw/PmsmFoc/EmbedSim/embed_sim_matrix.d ./AppSw/PmsmFoc/EmbedSim/embed_sim_matrix.o ./AppSw/PmsmFoc/EmbedSim/embed_sim_matrix.src ./AppSw/PmsmFoc/EmbedSim/embed_sim_motor_utility_blocks.d ./AppSw/PmsmFoc/EmbedSim/embed_sim_motor_utility_blocks.o ./AppSw/PmsmFoc/EmbedSim/embed_sim_motor_utility_blocks.src ./AppSw/PmsmFoc/EmbedSim/embed_sim_smc_controller.d ./AppSw/PmsmFoc/EmbedSim/embed_sim_smc_controller.o ./AppSw/PmsmFoc/EmbedSim/embed_sim_smc_controller.src ./AppSw/PmsmFoc/EmbedSim/embed_sim_sv_pwm.d ./AppSw/PmsmFoc/EmbedSim/embed_sim_sv_pwm.o ./AppSw/PmsmFoc/EmbedSim/embed_sim_sv_pwm.src ./AppSw/PmsmFoc/EmbedSim/embedsim_step.d ./AppSw/PmsmFoc/EmbedSim/embedsim_step.o ./AppSw/PmsmFoc/EmbedSim/embedsim_step.src

.PHONY: clean-AppSw-2f-PmsmFoc-2f-EmbedSim

