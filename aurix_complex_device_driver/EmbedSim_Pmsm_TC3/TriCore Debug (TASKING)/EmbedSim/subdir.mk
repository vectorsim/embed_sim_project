################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../EmbedSim/embed_sim_coordinate_transform.c" \
"../EmbedSim/embed_sim_dfc_controller.c" \
"../EmbedSim/embed_sim_matrix.c" \
"../EmbedSim/embed_sim_motor_utility_blocks.c" \
"../EmbedSim/embed_sim_mpc_controller.c" \
"../EmbedSim/embed_sim_smc_controller.c" \
"../EmbedSim/embed_sim_sv_pwm.c" \
"../EmbedSim/embedsim_step.c" 

COMPILED_SRCS += \
"EmbedSim/embed_sim_coordinate_transform.src" \
"EmbedSim/embed_sim_dfc_controller.src" \
"EmbedSim/embed_sim_matrix.src" \
"EmbedSim/embed_sim_motor_utility_blocks.src" \
"EmbedSim/embed_sim_mpc_controller.src" \
"EmbedSim/embed_sim_smc_controller.src" \
"EmbedSim/embed_sim_sv_pwm.src" \
"EmbedSim/embedsim_step.src" 

C_DEPS += \
"./EmbedSim/embed_sim_coordinate_transform.d" \
"./EmbedSim/embed_sim_dfc_controller.d" \
"./EmbedSim/embed_sim_matrix.d" \
"./EmbedSim/embed_sim_motor_utility_blocks.d" \
"./EmbedSim/embed_sim_mpc_controller.d" \
"./EmbedSim/embed_sim_smc_controller.d" \
"./EmbedSim/embed_sim_sv_pwm.d" \
"./EmbedSim/embedsim_step.d" 

OBJS += \
"EmbedSim/embed_sim_coordinate_transform.o" \
"EmbedSim/embed_sim_dfc_controller.o" \
"EmbedSim/embed_sim_matrix.o" \
"EmbedSim/embed_sim_motor_utility_blocks.o" \
"EmbedSim/embed_sim_mpc_controller.o" \
"EmbedSim/embed_sim_smc_controller.o" \
"EmbedSim/embed_sim_sv_pwm.o" \
"EmbedSim/embedsim_step.o" 


# Each subdirectory must supply rules for building sources it contributes
"EmbedSim/embed_sim_coordinate_transform.src":"../EmbedSim/embed_sim_coordinate_transform.c" "EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"EmbedSim/embed_sim_coordinate_transform.o":"EmbedSim/embed_sim_coordinate_transform.src" "EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"EmbedSim/embed_sim_dfc_controller.src":"../EmbedSim/embed_sim_dfc_controller.c" "EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"EmbedSim/embed_sim_dfc_controller.o":"EmbedSim/embed_sim_dfc_controller.src" "EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"EmbedSim/embed_sim_matrix.src":"../EmbedSim/embed_sim_matrix.c" "EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"EmbedSim/embed_sim_matrix.o":"EmbedSim/embed_sim_matrix.src" "EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"EmbedSim/embed_sim_motor_utility_blocks.src":"../EmbedSim/embed_sim_motor_utility_blocks.c" "EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"EmbedSim/embed_sim_motor_utility_blocks.o":"EmbedSim/embed_sim_motor_utility_blocks.src" "EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"EmbedSim/embed_sim_mpc_controller.src":"../EmbedSim/embed_sim_mpc_controller.c" "EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"EmbedSim/embed_sim_mpc_controller.o":"EmbedSim/embed_sim_mpc_controller.src" "EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"EmbedSim/embed_sim_smc_controller.src":"../EmbedSim/embed_sim_smc_controller.c" "EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"EmbedSim/embed_sim_smc_controller.o":"EmbedSim/embed_sim_smc_controller.src" "EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"EmbedSim/embed_sim_sv_pwm.src":"../EmbedSim/embed_sim_sv_pwm.c" "EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"EmbedSim/embed_sim_sv_pwm.o":"EmbedSim/embed_sim_sv_pwm.src" "EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"EmbedSim/embedsim_step.src":"../EmbedSim/embedsim_step.c" "EmbedSim/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"EmbedSim/embedsim_step.o":"EmbedSim/embedsim_step.src" "EmbedSim/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-EmbedSim

clean-EmbedSim:
	-$(RM) ./EmbedSim/embed_sim_coordinate_transform.d ./EmbedSim/embed_sim_coordinate_transform.o ./EmbedSim/embed_sim_coordinate_transform.src ./EmbedSim/embed_sim_dfc_controller.d ./EmbedSim/embed_sim_dfc_controller.o ./EmbedSim/embed_sim_dfc_controller.src ./EmbedSim/embed_sim_matrix.d ./EmbedSim/embed_sim_matrix.o ./EmbedSim/embed_sim_matrix.src ./EmbedSim/embed_sim_motor_utility_blocks.d ./EmbedSim/embed_sim_motor_utility_blocks.o ./EmbedSim/embed_sim_motor_utility_blocks.src ./EmbedSim/embed_sim_mpc_controller.d ./EmbedSim/embed_sim_mpc_controller.o ./EmbedSim/embed_sim_mpc_controller.src ./EmbedSim/embed_sim_smc_controller.d ./EmbedSim/embed_sim_smc_controller.o ./EmbedSim/embed_sim_smc_controller.src ./EmbedSim/embed_sim_sv_pwm.d ./EmbedSim/embed_sim_sv_pwm.o ./EmbedSim/embed_sim_sv_pwm.src ./EmbedSim/embedsim_step.d ./EmbedSim/embedsim_step.o ./EmbedSim/embedsim_step.src

.PHONY: clean-EmbedSim

