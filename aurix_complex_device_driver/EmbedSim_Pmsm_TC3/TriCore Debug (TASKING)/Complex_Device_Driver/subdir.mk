################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../Complex_Device_Driver/cdd_app.c" \
"../Complex_Device_Driver/cdd_asm_functions.c" \
"../Complex_Device_Driver/cdd_evadc_app.c" \
"../Complex_Device_Driver/cdd_gpio_app.c" \
"../Complex_Device_Driver/cdd_gpt12_app.c" \
"../Complex_Device_Driver/cdd_gtm_app.c" \
"../Complex_Device_Driver/cdd_motor_command_queue.c" \
"../Complex_Device_Driver/cdd_qspi_app.c" \
"../Complex_Device_Driver/cdd_stm_app.c" \
"../Complex_Device_Driver/cdd_sys_utility.c" \
"../Complex_Device_Driver/cdd_task_handler_app.c" \
"../Complex_Device_Driver/cdd_tle9180_app.c" 

COMPILED_SRCS += \
"Complex_Device_Driver/cdd_app.src" \
"Complex_Device_Driver/cdd_asm_functions.src" \
"Complex_Device_Driver/cdd_evadc_app.src" \
"Complex_Device_Driver/cdd_gpio_app.src" \
"Complex_Device_Driver/cdd_gpt12_app.src" \
"Complex_Device_Driver/cdd_gtm_app.src" \
"Complex_Device_Driver/cdd_motor_command_queue.src" \
"Complex_Device_Driver/cdd_qspi_app.src" \
"Complex_Device_Driver/cdd_stm_app.src" \
"Complex_Device_Driver/cdd_sys_utility.src" \
"Complex_Device_Driver/cdd_task_handler_app.src" \
"Complex_Device_Driver/cdd_tle9180_app.src" 

C_DEPS += \
"./Complex_Device_Driver/cdd_app.d" \
"./Complex_Device_Driver/cdd_asm_functions.d" \
"./Complex_Device_Driver/cdd_evadc_app.d" \
"./Complex_Device_Driver/cdd_gpio_app.d" \
"./Complex_Device_Driver/cdd_gpt12_app.d" \
"./Complex_Device_Driver/cdd_gtm_app.d" \
"./Complex_Device_Driver/cdd_motor_command_queue.d" \
"./Complex_Device_Driver/cdd_qspi_app.d" \
"./Complex_Device_Driver/cdd_stm_app.d" \
"./Complex_Device_Driver/cdd_sys_utility.d" \
"./Complex_Device_Driver/cdd_task_handler_app.d" \
"./Complex_Device_Driver/cdd_tle9180_app.d" 

OBJS += \
"Complex_Device_Driver/cdd_app.o" \
"Complex_Device_Driver/cdd_asm_functions.o" \
"Complex_Device_Driver/cdd_evadc_app.o" \
"Complex_Device_Driver/cdd_gpio_app.o" \
"Complex_Device_Driver/cdd_gpt12_app.o" \
"Complex_Device_Driver/cdd_gtm_app.o" \
"Complex_Device_Driver/cdd_motor_command_queue.o" \
"Complex_Device_Driver/cdd_qspi_app.o" \
"Complex_Device_Driver/cdd_stm_app.o" \
"Complex_Device_Driver/cdd_sys_utility.o" \
"Complex_Device_Driver/cdd_task_handler_app.o" \
"Complex_Device_Driver/cdd_tle9180_app.o" 


# Each subdirectory must supply rules for building sources it contributes
"Complex_Device_Driver/cdd_app.src":"../Complex_Device_Driver/cdd_app.c" "Complex_Device_Driver/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Complex_Device_Driver/cdd_app.o":"Complex_Device_Driver/cdd_app.src" "Complex_Device_Driver/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Complex_Device_Driver/cdd_asm_functions.src":"../Complex_Device_Driver/cdd_asm_functions.c" "Complex_Device_Driver/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Complex_Device_Driver/cdd_asm_functions.o":"Complex_Device_Driver/cdd_asm_functions.src" "Complex_Device_Driver/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Complex_Device_Driver/cdd_evadc_app.src":"../Complex_Device_Driver/cdd_evadc_app.c" "Complex_Device_Driver/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Complex_Device_Driver/cdd_evadc_app.o":"Complex_Device_Driver/cdd_evadc_app.src" "Complex_Device_Driver/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Complex_Device_Driver/cdd_gpio_app.src":"../Complex_Device_Driver/cdd_gpio_app.c" "Complex_Device_Driver/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Complex_Device_Driver/cdd_gpio_app.o":"Complex_Device_Driver/cdd_gpio_app.src" "Complex_Device_Driver/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Complex_Device_Driver/cdd_gpt12_app.src":"../Complex_Device_Driver/cdd_gpt12_app.c" "Complex_Device_Driver/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Complex_Device_Driver/cdd_gpt12_app.o":"Complex_Device_Driver/cdd_gpt12_app.src" "Complex_Device_Driver/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Complex_Device_Driver/cdd_gtm_app.src":"../Complex_Device_Driver/cdd_gtm_app.c" "Complex_Device_Driver/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Complex_Device_Driver/cdd_gtm_app.o":"Complex_Device_Driver/cdd_gtm_app.src" "Complex_Device_Driver/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Complex_Device_Driver/cdd_motor_command_queue.src":"../Complex_Device_Driver/cdd_motor_command_queue.c" "Complex_Device_Driver/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Complex_Device_Driver/cdd_motor_command_queue.o":"Complex_Device_Driver/cdd_motor_command_queue.src" "Complex_Device_Driver/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Complex_Device_Driver/cdd_qspi_app.src":"../Complex_Device_Driver/cdd_qspi_app.c" "Complex_Device_Driver/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Complex_Device_Driver/cdd_qspi_app.o":"Complex_Device_Driver/cdd_qspi_app.src" "Complex_Device_Driver/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Complex_Device_Driver/cdd_stm_app.src":"../Complex_Device_Driver/cdd_stm_app.c" "Complex_Device_Driver/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Complex_Device_Driver/cdd_stm_app.o":"Complex_Device_Driver/cdd_stm_app.src" "Complex_Device_Driver/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Complex_Device_Driver/cdd_sys_utility.src":"../Complex_Device_Driver/cdd_sys_utility.c" "Complex_Device_Driver/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Complex_Device_Driver/cdd_sys_utility.o":"Complex_Device_Driver/cdd_sys_utility.src" "Complex_Device_Driver/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Complex_Device_Driver/cdd_task_handler_app.src":"../Complex_Device_Driver/cdd_task_handler_app.c" "Complex_Device_Driver/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Complex_Device_Driver/cdd_task_handler_app.o":"Complex_Device_Driver/cdd_task_handler_app.src" "Complex_Device_Driver/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Complex_Device_Driver/cdd_tle9180_app.src":"../Complex_Device_Driver/cdd_tle9180_app.c" "Complex_Device_Driver/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Complex_Device_Driver/cdd_tle9180_app.o":"Complex_Device_Driver/cdd_tle9180_app.src" "Complex_Device_Driver/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-Complex_Device_Driver

clean-Complex_Device_Driver:
	-$(RM) ./Complex_Device_Driver/cdd_app.d ./Complex_Device_Driver/cdd_app.o ./Complex_Device_Driver/cdd_app.src ./Complex_Device_Driver/cdd_asm_functions.d ./Complex_Device_Driver/cdd_asm_functions.o ./Complex_Device_Driver/cdd_asm_functions.src ./Complex_Device_Driver/cdd_evadc_app.d ./Complex_Device_Driver/cdd_evadc_app.o ./Complex_Device_Driver/cdd_evadc_app.src ./Complex_Device_Driver/cdd_gpio_app.d ./Complex_Device_Driver/cdd_gpio_app.o ./Complex_Device_Driver/cdd_gpio_app.src ./Complex_Device_Driver/cdd_gpt12_app.d ./Complex_Device_Driver/cdd_gpt12_app.o ./Complex_Device_Driver/cdd_gpt12_app.src ./Complex_Device_Driver/cdd_gtm_app.d ./Complex_Device_Driver/cdd_gtm_app.o ./Complex_Device_Driver/cdd_gtm_app.src ./Complex_Device_Driver/cdd_motor_command_queue.d ./Complex_Device_Driver/cdd_motor_command_queue.o ./Complex_Device_Driver/cdd_motor_command_queue.src ./Complex_Device_Driver/cdd_qspi_app.d ./Complex_Device_Driver/cdd_qspi_app.o ./Complex_Device_Driver/cdd_qspi_app.src ./Complex_Device_Driver/cdd_stm_app.d ./Complex_Device_Driver/cdd_stm_app.o ./Complex_Device_Driver/cdd_stm_app.src ./Complex_Device_Driver/cdd_sys_utility.d ./Complex_Device_Driver/cdd_sys_utility.o ./Complex_Device_Driver/cdd_sys_utility.src ./Complex_Device_Driver/cdd_task_handler_app.d ./Complex_Device_Driver/cdd_task_handler_app.o ./Complex_Device_Driver/cdd_task_handler_app.src ./Complex_Device_Driver/cdd_tle9180_app.d ./Complex_Device_Driver/cdd_tle9180_app.o ./Complex_Device_Driver/cdd_tle9180_app.src

.PHONY: clean-Complex_Device_Driver

