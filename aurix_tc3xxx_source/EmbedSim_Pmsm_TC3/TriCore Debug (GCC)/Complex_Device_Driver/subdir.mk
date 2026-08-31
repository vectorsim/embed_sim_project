################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Complex_Device_Driver/cdd_app.c \
../Complex_Device_Driver/cdd_asm_functions.c \
../Complex_Device_Driver/cdd_evadc_app.c \
../Complex_Device_Driver/cdd_gate_driver_9180.c \
../Complex_Device_Driver/cdd_gpio_app.c \
../Complex_Device_Driver/cdd_gpt12_app.c \
../Complex_Device_Driver/cdd_gtm_app.c \
../Complex_Device_Driver/cdd_motor_command_queue.c \
../Complex_Device_Driver/cdd_qspi_app.c \
../Complex_Device_Driver/cdd_qspi_init.c \
../Complex_Device_Driver/cdd_stm_app.c \
../Complex_Device_Driver/cdd_sys_utility.c \
../Complex_Device_Driver/cdd_task_handler_app.c 

C_DEPS += \
./Complex_Device_Driver/cdd_app.d \
./Complex_Device_Driver/cdd_asm_functions.d \
./Complex_Device_Driver/cdd_evadc_app.d \
./Complex_Device_Driver/cdd_gate_driver_9180.d \
./Complex_Device_Driver/cdd_gpio_app.d \
./Complex_Device_Driver/cdd_gpt12_app.d \
./Complex_Device_Driver/cdd_gtm_app.d \
./Complex_Device_Driver/cdd_motor_command_queue.d \
./Complex_Device_Driver/cdd_qspi_app.d \
./Complex_Device_Driver/cdd_qspi_init.d \
./Complex_Device_Driver/cdd_stm_app.d \
./Complex_Device_Driver/cdd_sys_utility.d \
./Complex_Device_Driver/cdd_task_handler_app.d 

OBJS += \
./Complex_Device_Driver/cdd_app.o \
./Complex_Device_Driver/cdd_asm_functions.o \
./Complex_Device_Driver/cdd_evadc_app.o \
./Complex_Device_Driver/cdd_gate_driver_9180.o \
./Complex_Device_Driver/cdd_gpio_app.o \
./Complex_Device_Driver/cdd_gpt12_app.o \
./Complex_Device_Driver/cdd_gtm_app.o \
./Complex_Device_Driver/cdd_motor_command_queue.o \
./Complex_Device_Driver/cdd_qspi_app.o \
./Complex_Device_Driver/cdd_qspi_init.o \
./Complex_Device_Driver/cdd_stm_app.o \
./Complex_Device_Driver/cdd_sys_utility.o \
./Complex_Device_Driver/cdd_task_handler_app.o 


# Each subdirectory must supply rules for building sources it contributes
Complex_Device_Driver/%.o: ../Complex_Device_Driver/%.c Complex_Device_Driver/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: AURIX GCC Compiler'
	tricore-elf-gcc -std=c99 "@C:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (GCC)/AURIX_GCC_Compiler-Include_paths__-I_.opt" -Og -g3 -gdwarf-3 -Wall -c -fmessage-length=0 -fno-common -fstrict-volatile-bitfields -fdata-sections -ffunction-sections -mtc162 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Complex_Device_Driver

clean-Complex_Device_Driver:
	-$(RM) ./Complex_Device_Driver/cdd_app.d ./Complex_Device_Driver/cdd_app.o ./Complex_Device_Driver/cdd_asm_functions.d ./Complex_Device_Driver/cdd_asm_functions.o ./Complex_Device_Driver/cdd_evadc_app.d ./Complex_Device_Driver/cdd_evadc_app.o ./Complex_Device_Driver/cdd_gate_driver_9180.d ./Complex_Device_Driver/cdd_gate_driver_9180.o ./Complex_Device_Driver/cdd_gpio_app.d ./Complex_Device_Driver/cdd_gpio_app.o ./Complex_Device_Driver/cdd_gpt12_app.d ./Complex_Device_Driver/cdd_gpt12_app.o ./Complex_Device_Driver/cdd_gtm_app.d ./Complex_Device_Driver/cdd_gtm_app.o ./Complex_Device_Driver/cdd_motor_command_queue.d ./Complex_Device_Driver/cdd_motor_command_queue.o ./Complex_Device_Driver/cdd_qspi_app.d ./Complex_Device_Driver/cdd_qspi_app.o ./Complex_Device_Driver/cdd_qspi_init.d ./Complex_Device_Driver/cdd_qspi_init.o ./Complex_Device_Driver/cdd_stm_app.d ./Complex_Device_Driver/cdd_stm_app.o ./Complex_Device_Driver/cdd_sys_utility.d ./Complex_Device_Driver/cdd_sys_utility.o ./Complex_Device_Driver/cdd_task_handler_app.d ./Complex_Device_Driver/cdd_task_handler_app.o

.PHONY: clean-Complex_Device_Driver

