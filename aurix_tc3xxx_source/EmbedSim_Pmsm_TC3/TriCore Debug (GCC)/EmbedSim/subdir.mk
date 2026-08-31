################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../EmbedSim/embed_sim_coordinate_transform.c \
../EmbedSim/embed_sim_dfc_controller.c \
../EmbedSim/embed_sim_matrix.c \
../EmbedSim/embed_sim_motor_utility_blocks.c \
../EmbedSim/embed_sim_mpc_controller.c \
../EmbedSim/embed_sim_smc_controller.c \
../EmbedSim/embed_sim_sv_pwm.c \
../EmbedSim/embedsim_step.c 

C_DEPS += \
./EmbedSim/embed_sim_coordinate_transform.d \
./EmbedSim/embed_sim_dfc_controller.d \
./EmbedSim/embed_sim_matrix.d \
./EmbedSim/embed_sim_motor_utility_blocks.d \
./EmbedSim/embed_sim_mpc_controller.d \
./EmbedSim/embed_sim_smc_controller.d \
./EmbedSim/embed_sim_sv_pwm.d \
./EmbedSim/embedsim_step.d 

OBJS += \
./EmbedSim/embed_sim_coordinate_transform.o \
./EmbedSim/embed_sim_dfc_controller.o \
./EmbedSim/embed_sim_matrix.o \
./EmbedSim/embed_sim_motor_utility_blocks.o \
./EmbedSim/embed_sim_mpc_controller.o \
./EmbedSim/embed_sim_smc_controller.o \
./EmbedSim/embed_sim_sv_pwm.o \
./EmbedSim/embedsim_step.o 


# Each subdirectory must supply rules for building sources it contributes
EmbedSim/%.o: ../EmbedSim/%.c EmbedSim/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: AURIX GCC Compiler'
	tricore-elf-gcc -std=c99 "@C:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (GCC)/AURIX_GCC_Compiler-Include_paths__-I_.opt" -Og -g3 -gdwarf-3 -Wall -c -fmessage-length=0 -fno-common -fstrict-volatile-bitfields -fdata-sections -ffunction-sections -mtc162 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-EmbedSim

clean-EmbedSim:
	-$(RM) ./EmbedSim/embed_sim_coordinate_transform.d ./EmbedSim/embed_sim_coordinate_transform.o ./EmbedSim/embed_sim_dfc_controller.d ./EmbedSim/embed_sim_dfc_controller.o ./EmbedSim/embed_sim_matrix.d ./EmbedSim/embed_sim_matrix.o ./EmbedSim/embed_sim_motor_utility_blocks.d ./EmbedSim/embed_sim_motor_utility_blocks.o ./EmbedSim/embed_sim_mpc_controller.d ./EmbedSim/embed_sim_mpc_controller.o ./EmbedSim/embed_sim_smc_controller.d ./EmbedSim/embed_sim_smc_controller.o ./EmbedSim/embed_sim_sv_pwm.d ./EmbedSim/embed_sim_sv_pwm.o ./EmbedSim/embedsim_step.d ./EmbedSim/embedsim_step.o

.PHONY: clean-EmbedSim

