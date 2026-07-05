################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Configurations/Ifx_Cfg_Ssw.c \
../Configurations/Ifx_Cfg_SswBmhd.c 

C_DEPS += \
./Configurations/Ifx_Cfg_Ssw.d \
./Configurations/Ifx_Cfg_SswBmhd.d 

OBJS += \
./Configurations/Ifx_Cfg_Ssw.o \
./Configurations/Ifx_Cfg_SswBmhd.o 


# Each subdirectory must supply rules for building sources it contributes
Configurations/%.o: ../Configurations/%.c Configurations/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: AURIX GCC Compiler'
	tricore-elf-gcc -std=c99 "@C:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (GCC)/AURIX_GCC_Compiler-Include_paths__-I_.opt" -Og -g3 -gdwarf-3 -Wall -c -fmessage-length=0 -fno-common -fstrict-volatile-bitfields -fdata-sections -ffunction-sections -mtc162 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Configurations

clean-Configurations:
	-$(RM) ./Configurations/Ifx_Cfg_Ssw.d ./Configurations/Ifx_Cfg_Ssw.o ./Configurations/Ifx_Cfg_SswBmhd.d ./Configurations/Ifx_Cfg_SswBmhd.o

.PHONY: clean-Configurations

