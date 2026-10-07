################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/Service/CpuGeneric/SysSe/General/Ifx_GlobalResources.c 

C_DEPS += \
./Libraries/Service/CpuGeneric/SysSe/General/Ifx_GlobalResources.d 

OBJS += \
./Libraries/Service/CpuGeneric/SysSe/General/Ifx_GlobalResources.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/Service/CpuGeneric/SysSe/General/%.o: ../Libraries/Service/CpuGeneric/SysSe/General/%.c Libraries/Service/CpuGeneric/SysSe/General/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: AURIX GCC Compiler'
	tricore-elf-gcc -std=c99 "@C:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (GCC)/AURIX_GCC_Compiler-Include_paths__-I_.opt" -Og -g3 -gdwarf-3 -Wall -c -fmessage-length=0 -fno-common -fstrict-volatile-bitfields -fdata-sections -ffunction-sections -mtc162 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-Service-2f-CpuGeneric-2f-SysSe-2f-General

clean-Libraries-2f-Service-2f-CpuGeneric-2f-SysSe-2f-General:
	-$(RM) ./Libraries/Service/CpuGeneric/SysSe/General/Ifx_GlobalResources.d ./Libraries/Service/CpuGeneric/SysSe/General/Ifx_GlobalResources.o

.PHONY: clean-Libraries-2f-Service-2f-CpuGeneric-2f-SysSe-2f-General

