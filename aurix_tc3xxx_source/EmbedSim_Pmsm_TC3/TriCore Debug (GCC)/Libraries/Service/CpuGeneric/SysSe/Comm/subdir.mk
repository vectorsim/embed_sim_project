################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.c \
../Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.c 

C_DEPS += \
./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.d \
./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.d 

OBJS += \
./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.o \
./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/Service/CpuGeneric/SysSe/Comm/%.o: ../Libraries/Service/CpuGeneric/SysSe/Comm/%.c Libraries/Service/CpuGeneric/SysSe/Comm/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: AURIX GCC Compiler'
	tricore-elf-gcc -std=c99 "@C:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (GCC)/AURIX_GCC_Compiler-Include_paths__-I_.opt" -Og -g3 -gdwarf-3 -Wall -c -fmessage-length=0 -fno-common -fstrict-volatile-bitfields -fdata-sections -ffunction-sections -mtc162 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-Service-2f-CpuGeneric-2f-SysSe-2f-Comm

clean-Libraries-2f-Service-2f-CpuGeneric-2f-SysSe-2f-Comm:
	-$(RM) ./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.d ./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.o ./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.d ./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.o

.PHONY: clean-Libraries-2f-Service-2f-CpuGeneric-2f-SysSe-2f-Comm

