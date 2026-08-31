################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/iLLD/TC3xx/Tricore/Eray/Eray/IfxEray_Eray.c 

C_DEPS += \
./Libraries/iLLD/TC3xx/Tricore/Eray/Eray/IfxEray_Eray.d 

OBJS += \
./Libraries/iLLD/TC3xx/Tricore/Eray/Eray/IfxEray_Eray.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/iLLD/TC3xx/Tricore/Eray/Eray/%.o: ../Libraries/iLLD/TC3xx/Tricore/Eray/Eray/%.c Libraries/iLLD/TC3xx/Tricore/Eray/Eray/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: AURIX GCC Compiler'
	tricore-elf-gcc -std=c99 "@C:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (GCC)/AURIX_GCC_Compiler-Include_paths__-I_.opt" -Og -g3 -gdwarf-3 -Wall -c -fmessage-length=0 -fno-common -fstrict-volatile-bitfields -fdata-sections -ffunction-sections -mtc162 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Eray-2f-Eray

clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Eray-2f-Eray:
	-$(RM) ./Libraries/iLLD/TC3xx/Tricore/Eray/Eray/IfxEray_Eray.d ./Libraries/iLLD/TC3xx/Tricore/Eray/Eray/IfxEray_Eray.o

.PHONY: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Eray-2f-Eray

