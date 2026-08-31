################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/iLLD/TC3xx/Tricore/Mtu/Std/IfxMtu.c 

C_DEPS += \
./Libraries/iLLD/TC3xx/Tricore/Mtu/Std/IfxMtu.d 

OBJS += \
./Libraries/iLLD/TC3xx/Tricore/Mtu/Std/IfxMtu.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/iLLD/TC3xx/Tricore/Mtu/Std/%.o: ../Libraries/iLLD/TC3xx/Tricore/Mtu/Std/%.c Libraries/iLLD/TC3xx/Tricore/Mtu/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: AURIX GCC Compiler'
	tricore-elf-gcc -std=c99 "@C:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (GCC)/AURIX_GCC_Compiler-Include_paths__-I_.opt" -Og -g3 -gdwarf-3 -Wall -c -fmessage-length=0 -fno-common -fstrict-volatile-bitfields -fdata-sections -ffunction-sections -mtc162 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Mtu-2f-Std

clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Mtu-2f-Std:
	-$(RM) ./Libraries/iLLD/TC3xx/Tricore/Mtu/Std/IfxMtu.d ./Libraries/iLLD/TC3xx/Tricore/Mtu/Std/IfxMtu.o

.PHONY: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Mtu-2f-Std

