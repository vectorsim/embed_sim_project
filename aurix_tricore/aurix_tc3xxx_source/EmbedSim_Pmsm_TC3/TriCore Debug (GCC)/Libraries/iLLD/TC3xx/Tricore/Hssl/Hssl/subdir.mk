################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/iLLD/TC3xx/Tricore/Hssl/Hssl/IfxHssl_Hssl.c 

C_DEPS += \
./Libraries/iLLD/TC3xx/Tricore/Hssl/Hssl/IfxHssl_Hssl.d 

OBJS += \
./Libraries/iLLD/TC3xx/Tricore/Hssl/Hssl/IfxHssl_Hssl.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/iLLD/TC3xx/Tricore/Hssl/Hssl/%.o: ../Libraries/iLLD/TC3xx/Tricore/Hssl/Hssl/%.c Libraries/iLLD/TC3xx/Tricore/Hssl/Hssl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: AURIX GCC Compiler'
	tricore-elf-gcc -std=c99 "@C:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (GCC)/AURIX_GCC_Compiler-Include_paths__-I_.opt" -Og -g3 -gdwarf-3 -Wall -c -fmessage-length=0 -fno-common -fstrict-volatile-bitfields -fdata-sections -ffunction-sections -mtc162 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Hssl-2f-Hssl

clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Hssl-2f-Hssl:
	-$(RM) ./Libraries/iLLD/TC3xx/Tricore/Hssl/Hssl/IfxHssl_Hssl.d ./Libraries/iLLD/TC3xx/Tricore/Hssl/Hssl/IfxHssl_Hssl.o

.PHONY: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Hssl-2f-Hssl

