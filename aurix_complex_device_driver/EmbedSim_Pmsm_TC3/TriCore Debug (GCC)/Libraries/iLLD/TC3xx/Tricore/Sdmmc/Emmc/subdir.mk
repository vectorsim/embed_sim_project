################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/iLLD/TC3xx/Tricore/Sdmmc/Emmc/IfxSdmmc_Emmc.c 

C_DEPS += \
./Libraries/iLLD/TC3xx/Tricore/Sdmmc/Emmc/IfxSdmmc_Emmc.d 

OBJS += \
./Libraries/iLLD/TC3xx/Tricore/Sdmmc/Emmc/IfxSdmmc_Emmc.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/iLLD/TC3xx/Tricore/Sdmmc/Emmc/%.o: ../Libraries/iLLD/TC3xx/Tricore/Sdmmc/Emmc/%.c Libraries/iLLD/TC3xx/Tricore/Sdmmc/Emmc/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: AURIX GCC Compiler'
	tricore-elf-gcc -std=c99 "@C:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (GCC)/AURIX_GCC_Compiler-Include_paths__-I_.opt" -Og -g3 -gdwarf-3 -Wall -c -fmessage-length=0 -fno-common -fstrict-volatile-bitfields -fdata-sections -ffunction-sections -mtc162 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Sdmmc-2f-Emmc

clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Sdmmc-2f-Emmc:
	-$(RM) ./Libraries/iLLD/TC3xx/Tricore/Sdmmc/Emmc/IfxSdmmc_Emmc.d ./Libraries/iLLD/TC3xx/Tricore/Sdmmc/Emmc/IfxSdmmc_Emmc.o

.PHONY: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Sdmmc-2f-Emmc

