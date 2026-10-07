################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/iLLD/TC4xx/CpuGeneric/Asclin/Spi/IfxAsclin_Spi.c 

C_DEPS += \
./Libraries/iLLD/TC4xx/CpuGeneric/Asclin/Spi/IfxAsclin_Spi.d 

OBJS += \
./Libraries/iLLD/TC4xx/CpuGeneric/Asclin/Spi/IfxAsclin_Spi.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/iLLD/TC4xx/CpuGeneric/Asclin/Spi/%.o: ../Libraries/iLLD/TC4xx/CpuGeneric/Asclin/Spi/%.c Libraries/iLLD/TC4xx/CpuGeneric/Asclin/Spi/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Compiler'
	"C:\Infineon\AURIX-Configuration-Studio-1.0.24\eclipse\tools/ccache/ccache.exe" tricore-elf-gcc.exe -MMD -MT "$@" "@C:/Users/CSO212/ACS-v1.0.24-workspace/EmbedSim_TC4/EmbedSim_TC4_TC4/Debug/TriCore_GCC_C_Compiler-Include_paths__-I_.opt" -Og -fdata-sections -ffunction-sections -fstrict-volatile-bitfields -g -gdwarf-2 -Wall -std=gnu99 -Wa,-adhlns="$@.lst" -pipe -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -mtc18 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Asclin/Spi/IfxAsclin_Spi: ./Libraries/iLLD/TC4xx/CpuGeneric/Asclin/Spi/IfxAsclin_Spi.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Asclin/Spi/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-iLLD-2f-TC4xx-2f-CpuGeneric-2f-Asclin-2f-Spi

clean-Libraries-2f-iLLD-2f-TC4xx-2f-CpuGeneric-2f-Asclin-2f-Spi:
	-$(RM) ./Libraries/iLLD/TC4xx/CpuGeneric/Asclin/Spi/IfxAsclin_Spi ./Libraries/iLLD/TC4xx/CpuGeneric/Asclin/Spi/IfxAsclin_Spi.d ./Libraries/iLLD/TC4xx/CpuGeneric/Asclin/Spi/IfxAsclin_Spi.o

.PHONY: clean-Libraries-2f-iLLD-2f-TC4xx-2f-CpuGeneric-2f-Asclin-2f-Spi

