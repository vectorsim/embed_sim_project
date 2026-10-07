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
	@echo 'Invoking: TriCore GCC C Compiler'
	"C:\Infineon\AURIX-Configuration-Studio-1.0.24\eclipse\tools/ccache/ccache.exe" tricore-elf-gcc.exe -MMD -MT "$@" "@C:/Users/CSO212/ACS-v1.0.24-workspace/EmbedSim_TC4/EmbedSim_TC4_TC4/Debug/TriCore_GCC_C_Compiler-Include_paths__-I_.opt" -Og -fdata-sections -ffunction-sections -fstrict-volatile-bitfields -g -gdwarf-2 -Wall -std=gnu99 -Wa,-adhlns="$@.lst" -pipe -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -mtc18 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Configurations/Ifx_Cfg_Ssw: ./Configurations/Ifx_Cfg_Ssw.o $(USER_OBJS) $(OBJS) Configurations/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Configurations/Ifx_Cfg_SswBmhd: ./Configurations/Ifx_Cfg_SswBmhd.o $(USER_OBJS) $(OBJS) Configurations/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Configurations

clean-Configurations:
	-$(RM) ./Configurations/Ifx_Cfg_Ssw ./Configurations/Ifx_Cfg_Ssw.d ./Configurations/Ifx_Cfg_Ssw.o ./Configurations/Ifx_Cfg_SswBmhd ./Configurations/Ifx_Cfg_SswBmhd.d ./Configurations/Ifx_Cfg_SswBmhd.o

.PHONY: clean-Configurations

