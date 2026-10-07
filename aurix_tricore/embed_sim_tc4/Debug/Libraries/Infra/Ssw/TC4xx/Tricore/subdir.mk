################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Infra.c \
../Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc0.c \
../Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc1.c \
../Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc2.c \
../Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc3.c \
../Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc4.c \
../Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc5.c 

C_DEPS += \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Infra.d \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc0.d \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc1.d \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc2.d \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc3.d \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc4.d \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc5.d 

OBJS += \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Infra.o \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc0.o \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc1.o \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc2.o \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc3.o \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc4.o \
./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc5.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/Infra/Ssw/TC4xx/Tricore/%.o: ../Libraries/Infra/Ssw/TC4xx/Tricore/%.c Libraries/Infra/Ssw/TC4xx/Tricore/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Compiler'
	"C:\Infineon\AURIX-Configuration-Studio-1.0.24\eclipse\tools/ccache/ccache.exe" tricore-elf-gcc.exe -MMD -MT "$@" "@C:/Users/CSO212/ACS-v1.0.24-workspace/EmbedSim_TC4/EmbedSim_TC4_TC4/Debug/TriCore_GCC_C_Compiler-Include_paths__-I_.opt" -Og -fdata-sections -ffunction-sections -fstrict-volatile-bitfields -g -gdwarf-2 -Wall -std=gnu99 -Wa,-adhlns="$@.lst" -pipe -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -mtc18 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Infra: ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Infra.o $(USER_OBJS) $(OBJS) Libraries/Infra/Ssw/TC4xx/Tricore/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc0: ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc0.o $(USER_OBJS) $(OBJS) Libraries/Infra/Ssw/TC4xx/Tricore/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc1: ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc1.o $(USER_OBJS) $(OBJS) Libraries/Infra/Ssw/TC4xx/Tricore/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc2: ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc2.o $(USER_OBJS) $(OBJS) Libraries/Infra/Ssw/TC4xx/Tricore/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc3: ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc3.o $(USER_OBJS) $(OBJS) Libraries/Infra/Ssw/TC4xx/Tricore/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc4: ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc4.o $(USER_OBJS) $(OBJS) Libraries/Infra/Ssw/TC4xx/Tricore/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc5: ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc5.o $(USER_OBJS) $(OBJS) Libraries/Infra/Ssw/TC4xx/Tricore/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-Infra-2f-Ssw-2f-TC4xx-2f-Tricore

clean-Libraries-2f-Infra-2f-Ssw-2f-TC4xx-2f-Tricore:
	-$(RM) ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Infra ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Infra.d ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Infra.o ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc0 ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc0.d ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc0.o ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc1 ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc1.d ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc1.o ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc2 ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc2.d ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc2.o ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc3 ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc3.d ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc3.o ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc4 ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc4.d ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc4.o ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc5 ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc5.d ./Libraries/Infra/Ssw/TC4xx/Tricore/Ifx_Ssw_Tc5.o

.PHONY: clean-Libraries-2f-Infra-2f-Ssw-2f-TC4xx-2f-Tricore

