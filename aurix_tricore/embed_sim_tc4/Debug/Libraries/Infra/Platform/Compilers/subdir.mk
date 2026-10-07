################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/Infra/Platform/Compilers/CompilerGcc.c \
../Libraries/Infra/Platform/Compilers/CompilerGhs.c \
../Libraries/Infra/Platform/Compilers/CompilerGnuc.c \
../Libraries/Infra/Platform/Compilers/CompilerHighTec.c \
../Libraries/Infra/Platform/Compilers/CompilerLlvm.c \
../Libraries/Infra/Platform/Compilers/CompilerMw.c \
../Libraries/Infra/Platform/Compilers/CompilerTasking.c \
../Libraries/Infra/Platform/Compilers/CompilerWindriver.c 

C_DEPS += \
./Libraries/Infra/Platform/Compilers/CompilerGcc.d \
./Libraries/Infra/Platform/Compilers/CompilerGhs.d \
./Libraries/Infra/Platform/Compilers/CompilerGnuc.d \
./Libraries/Infra/Platform/Compilers/CompilerHighTec.d \
./Libraries/Infra/Platform/Compilers/CompilerLlvm.d \
./Libraries/Infra/Platform/Compilers/CompilerMw.d \
./Libraries/Infra/Platform/Compilers/CompilerTasking.d \
./Libraries/Infra/Platform/Compilers/CompilerWindriver.d 

OBJS += \
./Libraries/Infra/Platform/Compilers/CompilerGcc.o \
./Libraries/Infra/Platform/Compilers/CompilerGhs.o \
./Libraries/Infra/Platform/Compilers/CompilerGnuc.o \
./Libraries/Infra/Platform/Compilers/CompilerHighTec.o \
./Libraries/Infra/Platform/Compilers/CompilerLlvm.o \
./Libraries/Infra/Platform/Compilers/CompilerMw.o \
./Libraries/Infra/Platform/Compilers/CompilerTasking.o \
./Libraries/Infra/Platform/Compilers/CompilerWindriver.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/Infra/Platform/Compilers/%.o: ../Libraries/Infra/Platform/Compilers/%.c Libraries/Infra/Platform/Compilers/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Compiler'
	"C:\Infineon\AURIX-Configuration-Studio-1.0.24\eclipse\tools/ccache/ccache.exe" tricore-elf-gcc.exe -MMD -MT "$@" "@C:/Users/CSO212/ACS-v1.0.24-workspace/EmbedSim_TC4/EmbedSim_TC4_TC4/Debug/TriCore_GCC_C_Compiler-Include_paths__-I_.opt" -Og -fdata-sections -ffunction-sections -fstrict-volatile-bitfields -g -gdwarf-2 -Wall -std=gnu99 -Wa,-adhlns="$@.lst" -pipe -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -mtc18 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Platform/Compilers/CompilerGcc: ./Libraries/Infra/Platform/Compilers/CompilerGcc.o $(USER_OBJS) $(OBJS) Libraries/Infra/Platform/Compilers/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Platform/Compilers/CompilerGhs: ./Libraries/Infra/Platform/Compilers/CompilerGhs.o $(USER_OBJS) $(OBJS) Libraries/Infra/Platform/Compilers/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Platform/Compilers/CompilerGnuc: ./Libraries/Infra/Platform/Compilers/CompilerGnuc.o $(USER_OBJS) $(OBJS) Libraries/Infra/Platform/Compilers/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Platform/Compilers/CompilerHighTec: ./Libraries/Infra/Platform/Compilers/CompilerHighTec.o $(USER_OBJS) $(OBJS) Libraries/Infra/Platform/Compilers/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Platform/Compilers/CompilerLlvm: ./Libraries/Infra/Platform/Compilers/CompilerLlvm.o $(USER_OBJS) $(OBJS) Libraries/Infra/Platform/Compilers/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Platform/Compilers/CompilerMw: ./Libraries/Infra/Platform/Compilers/CompilerMw.o $(USER_OBJS) $(OBJS) Libraries/Infra/Platform/Compilers/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Platform/Compilers/CompilerTasking: ./Libraries/Infra/Platform/Compilers/CompilerTasking.o $(USER_OBJS) $(OBJS) Libraries/Infra/Platform/Compilers/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Infra/Platform/Compilers/CompilerWindriver: ./Libraries/Infra/Platform/Compilers/CompilerWindriver.o $(USER_OBJS) $(OBJS) Libraries/Infra/Platform/Compilers/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-Infra-2f-Platform-2f-Compilers

clean-Libraries-2f-Infra-2f-Platform-2f-Compilers:
	-$(RM) ./Libraries/Infra/Platform/Compilers/CompilerGcc ./Libraries/Infra/Platform/Compilers/CompilerGcc.d ./Libraries/Infra/Platform/Compilers/CompilerGcc.o ./Libraries/Infra/Platform/Compilers/CompilerGhs ./Libraries/Infra/Platform/Compilers/CompilerGhs.d ./Libraries/Infra/Platform/Compilers/CompilerGhs.o ./Libraries/Infra/Platform/Compilers/CompilerGnuc ./Libraries/Infra/Platform/Compilers/CompilerGnuc.d ./Libraries/Infra/Platform/Compilers/CompilerGnuc.o ./Libraries/Infra/Platform/Compilers/CompilerHighTec ./Libraries/Infra/Platform/Compilers/CompilerHighTec.d ./Libraries/Infra/Platform/Compilers/CompilerHighTec.o ./Libraries/Infra/Platform/Compilers/CompilerLlvm ./Libraries/Infra/Platform/Compilers/CompilerLlvm.d ./Libraries/Infra/Platform/Compilers/CompilerLlvm.o ./Libraries/Infra/Platform/Compilers/CompilerMw ./Libraries/Infra/Platform/Compilers/CompilerMw.d ./Libraries/Infra/Platform/Compilers/CompilerMw.o ./Libraries/Infra/Platform/Compilers/CompilerTasking ./Libraries/Infra/Platform/Compilers/CompilerTasking.d ./Libraries/Infra/Platform/Compilers/CompilerTasking.o ./Libraries/Infra/Platform/Compilers/CompilerWindriver ./Libraries/Infra/Platform/Compilers/CompilerWindriver.d ./Libraries/Infra/Platform/Compilers/CompilerWindriver.o

.PHONY: clean-Libraries-2f-Infra-2f-Platform-2f-Compilers

