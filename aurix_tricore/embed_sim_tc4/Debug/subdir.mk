################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../AllowAccess.c \
../Cpu0_Main.c \
../Cpu1_Main.c \
../Cpu2_Main.c \
../Cpu3_Main.c \
../Cpu4_Main.c \
../Cpu5_Main.c \
../PpuInterface.c 

C_DEPS += \
./AllowAccess.d \
./Cpu0_Main.d \
./Cpu1_Main.d \
./Cpu2_Main.d \
./Cpu3_Main.d \
./Cpu4_Main.d \
./Cpu5_Main.d \
./PpuInterface.d 

OBJS += \
./AllowAccess.o \
./Cpu0_Main.o \
./Cpu1_Main.o \
./Cpu2_Main.o \
./Cpu3_Main.o \
./Cpu4_Main.o \
./Cpu5_Main.o \
./PpuInterface.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Compiler'
	"C:\Infineon\AURIX-Configuration-Studio-1.0.24\eclipse\tools/ccache/ccache.exe" tricore-elf-gcc.exe -MMD -MT "$@" "@C:/Users/CSO212/ACS-v1.0.24-workspace/EmbedSim_TC4/EmbedSim_TC4_TC4/Debug/TriCore_GCC_C_Compiler-Include_paths__-I_.opt" -Og -fdata-sections -ffunction-sections -fstrict-volatile-bitfields -g -gdwarf-2 -Wall -std=gnu99 -Wa,-adhlns="$@.lst" -pipe -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -mtc18 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

AllowAccess: ./AllowAccess.o $(USER_OBJS) $(OBJS) subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Cpu0_Main: ./Cpu0_Main.o $(USER_OBJS) $(OBJS) subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Cpu1_Main: ./Cpu1_Main.o $(USER_OBJS) $(OBJS) subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Cpu2_Main: ./Cpu2_Main.o $(USER_OBJS) $(OBJS) subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Cpu3_Main: ./Cpu3_Main.o $(USER_OBJS) $(OBJS) subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Cpu4_Main: ./Cpu4_Main.o $(USER_OBJS) $(OBJS) subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Cpu5_Main: ./Cpu5_Main.o $(USER_OBJS) $(OBJS) subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

PpuInterface: ./PpuInterface.o $(USER_OBJS) $(OBJS) subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '


clean: clean--2e-

clean--2e-:
	-$(RM) ./AllowAccess ./AllowAccess.d ./AllowAccess.o ./Cpu0_Main ./Cpu0_Main.d ./Cpu0_Main.o ./Cpu1_Main ./Cpu1_Main.d ./Cpu1_Main.o ./Cpu2_Main ./Cpu2_Main.d ./Cpu2_Main.o ./Cpu3_Main ./Cpu3_Main.d ./Cpu3_Main.o ./Cpu4_Main ./Cpu4_Main.d ./Cpu4_Main.o ./Cpu5_Main ./Cpu5_Main.d ./Cpu5_Main.o ./PpuInterface ./PpuInterface.d ./PpuInterface.o

.PHONY: clean--2e-

