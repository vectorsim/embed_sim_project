################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Atom.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Cmu.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Dtm.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Spe.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tbu.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tim.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tom.c 

C_DEPS += \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Atom.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Cmu.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Dtm.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Spe.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tbu.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tim.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tom.d 

OBJS += \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Atom.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Cmu.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Dtm.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Spe.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tbu.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tim.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tom.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/%.o: ../Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/%.c Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Compiler'
	"C:\Infineon\AURIX-Configuration-Studio-1.0.24\eclipse\tools/ccache/ccache.exe" tricore-elf-gcc.exe -MMD -MT "$@" "@C:/Users/CSO212/ACS-v1.0.24-workspace/EmbedSim_TC4/EmbedSim_TC4_TC4/Debug/TriCore_GCC_C_Compiler-Include_paths__-I_.opt" -Og -fdata-sections -ffunction-sections -fstrict-volatile-bitfields -g -gdwarf-2 -Wall -std=gnu99 -Wa,-adhlns="$@.lst" -pipe -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -mtc18 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm: ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Atom: ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Atom.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Cmu: ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Cmu.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Dtm: ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Dtm.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Spe: ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Spe.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tbu: ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tbu.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tim: ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tim.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tom: ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tom.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-iLLD-2f-TC4xx-2f-CpuGeneric-2f-Egtm-2f-Std

clean-Libraries-2f-iLLD-2f-TC4xx-2f-CpuGeneric-2f-Egtm-2f-Std:
	-$(RM) ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm.d ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm.o ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Atom ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Atom.d ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Atom.o ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Cmu ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Cmu.d ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Cmu.o ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Dtm ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Dtm.d ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Dtm.o ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Spe ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Spe.d ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Spe.o ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tbu ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tbu.d ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tbu.o ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tim ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tim.d ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tim.o ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tom ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tom.d ./Libraries/iLLD/TC4xx/CpuGeneric/Egtm/Std/IfxEgtm_Tom.o

.PHONY: clean-Libraries-2f-iLLD-2f-TC4xx-2f-CpuGeneric-2f-Egtm-2f-Std

