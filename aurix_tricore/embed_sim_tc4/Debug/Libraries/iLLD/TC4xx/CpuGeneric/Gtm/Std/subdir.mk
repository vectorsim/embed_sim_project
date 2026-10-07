################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Atom.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Cmu.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Dtm.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Psm.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Spe.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tbu.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tim.c \
../Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tom.c 

C_DEPS += \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Atom.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Cmu.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Dtm.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Psm.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Spe.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tbu.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tim.d \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tom.d 

OBJS += \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Atom.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Cmu.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Dtm.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Psm.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Spe.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tbu.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tim.o \
./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tom.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/%.o: ../Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/%.c Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Compiler'
	"C:\Infineon\AURIX-Configuration-Studio-1.0.24\eclipse\tools/ccache/ccache.exe" tricore-elf-gcc.exe -MMD -MT "$@" "@C:/Users/CSO212/ACS-v1.0.24-workspace/EmbedSim_TC4/EmbedSim_TC4_TC4/Debug/TriCore_GCC_C_Compiler-Include_paths__-I_.opt" -Og -fdata-sections -ffunction-sections -fstrict-volatile-bitfields -g -gdwarf-2 -Wall -std=gnu99 -Wa,-adhlns="$@.lst" -pipe -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -mtc18 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm: ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Atom: ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Atom.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Cmu: ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Cmu.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Dtm: ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Dtm.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Psm: ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Psm.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Spe: ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Spe.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tbu: ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tbu.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tim: ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tim.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tom: ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tom.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-iLLD-2f-TC4xx-2f-CpuGeneric-2f-Gtm-2f-Std

clean-Libraries-2f-iLLD-2f-TC4xx-2f-CpuGeneric-2f-Gtm-2f-Std:
	-$(RM) ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm.d ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm.o ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Atom ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Atom.d ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Atom.o ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Cmu ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Cmu.d ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Cmu.o ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Dtm ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Dtm.d ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Dtm.o ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Psm ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Psm.d ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Psm.o ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Spe ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Spe.d ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Spe.o ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tbu ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tbu.d ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tbu.o ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tim ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tim.d ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tim.o ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tom ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tom.d ./Libraries/iLLD/TC4xx/CpuGeneric/Gtm/Std/IfxGtm_Tom.o

.PHONY: clean-Libraries-2f-iLLD-2f-TC4xx-2f-CpuGeneric-2f-Gtm-2f-Std

