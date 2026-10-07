################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../complex_device_driver/cdd_app.c \
../complex_device_driver/cdd_egtm_app.c \
../complex_device_driver/cdd_gpio_app.c \
../complex_device_driver/cdd_sys_util.c 

C_DEPS += \
./complex_device_driver/cdd_app.d \
./complex_device_driver/cdd_egtm_app.d \
./complex_device_driver/cdd_gpio_app.d \
./complex_device_driver/cdd_sys_util.d 

OBJS += \
./complex_device_driver/cdd_app.o \
./complex_device_driver/cdd_egtm_app.o \
./complex_device_driver/cdd_gpio_app.o \
./complex_device_driver/cdd_sys_util.o 


# Each subdirectory must supply rules for building sources it contributes
complex_device_driver/%.o: ../complex_device_driver/%.c complex_device_driver/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Compiler'
	"C:\Infineon\AURIX-Configuration-Studio-1.0.24\eclipse\tools/ccache/ccache.exe" tricore-elf-gcc.exe -MMD -MT "$@" "@C:/Users/CSO212/ACS-v1.0.24-workspace/EmbedSim_TC4/EmbedSim_TC4_TC4/Debug/TriCore_GCC_C_Compiler-Include_paths__-I_.opt" -Og -fdata-sections -ffunction-sections -fstrict-volatile-bitfields -g -gdwarf-2 -Wall -std=gnu99 -Wa,-adhlns="$@.lst" -pipe -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -mtc18 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

complex_device_driver/cdd_app: ./complex_device_driver/cdd_app.o $(USER_OBJS) $(OBJS) complex_device_driver/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

complex_device_driver/cdd_egtm_app: ./complex_device_driver/cdd_egtm_app.o $(USER_OBJS) $(OBJS) complex_device_driver/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

complex_device_driver/cdd_gpio_app: ./complex_device_driver/cdd_gpio_app.o $(USER_OBJS) $(OBJS) complex_device_driver/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

complex_device_driver/cdd_sys_util: ./complex_device_driver/cdd_sys_util.o $(USER_OBJS) $(OBJS) complex_device_driver/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-complex_device_driver

clean-complex_device_driver:
	-$(RM) ./complex_device_driver/cdd_app ./complex_device_driver/cdd_app.d ./complex_device_driver/cdd_app.o ./complex_device_driver/cdd_egtm_app ./complex_device_driver/cdd_egtm_app.d ./complex_device_driver/cdd_egtm_app.o ./complex_device_driver/cdd_gpio_app ./complex_device_driver/cdd_gpio_app.d ./complex_device_driver/cdd_gpio_app.o ./complex_device_driver/cdd_sys_util ./complex_device_driver/cdd_sys_util.d ./complex_device_driver/cdd_sys_util.o

.PHONY: clean-complex_device_driver

