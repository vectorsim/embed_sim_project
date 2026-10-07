################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm.c \
../Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Atom.c \
../Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Cmu.c \
../Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Dpll.c \
../Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Dtm.c \
../Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Psm.c \
../Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Spe.c \
../Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tbu.c \
../Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tim.c \
../Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tom.c 

C_DEPS += \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm.d \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Atom.d \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Cmu.d \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Dpll.d \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Dtm.d \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Psm.d \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Spe.d \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tbu.d \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tim.d \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tom.d 

OBJS += \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm.o \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Atom.o \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Cmu.o \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Dpll.o \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Dtm.o \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Psm.o \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Spe.o \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tbu.o \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tim.o \
./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tom.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/iLLD/TC3xx/Tricore/Gtm/Std/%.o: ../Libraries/iLLD/TC3xx/Tricore/Gtm/Std/%.c Libraries/iLLD/TC3xx/Tricore/Gtm/Std/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: AURIX GCC Compiler'
	tricore-elf-gcc -std=c99 "@C:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (GCC)/AURIX_GCC_Compiler-Include_paths__-I_.opt" -Og -g3 -gdwarf-3 -Wall -c -fmessage-length=0 -fno-common -fstrict-volatile-bitfields -fdata-sections -ffunction-sections -mtc162 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Gtm-2f-Std

clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Gtm-2f-Std:
	-$(RM) ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm.d ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm.o ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Atom.d ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Atom.o ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Cmu.d ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Cmu.o ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Dpll.d ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Dpll.o ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Dtm.d ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Dtm.o ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Psm.d ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Psm.o ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Spe.d ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Spe.o ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tbu.d ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tbu.o ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tim.d ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tim.o ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tom.d ./Libraries/iLLD/TC3xx/Tricore/Gtm/Std/IfxGtm_Tom.o

.PHONY: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Gtm-2f-Std

