# FIXME: before you push into master...
RUNTIMEDIR=C:/Program Files/OpenModelica1.25.7-64bit/include/omc/c/
#COPY_RUNTIMEFILES=$(FMI_ME_OBJS:%= && (OMCFILE=% && cp $(RUNTIMEDIR)/$$OMCFILE.c $$OMCFILE.c))

fmu:
	rm -f 622.fmutmp/sources/ThreePhaseMotor_init.xml
	cp -a "C:/Program Files/OpenModelica1.25.7-64bit/share/omc/runtime/c/fmi/buildproject/"* 622.fmutmp/sources
	cp -a ThreePhaseMotor_FMU.libs 622.fmutmp/sources/

