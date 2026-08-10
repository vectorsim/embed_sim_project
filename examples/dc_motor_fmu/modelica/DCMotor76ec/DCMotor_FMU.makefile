# FIXME: before you push into master...
RUNTIMEDIR=/usr/bin/../include/omc/c/
#COPY_RUNTIMEFILES=$(FMI_ME_OBJS:%= && (OMCFILE=% && cp $(RUNTIMEDIR)/$$OMCFILE.c $$OMCFILE.c))

fmu:
	rm -f 229.fmutmp/sources/DCMotor_init.xml
	cp -a "/usr/bin/../share/omc/runtime/c/fmi/buildproject/"* 229.fmutmp/sources
	cp -a DCMotor_FMU.libs 229.fmutmp/sources/

