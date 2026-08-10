# FIXME: before you push into master...
RUNTIMEDIR=/usr/bin/../include/omc/c/
#COPY_RUNTIMEFILES=$(FMI_ME_OBJS:%= && (OMCFILE=% && cp $(RUNTIMEDIR)/$$OMCFILE.c $$OMCFILE.c))

fmu:
	rm -f 354.fmutmp/sources/PMSM_Plant_FMU_init.xml
	cp -a "/usr/bin/../share/omc/runtime/c/fmi/buildproject/"* 354.fmutmp/sources
	cp -a PMSM_Plant_FMU_FMU.libs 354.fmutmp/sources/

