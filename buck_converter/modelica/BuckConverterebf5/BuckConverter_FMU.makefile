# FIXME: before you push into master...
RUNTIMEDIR=/usr/bin/../include/omc/c/
#COPY_RUNTIMEFILES=$(FMI_ME_OBJS:%= && (OMCFILE=% && cp $(RUNTIMEDIR)/$$OMCFILE.c $$OMCFILE.c))

fmu:
	rm -f 330.fmutmp/sources/BuckConverter_init.xml
	cp -a "/usr/bin/../share/omc/runtime/c/fmi/buildproject/"* 330.fmutmp/sources
	cp -a BuckConverter_FMU.libs 330.fmutmp/sources/

