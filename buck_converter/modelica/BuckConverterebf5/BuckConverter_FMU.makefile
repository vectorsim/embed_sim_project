# FIXME: before you push into master...
RUNTIMEDIR=C:/Program Files/OpenModelica1.25.7-64bit/include/omc/c/
#COPY_RUNTIMEFILES=$(FMI_ME_OBJS:%= && (OMCFILE=% && cp $(RUNTIMEDIR)/$$OMCFILE.c $$OMCFILE.c))

fmu:
	rm -f 158.fmutmp/sources/BuckConverter_init.xml
	cp -a "C:/Program Files/OpenModelica1.25.7-64bit/share/omc/runtime/c/fmi/buildproject/"* 158.fmutmp/sources
	cp -a BuckConverter_FMU.libs 158.fmutmp/sources/

