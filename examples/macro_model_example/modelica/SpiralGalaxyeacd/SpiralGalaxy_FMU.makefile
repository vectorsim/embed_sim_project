# FIXME: before you push into master...
RUNTIMEDIR=C:/Program Files/OpenModelica1.25.7-64bit/include/omc/c/
#COPY_RUNTIMEFILES=$(FMI_ME_OBJS:%= && (OMCFILE=% && cp $(RUNTIMEDIR)/$$OMCFILE.c $$OMCFILE.c))

fmu:
	rm -f 102.fmutmp/sources/SpiralGalaxy_init.xml
	cp -a "C:/Program Files/OpenModelica1.25.7-64bit/share/omc/runtime/c/fmi/buildproject/"* 102.fmutmp/sources
	cp -a SpiralGalaxy_FMU.libs 102.fmutmp/sources/

