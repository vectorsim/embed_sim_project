# FIXME: before you push into master...
RUNTIMEDIR=C:/Program Files/OpenModelica1.26.3-64bit/include/omc/c/
#COPY_RUNTIMEFILES=$(FMI_ME_OBJS:%= && (OMCFILE=% && cp $(RUNTIMEDIR)/$$OMCFILE.c $$OMCFILE.c))

fmu:
	rm -f 642.fmutmp/sources/PMSM_Motor_init.xml
	cp -a "C:/Program Files/OpenModelica1.26.3-64bit/share/omc/runtime/c/fmi/buildproject/"* 642.fmutmp/sources
	cp -a PMSM_Motor_FMU.libs 642.fmutmp/sources/

