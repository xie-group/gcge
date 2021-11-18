.SUFFIXES: .cxx

CPP		= /usr/bin/mpicc -E
CPPFLAGS	= -I/home/hhxie/package/wzj/phg-0.9.6/include  -I/home/hhxie/package/wzj/petsc-3.15.3/include -I/home/hhxie/package/wzj/petsc-3.15.3/linux-gnu-c-debug/include -I/home/hhxie/package/wzj/slepc-3.15.1/include -I/home/hhxie/package/wzj/slepc-3.15.1/linux-gnu-c-debug/include -D__PHG__
BUILD_CPPFLAGS	= -DMPI_NO_CPPBIND   -DVTK_DIR="\"\""  -DGZIP_PROG="\"/usr/bin/gzip\"" -DBZIP2_PROG="\"/usr/bin/bzip2\""

CC		= /usr/bin/mpicc 
CFLAGS		= -g -O2 -Wall -Wmissing-field-initializers -fopenmp
BUILD_CFLAGS	= -fPIC
USER_CFLAGS	=

CXX		= /usr/bin/mpicxx 
CXXFLAGS	= -g -O2 -Wno-deprecated -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX -fopenmp
BUILD_CXXFLAGS	= -fPIC
USER_CXXFLAGS	=

FC		= /usr/bin/mpif90 
FCFLAGS		= -g -O2
USER_FCFLAGS	=

F77		= /usr/bin/mpif77 
FFLAGS		= -g -O2
USER_FFLAGS	=

LINKER		= /usr/bin/mpicxx 
LDFLAGS		= -L/home/hhxie/package/wzj/phg-0.9.6/src  -fopenmp -L/home/hhxie/package/wzj/petsc-3.15.3/linux-gnu-c-debug/lib -L/home/hhxie/package/wzj/slepc-3.15.1/linux-gnu-c-debug/lib
USER_LDFLAGS	=

LIBS		= -Wl,-rpath,/home/hhxie/package/wzj/phg-0.9.6/src -lphg  -Wl,-rpath,/home/hhxie/package/wzj/slepc-3.15.1/linux-gnu-c-debug/lib  -lslepc  -lsmumps -ldmumps -lmumps_common -lpord  -Wl,-rpath,/home/hhxie/package/wzj/petsc-3.15.3/linux-gnu-c-debug/lib -L/home/hhxie/package/wzj/petsc-3.15.3/linux-gnu-c-debug/lib -Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 -Wl,-rpath,/usr/lib/x86_64-linux-gnu -Wl,-rpath,/lib/x86_64-linux-gnu -lpetsc -lcmumps -ldmumps -lsmumps -lzmumps -lmumps_common -lpord -lscalapack -lsuperlu_dist -lflapack -lfblas -lmetis -lm -lX11 -lstdc++ -ldl -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lpthread -lquadmath -lstdc++ -ldl  -lm  -L/usr/lib/x86_64-linux-gnu/openmpi/lib -L/usr/lib/gcc/x86_64-linux-gnu/9 -L/usr/lib/gcc/x86_64-linux-gnu/9/../../../x86_64-linux-gnu -L/usr/lib/gcc/x86_64-linux-gnu/9/../../../../lib -L/lib/x86_64-linux-gnu -L/lib/../lib -L/usr/lib/x86_64-linux-gnu -L/usr/lib/../lib -L/usr/lib/gcc/x86_64-linux-gnu/9/../../.. -lgfortran -lm -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lquadmath -lpthread  -lgfortran -lm -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lquadmath -lpthread -Wl,-rpath,/usr/local/lib
USER_LIBS	=

AR		= /usr/bin/ar 
RANLIB		= /usr/bin/ranlib 
BUILD_SHARED	= gcc -shared
BUILD_SHARED_LIBS = 
LIB_SUFFIX	= .so

.c.o:
	${CC} ${CFLAGS} ${CPPFLAGS} ${USER_CFLAGS} -c $*.c

.cxx.o:
	${CXX} ${CXXFLAGS} ${CPPFLAGS} ${USER_CXXFLAGS} -c $*.cxx

.cpp.o:
	${CXX} ${CXXFLAGS} ${CPPFLAGS} ${USER_CXXFLAGS} -c $*.cpp

.C.o:
	${CXX} ${CXXFLAGS} ${CPPFLAGS} ${USER_CXXFLAGS} -c $*.C

.f.o:
	${FC} ${FCFLAGS} ${USER_FCFLAGS} -c $*.f

% : %.o
	${LINKER} ${USER_LDFLAGS} ${LDFLAGS} -o $@ $*.o ${USER_LIBS} ${LIBS}

% : %.c
	${CC} ${CFLAGS} ${CPPFLAGS} ${USER_CFLAGS} -c $*.c
	${LINKER} ${USER_LDFLAGS} ${LDFLAGS} -o $@ $*.o ${USER_LIBS} ${LIBS}

% : %.cxx
	${CXX} ${CXXFLAGS} ${CPPFLAGS} ${USER_CXXFLAGS} -c $*.cxx
	${CXX} ${USER_LDFLAGS} ${LDFLAGS} -o $@ $*.o ${USER_LIBS} ${LIBS}

% : %.cpp
	${CXX} ${CXXFLAGS} ${CPPFLAGS} ${USER_CXXFLAGS} -c $*.cpp
	${CXX} ${USER_LDFLAGS} ${LDFLAGS} -o $@ $*.o ${USER_LIBS} ${LIBS}

% : %.C
	${CXX} ${CXXFLAGS} ${CPPFLAGS} ${USER_CXXFLAGS} -c $*.C
	${CXX} ${USER_LDFLAGS} ${LDFLAGS} -o $@ $*.o ${USER_LIBS} ${LIBS}

% : %.f
	${FC} ${FCFLAGS} ${USER_FCFLAGS} -c $*.f
	${LINKER} ${USER_LDFLAGS} ${LDFLAGS} -o $@ $*.o ${USER_LIBS} ${LIBS}

CC:
	@echo $(CC)

CPP:
	@echo $(CPP)

CXX:
	@echo $(CXX)

FC:
	@echo $(FC)

CPPFLAGS:
	@echo $(CPPFLAGS)

CFLAGS:
	@echo $(CFLAGS)

CXXFLAGS:
	@echo $(CXXFLAGS)

FFLAGS:
	@echo $(FFLAGS)

FCFLAGS:
	@echo $(FCFLAGS)

LDFLAGS:
	@echo $(LDFLAGS)

LIBS:
	@echo $(LIBS)

LINKER:
	@echo $(LINKER)
