PETSCFLAGS = -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3
PETSCINC   = -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include
LIBPETSC   = -L${PETSC_DIR}/${PETSC_ARCH}/lib -Wl,-rpath,${PETSC_DIR}/${PETSC_ARCH}/lib -Wl,-rpath,/usr/local -L/usr/local -L/usr/local/lib -L/usr/lib/gcc/x86_64-linux-gnu/7 -L/usr/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -Wl,-rpath,/usr/local/lib -lpetsc -llapack -lblas -lpthread -lm -lstdc++ -ldl -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lstdc++ -ldl

SLEPCFLAGS = -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3
SLEPCINC   = -I${SLEPC_DIR}/include -I${SLEPC_DIR}/${PETSC_ARCH}/include -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include
LIBSLEPC   = -Wl,-rpath,${SLEPC_DIR}/${PETSC_ARCH}/lib -L${SLEPC_DIR}/${PETSC_ARCH}/lib -L${PETSC_DIR}/${PETSC_ARCH}/lib -Wl,-rpath,${PETSC_DIR}/${PETSC_ARCH}/lib -L/usr/local/lib -L/usr/lib/gcc/x86_64-linux-gnu/9 -L/usr/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -Wl,-rpath,/usr/local/lib -lslepc -lpetsc -llapack -lblas -lpthread -lm -lstdc++ -ldl -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lstdc++ -ldl

CPP      = mpic++ -D__DEBUG__
CC       = mpicc  -D__DEBUG__
