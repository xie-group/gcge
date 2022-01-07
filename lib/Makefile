# Project: TestOPS
#
# Makefile created by Dev-C++ 5.11

UNAME ?= $(shell uname)
VERSION := 3.0

OBJ      = ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_lapack.o ../app/app_ccs.o
LINKOBJ  = ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_lapack.o ../app/app_ccs.o

LIBBLASLAPACK = -lblas -llapack
LIB = -lpthread -lmpi -lm -lstdc++ -ldl
ifeq ($(UNAME), Darwin)
	LIB +=
else
	LIB += -lgfortran -lgcc_s -lquadmath
endif
LIBS = $(LIBBLASLAPACK) $(LIB)

INCS     = -I../src -I../app
CXXINCS  = 
CXXFLAGS = $(CXXINCS) -O2 -Wall -fPIC -g
CFLAGS   = $(INCS) -O2 -Wall -fPIC -g
RM       = rm -f
ifeq ($(UNAME), Darwin)
	CFLAGS +=
	SOLIB_SUFFIX ?= dylib
	SOLIB ?= libgcge.$(SOLIB_SUFFIX)
	SONAMELIB ?= libgcge.$(VERSION).$(SOLIB_SUFFIX)
	SOFLAGS ?= -dynamiclib -Wl,-install_name,$(SONAMELIB) -current_version $(VERSION) -Wl,-undefined -Wl,dynamic_lookup
	LDFLAGS ?= -Wl,-framework -Wl,Accelerate -m64
else
	CFLAGS += -fopenmp #-qopenmp -DMEMWATCH -DMW_STDIO ##
	SOLIB_SUFFIX ?= so
	SOLIB ?= libgcge.$(SOLIB_SUFFIX)
	SONAMELIB ?= libgcge.$(VERSION).$(SOLIB_SUFFIX)
	SOFLAGS ?= -shared -Wl,-soname,$(SONAMELIB)
	LDFLAGS ?=
endif

CPP      = mpic++ -D__DEBUG__
CC       = mpicc  -D__DEBUG__

all: $(OBJ)
	@echo "Building libgcge.$(SOLIB_SUFFIX) ..."
	@$(CC) $(SOFLAGS) -o $(SONAMELIB) $(OBJ) $(LIBS) $(LDFLAGS)
ifneq ($(SONAMELIB), $(SOLIB))
	@ln -sf $(SONAMELIB) $(SOLIB)
endif

test_eig_sol_gcg.o: test_eig_sol_gcg.c
	$(CC) -c test_eig_sol_gcg.c -o test_eig_sol_gcg.o $(CFLAGS)

test_lin_sol.o: test_lin_sol.c
	$(CC) -c test_lin_sol.c -o test_lin_sol.o $(CFLAGS)

test_multi_vec.o: test_multi_vec.c
	$(CC) -c test_multi_vec.c -o test_multi_vec.o $(CFLAGS)

test_orth.o: test_orth.c
	$(CC) -c test_orth.c -o test_orth.o $(CFLAGS)

test_vec.o: test_vec.c
	$(CC) -c test_vec.c -o test_vec.o $(CFLAGS)

../src/ops.o: ../src/ops.c
	$(CC) -c ../src/ops.c -o ../src/ops.o $(CFLAGS)

../src/ops_eig_sol_gcg.o: ../src/ops_eig_sol_gcg.c
	$(CC) -c ../src/ops_eig_sol_gcg.c -o ../src/ops_eig_sol_gcg.o $(CFLAGS)

../src/ops_lin_sol.o: ../src/ops_lin_sol.c
	$(CC) -c ../src/ops_lin_sol.c -o ../src/ops_lin_sol.o $(CFLAGS)

../src/ops_multi_vec.o: ../src/ops_multi_vec.c
	$(CC) -c ../src/ops_multi_vec.c -o ../src/ops_multi_vec.o $(CFLAGS)

../src/ops_orth.o: ../src/ops_orth.c
	$(CC) -c ../src/ops_orth.c -o ../src/ops_orth.o $(CFLAGS)

../app/app_ccs.o: ../app/app_ccs.c
	$(CC) -c ../app/app_ccs.c -o ../app/app_ccs.o $(CFLAGS)

clean: 
	-@$(RM) *.$(SOLIB_SUFFIX)

.PHONY: all clean
