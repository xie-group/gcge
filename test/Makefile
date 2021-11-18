# Project: TestOPS
# Makefile created by Dev-C++ 5.11

include ../config/make.MPI.inc

OBJ      = main.o test_app_slepc.o test_eig_sol_gcg.o ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o  ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_lapack.o ../app/app_slepc.o 
LINKOBJ  = main.o test_app_slepc.o test_eig_sol_gcg.o ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o  ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_lapack.o ../app/app_slepc.o  
LIBS     = $(LIBSLEPC) $(LIBPETSC) $(LIBBLASLAPACK)
INCS     = -I../src -I../app
CXXINCS  = 
BIN      = TestGCGE
CXXFLAGS = $(CXXINCS) -O2 -Wall -fPIC -g
CFLAGS   = $(INCS) -O2 -Wall -fPIC -g -fopenmp#-fopenmp 
RM       = rm -f

.PHONY: all all-before all-after clean clean-custom

all: all-before $(BIN) all-after

clean: clean-custom
	${RM} $(OBJ) $(BIN) 

$(BIN): $(OBJ) 
	$(CC) $(LINKOBJ) -o $(BIN) $(LIBS)

main.o: main.c
	$(CC) -c main.c -o main.o $(CFLAGS) $(PETSCINC) $(SLEPCINC)

test_app_slepc.o: test_app_slepc.c
	$(CC) -c test_app_slepc.c -o test_app_slepc.o $(CFLAGS) $(SLEPCFLAGS) $(SLEPCINC)

test_eig_sol_gcg.o: test_eig_sol_gcg.c
	$(CC) -c test_eig_sol_gcg.c -o test_eig_sol_gcg.o $(CFLAGS)

../src/ops.o: ../src/ops.c
	$(CC) -c ../src/ops.c -o ../src/ops.o $(CFLAGS)

../src/ops_eig_sol_gcg.o: ../src/ops_eig_sol_gcg.c
	$(CC) -c ../src/ops_eig_sol_gcg.c -o ../src/ops_eig_sol_gcg.o $(CFLAGS) $(PETSCINC) $(SLEPCINC)

../src/ops_lin_sol.o: ../src/ops_lin_sol.c
	$(CC) -c ../src/ops_lin_sol.c -o ../src/ops_lin_sol.o $(CFLAGS)

../src/ops_orth.o: ../src/ops_orth.c
	$(CC) -c ../src/ops_orth.c -o ../src/ops_orth.o $(CFLAGS)

../app/app_slepc.o: ../app/app_slepc.c
	$(CC) -c ../app/app_slepc.c -o ../app/app_slepc.o $(CFLAGS) $(SLEPCFLAGS) $(SLEPCINC)

