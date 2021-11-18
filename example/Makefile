# Project: TestOPS
# Makefile created by Dev-C++ 5.11

include ../config/make.MPI.inc

OBJ1      = ex1.o ../test/test_eig_sol_gcg.o ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o  ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_lapack.o ../app/app_slepc.o 
LINKOBJ1  = ex1.o ../test/test_eig_sol_gcg.o ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o  ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_lapack.o ../app/app_slepc.o  
OBJ2      = ex2.o ../test/test_eig_sol_gcg.o ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o  ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_lapack.o ../app/app_slepc.o  
LINKOBJ2  = ex2.o ../test/test_eig_sol_gcg.o ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o  ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_lapack.o ../app/app_slepc.o
OBJ3      = ex3.o ../test/test_eig_sol_gcg.o ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o  ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_lapack.o ../app/app_slepc.o
LINKOBJ3  = ex3.o ../test/test_eig_sol_gcg.o ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o  ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_lapack.o ../app/app_slepc.o


LIBS     = $(LIBSLEPC) $(LIBPETSC) 
INCS     = -I../src -I../app
CXXINCS  = 
BIN1     = ex1 
BIN2     = ex2
BIN3     = ex3
CXXFLAGS = $(CXXINCS) -O2 -Wall -fPIC -g
CFLAGS   = $(INCS) -O2 -Wall -fPIC -g -fopenmp 
RM       = rm -f


all: all-before $(BIN1) $(BIN2) $(BIN3) all-after

.PHONY: all all-before all-after clean clean-custom

clean: clean-custom
	${RM} $(OBJ1) $(OBJ2) $(OBJ3) $(BIN1) $(BIN2) $(BIN3)

$(BIN1): $(OBJ1)  
	$(CC) $(LINKOBJ1) -o $(BIN1) $(LIBS) 
$(BIN2): $(OBJ2)
	$(CC) $(LINKOBJ2) -o $(BIN2) $(LIBS) 	
$(BIN3): $(OBJ3)
	$(CC) $(LINKOBJ3) -o $(BIN3) $(LIBS) 

ex1.o: ex1.c
	$(CC) -c ex1.c -o ex1.o  $(CFLAGS) $(SLEPCFLAGS) $(SLEPCINC)

ex2.o: ex2.c
	$(CC) -c ex2.c -o ex2.o  $(CFLAGS) $(SLEPCFLAGS) $(SLEPCINC)

ex3.o: ex3.c
	$(CC) -c ex3.c -o ex3.o  $(CFLAGS) $(SLEPCFLAGS) $(SLEPCINC)

../test/test_eig_sol_gcg.o: ../test/test_eig_sol_gcg.c
	$(CC) -c ../test/test_eig_sol_gcg.c -o ../test/test_eig_sol_gcg.o $(CFLAGS)

../src/ops.o: ../src/ops.c
	$(CC) -c ../src/ops.c -o ../src/ops.o $(CFLAGS)

../src/ops_eig_sol_gcg.o: ../src/ops_eig_sol_gcg.c
	$(CC) -c ../src/ops_eig_sol_gcg.c -o ../src/ops_eig_sol_gcg.o $(CFLAGS)

../src/ops_lin_sol.o: ../src/ops_lin_sol.c
	$(CC) -c ../src/ops_lin_sol.c -o ../src/ops_lin_sol.o $(CFLAGS)

../src/ops_orth.o: ../src/ops_orth.c
	$(CC) -c ../src/ops_orth.c -o ../src/ops_orth.o $(CFLAGS)

../app/app_slepc.o: ../app/app_slepc.c
	$(CC) -c ../app/app_slepc.c -o ../app/app_slepc.o $(CFLAGS) $(SLEPCFLAGS) $(SLEPCINC)

