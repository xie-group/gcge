
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "ops.h"
#include "app_slepc.h"
#include <slepceps.h>

int TestEigenSolverGCG   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);
static void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m);
static char help[] = "Test App of SLEPC.\n";
int main(int argc, char *argv[]) 
{

	SlepcInitialize(&argc,&argv,(char*)0,help);
	PetscMPIInt   rank, size;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);		
	OPS *slepc_ops = NULL;
	OPS_Create (&slepc_ops);
	OPS_SLEPC_Set (slepc_ops);
	OPS_Setup (slepc_ops);
	void *matA, *matB; OPS *ops;
	Mat      slepc_matA, slepc_matB;
	int flag = 0;
	PetscInt n = 300, m = 300;
	GetPetscMat(&slepc_matA, &slepc_matB, n, m);
	MatAssemblyBegin(slepc_matA, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(slepc_matA, MAT_FINAL_ASSEMBLY);
	ops = slepc_ops; matA = (void*)(slepc_matA); matB = (void*)(slepc_matB);

	TestEigenSolverGCG(matA,matB,flag,argc,argv,ops);

	MatDestroy(&slepc_matA);
	MatDestroy(&slepc_matB);	
	OPS_Destroy (&slepc_ops);
	SlepcFinalize();	
	return 0;
}

/* 创建 2-D possion 差分矩阵 A B */
static void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m)
{
	assert(n==m);
	PetscInt N = n*m;
	PetscInt Istart, Iend, II, i, j;
	PetscReal h = 1.0/(n+1);
	MatCreate(PETSC_COMM_WORLD,A);
	MatSetSizes(*A,PETSC_DECIDE,PETSC_DECIDE,N,N);
	//MatSetFromOptions(*A);
	MatSetUp(*A);
	MatGetOwnershipRange(*A,&Istart,&Iend);
	for (II=Istart;II<Iend;II++) {
		i = II/n; j = II-i*n;
		if (i>0)   { MatSetValue(*A,II,II-n,-1.0/h,INSERT_VALUES); }
		if (i<m-1) { MatSetValue(*A,II,II+n,-1.0/h,INSERT_VALUES); }
		if (j>0)   { MatSetValue(*A,II,II-1,-1.0/h,INSERT_VALUES); }
		if (j<n-1) { MatSetValue(*A,II,II+1,-1.0/h,INSERT_VALUES); }
		MatSetValue(*A,II,II,4.0/h,INSERT_VALUES);
	}
	MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);

	MatCreate(PETSC_COMM_WORLD,B);
	MatSetSizes(*B,PETSC_DECIDE,PETSC_DECIDE,N,N);
	//MatSetFromOptions(*B);
	MatSetUp(*B);
	MatGetOwnershipRange(*B,&Istart,&Iend);
	for (II=Istart;II<Iend;II++) {
		MatSetValue(*B,II,II,1.0*h,INSERT_VALUES);
	}
	MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);
	return;
}

