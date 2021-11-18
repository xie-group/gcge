#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <slepceps.h>
#include "ops.h"
#include "app_slepc.h"

int TestEigenSolverGCG   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);
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
	slepc_ops->Printf("%s", help);
	
	void *matA, *matB; OPS *ops;

    	Mat      slepc_matA, slepc_matB;
    	PetscBool flg;
	char filename_matA[PETSC_MAX_PATH_LEN];
	PetscOptionsGetString(NULL,NULL,"-filename_matA",filename_matA,sizeof(filename_matA),&flg);
	if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name with the -filename_matA option");
	slepc_ops->Printf("%s\n",filename_matA);
	
	PetscInt nrows, ncols;
	PetscViewer    viewer;
	PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename_matA,FILE_MODE_READ,&viewer); 
	MatCreate(PETSC_COMM_WORLD,&slepc_matA); 
	MatSetFromOptions(slepc_matA);
	MatLoad(slepc_matA,viewer); 
	PetscViewerDestroy(&viewer);
	MatGetSize(slepc_matA,&nrows,&ncols);

	char filename_matB[PETSC_MAX_PATH_LEN];
	PetscOptionsGetString(NULL,NULL,"-filename_matB",filename_matB,sizeof(filename_matA),&flg);
	if (flg) {
		slepc_ops->Printf("%s\n",filename_matB);
		PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename_matB,FILE_MODE_READ,&viewer); 
		MatCreate(PETSC_COMM_WORLD,&slepc_matB); 
		MatLoad(slepc_matB,viewer); 
		PetscViewerDestroy(&viewer);
		MatGetSize(slepc_matB,&nrows,&ncols);
		slepc_ops->Printf("matrix B %d, %d\n",nrows,ncols);
	}
	else {
		slepc_matB = NULL;
	}

	int flag=0;
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
/*
	args:
	-filename_matA ${SLEPC_DIR}/share/slepc/datafiles/matrices/rdb200.petsc  
*/ 
