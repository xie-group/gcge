#include <slepceps.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "ops.h"
#include "app_slepc.h"

int TestEPS(void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);

static char help[] = "Test App of SLEPC.\n";
static void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m);

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

   /* 得到PETSC矩阵A, B, 规模为n*m */
   Mat      slepc_matA, slepc_matB;
   PetscInt n = 500, m = 500;
   GetPetscMat(&slepc_matA, &slepc_matB, n, m);

   MatAssemblyBegin(slepc_matA, MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(slepc_matA, MAT_FINAL_ASSEMBLY);
   ops = slepc_ops; matA = (void*)(slepc_matA); matB = (void*)(slepc_matB);
   int use_slepc_eps = 6;
   TestEPS(matA,matB,use_slepc_eps,argc,argv,ops);

   MatDestroy(&slepc_matA);
   MatDestroy(&slepc_matB);
   OPS_Destroy (&slepc_ops);
   SlepcFinalize();

   return 0;
}

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

int TestEPS(void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops)
{
   EPS eps; EPSType type; 
   PetscInt nev, ncv, mpd, max_it, nconv, its;
   PetscReal tol;
   nev = 30; ncv = 100; mpd = 48;
   tol = 1e-8; max_it = 2000;
   EPSCreate(PETSC_COMM_WORLD,&eps);
   EPSSetOperators(eps,(Mat)A,(Mat)B);
   if (B==NULL)
      EPSSetProblemType(eps,EPS_HEP);
   else 
      EPSSetProblemType(eps,EPS_GHEP);
   switch (flag) {
      case 1:
         EPSSetType(eps,EPSLANCZOS);
         break;
      case 2:
         EPSSetType(eps,EPSKRYLOVSCHUR);
         break;
      case 3:
         EPSSetType(eps,EPSGD);
         break;
      case 4:
         EPSSetType(eps,EPSJD);
         break;
      case 5:
         EPSSetType(eps,EPSRQCG);
         break;
      case 6:
         EPSSetType(eps,EPSLOBPCG);
         break;
      default:
         EPSSetType(eps,EPSKRYLOVSCHUR);
         break;
   }
   EPSSetDimensions(eps,nev,ncv,mpd);
   EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);
   EPSSetTolerances(eps, tol, max_it);
   EPSSetConvergenceTest(eps,EPS_CONV_ABS);

   EPSSetFromOptions(eps);
   EPSSetUp(eps);
   EPSView(eps,PETSC_VIEWER_STDOUT_WORLD);
   ST st; KSP ksp;
   EPSGetST(eps,&st);
   STGetKSP(st,&ksp);
   KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);
   double time_start, time_interval;
   time_start = ops->GetWtime();

   EPSSolve(eps);

   time_interval = ops->GetWtime() - time_start;
   ops->Printf("Time is %.3f\n", time_interval);

   EPSGetType(eps,&type);
   EPSGetConverged(eps,&nconv);
   EPSGetIterationNumber(eps, &its);
   PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);
   PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);
   PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);
   PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);
   PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);
   EPSErrorView(eps,EPS_ERROR_ABSOLUTE,PETSC_VIEWER_STDOUT_WORLD);
   EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);
   PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);
   EPSDestroy(&eps);
   return 0;
}







