/**
 *    @file  test_app_slepc.c
 *   @brief  test app of SLEPC 
 *
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/17
 *      Revision:  none
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "ops.h"
#include "app_slepc.h"

#if OPS_USE_MUMPS
#include "dmumps_c.h"
#define ICNTL(I) icntl[(I)-1] 
#define INFOG(I) infog[(I)-1] 
#define CNTL(I)  cntl[(I)-1] 
#define INFO(I)  info[(I)-1]
//#include "petscmat.h"
#endif


/* run this program using the console pauser or add your own getch, system("pause") or input loop */
int TestEigenSolverGCG   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);

/* test EPS in SLEPc */
int TestEPS(void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops);

#define OPS_USE_FILE_MAT 0

#if OPS_USE_SLEPC
#include <slepceps.h>

/*
  Create an application context to contain data needed by the
  application-provided call-back routines, ops->MultiLinearSolver().
*/
#if 0
typedef struct {
	KSP ksp;
	Vec rhs;
	Vec sol;
} AppCtx;
static void AppCtxCreate(AppCtx *user, Mat petsc_mat)
{
	double time_start, time_end; 
	time_start = MPI_Wtime();
	KSPCreate(PETSC_COMM_WORLD,&(user->ksp));
	KSP ksp = user->ksp;
	Mat A   = petsc_mat, F;
	PC  pc;

	KSPSetOperators(ksp,A,A);
	PetscBool flg_mumps = PETSC_TRUE, flg_mumps_ch = PETSC_FALSE;
	PetscOptionsGetBool(NULL,NULL,"-use_mumps_lu",&flg_mumps,NULL);
	PetscOptionsGetBool(NULL,NULL,"-use_mumps_ch",&flg_mumps_ch,NULL);
	if (flg_mumps || flg_mumps_ch) {
		KSPSetType(ksp,KSPPREONLY);
		KSPGetPC(ksp,&pc);
		if (flg_mumps) {
			PCSetType(pc,PCLU);
		} else if (flg_mumps_ch) {
			MatSetOption(A,MAT_SPD,PETSC_TRUE); /* set MUMPS id%SYM=1 */
			PCSetType(pc,PCCHOLESKY);
		}
		PCFactorSetMatSolverType(pc,MATSOLVERMUMPS);
		PCFactorSetUpMatSolverType(pc); /* call MatGetFactor() to create F */
		PCFactorGetMatrix(pc,&F);

		MatMumpsSetIcntl(F, 1,-1);  /* the output stream for error messages */
		MatMumpsSetIcntl(F, 2,-1);  /* the output stream for diagnostic printing and statistics local to each MPI process */
		MatMumpsSetIcntl(F, 3,-1);  /* the output stream for global information */
		MatMumpsSetIcntl(F, 4, 0);  /* errors, warnings and information on input, output parameters printed.*/
		//MatMumpsSetIcntl(F, 5, 0);  /* assembled format */
		MatMumpsSetIcntl(F, 9, 1);  /* AX = B is solved */
		MatMumpsSetIcntl(F,10, 0);  /* maximum number of steps of iterative refinement */
		//MatMumpsSetIcntl(F,18, 3);  /* the distributed matrix */
		//MatMumpsSetIcntl(F,20, 0);  /* the dense format of the right-hand side */
		//MatMumpsSetIcntl(F,21, 1);  /* the distributed format of the solution */

		MatMumpsSetCntl(F,2,0.0);   /* stopping criterion for iterative refinement */
	}
	KSPSetFromOptions(ksp);
	/* Get info from matrix factors */
	KSPSetUp(ksp);

	time_end = MPI_Wtime();
	int rank;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	if (rank == 0) {
		printf("FACTORIZATION time %f\n", time_end-time_start);
	}

	return;
}
static void AppCtxDestroy(AppCtx *user)
{
	KSPDestroy(&(user->ksp));
	return;
}
static void KSP_MultiLinearSolver(void *mat, void **b, void **x, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0] == end[1]-start[1]);
	AppCtx *user = (AppCtx*)ops->multi_linear_solver_workspace;
	int i, length = end[0] - start[0];
	for (i = 0; i < length; ++i) {
		BVGetColumn((BV)b,start[0]+i,&(user->rhs));
		BVGetColumn((BV)x,start[1]+i,&(user->sol));
		KSPSolve(user->ksp,user->rhs,user->sol);
		BVRestoreColumn((BV)b,start[0]+i,&(user->rhs));
		BVRestoreColumn((BV)x,start[1]+i,&(user->sol));
	}
	return;
}
#endif

#if OPS_USE_MUMPS
typedef struct {
	DMUMPS_STRUC_C mumps;
	MPI_Comm       comm;

	int rank   ; int nprocs;
	int nglobal; int nlocal; int nnz_local;
	int *rows  ; int *cols ; double *values;
	double *sol;
} AppCtx;

static void AppCtxCreate(AppCtx *user, Mat petsc_mat)
{
	//printf("AppCtxCreate\n");
	double time_start, time_end; 
	time_start = MPI_Wtime();

	user->comm = PetscObjectComm((PetscObject)petsc_mat);
	MPI_Comm_rank(user->comm,&user->rank);
	MPI_Comm_size(user->comm,&user->nprocs);

	user->mumps.comm_fortran = MPI_Comm_c2f(user->comm);
	user->mumps.par =  1;
	user->mumps.sym =  0; /* unsymmetric.*/
	user->mumps.job = -1; /* initializes an instance of the package */
	dmumps_c(&(user->mumps));

	PetscInt          row, row_start, row_end, nnz;
	const PetscInt    *cols;
	const PetscScalar *values;

	MatGetSize     (petsc_mat,&user->nglobal,NULL);
	MatGetLocalSize(petsc_mat,&user->nlocal ,NULL);
	MatGetOwnershipRange(petsc_mat,&row_start,&row_end);

	user->nnz_local = 0;
	for (row = row_start; row < row_end; ++row) {
		MatGetRow(petsc_mat, row, &nnz, NULL, NULL);	
		user->nnz_local += nnz;
		MatRestoreRow(petsc_mat, row, &nnz, NULL, NULL);	
	}
	//printf("nlocal = %d, nglobal = %d, rows_local = %d,%d, nnz_local = %d\n",
	//	user->nlocal,user->nglobal,row_start,row_end,user->nnz_local);

	user->rows   = malloc(user->nnz_local*sizeof(int));
	user->cols   = malloc(user->nnz_local*sizeof(int));
	user->values = malloc(user->nnz_local*sizeof(double));

	int k = 0, i;
	for (row = row_start; row < row_end; ++row) {
		MatGetRow(petsc_mat, row, &nnz, &cols, &values);
		//printf("row = %d, nnz = %d, cols = %d, val = %f\n",row,nnz,cols[0],values[0]);
		for (i = 0; i < nnz; ++i) {
			user->rows[k]   = 1 + row;
			user->cols[k]   = 1 + cols[i];
			user->values[k] = values[i];
			++k;
		}
		MatRestoreRow(petsc_mat, row, &nnz, &cols, &values);	
	}

	user->mumps.n       = user->nglobal;
	user->mumps.nnz_loc = user->nnz_local;
	user->mumps.irn_loc = user->rows;
	user->mumps.jcn_loc = user->cols;
	user->mumps.a_loc   = user->values;
#if 0
	user->mumps.ICNTL(1)  =  6; /* the output stream for error messages */
	user->mumps.ICNTL(2)  =  1; /* the output stream for diagnostic printing and statistics local to each MPI process */
	user->mumps.ICNTL(3)  =  6; /* the output stream for global information */
	user->mumps.ICNTL(4)  =  4; /* errors, warnings and information on input, output parameters printed.*/
#else
	user->mumps.ICNTL(1)  = -1; /* the output stream for error messages */
	user->mumps.ICNTL(2)  = -1; /* the output stream for diagnostic printing and statistics local to each MPI process */
	user->mumps.ICNTL(3)  = -1; /* the output stream for global information */
	user->mumps.ICNTL(4)  =  0; /* errors, warnings and information on input, output parameters printed.*/
#endif
	user->mumps.ICNTL(5)  =  0; /* assembled format */
	user->mumps.ICNTL(9)  =  1; /* AX = B is solved */
	user->mumps.ICNTL(10) =  0; /* maximum number of steps of iterative refinement */
  	user->mumps.CNTL(2)   =0.0; /* stopping criterion for iterative refinement */
	user->mumps.ICNTL(18) =  3; /* the distributed matrix */
	user->mumps.ICNTL(20) =  0; /* the dense format of the right-hand side */
	user->mumps.ICNTL(21) =  0; /* the centralized format of the solution */
	/* factorization */
	user->mumps.job = 4; /* perform the analysis and the factorization */
	dmumps_c(&(user->mumps));
	if (user->mumps.INFOG(1)<0)
		printf("\n (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
				user->rank, user->mumps.INFOG(1), user->mumps.INFOG(2));

	if (user->rank == 0) {
		/* 160 表示求解时, 至多160个向量一起算, blockSize<=160 */
		user->sol = malloc(160*user->nglobal*sizeof(double));
	}
	else {
		user->sol = NULL;
	}
	time_end = MPI_Wtime();
	if (user->rank == 0) {
		printf("FACTORIZATION time %f\n", time_end-time_start);
	}
#if 0
if (user->rank == 0) {
	printf("%d,%d,%d,%d\n",sizeof(int),sizeof(MUMPS_INT),sizeof(MUMPS_INT8),sizeof(double));
	printf("%d n = %d, nnz_loc = %ld, nnz_local = %d\n",user->rank, user->mumps.n, user->mumps.nnz_loc,nnz_local);
	printf("%d nlocal = %d, nglobal = %d\n",user->rank, user->nlocal, user->nglobal);
	for (j = 0; j < nnz_local; j++) {
		printf("%d (%d,%d) %.4e\n", user->rank,
				user->mumps.irn_loc[j],user->mumps.jcn_loc[j],user->mumps.a_loc[j]);
	}
}
#endif

	return;
}
static void AppCtxDestroy(AppCtx *user)
{
	user->mumps.job = -2; /* terminates an instance of the package */
	dmumps_c(&(user->mumps));
	if (user->sol!=NULL) free(user->sol);
	user->sol = NULL;
	free(user->rows) ; free(user->cols) ; free(user->values) ;
	user->rows = NULL; user->cols = NULL; user->values = NULL;
	return;
}
static void MUMPS_MultiLinearSolver(void *mat, void **b, void **x, int *start, int *end, struct OPS_ *ops)
{
   //ops->Printf("MUMPS_MultiLinearSolver\n");
   assert(end[0]-start[0]==end[1]-start[1]);
   int nvec = end[0]-start[0];
   double *data_b, *data_x;
   AppCtx *user     = (AppCtx*)ops->multi_linear_solver_workspace;
   Mat    petsc_mat = (Mat)mat;
   BV     bv_b      = (BV)b;
   BV     bv_x      = (BV)x;

   BVGetArray(bv_b,&data_b);
   BVGetArray(bv_x,&data_x);

   const PetscInt *ranges;
   MatGetOwnershipRanges(petsc_mat,&ranges);
   int nlocal = user->nlocal;
   int *cnts  = malloc(2 * user->nprocs * sizeof(*cnts));
   int *dsps  = cnts + user->nprocs;
   int i;

   //double time_start, time_end; 
   //time_start = MPI_Wtime();
   for (i = 0; i < user->nprocs; i++) {
	   cnts[i] = ranges[i + 1] - ranges[i];
	   dsps[i] = ranges[i];
   }
   MPI_Datatype *rowType = malloc(user->nprocs*sizeof(MPI_Datatype)); 
   MPI_Request  *request = malloc(user->nprocs*sizeof(MPI_Request ));
   for (i = 0; i < user->nprocs; ++i) {
	   MPI_Type_vector(nvec, cnts[i], user->nglobal, MPI_DOUBLE, rowType+i);
	   MPI_Type_commit(rowType+i);
   }

   assert(cnts[user->rank] == nlocal);

   MPI_Isend(data_b+start[0]*nlocal, nvec*cnts[user->rank], MPI_DOUBLE, 0, user->rank, user->comm, request+user->rank);
   if (user->rank == 0) {
	   for (i = 0; i < user->nprocs; ++i) {
		   MPI_Irecv(user->sol+dsps[i], 1, rowType[i], i, i, user->comm, request+i);
	   }
	   for (i = 0; i < user->nprocs; ++i) {
		   MPI_Wait(request+i,MPI_STATUS_IGNORE);
	   }
   }
   else {
	   MPI_Wait(request+user->rank,MPI_STATUS_IGNORE);
   }
   //time_end = MPI_Wtime();
   //ops->Printf("GATHER time %f\n", time_end-time_start);

#if 0
   int k;
   if (phg_mat->cmap->rank == 0) {
	   printf("==========rhs========\n");
	   for (k = 0; k < user->nglobal; ++k) {
		   printf("%.4e\n", user->sol[k]);
	   }
   }
#endif
   //time_start = MPI_Wtime();

   user->mumps.nrhs = nvec;
   user->mumps.lrhs = user->nglobal;
   user->mumps.rhs  = user->sol;
   user->mumps.job  = 3;
   dmumps_c(&(user->mumps));
   if (user->mumps.infog[0]<0)
	   printf("\n (PROC %d) ERROR RETURN: \tINFOG(1)= %d\n\t\t\t\tINFOG(2)= %d\n",
			   user->rank, user->mumps.INFOG(1), user->mumps.INFOG(2));
#if 0
   if (user->rank == 0) {
	   printf("==========sol========\n");
	   for (k = 0; k < user->nglobal; ++k) {
		   printf("%.4e\n", user->sol[k]);
	   }
   }
#endif
   //time_end = MPI_Wtime();
   //ops->Printf("CALCULATE time %f\n", time_end-time_start);

   //time_start = MPI_Wtime();
   if (user->rank == 0) {
	   for (i = 0; i < user->nprocs; ++i) {
		   MPI_Isend(user->sol+dsps[i], 1, rowType[i], i, i, user->comm, request+i);
	   }
   }
   MPI_Irecv(data_x+start[1]*nlocal, nvec*cnts[user->rank], MPI_DOUBLE, 0, user->rank, user->comm, request+user->rank);
   if (user->rank == 0) {
	   for (i = 0; i < user->nprocs; ++i) {
		   MPI_Wait(request+i,MPI_STATUS_IGNORE);
	   }
   }
   else {
	   MPI_Wait(request+user->rank,MPI_STATUS_IGNORE);
   }

   for (i = 0; i < user->nprocs; ++i) {
	   MPI_Type_free(rowType+i);
   }
   free(rowType);
   free(request);

   //time_end = MPI_Wtime();
   //ops->Printf("SCATTER time %f\n", time_end-time_start);

#if 0
   if (user->rank == 0) {
	   printf("==========x========\n");
	   for (k = 0; k < nlocal; ++k) {
		   printf("%.4e\n", data_x[k]);
	   }
   }
#endif

    BVRestoreArray(bv_b,&data_b);
    BVRestoreArray(bv_x,&data_x);

   //ops->Printf("MUMPS_MultiLinearSolver\n");
   return;
}
#endif




static char help[] = "Test App of SLEPC.\n";
static void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m);
int TestAppSLEPC(int argc, char *argv[]) 
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
   	PetscBool flg;
#if OPS_USE_FILE_MAT
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
	slepc_ops->Printf("matrix A %d, %d\n",nrows,ncols);

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

#else
   	//PetscInt n = 3750, m = 3750;
   	PetscInt n = 120, m = 120;
   	GetPetscMat(&slepc_matA, &slepc_matB, n, m);
#endif
	//slepc_ops->MatView((void*)slepc_matA, slepc_ops);
	//slepc_ops->MatView((void*)slepc_matB, slepc_ops);

	int row_start, row_end, i;
//	PetscInt           nc;
//	const PetscInt    *aj;
//	const PetscScalar *aa;
#if 0
	MatGetOwnershipRange(slepc_matA,&row_start,&row_end);
	for (i = row_start; i < row_end; ++i) {
		MatGetRow(slepc_matA,i,&nc,&aj,&aa);
		PetscPrintf(PETSC_COMM_WORLD,"row %d nc %d\n",i,nc);
		MatRestoreRow(slepc_matA,i,&nc,&aj,&aa);
	}
	MatGetOwnershipRange(slepc_matB,&row_start,&row_end);
	for (i = row_start; i < row_end; ++i) {
		MatGetRow(slepc_matB,i,&nc,&aj,&aa);
		PetscPrintf(PETSC_COMM_WORLD,"row %d nc %d\n",i,nc);
		MatRestoreRow(slepc_matB,i,&nc,&aj,&aa);
	}
#endif

	PetscReal shift = 0.0;
	PetscOptionsGetReal(NULL,NULL,"-shift",&shift,&flg);
	if (slepc_matB==NULL) {
		if (flg) {
			MatGetOwnershipRange(slepc_matA,&row_start,&row_end);
			for (i = row_start; i < row_end; ++i) {
				MatSetValue(slepc_matA, i, i, shift, ADD_VALUES);
			}
		}
	}
	else {
		/* should confirm A and B have same non-zero structure */
		if (flg) {
			/*A <- A + shift*B */
			/* SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN or SUBSET_NONZERO_PATTERN */
#if 1
			MatAXPY(slepc_matA,shift,slepc_matB,DIFFERENT_NONZERO_PATTERN);
#else

			MatGetOwnershipRange(slepc_matB,&row_start,&row_end);
			for (i = row_start; i < row_end; ++i) {
				MatGetRow(slepc_matB,i,&nc,&aj,&aa);
				int inc = 1;
				double *tmp_aa = malloc(nc*sizeof(double));
				memset(tmp_aa,0,nc*sizeof(double));
				/* tmp_aa = shift*aa */
				daxpy(&nc,&shift,aa,&inc,tmp_aa,&inc);
				MatSetValues(slepc_matA,1,&i,nc,aj,tmp_aa,ADD_VALUES);
				free(tmp_aa);

				MatRestoreRow(slepc_matB,i,&nc,&aj,&aa);
			}
#endif
		}
	}
	MatAssemblyBegin(slepc_matA, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(slepc_matA, MAT_FINAL_ASSEMBLY);


	ops = slepc_ops; matA = (void*)(slepc_matA); matB = (void*)(slepc_matB);
	
	PetscInt use_slepc_eps = 0;
	PetscOptionsGetInt(NULL,NULL,"-use_slepc_eps",&use_slepc_eps,&flg);
	if (use_slepc_eps)
		TestEPS(matA,matB,use_slepc_eps,argc,argv,ops);
	else {
		int flag = 0;
#if OPS_USE_MUMPS
		AppCtx user; flag = 0;
		if (flag != 0) {
			AppCtxCreate(&user, (Mat)matA);
			ops->multi_linear_solver_workspace = (void*)&user;
			ops->MultiLinearSolver = MUMPS_MultiLinearSolver;
		}
#endif
		TestEigenSolverGCG(matA,matB,flag,argc,argv,ops);
#if OPS_USE_MUMPS
		if (flag != 0) {
			AppCtxDestroy(&user);
		}
#endif
		//TestEigenSolverGCG(matA,matB,0,argc,argv,ops);
		//TestEigenSolverPAS(matA,matB,0,argc,argv,ops);	
	}
		
	//TestMultiGrid(matA,matB,ops);

	/* 销毁petsc矩阵 */
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


int TestEPS(void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops)
{
	EPS eps; EPSType type; 
	PetscInt nev, ncv, mpd, max_it, nconv, its;
	PetscReal tol;
	nev = 800; ncv = nev+nev/5; mpd = ncv;
	tol = 1e-12; max_it = 2000;
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
			//EPSSetType(eps,EPSLOBPCG);
			break;
	}
	EPSSetDimensions(eps,nev,ncv,mpd);
	EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);
	EPSSetTolerances(eps, tol, max_it);
	//EPSSetConvergenceTest(eps,EPS_CONV_REL);
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
#if 0
	int i; PetscScalar eigr;
	for (i = 0; i < nconv; ++i) {
		EPSGetEigenvalue(eps,i,&eigr,NULL);
		PetscPrintf(PETSC_COMM_WORLD,"%d: %6.14e\n",1+i,eigr);
	}
#else
	PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);
	//EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);
	EPSErrorView(eps,EPS_ERROR_ABSOLUTE,PETSC_VIEWER_STDOUT_WORLD);
	EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);
	PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);
#endif
	EPSDestroy(&eps);
	return 0;
}






#endif
