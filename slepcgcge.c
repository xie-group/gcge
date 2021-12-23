#include	<stdio.h>
#include	<stdlib.h>
#include	<assert.h>
#include  	<math.h>
#include   	<memory.h>
#include    	<assert.h>
#include    	<float.h> 
#include	"slepcgcge.h"
#include    	"ops.h"
#include    	"ops_eig_sol_gcg.h"




/* 进程分组, 主要用于 AMG, 默认最大层数是16 */ 
int       MG_COMM_COLOR[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
/* 能否这样赋初值 尤其时 MPI_COMM_WORLD 
 * 另外, 这些创建出来的通讯域可以 MPI_Comm_free 吗? 何时 */ 
MPI_Comm  MG_COMM[16][2] = {
	{MPI_COMM_NULL,MPI_COMM_NULL},{MPI_COMM_NULL,MPI_COMM_NULL},
	{MPI_COMM_NULL,MPI_COMM_NULL},{MPI_COMM_NULL,MPI_COMM_NULL},
	{MPI_COMM_NULL,MPI_COMM_NULL},{MPI_COMM_NULL,MPI_COMM_NULL},
	{MPI_COMM_NULL,MPI_COMM_NULL},{MPI_COMM_NULL,MPI_COMM_NULL},
	{MPI_COMM_NULL,MPI_COMM_NULL},{MPI_COMM_NULL,MPI_COMM_NULL},
	{MPI_COMM_NULL,MPI_COMM_NULL},{MPI_COMM_NULL,MPI_COMM_NULL},
	{MPI_COMM_NULL,MPI_COMM_NULL},{MPI_COMM_NULL,MPI_COMM_NULL},
	{MPI_COMM_NULL,MPI_COMM_NULL},{MPI_COMM_NULL,MPI_COMM_NULL}
};
MPI_Comm  MG_INTERCOMM[16] = {
	MPI_COMM_NULL,MPI_COMM_NULL,MPI_COMM_NULL,MPI_COMM_NULL,
	MPI_COMM_NULL,MPI_COMM_NULL,MPI_COMM_NULL,MPI_COMM_NULL,
	MPI_COMM_NULL,MPI_COMM_NULL,MPI_COMM_NULL,MPI_COMM_NULL,
	MPI_COMM_NULL,MPI_COMM_NULL,MPI_COMM_NULL,MPI_COMM_NULL
};


/* multi-vec */
static void MultiVecCreateByMat (BV *des_bv, int num_vec, Mat src_mat, struct OPS_ *ops)
{
	Vec vector;
    MatCreateVecs(src_mat,NULL,&vector);
    BVCreate(PETSC_COMM_WORLD, des_bv);
    BVSetType(*des_bv,BVMAT);
    BVSetSizesFromVec(*des_bv,vector,num_vec);
    VecDestroy(&vector);
	BVSetActiveColumns(*des_bv,0,num_vec);
	BVSetRandom(*des_bv);
	return;
}
static void MultiVecDestroy (BV *des_bv, int num_vec, struct OPS_ *ops)
{ 
	BVDestroy(des_bv);
	return;
}
static void MultiVecView (BV x, int start, int end, struct OPS_ *ops)
{
	BVSetActiveColumns(x,start,end);
	BVView(x,PETSC_VIEWER_STDOUT_WORLD);
	return;
}
static void MultiVecLocalInnerProd (char nsdIP, 
		BV x, BV y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	assert(is_vec==0);
	
    const PetscScalar *x_array, *y_array;
    int x_nrows, x_ncols, y_nrows, y_ncols;
    BVGetArrayRead(x,&x_array);
    BVGetSizes(x,&x_nrows,NULL,&x_ncols);
    if(is_vec == 0) {
		BVGetArrayRead(y,&y_array);
		BVGetSizes(y,&y_nrows,NULL,&y_ncols);
		LAPACKVEC x_vec, y_vec;
		x_vec.nrows = x_nrows; y_vec.nrows = y_nrows;
		x_vec.ncols = x_ncols; y_vec.ncols = y_ncols;
		x_vec.ldd   = x_nrows; y_vec.ldd   = y_nrows;
		x_vec.data  = (double *)x_array; 
		y_vec.data  = (double *)y_array;
		ops->lapack_ops->MultiVecLocalInnerProd(nsdIP,
				(void**)&x_vec,(void**)&y_vec,is_vec,
				start,end,inner_prod,ldIP,ops->lapack_ops);
		BVRestoreArrayRead(y, &y_array);
    }
    BVRestoreArrayRead(x,&x_array);
	return;
}
static void MultiVecSetRandomValue (BV x, int start, int end, struct OPS_ *ops)
{
	BVSetActiveColumns(x,start,end);
	BVSetRandom(x);
	return;
}
static void MultiVecAxpby (double alpha, BV x, 
		double beta, BV y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
    PetscScalar *y_array;
    int x_nrows, x_ncols, y_nrows, y_ncols;

    BVGetArray(y,&y_array);
    BVGetSizes(y,&y_nrows,NULL,&y_ncols);
    LAPACKVEC y_vec;
    y_vec.nrows = y_nrows; y_vec.ncols = y_ncols;
    y_vec.ldd   = y_nrows; y_vec.data  = y_array;
    if (x==NULL) {
#if 0
       ops->lapack_ops->MultiVecAxpby(alpha,
	     NULL,beta,(void**)&y_vec,start,end,ops->lapack_ops);
#else
       BVSetActiveColumns(y,start[1],end[1]);
       BVScale(y,beta);
#endif
    }
	else {
		if (x!=y) {
			BVSetActiveColumns(x,start[0],end[0]);
			BVSetActiveColumns(y,start[1],end[1]);
			BVMult(y,alpha,beta,x,NULL);
		}
		else if (start[0]==start[1]) {
			BVSetActiveColumns(y,start[1],end[1]);
			BVScale(y,(alpha+beta));
		}
		else {
			assert(end[0]<=start[1]||end[1]<=start[0]);
			const PetscScalar *x_array;
			LAPACKVEC x_vec;
			BVGetArrayRead(x,&x_array);
			BVGetSizes(x,&x_nrows,NULL,&x_ncols);
			x_vec.nrows = x_nrows;
			x_vec.ncols = x_ncols;
			x_vec.ldd   = x_nrows;
			x_vec.data  = (double *)x_array; 
			ops->lapack_ops->MultiVecAxpby(alpha,
					(void**)&x_vec,beta,(void**)&y_vec,start,end,ops->lapack_ops);
			BVRestoreArrayRead(x, &x_array);
		}
	}
    BVRestoreArray(y, &y_array);

    return;
}
static void MatDotMultiVec (Mat mat, BV x, 
		BV y, int *start, int *end, struct OPS_ *ops)
{
      
	assert(end[0]-start[0]==end[1]-start[1]);
	int nrows_x, nrows_y;
	BVGetSizes(x, &nrows_x, NULL, NULL);
	BVGetSizes(y, &nrows_y, NULL, NULL);

	if (nrows_x==nrows_y) {
		if (mat==NULL) {
			MultiVecAxpby(1.0, x, 0.0, y, start, end, ops);
		}
		else {
			/* sometimes Active does not work */
			assert(x!=y);
			if (end[0]-start[0] < 5) {
				int ncols = end[1]-start[1], col;
				Vec vec_x, vec_y;      
				for (col = 0; col < ncols; ++col) {
					BVGetColumn(x, start[0]+col, &vec_x);
					BVGetColumn(y, start[1]+col, &vec_y);
					MatMult(mat, vec_x, vec_y); 
					BVRestoreColumn(x, start[0]+col, &vec_x);
					BVRestoreColumn(y, start[1]+col, &vec_y);
				}
			}
			else {
				BVSetActiveColumns(x,start[0],end[0]);
				BVSetActiveColumns(y,start[1],end[1]);
				BVMatMult(x,mat,y);
			}
		}
	}
	else {
		assert(mat!=NULL);
		Vec vec_x, vec_y;      
		int ncols = end[1]-start[1], col;
		for (col = 0; col < ncols; ++col) {
			BVGetColumn(x, start[0]+col, &vec_x);
			BVGetColumn(y, start[1]+col, &vec_y);
			MatMult(mat, vec_x, vec_y); 
			BVRestoreColumn(x, start[0]+col, &vec_x);
			BVRestoreColumn(y, start[1]+col, &vec_y);
		}
	}
	return;
}
static void MatTransDotMultiVec (Mat mat, BV x, 
		BV y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
	int nrows_x, nrows_y;
	BVGetSizes(x, &nrows_x, NULL, NULL);
	BVGetSizes(y, &nrows_y, NULL, NULL);
	if (nrows_x==nrows_y) {
		if (mat==NULL) {
			MultiVecAxpby(1.0, x, 0.0, y, start, end, ops);
		}
		else {
			BVSetActiveColumns(x,start[0],end[0]);
			BVSetActiveColumns(y,start[1],end[1]);
			BVMatMultTranspose(x,mat,y);
		}
	}
	else {
		Vec vec_x, vec_y;
		assert(end[0]-start[0]==end[1]-start[1]);
		int ncols = end[1]-start[1], col;
		for (col = 0; col < ncols; ++col) {
			BVGetColumn(x, start[0]+col, &vec_x);
			BVGetColumn(y, start[1]+col, &vec_y);
			MatMultTranspose(mat, vec_x, vec_y); 
			BVRestoreColumn(x, start[0]+col, &vec_x);
			BVRestoreColumn(y, start[1]+col, &vec_y);
		}
	}
	return;
}
static void MultiVecLinearComb (BV x, BV y, int is_vec, 
		int    *start, int *end, 
		double *coef , int ldc , 
		double *beta , int incb, struct OPS_ *ops)
{
    assert(is_vec==0);
    PetscScalar *y_array;
    int x_nrows, x_ncols, y_nrows, y_ncols;

    BVGetArray(y,&y_array);
    BVGetSizes(y,&y_nrows,NULL,&y_ncols);
    LAPACKVEC y_vec;
    y_vec.nrows = y_nrows;
    y_vec.ncols = y_ncols;
    y_vec.ldd   = y_nrows;
    y_vec.data  = (double *)y_array;
    if (x==NULL) {
		ops->lapack_ops->MultiVecLinearComb(
			NULL,(void**)&y_vec,is_vec,
			start,end,coef,ldc,beta,incb,ops->lapack_ops);
    }
    else {
       //assert(end[0]<=start[1]||end[1]<=start[0]);
		const PetscScalar *x_array;
		LAPACKVEC x_vec;
		BVGetArrayRead(x,&x_array);
		BVGetSizes(x,&x_nrows,NULL,&x_ncols);
		x_vec.nrows = x_nrows; 
		x_vec.ncols = x_ncols; 
		x_vec.ldd   = x_nrows; 
		x_vec.data  = (double*)x_array; 
		ops->lapack_ops->MultiVecLinearComb(
			(void**)&x_vec,(void**)&y_vec,is_vec,
			start,end,coef,ldc,beta,incb,ops->lapack_ops);
		BVRestoreArrayRead(x, &x_array);
    }
    BVRestoreArray(y, &y_array);
    return;
}
/* Encapsulation */
static void SLEPC_MatView (void *mat, struct OPS_ *ops)
{
	MatView((Mat)mat,PETSC_VIEWER_STDOUT_WORLD);
	return;
}
static void SLEPC_MatAxpby (double alpha, void *matX, 
		double beta, void *matY, struct OPS_ *ops)
{
	/* y = alpha x + beta y */
	if (beta == 1.0) {
		/* SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN or SUBSET_NONZERO_PATTERN */
		/* y = alpha x + y */
		MatAXPY((Mat)matY,alpha,(Mat)matX,SUBSET_NONZERO_PATTERN);
	}
	else if (alpha == 1.0) {
		/* y = x + beta y */
		MatAYPX((Mat)matY,beta,(Mat)matX,SUBSET_NONZERO_PATTERN);
	}
	else {
		if (beta == 0.0) {
			MatCopy((Mat)matX,(Mat)matY,DIFFERENT_NONZERO_PATTERN);
			MatScale((Mat)matY,alpha);
		}
		else {
			MatAXPY((Mat)matY,(alpha-1.0)/beta,(Mat)matX,SUBSET_NONZERO_PATTERN);
			MatAYPX((Mat)matY,beta,(Mat)matX,SUBSET_NONZERO_PATTERN);
		}
	}
	return;
}
/* multi-vec */
static void SLEPC_MultiVecCreateByMat (void ***des_vec, int num_vec, void *src_mat, struct OPS_ *ops)
{
	MultiVecCreateByMat ((BV*)des_vec,num_vec,(Mat)src_mat,ops);		
	return;
}
static void SLEPC_MultiVecDestroy (void ***des_vec, int num_vec, struct OPS_ *ops)
{
	MultiVecDestroy ((BV*)des_vec,num_vec,ops);
	return;
}
static void SLEPC_MultiVecView (void **x, int start, int end, struct OPS_ *ops)
{
	MultiVecView ((BV)x,start,end,ops);
	return;
}
static void SLEPC_MultiVecLocalInnerProd (char nsdIP, 
		void **x, void **y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	MultiVecLocalInnerProd (nsdIP, 
			(BV)x,(BV)y,is_vec,start,end, 
			inner_prod,ldIP,ops);
	return;
}
static void SLEPC_MultiVecSetRandomValue (void **x, int start, int end, struct OPS_ *ops)
{
	MultiVecSetRandomValue ((BV)x,start,end,ops);
	return;
}
static void SLEPC_MultiVecAxpby (double alpha, void **x, 
		double beta, void **y, int *start, int *end, struct OPS_ *ops)
{
	MultiVecAxpby (alpha,(BV)x,beta,(BV)y,start,end,ops);
	return;
}
static void SLEPC_MatDotMultiVec (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatDotMultiVec ((Mat)mat,(BV)x,(BV)y,start,end,ops);
	return;
}
static void SLEPC_MatTransDotMultiVec (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatTransDotMultiVec ((Mat)mat,(BV)x,(BV)y,start,end,ops);
	return;
}
static void SLEPC_MultiVecLinearComb (
		void **x , void **y, int is_vec, 
		int    *start, int  *end, 
		double *coef , int  ldc , 
		double *beta , int  incb, struct OPS_ *ops)
{
        //assert(x!=y);
	MultiVecLinearComb (
			(BV)x, (BV)y, is_vec, 
			start, end , 
			coef , ldc , 
			beta , incb, ops);
	return;
}
static void SLEPC_MultiVecQtAP (char ntsA, char nsdQAP, 
		void **mvQ  , void *matA, void   **mvP, int is_vec, 
		int  *start , int  *end , double *qAp , int ldQAP ,
		void **mv_ws, struct OPS_ *ops)
{
	assert(nsdQAP!='T');
	assert(is_vec==0);
	if ( nsdQAP=='D' || ( mvQ==mvP&&(start[0]!=start[1]||end[0]!=end[1]) ) ) {
		DefaultMultiVecQtAP (ntsA, nsdQAP, 
				mvQ, matA, mvP, is_vec, 
				start, end, qAp, ldQAP,
				mv_ws, ops);
	}
	else {
		BVSetActiveColumns((BV)mvQ, start[0], end[0]);
		BVSetActiveColumns((BV)mvP, start[1], end[1]);
		BVSetMatrix((BV)mvP,(Mat)matA,PETSC_FALSE);
		BVSetMatrix((BV)mvQ,(Mat)matA,PETSC_FALSE);
		Mat dense_mat; const double *source;
		int nrows = end[0]-start[0], ncols = end[1]-start[1], col;		
		MatCreateSeqDense(PETSC_COMM_SELF,end[0],end[1],NULL,&dense_mat);        
		/* Qt A P */
		/* M must be a sequential dense Mat with dimensions m,n at least, 
		 * where m is the number of active columns of Q 
		 * and n is the number of active columns of P. 
		 * Only rows (resp. columns) of M starting from ly (resp. lx) are computed, 
		 * where ly (resp. lx) is the number of leading columns of Q (resp. P). */		
		BVDot((BV)mvP, (BV)mvQ, dense_mat);		
		MatDenseGetArrayRead(dense_mat, &source);        
		/* 当 qAp 连续存储 */

		if (start[0]==0&&ldQAP==nrows) {
		   memcpy(qAp,source+nrows*start[1],nrows*ncols*sizeof(double)); 	
		}
		else {
		   for(col = 0; col < ncols; ++col) {
		      memcpy(qAp+ldQAP*col, source+end[0]*(start[1]+col)+start[0], nrows*sizeof(double));
		   }
		}
		MatDenseRestoreArrayRead(dense_mat, &source);
		MatDestroy(&dense_mat);		
	}	
	return;
}

static void SLEPC_MultiVecInnerProd      (char nsdIP, void **x, void **y, int is_vec, int *start, int *end, 
	double *inner_prod, int ldIP, struct OPS_ *ops)
{
	if ( nsdIP=='D' || ( x==y&&(start[0]!=start[1]||end[0]!=end[1]) ) ) {
		DefaultMultiVecInnerProd (nsdIP, x, y, is_vec, start, end, 
			inner_prod, ldIP, ops);
	}
	else {
	   BVSetActiveColumns((BV)x, start[0], end[0]);
	   BVSetActiveColumns((BV)y, start[1], end[1]);
	   BVSetMatrix((BV)y,NULL,PETSC_FALSE);
	   BVSetMatrix((BV)x,NULL,PETSC_FALSE);
	   Mat dense_mat; const double *source;
	   int nrows = end[0]-start[0], ncols = end[1]-start[1], col;		
	   MatCreateSeqDense(PETSC_COMM_SELF,end[0],end[1],NULL,&dense_mat);        
	   BVDot((BV)y, (BV)x, dense_mat);		
	   MatDenseGetArrayRead(dense_mat, &source);        
	   /* 当 inner_prod 连续存储 */
	   if (start[0]==0&&ldIP==nrows) {
	      memcpy(inner_prod,source+nrows*start[1],nrows*ncols*sizeof(double)); 	
	   }
	   else {
	      for(col = 0; col < ncols; ++col) {
		 memcpy(inner_prod+ldIP*col, source+end[0]*(start[1]+col)+start[0], nrows*sizeof(double));
	      }
	   }
	   MatDenseRestoreArrayRead(dense_mat, &source);
	   MatDestroy(&dense_mat);		
	}
	return;
}



static int SLEPC_GetOptionFromCommandLine (
		const char *name, char type, void *value,
		int argc, char* argv[], struct OPS_ *ops)
{
	PetscBool set;
	int *int_value; double *dbl_value; char *str_value; 
	switch (type) {
		case 'i':
			int_value = (int*)value; 
			PetscOptionsGetInt(NULL, NULL, name, int_value, &set);
			break;
		case 'f':
			dbl_value = (double*)value; 
			PetscOptionsGetReal(NULL, NULL, name, dbl_value, &set);
			break;
		case 's':
			str_value = (char*) value;
			PetscOptionsGetString(NULL, NULL, name, str_value, 8, &set);
			//set = DefaultGetOptionFromCommandLine(name, type, value, argc, argv, ops);
			break;
			default:
		break;
	}	        
	return set;
}



void OPS_SLEPC_Set (struct OPS_ *ops)
{
	ops->GetOptionFromCommandLine = SLEPC_GetOptionFromCommandLine;
	/* mat */
	ops->MatAxpby               = SLEPC_MatAxpby;
	ops->MatView                = SLEPC_MatView;
	/* multi-vec */
	ops->MultiVecCreateByMat    = SLEPC_MultiVecCreateByMat   ;
	ops->MultiVecDestroy        = SLEPC_MultiVecDestroy       ;
	ops->MultiVecView           = SLEPC_MultiVecView          ;
	ops->MultiVecLocalInnerProd = SLEPC_MultiVecLocalInnerProd;
	ops->MultiVecSetRandomValue = SLEPC_MultiVecSetRandomValue;
	ops->MultiVecAxpby          = SLEPC_MultiVecAxpby         ;
	ops->MatDotMultiVec         = SLEPC_MatDotMultiVec        ;
	ops->MatTransDotMultiVec    = SLEPC_MatTransDotMultiVec   ;
	ops->MultiVecLinearComb     = SLEPC_MultiVecLinearComb    ;
	if (0) {// no efficiency
	   ops->MultiVecQtAP        = SLEPC_MultiVecQtAP          ;
	   ops->MultiVecInnerProd   = SLEPC_MultiVecInnerProd     ; 
	}

	return;
}


void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m)
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
	void *A, *B; OPS *ops;
	Mat      slepc_matA, slepc_matB;
	int flag = 0;	
	PetscInt n = 200, m = 200;
	GetPetscMat(&slepc_matA, &slepc_matB, n, m);
	MatAssemblyBegin(slepc_matA, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(slepc_matA, MAT_FINAL_ASSEMBLY);
	ops = slepc_ops; A = (void*)(slepc_matA); B = (void*)(slepc_matB);
	/*
		nevConv: the number of the required eigenpairs
		tol_gcg[0]: corresponding to the absolute error
		tol_gcg[1]: corresponding to the relative error
		max_iter_gcg: the maximum iteration of GCG 
	*/
	int nevConv  = 30;
	double gapMin = 1e-5;
	int nevGiven = 0, block_size = nevConv/5, nevMax = 50;
	int nevInit = 20;
	nevInit = nevInit<nevMax?nevInit:nevMax;
	int max_iter_gcg = 500; double tol_gcg[2] = {1e-1,1e-8};
	
	double *eval; void **evec;
	eval = malloc(nevMax*sizeof(double));
	memset(eval,0,nevMax*sizeof(double));
	ops->MultiVecCreateByMat(&evec,nevMax,A,ops);
	ops->MultiVecSetRandomValue(evec,0,nevMax,ops);
	void **gcg_mv_ws[4]; double *dbl_ws = NULL; int *int_ws = NULL;
	GCGE_Create(A, nevMax, block_size, nevInit, gcg_mv_ws, dbl_ws, int_ws, ops);
	int sizeV = nevInit + 2*block_size;
	int length_dbl_ws = 2*sizeV*sizeV+10*sizeV
		+(nevMax+2*block_size)+(nevMax)*block_size;
	int length_int_ws = 6*sizeV+2*(block_size+3);
	dbl_ws = malloc(length_dbl_ws*sizeof(double));
	memset(dbl_ws,0,length_dbl_ws*sizeof(double));
	int_ws = malloc(length_int_ws*sizeof(int));
	memset(int_ws,0,length_int_ws*sizeof(int));
	srand(0);
	double time_start, time_interval;
	time_start = ops->GetWtime();
	ops->Printf("===============================================\n");
	ops->Printf("GCG Eigen Solver\n");
	EigenSolverSetup_GCG(gapMin,nevInit,nevMax,block_size,
		tol_gcg,max_iter_gcg,flag,gcg_mv_ws,dbl_ws,int_ws,ops);
	GCGE_Setparameters(gapMin,ops);
	ops->EigenSolver(A,B,eval,evec,nevGiven,&nevConv,ops);
	ops->Printf("numIter = %d, nevConv = %d\n",
			((GCGSolver*)ops->eigen_solver_workspace)->numIter, nevConv);
	ops->Printf("++++++++++++++++++++++++++++++++++++++++++++++\n");
	time_interval = ops->GetWtime() - time_start;
	ops->Printf("Time is %.3f\n", time_interval);
	GCGE_Destroymvws(gcg_mv_ws, dbl_ws, int_ws, nevMax, block_size,ops);
	ops->Printf("eigenvalues\n");
	int idx;
	for (idx = 0; idx < nevConv; ++idx) {
		ops->Printf("%d: %6.14e\n",idx+1,eval[idx]);
	}
	ops->MultiVecDestroy(&(evec),nevMax,ops);
	free(eval);
	MatDestroy(&slepc_matA);
	MatDestroy(&slepc_matB);	
	OPS_Destroy (&slepc_ops);
	SlepcFinalize();	
	return 0;
}


