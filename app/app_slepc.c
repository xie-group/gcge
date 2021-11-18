/**
 *    @file  app_slepc.c
 *   @brief  app of slecp 
 *
 *  不支持单向量操作 
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/9/13
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<assert.h>
#include  	<math.h>
#include   	<memory.h>
 
#include	"app_slepc.h"

#if OPS_USE_SLEPC
#define DEBUG 0



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
#if DEBUG
	int n, N, m;
	if (mat!=NULL) {
		MatGetSize(mat, &N, &m);
		PetscPrintf(PETSC_COMM_WORLD, "mat global, N = %d, m = %d\n", N, m);
		MatGetLocalSize(mat, &n, &m);
		PetscPrintf(PETSC_COMM_WORLD, "mat local , n = %d, m = %d\n", n, m);
	}
	BVGetSizes(x, &n, &N, &m);
	PetscPrintf(PETSC_COMM_WORLD, "x local n = %d, global N = %d, ncols = %d\n", n, N, m);
	BVGetSizes(y, &n, &N, &m);
	PetscPrintf(PETSC_COMM_WORLD, "y local n = %d, global N = %d, ncols = %d\n", n, N, m);
	ops->Printf("%d,%d, %d,%d\n", start[0],end[0],start[1],end[1]);
#endif
      
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
static void SLEPC_MultiGridCreate (void ***A_array, void ***B_array, void ***P_array,
		int *num_levels, void *A, void *B, struct OPS_ *ops)
{
	/* P 是行多列少, Px 是从粗到细 */
	PetscInt m, n, level;
	Mat   *petsc_A_array = NULL, *petsc_B_array = NULL, *petsc_P_array = NULL;
	PC    pc;
	Mat   *Aarr=NULL, *Parr=NULL;

	PCCreate(PETSC_COMM_WORLD,&pc);
	PCSetOperators(pc,(Mat)A,(Mat)A);
	PCSetType(pc,PCGAMG);
	//PCGAMGSetType(pc,PCGAMGAGG);
	PCGAMGSetType(pc,PCGAMGCLASSICAL);
	PetscPrintf(PETSC_COMM_WORLD, "num_levels = %d\n", *num_levels);
	PCGAMGSetNlevels(pc,*num_levels);
	/* not force coarse grid onto one processor */
	//PCGAMGSetUseParallelCoarseGridSolve(pc,PETSC_TRUE);
	/* this will generally improve the loading balancing of the work on each level 
	 * should use parmetis */
	//   PCGAMGSetRepartition(pc, PETSC_TRUE);
	//	type 	- PCGAMGAGG, PCGAMGGEO, or PCGAMGCLASSICAL
	/* Increasing the threshold decreases the rate of coarsening. 
	 * 0.0 means keep all nonzero entries in the graph; 
	 * negative means keep even zero entries in the graph */
	//PCGAMGSetThresholdScale(pc, 0.5);
	PetscReal th[16] = {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
	   0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};
	PCGAMGSetThreshold(pc, th, 16);
	//stop coarsening once the coarse grid has less than <100000> unknowns.
	//PCGAMGSetCoarseEqLim(pc, 50000);
	//there are around <1000> equations on each process
	//PCGAMGSetProcEqLim(pc, 1000);
	PetscPrintf(PETSC_COMM_WORLD, "before PCGAMG SetUp\n");
	PCSetUp(pc);
	PetscPrintf(PETSC_COMM_WORLD, "after  PCGAMG SetUp\n");
	/* the size of Aarr is num_levels-1, Aarr[0] is the coarsest matrix */
	PCGetCoarseOperators(pc, num_levels, &Aarr);
	PetscPrintf(PETSC_COMM_WORLD, "num_levels = %d\n", *num_levels);
	/* the size of Parr is num_levels-1 */
	PCGetInterpolations(pc, num_levels, &Parr);
	/* we should make that zero is the refinest level */
	/* when num_levels == 5, 1 2 3 4 of A_array == 3 2 1 0 of Aarr */
	petsc_A_array = malloc(sizeof(Mat)*(*num_levels));
	petsc_P_array = malloc(sizeof(Mat)*((*num_levels)-1));
	petsc_A_array[0] = (Mat)A;
	MatGetSize(petsc_A_array[0], &m, &n);
	PetscPrintf(PETSC_COMM_WORLD, "A_array[%d], m = %d, n = %d\n", 0, m, n );
	for (level = 1; level < (*num_levels); ++level) {
		petsc_A_array[level] = Aarr[(*num_levels)-level-1];
		MatGetSize(petsc_A_array[level], &m, &n);
		PetscPrintf(PETSC_COMM_WORLD, "A_array[%d], m = %d, n = %d\n", level, m, n );

		petsc_P_array[level-1] = Parr[(*num_levels)-level-1];
		MatGetSize(petsc_P_array[level-1], &m, &n);
		PetscPrintf(PETSC_COMM_WORLD, "P_array[%d], m = %d, n = %d\n", level-1, m, n );
	}
	(*A_array) = (void**)petsc_A_array;
	(*P_array) = (void**)petsc_P_array;

	PetscFree(Aarr);
	PetscFree(Parr);
	PCDestroy(&pc);

	if (B!=NULL) {
		petsc_B_array = malloc(sizeof(Mat)*(*num_levels));
		petsc_B_array[0] = (Mat)B;
		MatGetSize(petsc_B_array[0], &m, &n);
		PetscPrintf(PETSC_COMM_WORLD, "B_array[%d], m = %d, n = %d\n", 0, m, n );
		/* B0  P0^T B0 P0  P1^T B1 P1   P2^T B2 P2 */
		for ( level = 1; level < (*num_levels); ++level ) {
			MatPtAP(petsc_B_array[level-1], petsc_P_array[level-1], 
					MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_B_array[level]));
			MatGetSize(petsc_B_array[level], &m, &n);
			PetscPrintf(PETSC_COMM_WORLD, "B_array[%d], m = %d, n = %d\n", level, m, n );
		}
		(*B_array) = (void**)petsc_B_array;
	}
	return;
}
static void SLEPC_MultiGridDestroy (void ***A_array , void ***B_array, void ***P_array,
		int *num_levels, struct OPS_ *ops)
{
	Mat *petsc_A_array, *petsc_B_array, *petsc_P_array;
    petsc_A_array = (Mat *)(*A_array);
    petsc_P_array = (Mat *)(*P_array);
    int level; 
    for ( level = 1; level < (*num_levels); ++level ) {
        MatDestroy(&(petsc_A_array[level]));
        MatDestroy(&(petsc_P_array[level-1]));
    }
    free(petsc_A_array);
    free(petsc_P_array);
    (*A_array) = NULL;
    (*P_array) = NULL;

    if (B_array!=NULL) {
		petsc_B_array = (Mat *)(*B_array);
		for ( level = 1; level < (*num_levels); ++level ) {
			MatDestroy(&(petsc_B_array[level]));
		}
		free(petsc_B_array);
		(*B_array) = NULL;
	}
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
#if DEBUG
		int row;
		ops->Printf("(%d, %d), (%d, %d)\n", start[0], end[0], start[1], end[1]);
		for(row = 0; row < end[0]; ++row) {
		   for(col = 0; col < end[1]; ++col) {
		      ops->Printf("%6.4e\t", source[end[0]*col+row]);
		   }
		   ops->Printf("%\n");
		}
#endif
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
#if DEBUG
	   int row;
	   for(row = 0; row < end[0]; ++row) {
	      for(col = 0; col < end[1]; ++col) {
		 ops->Printf("%6.4e\t", source[end[0]*col+row]);
	      }
	      ops->Printf("%\n");
	   }
#endif
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
	/* multi grid */
	ops->MultiGridCreate        = SLEPC_MultiGridCreate ;
	ops->MultiGridDestroy       = SLEPC_MultiGridDestroy;
	return;
}

/**
 * @brief 
 *    nbigranks = ((PetscInt)((((PetscReal)size)*proc_rate[level])/((PetscReal)unit))) * (unit);
 *    if (nbigranks < unit) nbigranks = unit<size?unit:size;
 *
 * @param petsc_A_array
 * @param petsc_B_array
 * @param petsc_P_array
 * @param num_levels
 * @param proc_rate
 * @param unit           保证每层nbigranks是unit的倍数
 */
void PETSC_RedistributeDataOfMultiGridMatrixOnEachProcess(
     Mat  *petsc_A_array, Mat   *petsc_B_array, Mat *petsc_P_array, 
     PetscInt num_levels, PetscReal *proc_rate, PetscInt unit)
{
	PetscMPIInt   rank, size;
	//PetscViewer   viewer;
	
	PetscInt      level, row;
	Mat           new_P_H;
	PetscMPIInt   nbigranks;
	PetscInt      global_nrows, global_ncols; 
	PetscInt      local_nrows , local_ncols ;
	PetscInt      new_local_ncols;
	/* 保证每层nbigranks是unit的倍数 */
	PetscInt      rstart, rend, ncols;
	const PetscInt              *cols; 
	const PetscScalar           *vals;

	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);
	
	if (proc_rate[0]<=1.0 && proc_rate[0]>0.0) {
		PetscPrintf(PETSC_COMM_WORLD, "Warning the refinest matrix cannot be redistributed\n");
	}

	/* 不改变最细层的进程分布 */
	MPI_Comm_dup(PETSC_COMM_WORLD, &MG_COMM[0][0]);
	MG_COMM[0][1]    = MPI_COMM_NULL;
	MG_INTERCOMM[0]  = MPI_COMM_NULL;
	MG_COMM_COLOR[0] = 0;
	for (level = 1; level < num_levels; ++level) {
		MatGetSize(petsc_P_array[level-1], &global_nrows, &global_ncols);
		/* 在设定new_P_H的局部行时已经不能用以前P的局部行，因为当前层的A可能已经改变 */
		MatGetLocalSize(petsc_A_array[level-1], &local_nrows, &local_ncols);
		/* 应该通过ncols_P，即最粗层矩阵大小和进程总数size确定nbigranks */
		nbigranks = ((PetscInt)((((PetscReal)size)*proc_rate[level])/((PetscReal)unit))) * (unit);
		if (nbigranks < unit) nbigranks = unit<size?unit:size;
		/* 若proc_rate设为(0,1)之外，则不进行数据重分配/ */
		if (proc_rate[level]>1.0 || proc_rate[level]<=0.0 || nbigranks >= size || nbigranks <= 0) 		{
			PetscPrintf(PETSC_COMM_WORLD, "Retain data distribution of %D level\n", level);
			/* 创建分层矩阵的通信域 */
			MG_COMM_COLOR[level] = 0;
			/* TODO: 是否可以直接赋值
			 * MG_COMM[level][0] = PETSC_COMM_WORLD */ 
			MPI_Comm_dup(PETSC_COMM_WORLD, &MG_COMM[level][0]);
			MG_COMM[level][1]   = MPI_COMM_NULL;
			MG_INTERCOMM[level] = MPI_COMM_NULL;
			continue; /* 直接到下一次循环 */
		} else {
			PetscPrintf(PETSC_COMM_WORLD, "Redistribute data of %D level\n", level);
			PetscPrintf(PETSC_COMM_WORLD, "nbigranks[%D] = %D\n", level, nbigranks);
		}
		/* 上面的判断已经保证 0 < nbigranks < size */
		/* 创建分层矩阵的通信域 */
		int comm_color, local_leader, remote_leader;
		/* 对0到nbigranks-1进程平均分配global_ncols */
		new_local_ncols = 0;
		if (rank < nbigranks) {
			new_local_ncols = global_ncols/nbigranks;
			if (rank < global_ncols%nbigranks) {
				++new_local_ncols;
			}
			comm_color    = 0;
			local_leader  = 0;
			remote_leader = nbigranks;
		} else {
			comm_color    = 1;
			local_leader  = 0; /* 它的全局进程号是nbigranks */
			remote_leader = 0;
		}
      	/* 在不同进程中MG_COMM_COLOR[level]是不一样的值，它表征该进程属于哪个通讯域 */
      	MG_COMM_COLOR[level] = comm_color;
    	/* 分成两个子通讯域, MG_COMM[level][0]从0~(nbigranks-1)
    	 * MG_COMM[level][0]从nbigranks~(size-1) */
    	MPI_Comm_split(PETSC_COMM_WORLD, comm_color, rank, &MG_COMM[level][comm_color]);
    	MPI_Intercomm_create(MG_COMM[level][comm_color], local_leader, 
	    	PETSC_COMM_WORLD, remote_leader, level, &MG_INTERCOMM[level]);

      	int aux_size = -1, aux_rank = -1;
      	MPI_Comm_rank(MG_COMM[level][comm_color], &aux_rank);
      	MPI_Comm_size(MG_COMM[level][comm_color], &aux_size);
     	PetscPrintf(PETSC_COMM_SELF, "aux %D/%D, global %D/%D\n", 
			aux_rank, aux_size, rank, size);  

      	/* 创建新的延拓矩阵, 并用原始的P为之赋值
		 * 新的P与原来的P只有 局部列数new_local_ncols 不同 */
      	MatCreate(PETSC_COMM_WORLD, &new_P_H);
      	MatSetSizes(new_P_H, local_nrows, new_local_ncols, global_nrows, global_ncols);
      	//MatSetFromOptions(new_P_H);
      	/* can be improved */
      	//MatSeqAIJSetPreallocation(new_P_H, 5, NULL);
      	//MatMPIAIJSetPreallocation(new_P_H, 3, NULL, 2, NULL);
      	MatSetUp(new_P_H);
      	MatGetOwnershipRange(petsc_P_array[level-1], &rstart, &rend);
      	for(row = rstart; row < rend; ++row) {
			MatGetRow(petsc_P_array[level-1], row, &ncols, &cols, &vals);
			MatSetValues(new_P_H, 1, &row, ncols, cols, vals, INSERT_VALUES);
			MatRestoreRow(petsc_P_array[level-1], row, &ncols, &cols, &vals);
      	}
      	MatAssemblyBegin(new_P_H,MAT_FINAL_ASSEMBLY);
      	MatAssemblyEnd(new_P_H,MAT_FINAL_ASSEMBLY);

      	MatGetLocalSize(petsc_P_array[level-1], &local_nrows, &local_ncols);
      	PetscPrintf(PETSC_COMM_SELF, "[%D] original P_H[%D] local size %D * %D\n", 
	    	rank, level, local_nrows, local_ncols);
      	MatGetLocalSize(new_P_H, &local_nrows, &local_ncols);
      	PetscPrintf(PETSC_COMM_SELF, "[%D] new P_H[%D] local size %D * %D\n", 
	    	rank, level, local_nrows, local_ncols);
      	//MatView(petsc_P_array[level-1], viewer);
      	//MatView(new_P_H, viewer);
      	/* 销毁之前的P_H A_H B_H */
      	MatDestroy(&(petsc_P_array[level-1]));
      	MatDestroy(&(petsc_A_array[level]));
      	if (petsc_B_array!=NULL) {
	 		MatDestroy(&(petsc_B_array[level]));
      	}

      	petsc_P_array[level-1] = new_P_H;
      	MatPtAP(petsc_A_array[level-1], petsc_P_array[level-1],
	    	MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_A_array[level]));
      	if (petsc_B_array!=NULL) {
	 		MatPtAP(petsc_B_array[level-1], petsc_P_array[level-1],
	    		MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_B_array[level]));
      	}
      	//MatView(petsc_A_array[num_levels-1], viewer);
      	//MatView(petsc_B_array[num_levels-1], viewer);
      	/* 这里需要修改petsc_P_array[level], 原因是
       	 * petsc_A_array[level]修改后，
      	 * 它利用原来的petsc_P_array[level]插值上来的向量已经与petsc_A_array[level]不匹配
      	 * 所以在不修改level+1层的分布结构的情况下，需要对petsc_P_array[level]进行修改 */
     	/* 如果当前层不是最粗层，并且，下一层也不进行数据重分配 */
      	if (level+1<num_levels && (proc_rate[level+1]>1.0 || proc_rate[level+1]<=0.0) ) {
	 		MatGetSize(petsc_P_array[level], &global_nrows, &global_ncols);
	 		/*需要当前层A的列 作为P的行 */
	 		MatGetLocalSize(petsc_A_array[level],   &new_local_ncols, &local_ncols);
	 		/*需要下一层A的行 作为P的列 */
	 		MatGetLocalSize(petsc_A_array[level+1], &local_nrows, &new_local_ncols);
	 		/* 创建新的延拓矩阵, 并用原始的P为之赋值 */
	 		MatCreate(PETSC_COMM_WORLD, &new_P_H);
	 		MatSetSizes(new_P_H, local_ncols, local_nrows, global_nrows, global_ncols);
	 		//MatSetFromOptions(new_P_H);
	 		/* can be improved */
	 		//MatSeqAIJSetPreallocation(new_P_H, 5, NULL);
	 		//MatMPIAIJSetPreallocation(new_P_H, 3, NULL, 2, NULL);
	 		MatSetUp(new_P_H);
	 		MatGetOwnershipRange(petsc_P_array[level], &rstart, &rend);
			for(row = rstart; row < rend; ++row) {
				MatGetRow(petsc_P_array[level], row, &ncols, &cols, &vals);
				MatSetValues(new_P_H, 1, &row, ncols, cols, vals, INSERT_VALUES);
				MatRestoreRow(petsc_P_array[level], row, &ncols, &cols, &vals);
			}
			MatAssemblyBegin(new_P_H,MAT_FINAL_ASSEMBLY);
			MatAssemblyEnd(new_P_H,MAT_FINAL_ASSEMBLY);
			/* 销毁原始的 P_H */
			MatDestroy(&(petsc_P_array[level]));
			petsc_P_array[level] = new_P_H;
      	}
   	}
   	return; 
}

#endif
