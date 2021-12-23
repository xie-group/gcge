/**
 *    @file  app_ccs.c
 *   @brief  app of ccs  
 *
 *  单向量与多向量结构是统一的
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *           ZJ Wang, for OpenMP
 *
 *       Created:  2020/8/13
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<assert.h>
#include	<math.h>
#include	<memory.h>
 
#include	"app_ccs.h"

static void MatView (CCSMAT *mat, struct OPS_ *ops)
{
	/* 第 i_row[i] 行, 第 j 列 元素非零, i data[i]
	 * j_col[j] <= i < j_col[j+1] */
	LAPACKVEC *multi_vec;
	ops->MultiVecCreateByMat((void ***)(&multi_vec), mat->ncols, mat, ops);
	int col, i; double *destin; 
	for (col = 0; col < mat->ncols; ++col) {
		for (i = mat->j_col[col]; i < mat->j_col[col+1]; ++i) {
			destin  = multi_vec->data+(multi_vec->ldd)*col+mat->i_row[i];
			*destin = mat->data[i];
		}
	}
	ops->lapack_ops->MatView((void *)multi_vec, ops->lapack_ops);
	ops->lapack_ops->MultiVecDestroy((void ***)(&multi_vec), mat->ncols, ops->lapack_ops);
	return;
}
/* multi-vec */
static void MultiVecCreateByMat (LAPACKVEC **des_vec, int num_vec, CCSMAT *src_mat, struct OPS_ *ops)
{
	(*des_vec)        = malloc(sizeof(LAPACKVEC));
	(*des_vec)->nrows = src_mat->ncols   ; 
	(*des_vec)->ncols = num_vec          ;
	(*des_vec)->ldd   = (*des_vec)->nrows;
	(*des_vec)->data  = malloc(((*des_vec)->ldd)*((*des_vec)->ncols)*sizeof(double));
	memset((*des_vec)->data,0,((*des_vec)->ldd)*((*des_vec)->ncols)*sizeof(double));
	return;
}
static void MatDotMultiVec (CCSMAT *mat, LAPACKVEC *x, 
		LAPACKVEC *y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
	assert(y->nrows==y->ldd);
	assert(x->nrows==x->ldd);
	int num_vec = end[0]-start[0]; int col; 
	if (mat!=NULL) {
#if OPS_USE_INTEL_MKL
	sparse_matrix_t csrA;
	struct matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;
	/*
	 * sparse_status_t mkl_sparse_d_create_csr (
	 *       sparse_matrix_t *A,  
	 *       const sparse_index_base_t indexing,  
	 *       const MKL_INT rows,  const MKL_INT cols,  
	 *       MKL_INT *rows_start,  MKL_INT *rows_end,  MKL_INT *col_indx,  double *values);
	 * sparse_status_t mkl_sparse_destroy (sparse_matrix_t A);
	 * sparse_status_t mkl_sparse_d_mm (
	 *       const sparse_operation_t operation,  
	 *       const double alpha,  
	 *       const sparse_matrix_t A,  const struct matrix_descr descr,  const sparse_layout_t layout,  
	 *       const double *B,  const MKL_INT columns,  const MKL_INT ldb,  
	 *       const double beta,  double *C,  const MKL_INT ldc);
	 */

	/* in process */
	mkl_sparse_d_create_csr (
			&csrA,
			SPARSE_INDEX_BASE_ZERO,  
			mat->ncols, mat->nrows,  
			mat->j_col, mat->j_col+1, mat->i_row, mat->data);
#if OPS_USE_OMP
	#pragma omp parallel num_threads(OMP_NUM_THREADS)
	{
		int id, length, offset;
		id     = omp_get_thread_num();
		length = num_vec/OMP_NUM_THREADS;
		offset = length*id;
		if (id < num_vec%OMP_NUM_THREADS) {
			++length; offset += id;
		}
		else {
			offset += num_vec%OMP_NUM_THREADS;
		} 
		/* 假设 mat 是对称矩阵, 否则 SPARSE_OPERATION_NON_TRANSPOSE 改为 SPARSE_OPERATION_TRANSPOSE */
		mkl_sparse_d_mm (
				SPARSE_OPERATION_NON_TRANSPOSE,
				1.0,
				csrA, descr, SPARSE_LAYOUT_COLUMN_MAJOR,  
				     x->data+(start[0]+offset)*x->ldd, length, x->ldd,  
				0.0, y->data+(start[1]+offset)*y->ldd, y->ldd);
	}
#else
	/* 假设 mat 是对称矩阵, 否则 SPARSE_OPERATION_NON_TRANSPOSE 改为 SPARSE_OPERATION_TRANSPOSE */
	mkl_sparse_d_mm (
			SPARSE_OPERATION_NON_TRANSPOSE,
			1.0,
			csrA, descr, SPARSE_LAYOUT_COLUMN_MAJOR,
			     x->data+start[0]*x->ldd, num_vec, x->ldd,  
			0.0, y->data+start[1]*y->ldd, y->ldd);
#endif
	mkl_sparse_destroy (csrA);

#else
	memset(y->data+(y->ldd)*start[1],0,(y->ldd)*num_vec*sizeof(double));
#if OPS_USE_OMP
	#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
	for (col = 0; col < num_vec; ++col) {
		int i, j;
		double *dm, *dx, *dy; int *i_row;
		dm = mat->data; i_row = mat->i_row;
		dx = x->data+(x->ldd)*(start[0]+col);
		dy = y->data+(y->ldd)*(start[1]+col);
		for (j = 0; j < mat->ncols; ++j, ++dx) {
			for (i = mat->j_col[j]; i < mat->j_col[j+1]; ++i) {
				dy[*i_row++] += (*dm++)*(*dx);
			}
		}
	}
#endif
	}
	else {
		ops->lapack_ops->MultiVecAxpby (1.0, (void **)x, 0.0, (void **)y, 
				start, end, ops->lapack_ops);
	}
	return;
}
static void MatTransDotMultiVec (CCSMAT *mat, LAPACKVEC *x, 
		LAPACKVEC *y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
	assert(y->nrows==y->ldd);
	assert(x->nrows==x->ldd);
	assert(mat->nrows==mat->ncols);
	/* Only for 对称矩阵 */
	MatDotMultiVec (mat, x, y, start, end, ops);
	return;
}
static void VecCreateByMat (LAPACKVEC **des_vec, CCSMAT *src_mat, struct OPS_ *ops)
{
	MultiVecCreateByMat(des_vec,1,src_mat, ops);
	return;
}
static void MatDotVec (CCSMAT *mat, LAPACKVEC *x, LAPACKVEC *y, struct OPS_ *ops)
{
	int start[2] = {0,0}, end[2] = {1,1};
	MatDotMultiVec(mat,x,y,start,end, ops);
	return;
}
static void MatTransDotVec (CCSMAT *mat, LAPACKVEC *x, LAPACKVEC *y, struct OPS_ *ops)
{
	int start[2] = {0,0}, end[2] = {1,1};
	MatTransDotMultiVec(mat,x,y,start,end, ops);
	return;
}

/* Encapsulation */
static void CCS_MatView (void *mat, struct OPS_ *ops)
{
	MatView ((CCSMAT *)mat, ops);
	return;
}
/* vec */
static void CCS_VecCreateByMat (void **des_vec, void *src_mat, struct OPS_ *ops)
{
	VecCreateByMat ((LAPACKVEC **)des_vec, (CCSMAT *)src_mat, ops);
	return;
}
static void CCS_MatDotVec (void *mat, void *x, void *y, struct OPS_ *ops)
{
	MatDotVec ((CCSMAT *)mat, (LAPACKVEC *)x, (LAPACKVEC *)y, ops);
	return;
}
static void CCS_MatTransDotVec (void *mat, void *x, void *y, struct OPS_ *ops)
{
	MatTransDotVec ((CCSMAT *)mat, (LAPACKVEC *)x, (LAPACKVEC *)y, ops);
	return;
}
/* multi-vec */
static void CCS_MultiVecCreateByMat (void ***des_vec, int num_vec, void *src_mat, struct OPS_ *ops)
{
	MultiVecCreateByMat ((LAPACKVEC **)des_vec, num_vec, (CCSMAT *)src_mat, ops);		
	return;
}

static void CCS_MatDotMultiVec (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatDotMultiVec ((CCSMAT *)mat, (LAPACKVEC *)x, 
			(LAPACKVEC *)y, start, end, ops);
	return;
}
static void CCS_MatTransDotMultiVec (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatTransDotMultiVec ((CCSMAT *)mat, (LAPACKVEC *)x, 
			(LAPACKVEC *)y, start, end, ops);
	return;
}

void OPS_CCS_Set (struct OPS_ *ops)
{
	assert(ops->lapack_ops==NULL);
	OPS_Create (&(ops->lapack_ops));
	OPS_LAPACK_Set (ops->lapack_ops);
	ops->Printf                   = DefaultPrintf;
	ops->GetOptionFromCommandLine = DefaultGetOptionFromCommandLine;
	ops->GetWtime                 = DefaultGetWtime;
	ops->MatView                  = CCS_MatView;
	/* vec */
	ops->VecCreateByMat           = CCS_VecCreateByMat;
	ops->VecCreateByVec           = ops->lapack_ops->VecCreateByVec   ;
	ops->VecDestroy               = ops->lapack_ops->VecDestroy       ;
	ops->VecView                  = ops->lapack_ops->VecView          ;
	ops->VecInnerProd             = ops->lapack_ops->VecInnerProd     ;
	ops->VecLocalInnerProd        = ops->lapack_ops->VecLocalInnerProd;
	ops->VecSetRandomValue        = ops->lapack_ops->VecSetRandomValue;
	ops->VecAxpby                 = ops->lapack_ops->VecAxpby         ;
	ops->MatDotVec                = CCS_MatDotVec     ;
	ops->MatTransDotVec           = CCS_MatTransDotVec;
	/* multi-vec */
	ops->MultiVecCreateByMat      = CCS_MultiVecCreateByMat;
	ops->MultiVecCreateByVec      = ops->lapack_ops->MultiVecCreateByVec     ;
	ops->MultiVecCreateByMultiVec = ops->lapack_ops->MultiVecCreateByMultiVec;
	ops->MultiVecDestroy          = ops->lapack_ops->MultiVecDestroy         ;
	ops->GetVecFromMultiVec       = ops->lapack_ops->GetVecFromMultiVec      ;
	ops->RestoreVecForMultiVec    = ops->lapack_ops->RestoreVecForMultiVec   ;
	ops->MultiVecView             = ops->lapack_ops->MultiVecView            ;
	ops->MultiVecLocalInnerProd   = ops->lapack_ops->MultiVecLocalInnerProd  ;
	ops->MultiVecInnerProd        = ops->lapack_ops->MultiVecInnerProd       ;
	ops->MultiVecSetRandomValue   = ops->lapack_ops->MultiVecSetRandomValue  ;
	ops->MultiVecAxpby            = ops->lapack_ops->MultiVecAxpby           ;
	ops->MultiVecLinearComb       = ops->lapack_ops->MultiVecLinearComb      ;
	ops->MatDotMultiVec           = CCS_MatDotMultiVec     ;
	ops->MatTransDotMultiVec      = CCS_MatTransDotMultiVec;
	return;
}

