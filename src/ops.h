/**
 *    @file  ops.h
 *   @brief  operations
 *
 *  单向量多向量操作, 正交化操作, 多重网格操作, 线性求解器, 特征值求解器
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/13
 *      Revision:  none
 */
#ifndef  _OPS_H_
#define  _OPS_H_

#include "ops_config.h"

#if OPS_USE_OMP
#include <omp.h>
#endif

#if OPS_USE_INTEL_MKL
#include <omp.h>
#include <mkl.h>
#include <mkl_spblas.h>
#endif

#if OPS_USE_MPI
#include <mpi.h>

extern double *debug_ptr;
/* 矩阵块求和操作 */
int CreateMPIDataTypeSubMat(MPI_Datatype *submat_type,
	int nrows, int ncols, int ldA);
int DestroyMPIDataTypeSubMat(MPI_Datatype *submat_type);
int CreateMPIOpSubMatSum(MPI_Op *op);
int DestroyMPIOpSubMatSum(MPI_Op *op); 
#endif
/* 数据分组 */
int SplitDoubleArray(double *destin, int length, 
	int num_group, double min_gap, int min_num, int* displs, 
	double *dbl_ws, int *int_ws);

typedef struct OPS_ {
	void   (*Printf) (const char *fmt, ...);
	double (*GetWtime) (void);
	int    (*GetOptionFromCommandLine) (
			const char *name, char type, void *data,
			int argc, char* argv[], struct OPS_ *ops);	  
	/* mat */
	void (*MatView) (void *mat, struct OPS_ *ops);  
	/* y = alpha x + beta y */
	void (*MatAxpby) (double alpha, void *matX, double beta, void *matY, struct OPS_ *ops);
	/* vec */
	void (*VecCreateByMat) (void **des_vec, void *src_mat, struct OPS_ *ops);
	void (*VecCreateByVec) (void **des_vec, void *src_vec, struct OPS_ *ops);
	void (*VecDestroy)     (void **des_vec,                struct OPS_ *ops);
	void (*VecView)           (void *x, 	                         struct OPS_ *ops);
	/* inner_prod = x'y */
	void (*VecInnerProd)      (void *x, void *y, double *inner_prod, struct OPS_ *ops);
	/* inner_prod = x'y for each proc */
	void (*VecLocalInnerProd) (void *x, void *y, double *inner_prod, struct OPS_ *ops);
	void (*VecSetRandomValue) (void *x,                              struct OPS_ *ops);
	/* y = alpha x + beta y */
	void (*VecAxpby)          (double alpha, void *x, double beta, void *y, struct OPS_ *ops);
	/* y = mat  * x */
	void (*MatDotVec)      (void *mat, void *x, void *y, struct OPS_ *ops);
	/* y = mat' * x */
	void (*MatTransDotVec) (void *mat, void *x, void *y, struct OPS_ *ops);
	/* multi-vec */
	void (*MultiVecCreateByMat)      (void ***multi_vec, int num_vec, void *src_mat, struct OPS_ *ops);
	void (*MultiVecCreateByVec)      (void ***multi_vec, int num_vec, void *src_vec, struct OPS_ *ops);
	void (*MultiVecCreateByMultiVec) (void ***multi_vec, int num_vec, void **src_mv, struct OPS_ *ops);
	void (*MultiVecDestroy)          (void ***multi_vec, int num_vec,                struct OPS_ *ops);
	/* *vec = multi_vec[col] */
	void (*GetVecFromMultiVec)    (void **multi_vec, int col, void **vec, struct OPS_ *ops);
	void (*RestoreVecForMultiVec) (void **multi_vec, int col, void **vec, struct OPS_ *ops);
	void (*MultiVecView)           (void **x, int start, int end, struct OPS_ *ops);
	void (*MultiVecLocalInnerProd) (char nsdIP, 
			void **x, void **y, int is_vec, int *start, int *end, 
			double *inner_prod, int ldIP, struct OPS_ *ops);
	void (*MultiVecInnerProd)      (char nsdIP, 
			void **x, void **y, int is_vec, int *start, int *end, 
			double *inner_prod, int ldIP, struct OPS_ *ops);
	void (*MultiVecSetRandomValue) (void **multi_vec, 
			int    start , int  end , struct OPS_ *ops);
	void (*MultiVecAxpby)          (
			double alpha , void **x , double beta, void **y, 
			int    *start, int  *end, struct OPS_ *ops);
	/* y = x coef + y diag(beta) */
	void (*MultiVecLinearComb)     (
			void   **x   , void **y , int is_vec, 
			int    *start, int  *end, 
			double *coef , int  ldc , 
			double *beta , int  incb, struct OPS_ *ops);
	void (*MatDotMultiVec)      (void *mat, void **x, void **y, 
			int  *start, int *end, struct OPS_ *ops);
	void (*MatTransDotMultiVec) (void *mat, void **x, void **y, 
			int  *start, int *end, struct OPS_ *ops);
	/* qAp = Qt A P */
	void (*MultiVecQtAP)        (char ntsA, char ntsdQAP, 
			void **mvQ  , void   *matA  , void   **mvP, int is_vec, 
			int  *start , int    *end   , double *qAp , int ldQAP , 
			void **mv_ws, struct OPS_ *ops);
	/* Dense matrix vector ops */ 
	struct OPS_ *lapack_ops; /* 稠密矩阵向量的操作 */
	void (*DenseMatQtAP) (char ntluA, char nsdC,
			int nrowsA, int ncolsA, /* matA 的行列数 */
			int nrowsC, int ncolsC, /* matC 的行列数 */
			double alpha, double *matQ, int ldQ, 
		              double *matA, int ldA,
	                      double *matP, int ldP,
			double beta , double *matC, int ldC,
			double *dbl_ws);
	void (*DenseMatOrth) (double *mat, int nrows, int ldm, 
			int start, int *end, double orth_zero_tol,
			double *dbl_ws, int length, int *int_ws);
	/* linear solver */
	void (*LinearSolver)      (void *mat, void *b, void *x, 
			struct OPS_ *ops);
	void *linear_solver_workspace;
	void (*MultiLinearSolver) (void *mat, void **b, void **x, 
			int *start, int *end, struct OPS_ *ops); 
	void *multi_linear_solver_workspace;	
	/* orthonormal */
	void (*MultiVecOrth) (void **x, int start_x, int *end_x, 
			void *B, struct OPS_ *ops);
	void *orth_workspace;
	/* multi grid */
	/* get multigrid operator for num_levels = 4
 	 * P0     P1       P2
 	 * A0     A1       A2        A3
	 * B0  P0'B0P0  P1'B1P1   P2'B2P2 
	 * A0 is the original matrix */
	void (*MultiGridCreate)  (void ***A_array, void ***B_array, void ***P_array,
		int *num_levels, void *A, void *B, struct OPS_ *ops);
	/* free A1 A2 A3 B1 B2 B3 P0 P1 P2 
	 * A0 and B0 are just pointers */
	void (*MultiGridDestroy) (void ***A_array , void ***B_array, void ***P_array,
		int *num_levels, struct OPS_ *ops);
	void (*VecFromItoJ)      (void **P_array, int level_i, int level_j, 
		void *vec_i, void *vec_j, void **vec_ws, struct OPS_ *ops);
	void (*MultiVecFromItoJ) (void **P_array, int level_i, int level_j, 
		void **multi_vec_i, void **multi_vec_j, int *startIJ, int *endIJ, 
		void ***multi_vec_ws, struct OPS_ *ops);
	/* eigen solver */
	void (*EigenSolver) (void *A, void *B , double *eval, void **evec,
		int nevGiven, int *nevConv, struct OPS_ *ops);	
	void *eigen_solver_workspace;
	
	/* for pas */
	struct OPS_ *app_ops;
} OPS;

void OPS_Create  (OPS **ops);
void OPS_Setup   (OPS  *ops);
void OPS_Destroy (OPS **ops);

/* multi-vec */
void DefaultPrintf (const char *fmt, ...);
double DefaultGetWtime (void);
int  DefaultGetOptionFromCommandLine (
		const char *name, char type, void *value,
		int argc, char* argv[], struct OPS_ *ops);
void DefaultMultiVecCreateByVec      (void ***multi_vec, int num_vec, void *src_vec, struct OPS_ *ops);
void DefaultMultiVecCreateByMat      (void ***multi_vec, int num_vec, void *src_mat, struct OPS_ *ops);
void DefaultMultiVecCreateByMultiVec (void ***multi_vec, int num_vec, void **src_mv, struct OPS_ *ops);
void DefaultMultiVecDestroy          (void ***multi_vec, int num_vec, struct OPS_ *ops);
void DefaultGetVecFromMultiVec    (void **multi_vec, int col, void **vec, struct OPS_ *ops);
void DefaultRestoreVecForMultiVec (void **multi_vec, int col, void **vec, struct OPS_ *ops);
void DefaultMultiVecView           (void **x, int start, int end, struct OPS_ *ops);
void DefaultMultiVecLocalInnerProd (char nsdIP, void **x, void **y, int is_vec, int *start, int *end, 
	double *inner_prod, int ldIP, struct OPS_ *ops);
void DefaultMultiVecInnerProd      (char nsdIP, void **x, void **y, int is_vec, int *start, int *end, 
	double *inner_prod, int ldIP, struct OPS_ *ops);
void DefaultMultiVecSetRandomValue (void **x, int start, int end, struct OPS_ *ops);
void DefaultMultiVecAxpby          (
	double alpha , void **x , double beta, void **y, 
	int    *start, int  *end, struct OPS_ *ops);
void DefaultMultiVecLinearComb     (
	void   **x   , void **y , int is_vec, 
	int    *start, int  *end, 
	double *coef , int  ldc , 
	double *beta , int  incb, struct OPS_ *ops);
void DefaultMatDotMultiVec      (void *mat, void **x, void **y, 
	int  *start, int *end, struct OPS_ *ops);
void DefaultMatTransDotMultiVec (void *mat, void **x, void **y, 
	int  *start, int *end, struct OPS_ *ops);
void DefaultMultiVecQtAP        (char ntsA, char ntsdQAP, 
	void **mvQ   , void   *matA  , void   **mvP, int is_vec, 
	int  *startQP, int    *endQP , double *qAp , int ldQAP , 
	void **mv_ws, struct OPS_ *ops);
/* multi-grid */
void DefaultVecFromItoJ(void **P_array, int level_i, int level_j, 
	void *vec_i, void *vec_j, void **vec_ws, struct OPS_ *ops);
void DefaultMultiVecFromItoJ(void **P_array, int level_i, int level_j, 
	void **multi_vec_i, void **multi_vec_j, int *startIJ, int *endIJ, 
	void ***multi_vec_ws, struct OPS_ *ops);

#endif  /* -- #ifndef _OPS_H_ -- */
