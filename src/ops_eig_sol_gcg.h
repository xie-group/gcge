/**
 *    @file  ops_eig_sol_gcg.h
 *   @brief  GCG ����ֵ����� 
 *
 *  GCG ����ֵ�����
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/13
 *      Revision:  none
 */
#ifndef  _OPS_EIG_SOL_GCG_H_
#define  _OPS_EIG_SOL_GCG_H_

#include	"ops.h"
#include    "ops_orth.h"
#include    "ops_lin_sol.h"
#include    "app_lapack.h"

/* GCG �Ľṹ�� */
typedef struct GCGSolver_ {
	void   *A        ; void  *B      ; double sigma;
	double *eval     ; void  **evec  ; 
	int    nevMax    ; double gapMin;
	int    nevInit   ; int   nevGiven; int    nevConv;
	/* �������, ������ */
	int    block_size; double tol[2] ; int numIterMax; 
	int    numIter   ; int    sizeV  ;
	void   **mv_ws[4]; double *dbl_ws; int *int_ws;
	int    length_dbl_ws;
	int    print;
	/* �û��Զ�������Խⷨ��: 
	 * 1 ��ʾ�û��Զ���, ��ô�� ComputeW ʱ����Setup���Խⷨ�� 
	 *   ��ô, �û���Ҫ���� 
	 *   ops->MultiLinearSovler �Լ� ops->multi_linear_solver_workspace 
	 * 0 ��ʾĬ��ʹ�� BlockPCG */
	int    user_defined_multi_linear_solver;	
	/* --------�㷨�ڲ�����----------------------------------- */
	int  check_conv_max_num;
	char initX_orth_method[8] ; int    initX_orth_block_size; 
	int  initX_orth_max_reorth; double initX_orth_zero_tol;
	char compP_orth_method[8] ; int    compP_orth_block_size; 
	int  compP_orth_max_reorth; double compP_orth_zero_tol;
	char compW_orth_method[8] ; int    compW_orth_block_size; 
	int  compW_orth_max_reorth; double compW_orth_zero_tol;
	int  compW_cg_max_iter    ; double compW_cg_rate; 
	double compW_cg_tol       ; char   compW_cg_tol_type[8];
	int  compW_cg_auto_shift  ; double compW_cg_shift;
	int  compW_cg_order       ;
	int    compRR_min_num     ; double compRR_min_gap; 
	double compRR_tol; /*tol for dsyevx_ */	
} GCGSolver;

/* �趨 GCG �Ĺ����ռ� */
void EigenSolverSetup_GCG(
	double gapMin, 
	int    nevInit , int    nevMax, int block_size,
	double tol[2]  , int    numIterMax, int print,
	int    user_defined_multi_linear_solver,
	void **mv_ws[4], double *dbl_ws   , int *int_ws, 
	struct OPS_ *ops);
	
void EigenSolverCreateWorkspace_GCG(
	int nevInit, int nevMax, int block_size, void *mat,
	void ***mv_ws, double **dbl_ws, int **int_ws, 
	struct OPS_ *ops);

void EigenSolverDestroyWorkspace_GCG(
	int nevInit, int nevMax, int block_size, void *mat,
	void ***mv_ws, double **dbl_ws, int **int_ws, 
	struct OPS_ *ops);
	
void EigenSolverSetParameters_GCG(
	int    check_conv_max_num,
	const char *initX_orth_method, int initX_orth_block_size, int initX_orth_max_reorth, double initX_orth_zero_tol,
	const char *compP_orth_method, int compP_orth_block_size, int compP_orth_max_reorth, double compP_orth_zero_tol,
	const char *compW_orth_method, int compW_orth_block_size, int compW_orth_max_reorth, double compW_orth_zero_tol,
	int    compW_cg_max_iter , double compW_cg_rate, 
	double compW_cg_tol      , const char *compW_cg_tol_type, int compW_cg_auto_shift  ,
	int    compRR_min_num, double compRR_min_gap, double compRR_tol, 
	struct OPS_ *ops);

void EigenSolverSetParametersFromCommandLine_GCG(
	int argc, char* argv[], struct OPS_ *ops);


void GCGE_Create(void *A, int nevMax, int block_size, int nevInit, void ***gcg_mv_ws, double *dbl_ws, int *int_ws, 
	struct OPS_ *ops);

void GCGE_Setparameters(double gapMin, int shift, struct OPS_ *ops);

int gcgeprint(int flag);

void GCGE_Destroymvws(void ***gcg_mv_ws, double *dbl_ws, int *int_ws, int nevMax, int block_size, struct OPS_ *ops);

#endif  /* -- #ifndef _OPS_EIG_SOL_GCG_H_ -- */

