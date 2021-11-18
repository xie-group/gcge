/**
 *    @file  test_eig_sol.c
 *   @brief  特征值求解器测试
 *
 *  PASE and GCGE
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/14
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<float.h>
#include    <memory.h>
#include    <time.h>

#include    "ops.h"
#include    "ops_eig_sol_gcg.h"


#define DEBUG   0

/* flag == 0 表示不使用外部多向量线性求解器 
 * flag == 1 表示仅使用外部多向量线性求解器 
 * flag == 2 表示以外部多向量线性求解器为预条件子 */
int TestEigenSolverGCG(void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops) 
{
	/* 展示算法调用参数 */
	/* 用户希望收敛的特征对个数 nevConv, 最多返回 nevMax 
	 * 要求 block_size >= multiMax */ 
	int nevConv  = 30, multiMax = 1; double gapMin = 1e-5;
	int nevGiven = 0, block_size = nevConv/5, nevMax = 2*nevConv;
	/* 当特征值收敛 2*block_size 时, 将 P W 部分归入 X 部分, 
	 * 工作空间中的 X 不超过 nevInit (>=3*block_size) */
	//int nevInit = 3*block_size; 
	int nevInit = nevMax;

	ops->GetOptionFromCommandLine("-nevConv"  ,'i',&nevConv   ,argc,argv,ops);
	nevMax = 2*nevConv;
	ops->GetOptionFromCommandLine("-nevMax"   ,'i',&nevMax    ,argc,argv,ops);
	block_size = nevConv<30?(nevMax-nevConv):nevConv/5;
	ops->GetOptionFromCommandLine("-blockSize",'i',&block_size,argc,argv,ops);
	nevInit = nevMax;
	ops->GetOptionFromCommandLine("-nevInit"  ,'i',&nevInit   ,argc,argv,ops);
	/* 工作空间由 nevMax blockSize nevInit 决定 */ 
	nevInit = nevInit<nevMax?nevInit:nevMax;
	int max_iter_gcg = 500; double tol_gcg[2] = {1e-1,1e-8};
	/* 特征值 特征向量 长度为 nevMax */
	double *eval; void **evec;
	eval = malloc(nevMax*sizeof(double));
	memset(eval,0,nevMax*sizeof(double));
	ops->MultiVecCreateByMat(&evec,nevMax,A,ops);
	ops->MultiVecSetRandomValue(evec,0,nevMax,ops);
	/* 下述部分将被全封存在 EigenSolverCreateWorkspace_GCG */
	void **gcg_mv_ws[4]; double *dbl_ws; int *int_ws;
	/* 设定 GCG 的工作空间 nevMax+2*block_size, 
	 * block_size, block_size, block_size */
	ops->MultiVecCreateByMat(&gcg_mv_ws[0],nevMax+2*block_size,A,ops);				
	ops->MultiVecSetRandomValue(gcg_mv_ws[0],0,nevMax+2*block_size,ops);
	ops->MultiVecCreateByMat(&gcg_mv_ws[1],block_size,A,ops);				
	ops->MultiVecSetRandomValue(gcg_mv_ws[1],0,block_size,ops);
	ops->MultiVecCreateByMat(&gcg_mv_ws[2],block_size,A,ops);				
	ops->MultiVecSetRandomValue(gcg_mv_ws[2],0,block_size,ops);
	ops->MultiVecCreateByMat(&gcg_mv_ws[3],block_size,A,ops);				
	ops->MultiVecSetRandomValue(gcg_mv_ws[3],0,block_size,ops);
	int sizeV = nevInit + 2*block_size;
	int length_dbl_ws = 2*sizeV*sizeV+10*sizeV
		+(nevMax+2*block_size)+(nevMax)*block_size;
	ops->Printf ( "length_dbl_ws = %d\n", length_dbl_ws );
	int length_int_ws = 6*sizeV+2*(block_size+3);
	ops->Printf ( "length_int_ws = %d\n", length_int_ws );
	dbl_ws = malloc(length_dbl_ws*sizeof(double));
	memset(dbl_ws,0,length_dbl_ws*sizeof(double));
	int_ws = malloc(length_int_ws*sizeof(int));
	memset(int_ws,0,length_int_ws*sizeof(int));
	/* 上述部分将被全封存在 EigenSolverCreateWorkspace_GCG */

	ops->Printf("mat A:\n");
	//ops->MatView(A,ops);
	if (B!=NULL) {
		ops->Printf("mat B:\n");
	//	ops->MatView(B,ops);
	}

	srand(0);
	double time_start, time_interval;
	time_start = ops->GetWtime();
		
	ops->Printf("===============================================\n");
	ops->Printf("GCG Eigen Solver\n");
	/* 设定 ops 中的特征值求解器是 GCG */
	EigenSolverSetup_GCG(multiMax,gapMin,nevInit,nevMax,block_size,
		tol_gcg,max_iter_gcg,flag,gcg_mv_ws,dbl_ws,int_ws,ops);
	
	/* 展示算法所有参数 */
	int    check_conv_max_num    = 50   ;
		
	char   initX_orth_method[8]  = "mgs"; 
	int    initX_orth_block_size = 80   ; 
	int    initX_orth_max_reorth = 2    ; double initX_orth_zero_tol    = 2*DBL_EPSILON;//1e-12
	
	char   compP_orth_method[8]  = "mgs"; 
	int    compP_orth_block_size = -1   ; 
	int    compP_orth_max_reorth = 2    ; double compP_orth_zero_tol    = 2*DBL_EPSILON;//1e-12
	
	char   compW_orth_method[8]  = "mgs";
	int    compW_orth_block_size = 80   ; 	
	int    compW_orth_max_reorth = 2    ;  double compW_orth_zero_tol   = 2*DBL_EPSILON;//1e-12
	int    compW_bpcg_max_iter   = 30   ;  double compW_bpcg_rate       = 1e-2; 
	double compW_bpcg_tol        = 1e-14;  char   compW_bpcg_tol_type[8] = "abs";
	
	int    compRR_min_num        = -1   ;  double compRR_min_gap        = gapMin;
	double compRR_tol            = 2*DBL_EPSILON;
	//double compRR_tol            = 0.0  ; 
			
	/* 设定 GCG 的算法参数 */
	EigenSolverSetParameters_GCG(
			check_conv_max_num   ,
			initX_orth_method    , initX_orth_block_size, 
			initX_orth_max_reorth, initX_orth_zero_tol  ,
			compP_orth_method    , compP_orth_block_size, 
			compP_orth_max_reorth, compP_orth_zero_tol  ,
			compW_orth_method    , compW_orth_block_size, 
			compW_orth_max_reorth, compW_orth_zero_tol  ,
			compW_bpcg_max_iter  , compW_bpcg_rate      , 
			compW_bpcg_tol       , compW_bpcg_tol_type  , 1, // without shift
			compRR_min_num       , compRR_min_gap       ,
			compRR_tol           ,  
			ops);		

	/* 命令行获取 GCG 的算法参数 勿用 有 BUG, 
	 * 不应该改变 nevMax nevInit block_size, 这些与工作空间有关 */
	EigenSolverSetParametersFromCommandLine_GCG(argc,argv,ops);
	ops->Printf("nevGiven = %d, nevConv = %d, nevMax = %d, block_size = %d, nevInit = %d\n",
			nevGiven,nevConv,nevMax,block_size,nevInit);
	ops->EigenSolver(A,B,eval,evec,nevGiven,&nevConv,ops);
	ops->Printf("numIter = %d, nevConv = %d\n",
			((GCGSolver*)ops->eigen_solver_workspace)->numIter, nevConv);
	ops->Printf("++++++++++++++++++++++++++++++++++++++++++++++\n");

	time_interval = ops->GetWtime() - time_start;
	ops->Printf("Time is %.3f\n", time_interval);

	/* 下述部分将被全封存在 EigenSolverDestroyWorkspace_GCG */
	ops->MultiVecDestroy(&gcg_mv_ws[0],nevMax+2*block_size,ops);
	ops->MultiVecDestroy(&gcg_mv_ws[1],block_size,ops);
	ops->MultiVecDestroy(&gcg_mv_ws[2],block_size,ops);
	ops->MultiVecDestroy(&gcg_mv_ws[3],block_size,ops);
	free(dbl_ws); free(int_ws);
	/* 上述部分将被全封存在 EigenSolverDestroyWorkspace_GCG */

#if 1
	ops->Printf("eigenvalues\n");
	int idx;
	for (idx = 0; idx < nevConv; ++idx) {
		ops->Printf("%d: %6.14e\n",idx+1,eval[idx]);
		//if (idx > 0)
		//	ops->Printf("%d: %6.14e\n",idx+1,(eval[idx]-eval[idx-1])/(eval[idx]+1));
	}
	ops->Printf("eigenvectors\n");
	//ops->MultiVecView(evec,0,nevConv,ops);
#endif

	ops->MultiVecDestroy(&(evec),nevMax,ops);
	free(eval);
	return 0;
}
