/**
 *    @file  ops_lin_sol.c
 *   @brief  operations for linear solver 
 *
 *  线性求解器操作
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/15
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<math.h>
#include	<assert.h>
#include	<time.h>
#include    <string.h> 

#include    "ops_lin_sol.h"
#define     DEBUG 0

#define     TIME_BPCG 0
#define     TIME_BAMG 0

typedef struct TimeBlockPCG_ {
	double allreduce_time;
    double axpby_time;
    double innerprod_time;
    double matvec_time;        
    double time_total;
} TimeBlockPCG;

struct TimeBlockPCG_ time_bpcg = {0.0,0.0,0.0,0.0,0.0}; 

void PCG(void *mat, void *b, void *x, struct OPS_ *ops)
{
	PCGSolver *pcg = (PCGSolver*)ops->linear_solver_workspace;
	int    niter, max_iter = pcg->max_iter;
	double rate = pcg->rate, tol = pcg->tol;
	double alpha, beta, rho1, rho2, init_error, last_error, pTw;
	void   *r, *p, *w;
	/* CG迭代中用到的临时向量 */
	r = pcg->vec_ws[0]; //记录残差向量
	p = pcg->vec_ws[1]; //记录下降方向
	w = pcg->vec_ws[2]; //记录A*p
    
    // tol = tol*norm2(b)
	if (0 == strcmp("rel", pcg->tol_type)) {
		ops->VecInnerProd(b, b, &pTw, ops);
		tol = tol*sqrt(pTw);
	}
	ops->MatDotVec(mat, x, r, ops);
	ops->VecAxpby(1.0, b, -1.0, r, ops);//r = b-A*x 
	ops->VecInnerProd(r, r, &rho2, ops);//用残量的模来判断误差
	init_error = sqrt(rho2);	
	last_error = init_error;
	niter = 0;
	/* 当last_error< rate*init_error时停止迭代, 
	 * 即最后的误差下降到初始误差的rate倍时停止 */
	while( (last_error>rate*init_error)&&(last_error>tol)&&(niter<max_iter) ) {
		//compute the value of beta
		if(niter == 0) beta = 0.0;
		else           beta = rho2/rho1;
		//set rho1 as rho2
		rho1 = rho2;
		//compute the new direction: p = r + beta * p
		ops->VecAxpby(1.0, r, beta, p, ops);
		//compute the vector w = A*p
		ops->MatDotVec(mat, p, w, ops);	
		//compute the value pTw = p^T * w 
		ops->VecInnerProd(p, w, &pTw, ops);
		//compute the value of alpha
		alpha = rho2/pTw; 
		//compute the new solution x = alpha * p + x
		ops->VecAxpby( alpha, p, 1.0, x, ops);
		//compute the new residual: r = - alpha*w + r
		ops->VecAxpby(-alpha, w, 1.0, r, ops);
		//compute the new rho2
		ops->VecInnerProd(r, r, &rho2, ops);	
		last_error = sqrt(rho2);
		//update the iteration time
		++niter;   
	}
	pcg->niter = niter; pcg->residual = last_error;
	return;
}

/**
 * @brief 在调用LinearSolver之前需要设置LinearSolver
 *        再次调用LinearSolver时，如果参数和临时空间不变，无需再次调用
 */
void LinearSolverSetup_PCG(int max_iter, double rate, double tol,
		const char *tol_type, void *vec_ws[3], void *pc, struct OPS_ *ops)
{
	/* 只初始化一次，且全局可见 */
	static PCGSolver pcg_static = {
		.max_iter = 50, .rate = 1e-2, .tol=1e-12, .tol_type = "abs", 
		.vec_ws   = {}, .pc   = NULL};
	pcg_static.max_iter  = max_iter;
	pcg_static.rate      = rate    ;
	pcg_static.tol       = tol     ;
	strcpy(pcg_static.tol_type, tol_type);
	pcg_static.vec_ws[0] = vec_ws[0];
	pcg_static.vec_ws[1] = vec_ws[1];
	pcg_static.vec_ws[2] = vec_ws[2];
	pcg_static.niter     = 0   ;
	pcg_static.residual  = -1.0;

	ops->linear_solver_workspace = (void *)(&pcg_static);
	ops->LinearSolver = PCG;
	return;
}
void BlockPCG(void *mat, void **mv_b, void **mv_x, 
		int *start_bx, int *end_bx, struct OPS_ *ops) 
{
	BlockPCGSolver *bpcg = (BlockPCGSolver*)ops->multi_linear_solver_workspace;
	int    niter, max_iter = bpcg->max_iter, idx, col, length, start[2], end[2],
	       num_block, *block, pre_num_unconv, num_unconv, *unconv;
	double rate = bpcg->rate, tol = bpcg->tol;
	double alpha, beta, *rho1, *rho2, *pTw, *norm_b, *init_res, *last_res, *destin;
	void   **mv_r, **mv_p, **mv_w;
	mv_r = bpcg->mv_ws[0];
	mv_p = bpcg->mv_ws[1];
	mv_w = bpcg->mv_ws[2];
	
	assert(end_bx[0]-start_bx[0]==end_bx[1]-start_bx[1]);
	num_unconv = end_bx[0]-start_bx[0];
	norm_b   = bpcg->dbl_ws;
	rho1     = norm_b   + num_unconv; 
	rho2     = rho1     + num_unconv; 
	pTw      = rho2     + num_unconv;
	init_res = pTw      + num_unconv; 
	last_res = init_res + num_unconv;	
	unconv = bpcg->int_ws;
	block  = unconv + num_unconv;
	
#if TIME_BPCG
	time_bpcg.allreduce_time = 0.0;
	time_bpcg.axpby_time     = 0.0;
	time_bpcg.innerprod_time = 0.0;
	time_bpcg.matvec_time    = 0.0;
#endif
	
	/* 计算 norm of rhs */
	if (0 == strcmp("rel", bpcg->tol_type)) {		
#if TIME_BPCG
        time_bpcg.innerprod_time -= ops->GetWtime();
#endif 
		start[0] = start_bx[0]; end[0] = end_bx[0];
		start[1] = start_bx[0]; end[1] = end_bx[0];
		ops->MultiVecInnerProd('D',mv_b,mv_b,0,start,end,norm_b,1,ops);
#if TIME_BPCG
        time_bpcg.innerprod_time += ops->GetWtime();
#endif 
		for (idx = 0; idx < num_unconv; ++idx) {
			norm_b[idx] = sqrt(norm_b[idx]);
		}
	}
	else if (0 == strcmp("user", bpcg->tol_type)){
	   /* user defined norm_b */
	   for (idx = 0; idx < num_unconv; ++idx) {
	      norm_b[idx] = fabs(norm_b[idx]);
	      //ops->Printf("%e\n",norm_b[idx]);
	   }
	}
	else {
		for (idx = 0; idx < num_unconv; ++idx) {
			norm_b[idx] = 1.0;
		}
	}
	
#if TIME_BPCG
    time_bpcg.matvec_time -= ops->GetWtime();
#endif 
	/* 计算初始残量 */
	start[0] = start_bx[1]; end[0] = end_bx[1] ;
	start[1] = 0          ; end[1] = num_unconv;
	if (bpcg->MatDotMultiVec!=NULL) {
		bpcg->MatDotMultiVec(mv_x,mv_r,start,end,mv_p,0,ops);
	}
	else {
		ops->MatDotMultiVec(mat,mv_x,mv_r,start,end,ops);
	}
		
#if TIME_BPCG
    time_bpcg.matvec_time += ops->GetWtime();
#endif 
	
#if TIME_BPCG
    time_bpcg.axpby_time -= ops->GetWtime();
#endif	
	start[0] = start_bx[0]; end[0] = end_bx[0] ;
	start[1] = 0          ; end[1] = num_unconv;
	ops->MultiVecAxpby(1.0,mv_b,-1.0,mv_r,start,end,ops);
#if TIME_BPCG
    time_bpcg.axpby_time += ops->GetWtime();
#endif

#if TIME_BPCG
    time_bpcg.innerprod_time -= ops->GetWtime();
#endif 
	start[0] = 0          ; end[0] = num_unconv;
	start[1] = 0          ; end[1] = num_unconv;
	ops->MultiVecInnerProd('D',mv_r,mv_r,0,start,end,rho2,1,ops);
#if TIME_BPCG
    time_bpcg.innerprod_time += ops->GetWtime();
#endif 
	for (idx = 0; idx < num_unconv; ++idx) {
		init_res[idx] = sqrt(rho2[idx]);
	}
	/* 判断收敛性 */
	pre_num_unconv = num_unconv;
	num_unconv = 0;
	for (idx = 0; idx < pre_num_unconv; ++idx) {
		if (init_res[idx]>tol*norm_b[idx]) {
			unconv[num_unconv] = idx;
			rho2[num_unconv]   = rho2[idx];
			++num_unconv;
		}
	}
#if DEBUG
	if (num_unconv > 0) {
		ops->Printf("BlockPCG: initial residual[%d] = %6.4e\n",unconv[0],init_res[unconv[0]]/norm_b[unconv[0]]);	
	} else {
		ops->Printf("BlockPCG: initial residual[%d] = %6.4e\n",0,init_res[0]/norm_b[0]);
	}
#endif	
	niter = 0;
	while( niter<max_iter&&num_unconv>0 ) {
		num_block = 0;
		block[num_block] = 0;
		++num_block;
		for (idx = 1; idx < num_unconv; ++idx) {
			if (unconv[idx]-unconv[idx-1]>1) {
				block[num_block] = idx;
				++num_block;
			}
		}
		block[num_block] = num_unconv;
		/* for each block */
		destin = pTw;
		for (idx = 0; idx < num_block; ++idx) {
			length = block[idx+1] - block[idx];
			for (col = block[idx]; col < block[idx+1]; ++col) {
				if(niter == 0) beta = 0.0;
				else           beta = rho2[col]/rho1[col];
				
#if TIME_BPCG
    			time_bpcg.axpby_time -= ops->GetWtime();
#endif
				start[0] = unconv[col]; end[0] = unconv[col]+1;
				start[1] = unconv[col]; end[1] = unconv[col]+1;
				ops->MultiVecAxpby(1.0,mv_r,beta,mv_p,start,end,ops);
#if TIME_BPCG
    			time_bpcg.axpby_time += ops->GetWtime();
#endif
			}
#if TIME_BPCG
    		time_bpcg.matvec_time -= ops->GetWtime();
#endif
			//compute the vector w = A*p
			start[0] = unconv[block[idx]]; end[0] = unconv[block[idx+1]-1]+1;
			start[1] = unconv[block[idx]]; end[1] = unconv[block[idx+1]-1]+1;
			if (bpcg->MatDotMultiVec!=NULL) {
				bpcg->MatDotMultiVec(mv_p,mv_w,start,end,mv_b,start_bx[0],ops);
			}
			else {
				ops->MatDotMultiVec(mat,mv_p,mv_w,start,end,ops);
			}
			
#if TIME_BPCG
    		time_bpcg.matvec_time += ops->GetWtime();
#endif

#if TIME_BPCG
    		time_bpcg.innerprod_time -= ops->GetWtime();
#endif	
			//compute the value pTw = p^T * w 
			ops->MultiVecLocalInnerProd('D',mv_p,mv_w,0,start,end,destin,1,ops);
#if TIME_BPCG
    		time_bpcg.innerprod_time += ops->GetWtime();
#endif
			destin += length;
		}

#if OPS_USE_MPI
#if TIME_BPCG
    	time_bpcg.allreduce_time -= ops->GetWtime();
#endif
		MPI_Allreduce(MPI_IN_PLACE,pTw,num_unconv,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
#if TIME_BPCG
    	time_bpcg.allreduce_time += ops->GetWtime();
#endif
#endif

		//set rho1 as rho2
		int inc = 1;
		dcopy(&num_unconv,rho2,&inc,rho1,&inc);
		/* for each block */
		destin = rho2;
		for (idx = 0; idx < num_block; ++idx) {
			length = block[idx+1] - block[idx];
			for (col = block[idx]; col < block[idx+1]; ++col) {
				//compute the value of alpha
				alpha = rho2[col]/pTw[col];
#if TIME_BPCG
    			time_bpcg.axpby_time -= ops->GetWtime();
#endif
				//compute the new solution x = alpha * p + x
				start[0] = unconv[col]; end[0] = unconv[col]+1;
				start[1] = start_bx[1]+unconv[col]; 
				end[1]   = start_bx[1]+unconv[col]+1;
				ops->MultiVecAxpby(alpha,mv_p,1.0,mv_x,start,end,ops);
				//compute the new residual: r = - alpha*w + r
				start[0] = unconv[col]; end[0] = unconv[col]+1;
				start[1] = unconv[col]; end[1] = unconv[col]+1;
				ops->MultiVecAxpby(-alpha,mv_w,1.0,mv_r,start,end,ops);
#if TIME_BPCG
    			time_bpcg.axpby_time += ops->GetWtime();
#endif
			}
#if TIME_BPCG
    		time_bpcg.innerprod_time -= ops->GetWtime();
#endif
			//compute the new rho2
			start[0] = unconv[block[idx]]; end[0] = unconv[block[idx+1]-1]+1;
			start[1] = unconv[block[idx]]; end[1] = unconv[block[idx+1]-1]+1;
			ops->MultiVecLocalInnerProd('D',mv_r,mv_r,0,start,end,destin,1,ops);
#if TIME_BPCG
    		time_bpcg.innerprod_time += ops->GetWtime();
#endif	
			destin += length;
		}
#if OPS_USE_MPI
#if TIME_BPCG
    	time_bpcg.allreduce_time -= ops->GetWtime();
#endif
		MPI_Allreduce(MPI_IN_PLACE,rho2,num_unconv,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
#if TIME_BPCG
    	time_bpcg.allreduce_time += ops->GetWtime();
#endif
#endif
		for (idx = 0; idx < num_unconv; ++idx) {
		   //if (bpcg->tol_type=='U' && niter > 10) {
			//last_res[unconv[idx]] = (1.1*last_res[unconv[idx]])<sqrt(rho2[idx])?1e-16:sqrt(rho2[idx]);
		   //}
		   //else {
			last_res[unconv[idx]] = sqrt(rho2[idx]);
		   //}
		}
#if DEBUG
		ops->Printf("niter = %d, num_unconv = %d, residual[%d] = %6.4e\n",
		      niter+1,num_unconv,unconv[0],last_res[unconv[0]]/norm_b[unconv[0]]);
#endif
		/* 判断收敛性 */
		pre_num_unconv = num_unconv;
		num_unconv = 0;
		for (idx = 0; idx < pre_num_unconv; ++idx) {
			col = unconv[idx];
			if ((last_res[col]>rate*init_res[col])&&(last_res[col]>tol*norm_b[col])) {
				unconv[num_unconv] = col;
				/* 需将 rho1 rho2 未收敛部分顺序前移 */
				rho1[num_unconv]   = rho1[idx];
				rho2[num_unconv]   = rho2[idx];
				++num_unconv;
			}
		}
		//update the iteration time

		++niter;

#if DEBUG
		if (niter%5 == 0) {
		   ops->Printf("BlockPCG: niter = %d, num_unconv = %d, residual[%d] = %6.4e\n",
			 niter,num_unconv,unconv[0],last_res[unconv[0]]/norm_b[unconv[0]]);
		}
#endif
	}
	if (niter > 0) {
		bpcg->niter = niter; bpcg->residual = last_res[unconv[0]];	
	} else {
		bpcg->niter = niter; bpcg->residual = init_res[0];
	}
	
#if TIME_BPCG
	ops->Printf("|--BPCG----------------------------\n");
	time_bpcg.time_total = time_bpcg.allreduce_time
		+time_bpcg.axpby_time
		+time_bpcg.innerprod_time
		+time_bpcg.matvec_time;
	ops->Printf("|allreduce  axpby  inner_prod  matvec\n");
	ops->Printf("|%.2f\t%.2f\t%.2f\t%.2f\n",
		time_bpcg.allreduce_time,		
		time_bpcg.axpby_time,		
		time_bpcg.innerprod_time,		
		time_bpcg.matvec_time);
	ops->Printf("|%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\n",
		time_bpcg.allreduce_time/time_bpcg.time_total*100,
		time_bpcg.axpby_time    /time_bpcg.time_total*100,
		time_bpcg.innerprod_time/time_bpcg.time_total*100,
		time_bpcg.matvec_time   /time_bpcg.time_total*100);
	ops->Printf("|--BPCG----------------------------\n");
	time_bpcg.allreduce_time = 0.0;
	time_bpcg.axpby_time     = 0.0;
	time_bpcg.innerprod_time = 0.0;
	time_bpcg.matvec_time    = 0.0;	
#endif
	
	return;
}      
	      
void MultiLinearSolverSetup_BlockPCG(int max_iter, double rate, double tol,
		const char *tol_type, void **mv_ws[3], double *dbl_ws, int *int_ws,
		void *pc, void (*MatDotMultiVec)(void**x,void**y,int*,int*,void **z,int s,struct OPS_*),
		struct OPS_ *ops)
{
	/* 只初始化一次，且全局可见 */
	static BlockPCGSolver bpcg_static = {
		.max_iter = 50, .rate = 1e-2, .tol=1e-12, .tol_type = "abs", 
		.mv_ws    = {}, .pc = NULL, .MatDotMultiVec = NULL};
	bpcg_static.max_iter = max_iter;
	bpcg_static.rate     = rate    ;
	bpcg_static.tol      = tol     ;
	strcpy(bpcg_static.tol_type, tol_type);
	bpcg_static.mv_ws[0] = mv_ws[0];
	bpcg_static.mv_ws[1] = mv_ws[1];
	bpcg_static.mv_ws[2] = mv_ws[2];
	bpcg_static.dbl_ws   = dbl_ws  ;
	bpcg_static.int_ws   = int_ws  ;
	bpcg_static.MatDotMultiVec = MatDotMultiVec;
	
	bpcg_static.niter    = 0   ;
	bpcg_static.residual = -1.0;

	ops->multi_linear_solver_workspace = (void *)(&bpcg_static);
	ops->MultiLinearSolver = BlockPCG;
	return;
}
