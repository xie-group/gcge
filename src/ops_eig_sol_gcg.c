/**
 *    @file  ops_eig_sol_gcg.c
 *   @brief  特征值求解器 GCG 
 *
 *  特征值求解器 GCG
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/18
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<string.h>
#include    <math.h>
#include    <memory.h>
#include    <assert.h>
#include    <time.h>
#include    <float.h>
#include    "ops_eig_sol_gcg.h"

#define     DEBUG 0
#define     TIME_GCG 0
#define     PRINT_FIRST_UNCONV 0

typedef struct TimeGCG_ {
	double initX_time;
	double checkconv_time;
	double compP_time;
	double compRR_time;
	double rr_matW_time;
	double dsyevx_time;
	double compRV_time;
	double compW_time;
	double compX_time;            
	double linsol_time;
	double time_total;
} TimeGCG;

struct TimeGCG_ time_gcg = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};


static int sizeN, startN, endN;
static int sizeP, startP, endP;
static int sizeW, startW, endW;
static int sizeC, sizeX , sizeV, endX;


static void   **mv_ws[3]; 
static double *dbl_ws; 
static int    *int_ws;
static struct OPS_ *ops_gcg;
static struct GCGSolver_ *gcg_solver;

#if 0
static double tmp_sigma[200];
#endif

/* y = ( A+sigma B ) x 
 * Only for CG (A+sigma*B)y = (lambda+sigma) B x 
 * use z[s:end] as workspace, which is b or p in CG */ 
static void MatDotMultiVecShift(void **x, void **y, 
	int *start, int *end, void **z, int s, struct OPS_ *ops)
{
	void  *A = gcg_solver->A;
	void  *B = gcg_solver->B;
	
	double sigma = gcg_solver->sigma;
	ops->MatDotMultiVec(A,x,y,start,end,ops);
	if (sigma != 0.0) {
		if (B==NULL) {
#if 1
			ops->MultiVecAxpby(sigma,x,1.0,y,start,end,ops);
#else
			int ncols = end[0]-start[0], col;
			for (col = 0; col < ncols; ++col) {
				ops->MultiVecAxpby(tmp_sigma[col+start[1]],x,1.0,y,start,end,ops);
			}
#endif
		}		
		else {
			//void **z;
			//ops->MultiVecCreateByMat(&z,end[0]-start[0],A,ops);rr
			int start_tmp[2], end_tmp[2];
			start_tmp[0] = start[0]; end_tmp[0] = end[0];
			start_tmp[1] = s       ; end_tmp[1] = s+end[0]-start[0];
			ops->MatDotMultiVec(B,x,z,start_tmp,end_tmp,ops);
			start_tmp[0] = s       ; end_tmp[0] = s+end[0]-start[0];
			start_tmp[1] = start[1]; end_tmp[1] = end[1];
			ops->MultiVecAxpby(sigma,z,1.0,y,start_tmp,end_tmp,ops);
			//ops->MultiVecDestroy(&z,end[0]-start[0],ops);	
		}
	}
	return;
}




static void InitializeX(void **V, void **ritz_vec, void *B, int nevGiven)
{
#if TIME_GCG
    time_gcg.initX_time -= ops_gcg->GetWtime();
#endif
	int start[2], end[2];	
	start[0] = 0; end[0] = nevGiven;
	start[1] = 0; end[1] = nevGiven;
	ops_gcg->MultiVecAxpby(1.0,ritz_vec,0,V,start,end,ops_gcg);	
#if DEBUG
	ops_gcg->Printf("V\n");	
	ops_gcg->MultiVecView(V,0,sizeV,ops_gcg);
#endif
	ops_gcg->Printf("sizeX = %d, nevGiven = %d, %s\n",
		sizeX,nevGiven,gcg_solver->initX_orth_method);
	/* orth_dbl_ws begin from the end of ss_eval */
	double *orth_dbl_ws = gcg_solver->dbl_ws+gcg_solver->nevMax+2*gcg_solver->block_size;
	if (0 == strcmp("mgs", gcg_solver->initX_orth_method))
			MultiVecOrthSetup_ModifiedGramSchmidt(
				gcg_solver->initX_orth_block_size,
				gcg_solver->initX_orth_max_reorth,
				gcg_solver->initX_orth_zero_tol,
				//ritz_vec,gcg_solver->dbl_ws,ops_gcg);
				ritz_vec,orth_dbl_ws,ops_gcg);
	else if (0 == strcmp("bgs", gcg_solver->initX_orth_method))
			MultiVecOrthSetup_BinaryGramSchmidt(
				gcg_solver->initX_orth_block_size,
				gcg_solver->initX_orth_max_reorth,
				gcg_solver->initX_orth_zero_tol,
				//ritz_vec,gcg_solver->dbl_ws,ops_gcg);
				ritz_vec,orth_dbl_ws,ops_gcg);
	else
			MultiVecOrthSetup_ModifiedGramSchmidt(
				gcg_solver->initX_orth_block_size,
				gcg_solver->initX_orth_max_reorth,
				gcg_solver->initX_orth_zero_tol,
				//ritz_vec,gcg_solver->dbl_ws,ops_gcg);
				ritz_vec,orth_dbl_ws,ops_gcg);

	ops_gcg->MultiVecOrth(V,0,&nevGiven,B,ops_gcg);
	ops_gcg->MultiVecSetRandomValue(V,nevGiven,sizeX,ops_gcg);
	ops_gcg->MultiVecOrth(V,nevGiven,&endX,B,ops_gcg);
	assert(endX==sizeX);
	/* 多次正交化, 保证有 sizeX 个正交向量 */
	//int pre_endX;	
	//while (endX < sizeX) {
	//	ops_gcg->MultiVecSetRandomValue(V,endX,sizeX,ops_gcg);
	//	pre_endX = endX; endX = sizeX;
	//	ops_gcg->MultiVecOrth(V,pre_endX,&endX,B,ops_gcg);
	//}
#if DEBUG
	ops_gcg->MultiVecView(V,0,endX,ops_gcg);
#endif
#if TIME_GCG
    time_gcg.initX_time += ops_gcg->GetWtime();
#endif
	return;
}
static void ComputeRitzVec(void **ritz_vec, void **V, double *ss_evec)
{
#if TIME_GCG
    time_gcg.compRV_time -= ops_gcg->GetWtime();
#endif 
	int start[2], end[2]; double *coef;
	start[0] = startN; end[0] = endW;
	start[1] = startN; end[1] = endX;
	coef     = ss_evec;
#if DEBUG
	ops_gcg->Printf("startN = %d, endW = %d, endX = %d\n",startN,endW,endX);	
	ops_gcg->Printf("coef: (%d * %d)\n",end[0]-start[0],end[1]-start[1]);	
	int row, col;
	for (row = 0; row < end[0]-start[0]; ++row) {
		for (col = 0; col < end[1]-start[1]; ++col) {
			ops_gcg->Printf("%6.4e\t",coef[row+col*(sizeV-sizeC)]);
		}	
		ops_gcg->Printf("\n");
	}
	ops_gcg->Printf("V:\n");
	ops_gcg->MultiVecView(V,start[0],end[0],ops_gcg);
	ops_gcg->Printf("startN = %d, endW = %d, endX = %d\n",startN,endW,endX);	
	ops_gcg->Printf("V = %p, (%d, %d), ritz_vec = %p (%d, %d)\n",V,start[0],end[0],ritz_vec,start[1],end[1]);	
#endif
	ops_gcg->MultiVecLinearComb(V,ritz_vec,0, 
			start,end,coef,sizeV-sizeC,NULL,0,ops_gcg);
			
#if DEBUG	
	ops_gcg->Printf("ritz vec:\n");
	ops_gcg->MultiVecView(ritz_vec,start[1],end[1],ops_gcg);
#endif
#if TIME_GCG
    time_gcg.compRV_time += ops_gcg->GetWtime();
#endif
	return;
}
static int CheckConvergence(void *A, void *B, double *ss_eval, void **ritz_vec, 
	int numCheck, double *tol, int *offset)
{
#if TIME_GCG
    time_gcg.checkconv_time -= ops_gcg->GetWtime();
#endif
#if DEBUG
	ops_gcg->Printf("numCheck = %d\n", numCheck);
#endif
	int start[2], end[2], idx; double *inner_prod;
	int nevConv;
	start[0] = startN; end[0] = start[0]+numCheck;
	start[1] = 0     ; end[1] = numCheck;	
	ops_gcg->MatDotMultiVec(A,ritz_vec,mv_ws[0],start,end,ops_gcg);	
	ops_gcg->MatDotMultiVec(B,ritz_vec,mv_ws[1],start,end,ops_gcg);	
	/* lambda Bx */
	ops_gcg->MultiVecLinearComb(NULL,mv_ws[1],0,start,end,
			NULL,0,ss_eval+startN,1,ops_gcg);
	start[0] = 0     ; end[0] = numCheck;
	start[1] = 0     ; end[1] = numCheck;
	/* Ax - lambda Bx */
	ops_gcg->MultiVecAxpby(-1.0,mv_ws[1],1.0,mv_ws[0],start,end,ops_gcg);
	/* 不使用 ss_evec 部分 */
	inner_prod = dbl_ws+(sizeV-sizeC)*sizeW;
	ops_gcg->MultiVecInnerProd('D',mv_ws[0],mv_ws[0],0,
			start,end,inner_prod,1,ops_gcg);
	for (idx = 0; idx < numCheck; ++idx) {
		inner_prod[idx] = sqrt(inner_prod[idx]);
#if DEBUG 
		ops_gcg->Printf("GCG: [%d] %6.14e (%6.4e, %6.4e)\n",
			startN+idx,ss_eval[startN+idx],
			inner_prod[idx], inner_prod[idx]/fabs(ss_eval[startN+idx]));
#endif
	}
	for (idx = 0; idx < numCheck; ++idx) {
		/* 绝对残量 和 相对残量 需分别小于 tol[0] 和 tol[1] */
		if (fabs(ss_eval[startN+idx]) > tol[1]) {
			if (inner_prod[idx] > tol[0] || 
					inner_prod[idx] > fabs(ss_eval[startN+idx])*tol[1]) {
#if PRINT_FIRST_UNCONV
				ops_gcg->Printf("GCG: [%d] %6.14e (%6.4e, %6.4e)\n",
						startN+idx,ss_eval[startN+idx],
						inner_prod[idx], inner_prod[idx]/fabs(ss_eval[startN+idx]));
#endif
				break;
			}
		}
		else {
			if (inner_prod[idx] > tol[0]) {
#if PRINT_FIRST_UNCONV
				ops_gcg->Printf("GCG: [%d] %6.14e (%6.4e, %6.4e)\n",
						startN+idx,ss_eval[startN+idx],
						inner_prod[idx], inner_prod[idx]/fabs(ss_eval[startN+idx]));
#endif 
				break;
			}
		}
	}	
	for ( ; idx > 0; --idx) {
		/* 最后一个收敛的特征值与第一个不收敛的特征值不是重根 */
		if ( fabs((ss_eval[startN+idx-1]-ss_eval[startN+idx])/ss_eval[startN+idx-1]) 
		      > gcg_solver->gapMin) {
			break;
		}
	}
	nevConv = sizeC+idx;
	
	/* offset[0] 为未收敛块的个数, offset[2n-1] <= idx < offset[2n]
	 * idx 是不收敛的标号 1 <= n <= offset[0] */
	int state, num_unconv;
	/* 1 1 0 0 1 1 1 1 0 0 1 0 1 0 0 0 0 0 0 */
	offset[0] = 0; state = 1; num_unconv = 0; 
	for (idx = 0; idx < numCheck; ++idx) {
		/* 这一个是不收敛的 */
		if (inner_prod[idx] > tol[0] || 
				inner_prod[idx] > fabs(ss_eval[startN+idx])*tol[1]) {
			/* 上一个是收敛的 */
			if (state) {
				offset[ offset[0]*2+1 ] = startN+idx;
				state = 0;
			}
			++num_unconv;
			if (num_unconv == sizeN) {
				offset[ offset[0]*2+2 ] = startN+idx+1;
				++offset[0];
				break;
			}
		}
		else {
			/* 上一个是不收敛的 */
			if (!state) {
				offset[ offset[0]*2+2 ] = startN+idx;
				++offset[0];
				state = 1;
			}
		}
	}
	if (num_unconv < sizeN) {
		if (state == 1) {
			offset[ offset[0]*2+1 ] = startN+numCheck;	
		}
		offset[ offset[0]*2+2 ] = startN+numCheck+sizeN-num_unconv;
		offset[ offset[0]*2+2 ] = offset[ offset[0]*2+2 ] < endX?
				offset[ offset[0]*2+2 ]:endX;
		assert(offset[ offset[0]*2+1 ]<offset[ offset[0]*2+2 ]);
		++offset[0];	
	}
	
#if TIME_GCG
    time_gcg.checkconv_time += ops_gcg->GetWtime();
#endif

#if DEBUG
	for (idx = 0; idx < offset[0]; ++idx) {
		ops_gcg->Printf("offset [%d,%d)\n",
			offset[idx*2+1],offset[idx*2+2]);
	}
#endif
	assert(offset[0]>0); 
	return nevConv;
}
static void ComputeP(void **V, double *ss_evec, int *offset)
{
#if TIME_GCG
    time_gcg.compP_time -= ops_gcg->GetWtime();
#endif
	int length, incx, incy, ldm, block_size;
	int nrows, idx, col, start[2], end[2]; 
	double *source, *destin, *mat, *coef;
	
	/* 复制 n 部分对应的列 */
#if DEBUG
	ops_gcg->Printf("offset[0] = %d, sizeP = %d\n", offset[0], sizeP);	
#endif
	block_size = 0;
	for (idx = 0; idx < offset[0]; ++idx) {
		length = (sizeV-sizeC)*(offset[idx*2+2]-offset[idx*2+1]);		
		source = ss_evec+(sizeV-sizeC)*(offset[idx*2+1]-sizeC) ; incx = 1;
		destin = ss_evec+(sizeV-sizeC)*(sizeX-sizeC+block_size); incy = 1;
		dcopy(&length,source,&incx,destin,&incy);		
		block_size += offset[idx*2+2]-offset[idx*2+1];
#if DEBUG
		ops_gcg->Printf("offset [%d, %d)\n", offset[idx*2+1],offset[idx*2+2]);	
#endif
	}
	sizeP = block_size;
	/* 置零 np 部分 */
	for (idx = 0; idx < offset[0]; ++idx) {
		length = (offset[idx*2+2]-offset[idx*2+1]);
		destin = ss_evec+(sizeV-sizeC)*(sizeX-sizeC)+(offset[idx*2+1]-sizeC);
		for (col = 0; col < sizeP; ++col) {
			memset(destin,0,length*sizeof(double));
			destin += sizeV-sizeC;
		}			
	}
	
	
	/* 小规模正交化 */
	mat    = ss_evec; 
	nrows  = sizeV-sizeC; ldm  = sizeV-sizeC ;
	startP = sizeX-sizeC; endP = startP+sizeP;
#if DEBUG	
	ops_gcg->Printf("sizeC = %d, sizeN = %d, sizeX = %d, sizeP = %d, sizeW = %d\n",
			sizeC,sizeN,sizeX,sizeP,sizeW);	
	ops_gcg->Printf("startP = %d, endP = %d, startW = %d, endW = %d, sizeV = %d\n",
			startP,endP,startW,endW,sizeV);
	int row, ncols;
	for (row = 0; row < nrows; ++row) {
		for (col = 0; col < endP; ++col) {
			ops_gcg->Printf("%6.4e\t",mat[row+ldm*col]);
		}
		ops_gcg->Printf("\n");
	}
	ops_gcg->Printf("startP = %d, endP = %d, sizeP = %d, startW = %d, endW = %d, sizeW = %d, sizeV = %d\n",
			startP,endP,sizeP,startW,endW,sizeW,sizeV);
#endif
	double *orth_dbl_ws = ss_evec+ldm*endP;
	/* ss_diag ss_matA ss_evec 剩下的空间 */
	if (0 == strcmp("bqr", gcg_solver->compP_orth_method)) {
		int length_orth_dbl_ws = gcg_solver->length_dbl_ws-(orth_dbl_ws - gcg_solver->dbl_ws); 
		ops_gcg->DenseMatOrth(mat,nrows,ldm,startP,&endP,
			gcg_solver->compP_orth_zero_tol,
			orth_dbl_ws,length_orth_dbl_ws,int_ws);		
	}
	else {
		LAPACKVEC lapack_vec_P, lapack_vec_ws;
		lapack_vec_P.data   = mat;
		lapack_vec_P.ldd    = ldm;
		lapack_vec_P.ncols  = endP;
		lapack_vec_P.nrows  = nrows;
		
		lapack_vec_ws.data  = orth_dbl_ws;
		lapack_vec_ws.ldd   = ldm;
		lapack_vec_ws.ncols = endP-startP;
		lapack_vec_ws.nrows = nrows;
		if (0 == strcmp("mgs", gcg_solver->compP_orth_method))
			MultiVecOrthSetup_ModifiedGramSchmidt(
				gcg_solver->compP_orth_block_size,
				gcg_solver->compP_orth_max_reorth,
				gcg_solver->compP_orth_zero_tol,
				(void*)&lapack_vec_ws,orth_dbl_ws+ldm*(endP-startP),
				ops_gcg->lapack_ops);
		else if (0 == strcmp("bgs", gcg_solver->compP_orth_method))
			MultiVecOrthSetup_BinaryGramSchmidt(
				gcg_solver->compP_orth_block_size,
				gcg_solver->compP_orth_max_reorth,
				gcg_solver->compP_orth_zero_tol,
				(void*)&lapack_vec_ws,orth_dbl_ws+ldm*(endP-startP),
				ops_gcg->lapack_ops);
		else
			MultiVecOrthSetup_ModifiedGramSchmidt(
				gcg_solver->compP_orth_block_size,
				gcg_solver->compP_orth_max_reorth,
				gcg_solver->compP_orth_zero_tol,
				(void*)&lapack_vec_ws,orth_dbl_ws+ldm*(endP-startP),
				ops_gcg->lapack_ops);
					
		ops_gcg->lapack_ops->MultiVecOrth((void*)&lapack_vec_P,
			startP,&endP,NULL,ops_gcg->lapack_ops);		
	}
	startP += sizeC; endP += sizeC; sizeP = endP-startP;
	
#if DEBUG
	ops_gcg->Printf("startP = %d, endP = %d, sizeP = %d, startW = %d, endW = %d, sizeW = %d, sizeV = %d\n",
			startP,endP,sizeP,startW,endW,sizeW,sizeV);
				
	nrows = sizeV-sizeC; ncols = sizeV-sizeC;
	for (row = 0; row < nrows; ++row) {
		for (col = 0; col < ncols; ++col) {
			ops_gcg->Printf("%6.4e\t",mat[row+ldm*col]);
		}
		ops_gcg->Printf("\n");
	}
#endif
	/* 更新 P */
	start[0] = startN; end[0] = endW ;
	start[1] = 0     ; end[1] = sizeP;
	coef     = ss_evec+(sizeV-sizeC)*(sizeX-sizeC);
	ops_gcg->MultiVecLinearComb(V,mv_ws[0],0,start,end,
			coef,sizeV-sizeC,NULL,0,ops_gcg);
	start[0] = 0     ; end[0] = sizeP;
	start[1] = startP; end[1] = endP ;
	ops_gcg->MultiVecAxpby(1.0,mv_ws[0],0.0,V,start,end,ops_gcg);
	
#if DEBUG
	start[0] = startP; end[0] = endP;
	start[1] = startP; end[1] = endP;
	nrows = end[0]-start[0]; ncols = end[1]-start[1];
	ops_gcg->Printf("PtBP: nrows = %d, ncols = %d\n", nrows, ncols);
	ops_gcg->Printf("PtBP: start = %d,%d, end = %d,%d\n", start[0], start[1], end[0], end[1]);
	ops_gcg->MultiVecQtAP('N','N',V,NULL,V,0,start,end,dbl_ws,nrows,mv_ws[0],ops_gcg);
	for (row = 0; row < nrows; ++row) {
		for (col = 0; col < ncols; ++col) {
			ops_gcg->Printf("%6.4e\t",dbl_ws[row+nrows*col]);
		}
		ops_gcg->Printf("\n");
	}
#endif
#if TIME_GCG
    time_gcg.compP_time += ops_gcg->GetWtime();
#endif	
	return;	
}
static void ComputeX(void **V, void **ritz_vec)
{
#if TIME_GCG
    time_gcg.compX_time -= ops_gcg->GetWtime();
#endif
	int start[2], end[2];
	start[0] = startN; end[0] = endX;
	start[1] = startN; end[1] = endX;
	ops_gcg->MultiVecAxpby(1.0,ritz_vec,0.0,V,start,end,ops_gcg);
#if TIME_GCG
    time_gcg.compX_time += ops_gcg->GetWtime();
#endif
	return;
}
static void ComputeW(void **V, void *A, void *B,
	double *ss_eval, void **ritz_vec, int *offset)
{
#if TIME_GCG
	time_gcg.compW_time -= ops_gcg->GetWtime();
#endif	
	void **b = ritz_vec;
	int start[2], end[2], block_size, length, inc, idx;
	double *destin = dbl_ws;
	
	double sigma = 0.0;
	if (gcg_solver->compW_cg_auto_shift==1) {
		sigma = -ss_eval[sizeC]+((ss_eval[sizeC+1]-ss_eval[sizeC])*0.01);
#if 0
		if (sizeC<3)
			sigma = -ss_eval[sizeC]+(3*(ss_eval[1]-ss_eval[0])<0.1?3*(ss_eval[1]-ss_eval[0]):0.1);
		else 
			sigma = -ss_eval[sizeC]+((ss_eval[sizeC+1]-ss_eval[sizeC])<0.1?(ss_eval[sizeC+1]-ss_eval[sizeC]):0.1);
#endif
	}
	gcg_solver->sigma = gcg_solver->compW_cg_shift+sigma; sigma = gcg_solver->sigma;
#if DEBUG
	ops_gcg->Printf("ss_eval[%d] = %e, sigma = %e\n",sizeC,ss_eval[sizeC],gcg_solver->sigma);
#endif
	/* 不支持 同时使用 外部求解器 和 shift */
	assert(gcg_solver->compW_cg_auto_shift==0 || gcg_solver->user_defined_multi_linear_solver == 0);
	/* initialize */
	block_size = 0; startW = endP; inc = 1; 
	for (idx = 0; idx < offset[0]; ++idx) {
		length   = offset[idx*2+2]-offset[idx*2+1];
		/* initialize x */
		start[0] = offset[idx*2+1]  ; end[0] = offset[idx*2+2];
		start[1] = startW+block_size; end[1] = start[1]+length;
		ops_gcg->MultiVecAxpby(1.0,ritz_vec,0.0,V,start,end,ops_gcg);
#if 0
		/* 20210530 Ax = lambda Bx - theta Ax */
		int tmp_start[2], tmp_end[2]; double tmp_theta = 0.0;
		ops_gcg->MultiVecAxpby(0.0,NULL,1-tmp_theta,V,start,end,ops_gcg);
		/* 20210530 Ax = lambda Bx - theta Ax */
#endif

#if DEBUG
		ops_gcg->Printf("initial W:\n");		
		ops_gcg->MultiVecView(V,start[1],end[1],ops_gcg);	
#endif
		/* set b, b = (lambda+sigma) Bx */
		start[0] = offset[idx*2+1]     ; end[0] = offset[idx*2+2];
		start[1] = offset[1]+block_size; end[1] = start[1]+length;
		//ops_gcg->Printf("start = %d,%d, end = %d,%d\n",start[0],start[1],end[0],end[1]);
		ops_gcg->MatDotMultiVec(B,V,b,start,end,ops_gcg);
#if 0
		/* 20210530 Ax = lambda Bx - theta Ax */
		tmp_start[0] = start[0]; tmp_end[0] = end[0];
		tmp_start[1] = 0       ; tmp_end[1] = end[0]-start[0];
		ops_gcg->MatDotMultiVec(A,V,mv_ws[0],tmp_start,tmp_end,ops_gcg);
		/* 20210530 Ax = lambda Bx - theta Ax */
#endif
		
		int i;
#if 1
		/* shift eigenvalues with sigma */
		for (i = start[0]; i < end[0]; ++i) ss_eval[i] += sigma;
		ops_gcg->MultiVecLinearComb(NULL,b,0,start,end,
				NULL,0,ss_eval+start[0],1,ops_gcg);			
		dcopy(&length,ss_eval+start[0],&inc,destin,&inc);
		/* recover eigenvalues */
		for (i = start[0]; i < end[0]; ++i) ss_eval[i] -= sigma;

#if 0
		/* 20210530 Ax = lambda Bx - theta Ax */
		tmp_start[0] = 0       ; tmp_end[0] = end[0]-start[0];
		tmp_start[1] = start[1]; tmp_end[1] = end[1];
		ops_gcg->MultiVecAxpby(-tmp_theta,mv_ws[0],1.0,b,tmp_start,tmp_end,ops_gcg);
		/* 20210530 Ax = lambda Bx - theta Ax */
#endif
#else
		/* shift eigenvalues with sigma */
		for (i = 0; i < length; ++i) {
			if (start[0]+i > 0)
				tmp_sigma[block_size+i] = ss_eval[start[0]+i] - ss_eval[start[0]+i-1];
			else 
				tmp_sigma[block_size+i] = ss_eval[start[0]+i] + sigma;
		}
		ops_gcg->MultiVecLinearComb(NULL,b,0,start,end,
				NULL,0,tmp_sigma,1,ops_gcg);			
		dcopy(&length,tmp_sigma,&inc,destin,&inc);
		/* recover eigenvalues */
		for (i = 0; i < length; ++i) {
			tmp_sigma[block_size+i] -= ss_eval[start[0]+i];
			ops_gcg->Printf("%d, %d: %f\n",block_size+i,start[0]+i,tmp_sigma[block_size+i]);
		}
#endif
		destin += length;
		block_size += length;
#if DEBUG
		ops_gcg->Printf("initial b:\n");		
		ops_gcg->MultiVecView(b,start[1],end[1],ops_gcg);	
#endif	
	}
	endW = startW+block_size;	
	
	/* solve x */
	start[0] = offset[1]; end[0] = start[0]+block_size;
	start[1] = startW   ; end[1] = endW               ;
#if TIME_GCG
    	time_gcg.linsol_time -= ops_gcg->GetWtime();
#endif
	void(*lin_sol)(void*,void**,void**,int*,int*,struct OPS_*);
	void *ws;
	lin_sol = ops_gcg->MultiLinearSolver;
	ws      = ops_gcg->multi_linear_solver_workspace;
	/* b is set to (lambda+sigma) Bx */
	if (gcg_solver->user_defined_multi_linear_solver==2) {
	    ops_gcg->MultiLinearSolver(A,b,V,start,end,ops_gcg);
	}
#if TIME_GCG
    	time_gcg.linsol_time += ops_gcg->GetWtime();
#endif
	if (gcg_solver->user_defined_multi_linear_solver==0||
	    	gcg_solver->user_defined_multi_linear_solver==2) {	        
#if 1
		/* 20210628 A = sigma B + A */
		if (sigma!=0.0 && B!=NULL && ops_gcg->MatAxpby!=NULL) {
			ops_gcg->MatAxpby(sigma,B,1.0,A,ops_gcg);
			MultiLinearSolverSetup_BlockPCG(
					gcg_solver->compW_cg_max_iter,
					gcg_solver->compW_cg_rate,
					gcg_solver->compW_cg_tol,
					gcg_solver->compW_cg_tol_type,
					mv_ws,dbl_ws,int_ws,NULL,NULL,ops_gcg);
		}
		else {
#endif
			MultiLinearSolverSetup_BlockPCG(
					gcg_solver->compW_cg_max_iter,
					gcg_solver->compW_cg_rate,
					gcg_solver->compW_cg_tol,
					gcg_solver->compW_cg_tol_type,
					mv_ws,dbl_ws,int_ws,NULL,MatDotMultiVecShift,ops_gcg);
#if 1
		}
#endif
	}
#if TIME_GCG
    	time_gcg.linsol_time -= ops_gcg->GetWtime();
#endif
	ops_gcg->MultiLinearSolver(A,b,V,start,end,ops_gcg);
#if 1
	/* 20210628 recover A */
	if (sigma!=0.0 && B!=NULL && ops_gcg->MatAxpby!=NULL) {
		/* A = -sigma B + A */
		ops_gcg->MatAxpby(-sigma,B,1.0,A,ops_gcg);
	}
#endif
#if 0
	ops_gcg->Printf("=====b===========\n");
	ops_gcg->MultiVecView(b,start[0],end[0],ops_gcg);
	ops_gcg->Printf("=====x===========\n");
	ops_gcg->MultiVecView(V,start[1],end[1],ops_gcg);
#endif
#if TIME_GCG
    	time_gcg.linsol_time += ops_gcg->GetWtime();
#endif
	ops_gcg->MultiLinearSolver             = lin_sol;
	ops_gcg->multi_linear_solver_workspace = ws;


#if DEBUG
	ops_gcg->Printf("W = inv(A) b:\n");		
	ops_gcg->MultiVecView(V,startW,endW,ops_gcg);	
#endif
	/* orth W in V */
	if (0 == strcmp("mgs", gcg_solver->compW_orth_method))
		MultiVecOrthSetup_ModifiedGramSchmidt(
			gcg_solver->compW_orth_block_size,
			gcg_solver->compW_orth_max_reorth,
			gcg_solver->compW_orth_zero_tol,
			mv_ws[0],dbl_ws,ops_gcg);
	else if (0 == strcmp("bgs", gcg_solver->compW_orth_method))
		MultiVecOrthSetup_BinaryGramSchmidt(
			gcg_solver->compW_orth_block_size,
			gcg_solver->compW_orth_max_reorth,
			gcg_solver->compW_orth_zero_tol,
			mv_ws[0],dbl_ws,ops_gcg);
	else
		MultiVecOrthSetup_ModifiedGramSchmidt(
			gcg_solver->compW_orth_block_size,
			gcg_solver->compW_orth_max_reorth,
			gcg_solver->compW_orth_zero_tol,
			mv_ws[0],dbl_ws,ops_gcg);
	
	ops_gcg->MultiVecOrth(V,startW,&endW,B,ops_gcg);
#if DEBUG
	ops_gcg->Printf("Orth W in V, %d, %d\n", startW,endW);		
	ops_gcg->MultiVecView(V,startW,endW,ops_gcg);
	start[0] = startW; end[0] = endW;
	start[1] = startW; end[1] = endW;
	int nrows = end[0]-start[0], ncols = end[1]-start[1], row, col;
	ops_gcg->Printf("WtBW\n");
	ops_gcg->MultiVecQtAP('N','N',V,B,V,0,start,end,dbl_ws,nrows,mv_ws[0],ops_gcg);
	for (row = 0; row < nrows; ++row) {
		for (col = 0; col < ncols; ++col) {
			ops_gcg->Printf("%6.4e\t",dbl_ws[row+nrows*col]);
		}
		ops_gcg->Printf("\n");
	}	
#endif

	sizeW = endW-startW;

#if 0	
	if (sizeW<block_size) {
		ops_gcg->MultiVecSetRandomValue(V,endW,startW+block_size,ops_gcg);
		endW = startW+block_size;
		ops_gcg->MultiVecOrth(V,startW+sizeW,&endW,B,ops_gcg);
	}
	sizeW = endW-startW;
#endif

#if TIME_GCG
    	time_gcg.compW_time += ops_gcg->GetWtime();
#endif	
	return;
}

static void ComputeW12(void **V, void *A, void *B,
	double *ss_eval, void **ritz_vec, int *offset)
{
#if TIME_GCG
	time_gcg.compW_time -= ops_gcg->GetWtime();
#endif	
	assert(gcg_solver->user_defined_multi_linear_solver==0);
	void **b = ritz_vec;
	int start[2], end[2], block_size, length, inc, idx;
	double *destin = dbl_ws;

	double sigma = 0.0;
	if (gcg_solver->compW_cg_auto_shift==1) {
		sigma = -ss_eval[sizeC]+((ss_eval[sizeC+1]-ss_eval[sizeC])*0.01);
#if 0
		if (sizeC<3)
			sigma = -ss_eval[sizeC]+(3*(ss_eval[1]-ss_eval[0])>1?3*(ss_eval[1]-ss_eval[0]):1);
		else 
			sigma = -ss_eval[sizeC]+((ss_eval[sizeC]-ss_eval[sizeC-3])>1?(ss_eval[sizeC]-ss_eval[sizeC-3]):1);
#endif
	}
	gcg_solver->sigma = gcg_solver->compW_cg_shift+sigma; sigma = gcg_solver->sigma;
#if DEBUG
	ops_gcg->Printf("ss_eval[0] = %e, sigma = %e\n",ss_eval[0],gcg_solver->compW_cg_shift);
#endif

	void(*lin_sol)(void*,void**,void**,int*,int*,struct OPS_*);
	void *ws;
	lin_sol = ops_gcg->MultiLinearSolver;
	ws      = ops_gcg->multi_linear_solver_workspace;

	/* 20210628 A = sigma B + A */
	if (sigma!=0.0 && B!=NULL && ops_gcg->MatAxpby!=NULL) {
		ops_gcg->MatAxpby(sigma,B,1.0,A,ops_gcg);
		MultiLinearSolverSetup_BlockPCG(
				gcg_solver->compW_cg_max_iter,
				gcg_solver->compW_cg_rate,
				gcg_solver->compW_cg_tol,
				gcg_solver->compW_cg_tol_type,
				mv_ws,dbl_ws,int_ws,NULL,NULL,ops_gcg);
	}
	else {
		MultiLinearSolverSetup_BlockPCG(
				gcg_solver->compW_cg_max_iter,
				gcg_solver->compW_cg_rate,
				gcg_solver->compW_cg_tol,
				gcg_solver->compW_cg_tol_type,
				mv_ws,dbl_ws,int_ws,NULL,MatDotMultiVecShift,ops_gcg);
	}

	/* initialize */
	int total_length = 0;
	for (idx = 0; idx < offset[0]; ++idx) {
		total_length += offset[idx*2+2]-offset[idx*2+1];
	}

	block_size = 0; startW = endP; inc = 1; 
	for (idx = 0; idx < offset[0]; ++idx) {
		length   = offset[idx*2+2]-offset[idx*2+1];
		if (block_size+length >= total_length/2) {
			length = total_length/2 - block_size;
		}
		/* initialize x */
		start[0] = offset[idx*2+1]  ; end[0] = start[0]+length;
		start[1] = startW+block_size; end[1] = start[1]+length;
		ops_gcg->MultiVecAxpby(1.0,ritz_vec,0.0,V,start,end,ops_gcg);
		/* set b, b = (lambda+sigma) Bx */
		start[0] = offset[idx*2+1]     ; end[0] = start[0]+length;
		start[1] = offset[1]+block_size; end[1] = start[1]+length;
		ops_gcg->MatDotMultiVec(B,V,b,start,end,ops_gcg);
		
		int i;
		/* shift eigenvalues with sigma */
		for (i = start[0]; i < end[0]; ++i) ss_eval[i] += sigma;
		ops_gcg->MultiVecLinearComb(NULL,b,0,start,end,
				NULL,0,ss_eval+start[0],1,ops_gcg);			
		dcopy(&length,ss_eval+start[0],&inc,destin,&inc);
		/* recover eigenvalues */
		for (i = start[0]; i < end[0]; ++i) ss_eval[i] -= sigma;
		
		destin += length;
		block_size += length;
		if (block_size >= total_length/2) break;
	}
	/* solve x */
	start[0] = offset[1]; end[0] = start[0]+block_size;
	start[1] = startW   ; end[1] = start[1]+block_size;
#if TIME_GCG
    time_gcg.linsol_time -= ops_gcg->GetWtime();
#endif

#if DEBUG
	ops_gcg->Printf("b\n");
	ops_gcg->MultiVecView(b,offset[1],offset[1]+(total_length/2),ops_gcg);
#endif

	ops_gcg->MultiLinearSolver(A,b,V,start,end,ops_gcg);
#if TIME_GCG
    time_gcg.linsol_time += ops_gcg->GetWtime();
#endif
	endW = startW+block_size;	


	/* initialize x */
	start[0] = startW; end[0] = start[0]+block_size;
	start[1] = endW  ; end[1] = start[1]+block_size;
	ops_gcg->MultiVecAxpby(1.0,V,0.0,V,start,end,ops_gcg);

/* 是否按照原有 右端项 进行二次求解 */
#if 0
	/* set b, b = (lambda+sigma) Bx, where x is new */
	start[0] = endW     ; end[0] = start[0]+block_size;
	start[1] = offset[1]; end[1] = start[1]+block_size;
	ops_gcg->MatDotMultiVec(B,V,b,start,end,ops_gcg);
	block_size = 0; inc = 1; 
	for (idx = 0; idx < offset[0]; ++idx) {
		length   = offset[idx*2+2]-offset[idx*2+1];
		if (block_size+length >= total_length/2) {
			length = total_length/2 - block_size;
		}
		start[0] = offset[idx*2+1]     ; end[0] = start[0]+length;
		start[1] = offset[1]+block_size; end[1] = start[1]+length;
		
		int i;
		/* shift eigenvalues with sigma */
		for (i = start[0]; i < end[0]; ++i) ss_eval[i] += sigma;
		ops_gcg->MultiVecLinearComb(NULL,b,0,start,end,
				NULL,0,ss_eval+start[0],1,ops_gcg);			
		dcopy(&length,ss_eval+start[0],&inc,destin,&inc);
		/* recover eigenvalues */
		for (i = start[0]; i < end[0]; ++i) ss_eval[i] -= sigma;
		
		destin += length;
		block_size += length;
		if (block_size >= total_length/2) break;
	}
#else
/* 这种情况下, 在MatDotMultiVecShift中, 右端项会做为临时空间, 改变了值 */
if (sigma!=0.0 && B!=NULL && ops_gcg->MatAxpby==NULL) {
	block_size = 0; inc = 1; 
	for (idx = 0; idx < offset[0]; ++idx) {
		length   = offset[idx*2+2]-offset[idx*2+1];
		if (block_size+length >= total_length/2) {
			length = total_length/2 - block_size;
		}
		start[0] = offset[idx*2+1]     ; end[0] = start[0]+length;
		start[1] = offset[1]+block_size; end[1] = start[1]+length;
		ops_gcg->MatDotMultiVec(B,V,b,start,end,ops_gcg);	
		
		int i;
		/* shift eigenvalues with sigma */
		for (i = start[0]; i < end[0]; ++i) ss_eval[i] += sigma;
		ops_gcg->MultiVecLinearComb(NULL,b,0,start,end,
				NULL,0,ss_eval+start[0],1,ops_gcg);			
		dcopy(&length,ss_eval+start[0],&inc,destin,&inc);
		/* recover eigenvalues */
		for (i = start[0]; i < end[0]; ++i) ss_eval[i] -= sigma;
		
		destin += length;
		block_size += length;
		if (block_size >= total_length/2) break;
	}
}
#endif
	/* solve x */
	start[0] = offset[1]; end[0] = start[0]+block_size;
	start[1] = endW     ; end[1] = start[1]+block_size;
#if DEBUG
	ops_gcg->Printf("V\n");
	ops_gcg->MultiVecView(V,startW,startW+2*(total_length/2),ops_gcg);
	ops_gcg->Printf("b\n");
	ops_gcg->MultiVecView(b,offset[1],offset[1]+(total_length/2),ops_gcg);
#endif

	
#if TIME_GCG
    time_gcg.linsol_time -= ops_gcg->GetWtime();
#endif
	ops_gcg->MultiLinearSolver(A,b,V,start,end,ops_gcg);
#if TIME_GCG
    time_gcg.linsol_time += ops_gcg->GetWtime();
#endif
	endW += block_size;	
	assert(endW-startW <= total_length);

	ops_gcg->MultiLinearSolver             = lin_sol;
	ops_gcg->multi_linear_solver_workspace = ws;

	/* 20210628 recover A */
	if (sigma!=0.0 && B!=NULL && ops_gcg->MatAxpby!=NULL) {
		/* A = -sigma B + A */
		ops_gcg->MatAxpby(-sigma,B,1.0,A,ops_gcg);
	}

#if DEBUG
	ops_gcg->Printf("V\n")
	ops_gcg->MultiVecView(V,startW,endW,ops_gcg);
#endif

	/* orth W in V */
	if (0 == strcmp("mgs", gcg_solver->compW_orth_method))
		MultiVecOrthSetup_ModifiedGramSchmidt(
			gcg_solver->compW_orth_block_size,
			gcg_solver->compW_orth_max_reorth,
			gcg_solver->compW_orth_zero_tol,
			mv_ws[0],dbl_ws,ops_gcg);
	else if (0 == strcmp("bgs", gcg_solver->compW_orth_method))
		MultiVecOrthSetup_BinaryGramSchmidt(
			gcg_solver->compW_orth_block_size,
			gcg_solver->compW_orth_max_reorth,
			gcg_solver->compW_orth_zero_tol,
			mv_ws[0],dbl_ws,ops_gcg);
	else
		MultiVecOrthSetup_ModifiedGramSchmidt(
			gcg_solver->compW_orth_block_size,
			gcg_solver->compW_orth_max_reorth,
			gcg_solver->compW_orth_zero_tol,
			mv_ws[0],dbl_ws,ops_gcg);
	
	ops_gcg->MultiVecOrth(V,startW,&endW,B,ops_gcg);
	sizeW = endW-startW;

#if TIME_GCG
    	time_gcg.compW_time += ops_gcg->GetWtime();
#endif	
	return;
}

static void ComputeRayleighRitz(double *ss_matA, double *ss_eval, double *ss_evec, double tol,
		int nevConv, double *ss_diag, void *A, void **V)
{
#if TIME_GCG
    time_gcg.compRR_time -= ops_gcg->GetWtime();
#endif	
	int nrows, ncols, nrowsA, ncolsA, length, incx, incy, idx, start[2], end[2];
	double *source, *destin, alpha;
#if DEBUG
	ops_gcg->Printf("PtAP sizeP = %d\n", sizeP);
#endif
	if (sizeP>0) {
		/* 计算 PtAP 部分 */
		nrows  = sizeP      ; ncols  = sizeP      ;
		nrowsA = sizeV-sizeC; ncolsA = sizeV-sizeC;
		/* C = alpha*op(Q)*op(A)*op(P) + beta*C */
		/* dbl_ws: nrows*ncols+nrowA*ncols
		 *       <=(sizeV+sizeP)*sizeP */
		ops_gcg->DenseMatQtAP('L','S',nrowsA,ncolsA,nrows,ncols,
				1.0,ss_evec+(sizeV-sizeC)*(sizeX-sizeC),sizeV-sizeC, /* Q */
				    ss_matA                            ,sizeV-sizeC, /* A */
				    ss_evec+(sizeV-sizeC)*(sizeX-sizeC),sizeV-sizeC, /* P */
				0.0,dbl_ws                             ,nrows      , /* C */
				dbl_ws+nrows*ncols);		
	}	
	
	sizeV  = sizeX+sizeP+sizeW;
	startN = startN + (nevConv-sizeC);
	endN   = endN   + (nevConv-sizeC);
	endN   = (endN<endX)?endN:endX;

	sizeN  = endN - startN; 
	sizeC  = nevConv;

	/* 更新 ss_mat ss_evec */
	ss_matA = ss_diag+(sizeV-sizeC);
	ss_evec = ss_matA+(sizeV-sizeC)*(sizeV-sizeC); 
	
#if DEBUG
	ops_gcg->Printf("WtAW sizeW = %d\n", sizeW);
#endif

#if TIME_GCG
    time_gcg.rr_matW_time -= ops_gcg->GetWtime();
#endif
	if (sizeW>0) {
		/* 计算 VtAW 部分 */
		start[0] = startN; end[0] = endW;
		start[1] = startW; end[1] = endW;
		destin = ss_matA+(sizeV-sizeC)*(sizeX+sizeP-sizeC);
		/* (endW-startN)*(endW-startW) 个 double 
		 *               (endW-startW) 个 向量 */
		ops_gcg->MultiVecQtAP('S','N',V,A,V,0,start,end,destin,sizeV-sizeC,
				mv_ws[0],ops_gcg);
		/* 对称化 */
		length = sizeX+sizeP-sizeC;
		source = ss_matA+(sizeV-sizeC)*(sizeX+sizeP-sizeC); incx = 1; 
		destin = ss_matA+(sizeX+sizeP-sizeC); incy = sizeV-sizeC;
		for (idx = 0; idx < sizeW; ++idx) {
			dcopy(&length,source,&incx,destin,&incy);
			source += sizeV-sizeC; destin += 1;
		}
	}
#if TIME_GCG
    time_gcg.rr_matW_time += ops_gcg->GetWtime();
#endif
	
	if (sizeX == sizeV) {
#if DEBUG
		ops_gcg->Printf("V\n");
		ops_gcg->MultiVecView(V,0,sizeX,ops_gcg);
#endif		
		int block_size = gcg_solver->block_size;
		destin     = ss_matA;
		length     = sizeX-sizeC;
		block_size = block_size<length?block_size:length;
		start[0] = sizeC; end[0] = sizeX;
		start[1] = sizeC; end[1] = start[1]+block_size;
		while (length) {			
			ops_gcg->MultiVecQtAP('S','N',V,A,V,0,start,end,
				destin,sizeV-sizeC,mv_ws[0],ops_gcg);
			destin    += (sizeV-sizeC)*block_size;
			length    -= block_size;
			block_size = block_size<length?block_size:length;
			start[1] = end[1]; end[1] = start[1]+block_size;
		}	
	}
	else {
		/* 置零 X P 部分, 忽略 C 部分 */
		length = sizeX+sizeP-sizeC;
		destin = ss_matA;
		for (idx = 0; idx < length; ++idx) {
			memset(destin,0,length*sizeof(double));
			destin += sizeV-sizeC;
		}
		/* 赋值 X 部分的对角线 */
		length = sizeX-sizeC;
		source = ss_eval+sizeC; incx = 1              ; 
		destin = ss_matA      ; incy = (sizeV-sizeC)+1;
		dcopy(&length,source,&incx,destin,&incy);
		/* 更新 PtAP 部分*/
		length = sizeP;
		source = dbl_ws                                           ; incx = 1; 
		destin = ss_matA+(sizeV-sizeC)*(sizeX-sizeC)+(sizeX-sizeC); incy = 1;
		for (idx = 0; idx < length; ++idx) {
			dcopy(&length,source,&incx,destin,&incy);
			source += length; destin += sizeV-sizeC;
		}
	}
	
	/* 记录对角线部分 */
	length = sizeV-sizeC;
	source = ss_matA; incx = (sizeV-sizeC)+1; 
	destin = ss_diag; incy = 1              ;
	dcopy(&length,source,&incx,destin,&incy);
	
	/* 对 ss_matA 进行 shift */
	if (gcg_solver->compW_cg_shift != 0.0) {
		alpha = 1.0;
		length = sizeV-sizeC;
		source = &(gcg_solver->compW_cg_shift); incx = 0;
		destin = ss_matA             ; incy = (sizeV-sizeC)+1;
		daxpy(&length,&alpha,source,&incx,destin,&incy);
	}

#if DEBUG	
	int row, col;
	ops_gcg->Printf("ss_diag:\n");
	for (idx = 0; idx < length; ++idx) ops_gcg->Printf("%f\n",destin[idx]);
#endif	
	/* 计算小规模特征值问题 */
	char   JOBZ, RANGE, UPLO; 
	int    LDA, M, LDZ, INFO, N, LWORK, *IWORK, *IFAIL; 
	double ABSTOL, *AA, *W, *Z, *WORK;
	JOBZ   = 'V'        ; RANGE  = 'A'; UPLO  = 'U'        ;
	LDA    = sizeV-sizeC; ABSTOL = tol; LDZ   = sizeV-sizeC; 
	IWORK  = int_ws; INFO   = 0  ;
	/* 不再计算 C 部分 */
	N      = sizeV-sizeC; M = N;
	IFAIL  = int_ws+5*N; 
	AA     = ss_matA;
	W      = ss_eval+sizeC; 
	Z      = ss_evec; 
	WORK   = Z+LDZ*N;
	/* ss_diag ss_matA ss_evec 剩下的空间 */
	LWORK  = gcg_solver->length_dbl_ws-(WORK - gcg_solver->dbl_ws); 

#if DEBUG
	ops_gcg->Printf ( "LWORK = %d\n", LWORK );
	ops_gcg->Printf("dsyevx: AA\n"); 		
	for (row = 0; row < N; ++row) {
		for (col = 0; col < N; ++col) {
			ops_gcg->Printf("%6.4e\t",AA[row+col*LDA]);
		}
		ops_gcg->Printf("\n");
	}	
#endif


#if OPS_USE_MPI
	/* 当 PAS 调用 GCG 时, 且使用并行怎么办? 
	 * 没关系, PAS 需要保证每个进程都有特征向量 
	 * 同时, 这样的分批计算, 不仅仅是效率的提升
	 * 更重要的是, 保证, 每个进程的特征向量完全一致 */
	int *displs;
	int sendcount, *recvcounts;
	double *recvbuf;
	int IL, IU; int rank, nproc;

	/* 每列多一行, 将特征值拷贝至此, 进行通讯 */
	LDZ  = LDZ+1;
	/* 特征向量不包含 C 的部分 */
	Z    = ss_evec;	
	/* 重置工作空间 */ 
	WORK = Z+LDZ*N; LWORK = LWORK-N;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	/* 分类特征值 */
	destin = ss_eval+sizeC;
	length = N;
	/* 每组至少10个 */
	if (gcg_solver->compRR_min_num <= 0) {
	   gcg_solver->compRR_min_num = N/(nproc+2)>10?N/(nproc+2):10;
	}
	displs = malloc((2*nproc+1)*sizeof(int)); /* 长度需要 2*nproc+1 */
	if (rank == 0) {
	   SplitDoubleArray(destin,length,nproc,
		 gcg_solver->compRR_min_gap,
		 gcg_solver->compRR_min_num,
		 displs,dbl_ws,int_ws);
	}
	MPI_Bcast(displs,nproc+1,MPI_INT,0,MPI_COMM_WORLD);
 	sendcount  = displs[rank+1]-displs[rank];
	recvcounts = displs+nproc+1;
	for (idx = 0; idx < nproc; ++idx) {
		recvcounts[idx] = displs[idx+1]-displs[idx];
	}
	RANGE = 'I';
	/* 1 <= IL <= IU <= N */
	IL = displs[rank]+1; IU = displs[rank+1]  ;
	M  = IU-IL+1;
	/* 不同进程 W Z 不同 */
	W += displs[rank]  ; Z += LDZ*displs[rank];	

#if TIME_GCG
    	time_gcg.dsyevx_time -= ops_gcg->GetWtime();
#endif
	//printf("%d\n",sendcount);
	if (sendcount > 0) {
#if DEBUG
		ops_gcg->Printf("dsyevx: N   = %d, M  = %d, LDA = %d, IL = %d, IU  = %d, LDZ = %d\n", 
		      N, M, LDA, IL, IU, LDZ);
#endif
		dsyevx(&JOBZ,&RANGE,&UPLO,&N,AA,&LDA,
				NULL,NULL,&IL,&IU,&ABSTOL,&M,
				W,Z,&LDZ,WORK,&LWORK,IWORK,IFAIL,&INFO);
		assert(M==IU-IL+1);
		assert(INFO==0);
	}
#if TIME_GCG
    	time_gcg.dsyevx_time += ops_gcg->GetWtime();
	//ops_gcg->Printf("dsyevx = %.2f\n",time_gcg.dsyevx_time);
#endif
	/* 将计算得到的特征值复制到 Z 的最后一行 */
	length  = sendcount;
	source  = W      ; incx    = 1  ;
	destin  = Z+LDZ-1; incy    = LDZ;
	dcopy(&length,source,&incx,destin,&incy);
	recvbuf = ss_evec;
	sendcount *= LDZ;
	for (idx = 0; idx < nproc; ++idx) {
		recvcounts[idx] *= LDZ;
		displs[idx+1]   *= LDZ;
	}
	/* 全聚集特征对, 发送和接收都是连续数据 */

#if DEBUG
	ops_gcg->Printf("before allgaterv sendcount = %d\n", sendcount);
#endif
	MPI_Allgatherv(MPI_IN_PLACE,sendcount,MPI_DOUBLE,
		recvbuf,recvcounts,displs,MPI_DOUBLE,MPI_COMM_WORLD);
#if DEBUG
	ops_gcg->Printf("after  allgaterv sendcount = %d\n", sendcount);
#endif
	free(displs);
	/* 将 Z 的最后一行复制给特征值 */
	length = N;
	source = ss_evec+LDZ-1; incx = LDZ;
	destin = ss_eval+sizeC; incy = 1  ;
	dcopy(&length,source,&incx,destin,&incy);
	/* 移动特征向量 */
#if DEBUG
	ops_gcg->Printf("before memmove length = %d\n", length);
#endif
	length = N; destin = ss_evec; source = ss_evec; 
	for (idx = 0; idx < N; ++idx) {
		/* 保证 source 在被覆盖之前
		 * 将重叠区域的字节拷贝到 destin 中 */
		memmove(destin,source,length*sizeof(double));
		destin += N; source += LDZ;
	}
#if DEBUG
	ops_gcg->Printf("after  memmove length = %d\n", length);
#endif

#else

#if DEBUG
	ops_gcg->Printf("dsyevx: N = %d, M = %d\n", N, M);
#endif

#if TIME_GCG
    time_gcg.dsyevx_time -= ops_gcg->GetWtime();
#endif
	/* 保证 ss_evec 是正交归一的 */
	dsyevx(&JOBZ,&RANGE,&UPLO,&N,AA,&LDA,
			NULL,NULL,NULL,NULL,&ABSTOL,&M,
			W,Z,&LDZ,WORK,&LWORK,IWORK,IFAIL,&INFO);
	assert(INFO==0);
#if TIME_GCG
    time_gcg.dsyevx_time += ops_gcg->GetWtime();
#endif
#if DEBUG
	ops_gcg->Printf("dsyevx: N = %d, M = %d\n", N, M);
#endif
	assert(M==N);

#endif

	/* 恢复ss_matA对角线部分 */
	length = sizeV-sizeC;
	source = ss_diag; incx = 1              ;
	destin = ss_matA; incy = (sizeV-sizeC)+1;
	dcopy(&length,source,&incx,destin,&incy);

	/* 回复特征值 W */
	if (gcg_solver->compW_cg_shift != 0.0) {
		alpha  = -1.0;
		length = sizeV-sizeC;
		source = &(gcg_solver->compW_cg_shift); incx = 0;
		destin = ss_eval+sizeC       ; incy = 1;
		daxpy(&length,&alpha,source,&incx,destin,&incy);
	}
	
#if DEBUG
	ops_gcg->Printf("dsyevx: ss_evec\n");
	for (row = 0; row < N; ++row) {
		for (col = 0; col < M; ++col) {
			ops_gcg->Printf("%6.4e\t",Z[row+col*LDZ]);
		}	
		ops_gcg->Printf("\n");
	}
	ops_gcg->Printf("dsyevx: ss_eval\n");
	for (row = 0; row < M; ++row) ops_gcg->Printf("%6.4e\n",W[row]);
	ops_gcg->Printf("dsyevx: AA\n"); 		
	for (row = 0; row < N; ++row) {
		for (col = 0; col < N; ++col) {
			ops_gcg->Printf("%6.4e\t",AA[row+col*LDA]);
		}
		ops_gcg->Printf("\n");
	}
#endif
#if TIME_GCG
    time_gcg.compRR_time += ops_gcg->GetWtime();
#endif
	return;
}
static void GCG(void *A, void *B, double *eval, void **evec,
		int nevGiven, int *nevConv, struct OPS_ *ops)
{	
	/* offsetW[0] 表示有多少个块, 
	 * offsetW[1] <= idx < offsetW[2] 是未收敛的编号 */ 
	int *offsetP, *offsetW, *ptr_tmp;
	gcg_solver = (GCGSolver*)ops->eigen_solver_workspace;
	gcg_solver->A = A; gcg_solver->B = B; 
	gcg_solver->nevGiven = nevGiven;
	gcg_solver->nevConv  = *nevConv;	
	ops_gcg = ops;
	int    nevMax, block_size, nevInit, nev0, nev;
	int    numIterMax, numIter, numCheck;
	void   **V, **ritz_vec;
	double *ss_matA, *ss_diag, *ss_eval, *ss_evec, *tol;
	int    start[2], end[2], idx; double *coef;

	nevInit    = gcg_solver->nevInit   ;
	nevMax     = gcg_solver->nevMax    ; 
	block_size = gcg_solver->block_size; 
	/*  工作空间基于 nevInit nevMax block_size 分配 */
	assert(nevInit >= nevGiven);
	assert(nevInit <= nevMax);
	assert(nevInit >= 3*block_size || nevInit==nevMax);
	assert(nevMax  >= *nevConv+block_size);
	assert(nevMax  <= *nevConv+nevInit);
	/* 初始给出的 sizeX == nevInit 比最终要计算的 sizeX = nevMax 要小
	 * 这样的好处是, dsyevx_ 的规模较小, 但 gcg 整体迭代次数变大, 
	 * 当特征值个数真的非常大时会由效果 */

	numIterMax = gcg_solver->numIterMax; tol = gcg_solver->tol;
	/* 全局变量初始化 */
	sizeC  = 0    ; sizeN = block_size  ; 
	/* sizeX 需要大于 nevGiven */
	sizeX  = nevInit; sizeP  = 0; sizeW = 0; 
	sizeV  = sizeX+sizeP+sizeW;
	startN = sizeC; endN  = startN+sizeN; endX  = sizeX;
	startP = endX ; endP  = startP+sizeP;
	startW = endP ; endW  = startW+sizeW;
	/* workspace */
	V        = gcg_solver->mv_ws[0]; ritz_vec = evec;
	mv_ws[0] = gcg_solver->mv_ws[1]; mv_ws[1] = gcg_solver->mv_ws[2];
	mv_ws[2] = gcg_solver->mv_ws[3];
	ss_eval  = gcg_solver->dbl_ws; 
	for (idx = 0; idx < (nevMax+2*block_size); ++idx) {
	   ss_eval[idx] = 1.0;
	}
	ss_diag  = ss_eval+(nevMax+2*block_size);
	ss_matA  = ss_diag+(sizeV-sizeC);
	ss_evec  = ss_matA+(sizeV-sizeC)*(sizeV-sizeC); 

	int distance = (nevMax +2*block_size)                   /* ss_eval */ 
	              +(nevInit+2*block_size)                   /* ss_diag */ 
			+(nevInit+2*block_size)*(nevInit+2*block_size)  /* ss_matA */
			+(nevInit+2*block_size)*(nevInit+1*block_size); /* ss_evec */ 
	/* dbl_ws 包含 W 的部分 */
	dbl_ws = gcg_solver->dbl_ws+distance;
	gcg_solver->length_dbl_ws = (nevMax+2*block_size)                 /* ss_eval */
	                +2*(nevInit+2*block_size)*(nevInit+2*block_size)  /* ss_matA ss_evec */   
	                +10*(nevInit+2*block_size)                        /* ss_diag WORK */
			        +nevMax*block_size;                               /* for orth */
#if DEBUG
	ops_gcg->Printf ( "gcg_solver->length_dbl_ws = %d\n", gcg_solver->length_dbl_ws );
#endif

#if 1
	offsetP = gcg_solver->int_ws;
	offsetW = offsetP + block_size+3; 
	int_ws  = offsetW + block_size+3;
#else	
	int_ws  = gcg_solver->int_ws;
	offsetP = int_ws  + 6*(nevInit+2*block_size);
	offsetW = offsetP + block_size+2;
#endif

#if TIME_GCG
	time_gcg.checkconv_time = 0.0;
	time_gcg.compP_time     = 0.0; 
	time_gcg.compRR_time    = 0.0; 
	time_gcg.compRV_time    = 0.0;
	time_gcg.compW_time     = 0.0;
	time_gcg.compX_time     = 0.0;
	time_gcg.rr_matW_time   = 0.0;
	time_gcg.dsyevx_time    = 0.0;
	time_gcg.initX_time     = 0.0;
	time_gcg.linsol_time    = 0.0;
#endif
	
#if DEBUG
	ops_gcg->Printf("initial X\n");
#endif
	/* 对 X 赋随机初值且 B 正交归一化 */
	InitializeX(V,ritz_vec,B,nevGiven);	

#if DEBUG
	int row, col;
#endif

#if DEBUG
	ops_gcg->Printf("ComputeRayleighRitz\n");	
#endif
	ComputeRayleighRitz(ss_matA,ss_eval,ss_evec,
		gcg_solver->compRR_tol,0,ss_diag,A,V);	


	for (idx = sizeV; idx < (nevMax+2*block_size); ++idx) {
	   ss_eval[idx] = ss_eval[sizeV-1];
	}
	/* 更新 ss_mat ss_evec */
	ss_matA = ss_diag+(sizeV-sizeC);
	ss_evec = ss_matA+(sizeV-sizeC)*(sizeV-sizeC);

#if DEBUG
	ops_gcg->Printf("ComputeRitzVec\n");	
#endif
	ComputeRitzVec(ritz_vec,V,ss_evec);				
	
	*nevConv = (*nevConv)<nevMax?(*nevConv):nevMax;
	/* 用户希望收敛的特征对个数 */
	nev0 = *nevConv; *nevConv = 0; 
	/* 收敛个数达到 nev 后将 P 和 W 部分扩充为 X 部分 */
	nev  = nevInit<nevMax?2*block_size:nev0;
	nev  = nev<nev0?nev:nev0;
	numIter = 0; /* numIter 取负值时, 小于等于零的迭代不进行判断收敛性 */
#if PRINT_FIRST_UNCONV
	ops_gcg->Printf("------------------------------\n");
	ops_gcg->Printf("numIter\tnevConv\n",numIter, *nevConv);		
#endif
	do {
#if DEBUG
		ops_gcg->Printf("numIter = %d, sizeC = %d, sizeN = %d, sizeX = %d, sizeP = %d, sizeW = %d, sizeV = %d\n",
				numIter,sizeC,sizeN,sizeX,sizeP,sizeW,sizeV);
		ops_gcg->Printf("CheckConvergence\n");
#endif
		if (numIter <= 0) {
		   numCheck = 0;
		} 
		else {
		   numCheck = (startN+sizeN<endX)?(sizeN):(endX-startN);
		}
		numCheck = numCheck<gcg_solver->check_conv_max_num?numCheck:gcg_solver->check_conv_max_num;
		*nevConv = CheckConvergence(A,B,ss_eval,ritz_vec,numCheck,tol,offsetW);		
#if PRINT_FIRST_UNCONV
		ops_gcg->Printf("%d\t%d\n",numIter, *nevConv);		
#endif
		if (*nevConv >= nev) {
			if (*nevConv >= nev0) {
				break;
			}
			else {
				/* Update sizeX */
				nev   += sizeP+sizeW; 
				nev    = nev<nev0?nev:nev0;
				sizeX += sizeP+sizeW; 
				sizeX  = sizeX<nevMax?sizeX:nevMax;
				/* 将 P 和 W 部分写入 ritz_vec */
				start[0] = startN; end[0] = endW ;
				start[1] = endX  ; end[1] = sizeX; 
				coef     = ss_evec+(sizeV-sizeC)*(endX-sizeC);
				ops_gcg->MultiVecLinearComb(V,ritz_vec,0, 
						start,end,coef,sizeV-sizeC,NULL,0,ops_gcg);
			
				sizeP  = 0; sizeW = 0; sizeV = sizeX;
				startP = endX ; endP = startP;
				startW = endP ; endW = startW; 
				endX   = sizeX; 

				endN   = startN+block_size;
				endN   = endN<endX?endN:endX;
				sizeN  = endN-startN;	

				numIterMax -= numIter; numIter = 0;				
			}
		}
		if (numIter == 0)	{
			sizeP = 0; startP = endX; endP = startP+sizeP;
		}
		else {
#if DEBUG		
			ops_gcg->Printf("ComputeP\n");
#endif
			ComputeP(V,ss_evec,offsetP); /* update sizeP startP endP */
		}

#if DEBUG		
		ops_gcg->Printf("ComputeX\n");
#endif
		ComputeX(V,ritz_vec);

#if DEBUG		
		ops_gcg->MultiVecView(V,0,sizeX,ops_gcg);
#endif

#if DEBUG		
		ops_gcg->Printf("ComputeW\n");
#endif
		if (gcg_solver->compW_cg_order!=1) {
			ComputeW12(V,A,B,ss_eval,ritz_vec,offsetW); /* update sizeW startW endW */
		}
		else {
			ComputeW(V,A,B,ss_eval,ritz_vec,offsetW); /* update sizeW startW endW */
		}
		ptr_tmp = offsetP; offsetP = offsetW; offsetW = ptr_tmp;
		
#if DEBUG	
		ops_gcg->Printf("ComputeRayleighRitz\n");
		ops_gcg->Printf("VtAV\n");
		start[0] = 0; end[0] = sizeX+sizeP+sizeW; start[1] = 0; end[1] = sizeX+sizeP+sizeW;
		double *debug_dbl_ws = malloc((end[0]-start[0])*(end[1]-start[1])*sizeof(double));
		void **debug_mv_ws;
		ops_gcg->MultiVecCreateByMat(&debug_mv_ws,sizeX+sizeP+sizeW,A,ops_gcg);
		ops_gcg->MultiVecQtAP('N','N',V,A,V,0,start,end,
				debug_dbl_ws,sizeX+sizeP+sizeW,debug_mv_ws,ops_gcg);
		for (row = 0; row < end[0]-start[0]; ++row) {
			for (col = 0; col < end[1]-start[1]; ++col) {
				ops_gcg->Printf("%6.4e\t",debug_dbl_ws[row+col*(end[0]-start[0])]);
			}	
			ops_gcg->Printf("\n");
		}			
		ops_gcg->Printf("VtBV\n");
		ops_gcg->MultiVecQtAP('N','N',V,B,V,0,start,end,
				debug_dbl_ws,sizeX+sizeP+sizeW,debug_mv_ws,ops_gcg);
		for (row = 0; row < end[0]-start[0]; ++row) {
			for (col = 0; col < end[1]-start[1]; ++col) {
				ops_gcg->Printf("%6.4e\t",debug_dbl_ws[row+col*(end[0]-start[0])]);
			}	
			ops_gcg->Printf("\n");
		}
		free(debug_dbl_ws);
		ops_gcg->MultiVecDestroy(&debug_mv_ws,sizeX+sizeP+sizeW,ops_gcg);
#endif
	
		/* 计算完 PtAP 部分后再更新 sizeV */
		ComputeRayleighRitz(ss_matA,ss_eval,ss_evec,
			gcg_solver->compRR_tol,*nevConv,ss_diag,A,V); /* update sizeC startN endN sizeN */

		for (idx = sizeV; idx < (nevMax+2*block_size); ++idx) {
		   ss_eval[idx] = ss_eval[sizeV-1];
		}
		ss_matA = ss_diag+(sizeV-sizeC);
		ss_evec = ss_matA+(sizeV-sizeC)*(sizeV-sizeC);
		
#if DEBUG	
		ops_gcg->Printf("ComputeRitzVec\n");
#endif
		ComputeRitzVec(ritz_vec,V,ss_evec);
		
		++numIter;
	} while (numIter < numIterMax);
	
	gcg_solver->numIter = numIter+(gcg_solver->numIterMax-numIterMax);
	/* eval evec 都是 sizeX 长 */
	int inc = 1;
	dcopy(&sizeX,ss_eval,&inc,eval,&inc);
	
#if TIME_GCG
	ops_gcg->Printf("|--GCG----------------------------\n");
	time_gcg.time_total = time_gcg.checkconv_time
		+time_gcg.compP_time
		+time_gcg.compRR_time
		+time_gcg.compRV_time
		+time_gcg.compW_time
		+time_gcg.compX_time
		+time_gcg.initX_time;
	ops_gcg->Printf("|Total Time = %.2f, Avg Time per Iteration = %.2f\n",
		time_gcg.time_total,time_gcg.time_total/gcg_solver->numIter);
	ops_gcg->Printf("|checkconv   compP   compRR   (rr_matW   dsyexv)   compRV   compW   (linsol)   compX   initX\n");
	ops_gcg->Printf("|%.2f\t%.2f\t%.2f\t(%.2f\t%.2f)\t%.2f\t%.2f\t(%.2f)\t%.2f\t%.2f\n",
		time_gcg.checkconv_time,		
		time_gcg.compP_time,		
		time_gcg.compRR_time,
		time_gcg.rr_matW_time,
		time_gcg.dsyevx_time,		
		time_gcg.compRV_time,
		time_gcg.compW_time,
		time_gcg.linsol_time,
		time_gcg.compX_time,
		time_gcg.initX_time);	   	
	ops_gcg->Printf("|%.2f%%\t%.2f%%\t%.2f%%\t(%.2f%%\t%.2f%%)\t%.2f%%\t%.2f%%\t(%.2f%%)\t%.2f%%\t%.2f%%\n",
		time_gcg.checkconv_time/time_gcg.time_total*100,
		time_gcg.compP_time    /time_gcg.time_total*100,
		time_gcg.compRR_time   /time_gcg.time_total*100,
		time_gcg.rr_matW_time  /time_gcg.compRR_time*100,
		time_gcg.dsyevx_time   /time_gcg.compRR_time*100,
		time_gcg.compRV_time   /time_gcg.time_total*100,		
		time_gcg.compW_time    /time_gcg.time_total*100,
		time_gcg.linsol_time   /time_gcg.compW_time*100,
		time_gcg.compX_time    /time_gcg.time_total*100,
		time_gcg.initX_time    /time_gcg.time_total*100);
	ops_gcg->Printf("|--GCG----------------------------\n");
	time_gcg.checkconv_time = 0.0;
	time_gcg.compP_time     = 0.0; 
	time_gcg.compRR_time    = 0.0; 
	time_gcg.compRV_time    = 0.0;
	time_gcg.compW_time     = 0.0;
	time_gcg.compX_time     = 0.0;
	time_gcg.rr_matW_time   = 0.0;
	time_gcg.dsyevx_time    = 0.0;
	time_gcg.initX_time     = 0.0;
	time_gcg.linsol_time    = 0.0;
#endif
	
	return;
}

/* 设定 GCG 的工作空间 */
void EigenSolverSetup_GCG(
	double gapMin , 
	int    nevInit , int    nevMax , int block_size, 
	double tol[2]  , int    numIterMax, 
	int user_defined_multi_linear_solver,
	void **mv_ws[4], double *dbl_ws, int *int_ws,	
	struct OPS_ *ops)
{
	static GCGSolver gcg_solver_static = {
		.nevMax     = 3 , .gapMin = 0.01, 
		.nevInit    = 3 , .nevGiven   = 0,
		.block_size = 1 , .numIterMax = 4, .user_defined_multi_linear_solver = 0,
		.mv_ws      = {}, .dbl_ws  = NULL, .int_ws = NULL,		
		/* 算法内部参数 */		
		.initX_orth_method     = "mgs",
		.initX_orth_block_size = -1   ,
		.initX_orth_max_reorth = 1    ,
		.initX_orth_zero_tol   = 1e-14,
		.check_conv_max_num    = 15   ,	
		.compP_orth_method     = "mgs", 
		.compP_orth_block_size = -1   ,
		.compP_orth_max_reorth = 1    ,
		.compP_orth_zero_tol   = 1e-14,		
		.compW_orth_method     = "mgs",
		.compW_orth_block_size = -1   ,
		.compW_orth_max_reorth = 1    ,
		.compW_orth_zero_tol   = 1e-14,	
		.compW_cg_max_iter     = 40   ,
		.compW_cg_rate         = 1e-2 , 
		.compW_cg_tol          = 1e-8 ,
		.compW_cg_tol_type     = "abs",
		.compW_cg_auto_shift   = 0    ,	
		.compW_cg_shift        = 0.0  ,	
		.compW_cg_order        = 1    ,	
		.compRR_min_gap        = 0.01 ,
		.compRR_min_num        = -1   ,
		.compRR_tol            = 1e-16,
	};
		
	gcg_solver_static.gapMin     = gapMin;	 
	gcg_solver_static.nevInit    = nevInit;
	gcg_solver_static.nevMax     = nevMax;
	gcg_solver_static.block_size = block_size;
	gcg_solver_static.tol[0]     = tol[0];
	gcg_solver_static.tol[1]     = tol[1];
	gcg_solver_static.numIterMax = numIterMax;
	gcg_solver_static.mv_ws[0]   = mv_ws[0];
	gcg_solver_static.mv_ws[1]   = mv_ws[1];
	gcg_solver_static.mv_ws[2]   = mv_ws[2];
	gcg_solver_static.mv_ws[3]   = mv_ws[3];
	gcg_solver_static.dbl_ws     = dbl_ws;
 	gcg_solver_static.int_ws     = int_ws;
 	
 	gcg_solver_static.compRR_min_gap = gapMin;
 	gcg_solver_static.check_conv_max_num = block_size;
 	gcg_solver_static.user_defined_multi_linear_solver = user_defined_multi_linear_solver;
		
	ops->eigen_solver_workspace = (void *)(&gcg_solver_static);
	ops->EigenSolver            = GCG;
	return;	
}

void EigenSolverCreateWorkspace_GCG(
	int nevInit, int nevMax, int block_size, void *mat,
	void ***mv_ws, double **dbl_ws, int **int_ws, 
	struct OPS_ *ops)
{
	assert(mv_ws!=NULL);
	int sizeV = nevMax+2*block_size; 
	ops->MultiVecCreateByMat(&mv_ws[0],sizeV,mat,ops);				
	ops->MultiVecSetRandomValue(mv_ws[0],0,sizeV,ops);
	ops->MultiVecCreateByMat(&mv_ws[1],block_size,mat,ops);				
	ops->MultiVecSetRandomValue(mv_ws[1],0,block_size,ops);
	ops->MultiVecCreateByMat(&mv_ws[2],block_size,mat,ops);				
	ops->MultiVecSetRandomValue(mv_ws[2],0,block_size,ops);
	ops->MultiVecCreateByMat(&mv_ws[3],block_size,mat,ops);				
	ops->MultiVecSetRandomValue(mv_ws[3],0,block_size,ops);

	/* 这里 nevInit 的设定要与 EigenSolverSetup_GCG 中 nevInit 一致 */
	sizeV = nevInit+2*block_size;
	int length_dbl_ws = 2*sizeV*sizeV+10*sizeV
		+(nevMax+2*block_size)+(nevMax)*block_size;
	ops->Printf ( "length_dbl_ws = %d\n", length_dbl_ws );
	int length_int_ws = 6*sizeV+2*(block_size+3);
	ops->Printf ( "length_int_ws = %d\n", length_int_ws );
	if (dbl_ws!=NULL) {
		*dbl_ws = malloc(length_dbl_ws*sizeof(double));
		memset(*dbl_ws,0,length_dbl_ws*sizeof(double));
	} 
	if (int_ws!=NULL) {
		*int_ws = malloc(length_int_ws*sizeof(int));	
		memset(*int_ws,0,length_int_ws*sizeof(int));	
	}	
	return;
}
void EigenSolverDestroyWorkspace_GCG(
	int nevInit, int nevMax, int block_size, void *mat,
	void ***mv_ws, double **dbl_ws, int **int_ws, 
	struct OPS_ *ops)
{
	assert(mv_ws!=NULL);
	ops->MultiVecDestroy(&mv_ws[0],nevMax+2*block_size,ops);
	ops->MultiVecDestroy(&mv_ws[1],block_size,ops);
	ops->MultiVecDestroy(&mv_ws[2],block_size,ops);
	ops->MultiVecDestroy(&mv_ws[3],block_size,ops);
	if (dbl_ws!=NULL) {
		free(*dbl_ws); *dbl_ws = NULL;
	}
	if (int_ws!=NULL) {
		free(*int_ws); *int_ws = NULL;
	}
	return;
}


/* 参数设定函数需要在 Setup 之后调用 */
void EigenSolverSetParameters_GCG(
	int check_conv_max_num,
	const char *initX_orth_method, int initX_orth_block_size, int initX_orth_max_reorth, double initX_orth_zero_tol,
	const char *compP_orth_method, int compP_orth_block_size, int compP_orth_max_reorth, double compP_orth_zero_tol,
	const char *compW_orth_method, int compW_orth_block_size, int compW_orth_max_reorth, double compW_orth_zero_tol,
	int compW_cg_max_iter, double compW_cg_rate, double compW_cg_tol, const char *compW_cg_tol_type, int compW_cg_auto_shift,
	int compRR_min_num, double compRR_min_gap, double compRR_tol, 
	struct OPS_ *ops)
{
	
	struct GCGSolver_ *gcg_solver = (GCGSolver*)ops->eigen_solver_workspace;
	if (check_conv_max_num>0)
		gcg_solver->check_conv_max_num = check_conv_max_num;
	if (initX_orth_method!=NULL)
		strcpy(gcg_solver->initX_orth_method, initX_orth_method);
	if (initX_orth_block_size>0)
		gcg_solver->initX_orth_block_size = initX_orth_block_size;
	if (initX_orth_max_reorth>=0)
		gcg_solver->initX_orth_max_reorth = initX_orth_max_reorth;
	if (initX_orth_zero_tol>0)
		gcg_solver->initX_orth_zero_tol   = initX_orth_zero_tol;
	
	if (compP_orth_method!=NULL)
		strcpy(gcg_solver->compP_orth_method, compP_orth_method);
	if (compP_orth_block_size>0)
		gcg_solver->compP_orth_block_size = compP_orth_block_size;
	if (compP_orth_max_reorth>=0)
		gcg_solver->compP_orth_max_reorth = compP_orth_max_reorth;
	if (compP_orth_zero_tol>0)
		gcg_solver->compP_orth_zero_tol   = compP_orth_zero_tol;	

	if (compW_orth_method!=NULL)
		strcpy(gcg_solver->compW_orth_method, compW_orth_method);
	if (compW_orth_block_size>0)
		gcg_solver->compW_orth_block_size = compW_orth_block_size;
	if (compW_orth_max_reorth>=0)
		gcg_solver->compW_orth_max_reorth = compW_orth_max_reorth;
	if (compW_orth_zero_tol>0)
		gcg_solver->compW_orth_zero_tol   = compW_orth_zero_tol;
	if (compW_cg_max_iter>0)	
		gcg_solver->compW_cg_max_iter = compW_cg_max_iter;
	if (compW_cg_rate>0)
		gcg_solver->compW_cg_rate     = compW_cg_rate;
	if (compW_cg_tol>0)
		gcg_solver->compW_cg_tol      = compW_cg_tol;
	if (compW_cg_tol_type!=NULL)
		strcpy(gcg_solver->compW_cg_tol_type, compW_cg_tol_type);
	gcg_solver->compW_cg_auto_shift       = compW_cg_auto_shift;	
		
	if (compRR_min_gap>0)
		gcg_solver->compRR_min_gap = compRR_min_gap;
	if (compRR_min_num>0)
		gcg_solver->compRR_min_num = compRR_min_num;
	if (compRR_tol>0)
		gcg_solver->compRR_tol     = compRR_tol;
	
	return;	
}

void EigenSolverSetParametersFromCommandLine_GCG(
	int argc, char* argv[], struct OPS_ *ops)
{
	struct GCGSolver_ *gcg_solver = (GCGSolver*)ops->eigen_solver_workspace;

	ops->GetOptionFromCommandLine("-gcge_min_gap"    ,'f',
		&gcg_solver->gapMin    ,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_given_nevec",'i',
		&gcg_solver->nevGiven  ,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_max_niter"  ,'i',
		&gcg_solver->numIterMax,argc,argv, ops);	
	ops->GetOptionFromCommandLine("-gcge_abs_tol"    ,'f',
		&gcg_solver->tol[0]    ,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_rel_tol"    ,'f',
		&gcg_solver->tol[1]    ,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_user_defined_multi_lin_sol",'i',
		&gcg_solver->user_defined_multi_linear_solver,argc,argv, ops);
	
	ops->GetOptionFromCommandLine("-gcge_initX_orth_method"    ,'s',
		&gcg_solver->initX_orth_method    ,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_initX_orth_block_size",'i',
		&gcg_solver->initX_orth_block_size,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_initX_orth_max_reorth",'i',
		&gcg_solver->initX_orth_max_reorth,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_initX_orth_zero_tol"  ,'f',
		&gcg_solver->initX_orth_zero_tol  ,argc,argv, ops);
	
	ops->GetOptionFromCommandLine("-gcge_check_conv_max_num"   ,'i',
		&gcg_solver->check_conv_max_num,argc,argv, ops);
	
	ops->GetOptionFromCommandLine("-gcge_compP_orth_method"    ,'s',
		&gcg_solver->compP_orth_method    ,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compP_orth_block_size",'i',
		&gcg_solver->compP_orth_block_size,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compP_orth_max_reorth",'i',
		&gcg_solver->compP_orth_max_reorth,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compP_orth_zero_tol"  ,'f',
		&gcg_solver->compP_orth_zero_tol  ,argc,argv, ops);
	
	ops->GetOptionFromCommandLine("-gcge_compW_orth_method"    ,'s',
		&gcg_solver->compW_orth_method    ,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compW_orth_block_size",'i',
		&gcg_solver->compW_orth_block_size,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compW_orth_max_reorth",'i',
		&gcg_solver->compW_orth_max_reorth,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compW_orth_zero_tol"  ,'f',
		&gcg_solver->compW_orth_zero_tol  ,argc,argv, ops);
	
	ops->GetOptionFromCommandLine("-gcge_compW_cg_max_iter"  ,'i',
		&gcg_solver->compW_cg_max_iter  ,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compW_cg_rate"      ,'f',
		&gcg_solver->compW_cg_rate      ,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compW_cg_tol"       ,'f',
		&gcg_solver->compW_cg_tol       ,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compW_cg_tol_type"  ,'s',
		&gcg_solver->compW_cg_tol_type  ,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compW_cg_auto_shift",'i',
		&gcg_solver->compW_cg_auto_shift,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compW_cg_shift"     ,'f',
		&gcg_solver->compW_cg_shift,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compW_cg_order"     ,'i',
		&gcg_solver->compW_cg_order     ,argc,argv, ops);
	
	ops->GetOptionFromCommandLine("-gcge_compRR_min_num",'i',
		&gcg_solver->compRR_min_num,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compRR_min_gap",'i',
		&gcg_solver->compRR_min_gap,argc,argv, ops);
	ops->GetOptionFromCommandLine("-gcge_compRR_tol    ",'f',
		&gcg_solver->compRR_tol    ,argc,argv, ops);
    return;
}
void Printconv_GCG(struct OPS_ *ops)
{
	struct GCGSolver_ *gcg_solver = (GCGSolver*)ops->eigen_solver_workspace;
	if (1) {
       ops->Printf("\n");
       ops->Printf("---------------------------------------------------------------------------------------------------\n");
       ops->Printf(" -gcge_min_gap      <f>: minimum of gap of eigenvalues relatively    %.2e (default: 1e-2)\n",gcg_solver->gapMin);
       ops->Printf("---------------------------------------------------------------------------------------------------\n");
       ops->Printf(" -gcge_max_niter    <i>: maximum of gcg iterations                   %d (default: 100)\n",gcg_solver->numIterMax);
       ops->Printf(" -gcge_given_nevec  <i>: number of given initial eigenvectors        %d (default: 0)\n",gcg_solver->nevGiven);
       ops->Printf(" -gcge_abs_tol      <f>: absolute convergence tolerance              %.2e (default: 1e-4)\n",gcg_solver->tol[0]);
       ops->Printf(" -gcge_rel_tol      <f>: relative convergence tolerance              %.2e (default: 1e-4)\n",gcg_solver->tol[1]);
       ops->Printf("---------------------------------------------------------------------------------------------------\n");
       ops->Printf(" -gcge_user_defined_multi_lin_sol  <i>: use user-defined multi linear solver  %d (default: 0[1])\n",gcg_solver->user_defined_multi_linear_solver);
       ops->Printf("---------------------------------------------------------------------------------------------------\n");
       ops->Printf(" -gcge_initX_orth_method  <s>: use which kind of orthogonalization for X  %s (default: mgs[bgs])\n",gcg_solver->initX_orth_method);
       ops->Printf(" -gcge_compP_orth_method  <s>: use which kind of orthogonalization for P  %s (default: bqr[bgs|mgs])\n",gcg_solver->compP_orth_method);
       ops->Printf(" -gcge_compW_orth_method  <s>: use which kind of orthogonalization for W  %s (default: mgs[bgs])\n",gcg_solver->compW_orth_method);
       ops->Printf("---------------------------------------------------------------------------------------------------\n");
       ops->Printf(" -gcge_initX_orth_block_size  <i>: size of vectors orthogonalized in one patch for X  %d (default: -1)\n",gcg_solver->initX_orth_block_size);
       ops->Printf(" -gcge_compP_orth_block_size  <i>: size of vectors orthogonalized in one patch for P  %d (default: -1)\n",gcg_solver->compP_orth_block_size);
       ops->Printf(" -gcge_compW_orth_block_size  <i>: size of vectors orthogonalized in one patch for W  %d (default: -1)\n",gcg_solver->compW_orth_block_size);
       ops->Printf("---------------------------------------------------------------------------------------------------\n");
       ops->Printf(" -gcge_initX_orth_zero_tol  <f>: zero tolerance in orthogonal for X  %.2e (default: 1e-16)\n",gcg_solver->initX_orth_zero_tol);
       ops->Printf(" -gcge_compP_orth_zero_tol  <f>: zero tolerance in orthogonal for P  %.2e (default: 1e-16)\n",gcg_solver->compP_orth_zero_tol);
       ops->Printf(" -gcge_compW_orth_zero_tol  <f>: zero tolerance in orthogonal for W  %.2e (default: 1e-16)\n",gcg_solver->compW_orth_zero_tol);
       ops->Printf("---------------------------------------------------------------------------------------------------\n");
       ops->Printf(" -gcge_initX_orth_max_reorth  <i>: maximum reorthogonal times for X  %d (default: 2)\n",gcg_solver->initX_orth_max_reorth);
       ops->Printf(" -gcge_compP_orth_max_reorth  <i>: maximum reorthogonal times for P  %d (default: 2)\n",gcg_solver->compP_orth_max_reorth);
       ops->Printf(" -gcge_compW_orth_max_reorth  <i>: maximum reorthogonal times for W  %d (default: 2)\n",gcg_solver->compW_orth_max_reorth);
       ops->Printf("---------------------------------------------------------------------------------------------------\n");
       ops->Printf(" -gcge_compW_cg_max_iter   <i>: maximum number of cg iteration       %d (default: 30)\n",gcg_solver->compW_cg_max_iter);
       ops->Printf(" -gcge_compW_cg_rate       <f>: descent rate of residual in cg       %.2e (default: 1e-2)\n",gcg_solver->compW_cg_rate);
       ops->Printf(" -gcge_compW_cg_tol        <f>: convergence tolerance in cg          %.2e (default: 1e-8)\n",gcg_solver->compW_cg_tol);
       ops->Printf(" -gcge_compW_cg_tol_type   <s>: type of convergence tolerance in cg  %s (default: abs[rel|user])\n",gcg_solver->compW_cg_tol_type);
       ops->Printf(" -gcge_compW_cg_order      <i>: order of krylov space for W in cg    %d (default: 1[2])\n",gcg_solver->compW_cg_order);
       ops->Printf(" -gcge_compW_cg_auto_shift <i>: shift automatically in cg            %d (default: 0[1])\n",gcg_solver->compW_cg_auto_shift);
       ops->Printf(" -gcge_compW_cg_shift      <f>: shift manually in cg                 %.2e (default: 0.0)\n",gcg_solver->compW_cg_shift);
       ops->Printf("---------------------------------------------------------------------------------------------------\n");
       ops->Printf(" -gcge_compRR_min_num  <i>: minimum number for splitting RR eval  %d (default: 10)\n",gcg_solver->compRR_min_num);
       ops->Printf(" -gcge_compRR_min_gap  <f>: minimum gap for splitting RR eval     %.2e (default: 1e-2)\n",gcg_solver->compRR_min_gap);
       ops->Printf(" -gcge_compRR_tol      <f>: convergence tolerance in RR           %.2e (default: 1e-16)\n",gcg_solver->compRR_tol);
       ops->Printf("---------------------------------------------------------------------------------------------------\n");
       ops->Printf(" -gcge_print_orth_zero  <i>: print the zero index in orthogonal      %d (default: 0[1])\n",1);
       ops->Printf(" -gcge_print_split      <i>: print the split information of RR eval  %d (default: 0[1])\n",0);
       ops->Printf(" -gcge_print_conv       <i>: print convergence in each iteration     %d (default: 1[0])\n",1);
       ops->Printf(" -gcge_print_eval       <i>: print the final eigenvalues             %d (default: 1[0])\n",1);
       ops->Printf(" -gcge_print_evec       <i>: print the final eigenvectors            %d (default: 0[1])\n",0);
       ops->Printf(" -gcge_print_time       <i>: print total time of each part           %d (default: 1[0])\n",1);
       ops->Printf(" -gcge_print_usage      <i>: print usage of gcg eigen solver         %d (default: 1[0])\n",1);
       ops->Printf("--------------------------------------------------------------------------------------------------\n");
       //ops->Printf(" -bpcg_print_res        <i>: print residual per five bpcg iteration  (default: 1[0])\n");
    }
	return;	
}

void GCGE_Create(void *A, int nevMax, int block_size, int nevInit, void ***gcg_mv_ws, double *dbl_ws, int *int_ws, 
	struct OPS_ *ops) 
{
	ops->MultiVecCreateByMat(&gcg_mv_ws[0],nevMax+2*block_size,A,ops);				
	ops->MultiVecSetRandomValue(gcg_mv_ws[0],0,nevMax+2*block_size,ops);
	ops->MultiVecCreateByMat(&gcg_mv_ws[1],block_size,A,ops);				
	ops->MultiVecSetRandomValue(gcg_mv_ws[1],0,block_size,ops);
	ops->MultiVecCreateByMat(&gcg_mv_ws[2],block_size,A,ops);				
	ops->MultiVecSetRandomValue(gcg_mv_ws[2],0,block_size,ops);
	ops->MultiVecCreateByMat(&gcg_mv_ws[3],block_size,A,ops);				
	ops->MultiVecSetRandomValue(gcg_mv_ws[3],0,block_size,ops);
	return;
}
void GCGE_Setparameters(double gapMin, struct OPS_ *ops) 
{
	int    check_conv_max_num    = 50   ;
		
	char   initX_orth_method[8]  = "mgs"; 
	int    initX_orth_block_size = -1   ; 
	int    initX_orth_max_reorth = 2    ; double initX_orth_zero_tol    = 2*DBL_EPSILON;//1e-12
	
	char   compP_orth_method[8]  = "mgs"; 
	int    compP_orth_block_size = -1   ; 
	int    compP_orth_max_reorth = 2    ; double compP_orth_zero_tol    = 2*DBL_EPSILON;//1e-12
	
	char   compW_orth_method[8]  = "mgs";
	int    compW_orth_block_size = -1   ; 	
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

	return;	
/* 命令行获取 GCG 的算法参数 勿用 有 BUG, 
	 * 不应该改变 nevMax nevInit block_size, 这些与工作空间有关 */
}
void GCGE_Destroymvws(void ***gcg_mv_ws, double *dbl_ws, int *int_ws, int nevMax, int block_size, struct OPS_ *ops) 
{
	ops->MultiVecDestroy(&gcg_mv_ws[0],nevMax+2*block_size,ops);
	ops->MultiVecDestroy(&gcg_mv_ws[1],block_size,ops);
	ops->MultiVecDestroy(&gcg_mv_ws[2],block_size,ops);
	ops->MultiVecDestroy(&gcg_mv_ws[3],block_size,ops);
	free(dbl_ws); free(int_ws);
	return;
}
