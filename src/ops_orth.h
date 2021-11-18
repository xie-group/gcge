/**
 *    @file  ops_orth.h
 *   @brief  正交化操作 
 *
 *  正交化操作
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/17
 *      Revision:  none
 */
#ifndef  _OPS_ORTH_H_
#define  _OPS_ORTH_H_

#include    "ops.h"
#include    "app_lapack.h"

typedef struct ModifiedGramSchmidtOrth_ {
	int    block_size;    /* 自身正交化大小 */ 
	int    max_reorth;
	double orth_zero_tol; /* 零向量误差     */ 
	double reorth_tol;
	void   **mv_ws;       /* 多向量工作空间 */
	double *dbl_ws;      /* 浮点型工作空间 */
} ModifiedGramSchmidtOrth;

typedef struct BinaryGramSchmidtOrth_ {
	int    block_size;    /* 自身正交化大小 */ 
	int    max_reorth;
	double orth_zero_tol; /* 零向量误差     */ 
	double reorth_tol;
	void   **mv_ws;       /* 多向量工作空间 */
	double *dbl_ws;      /* 浮点型工作空间 */
} BinaryGramSchmidtOrth;

void MultiVecOrthSetup_ModifiedGramSchmidt(
	int block_size, int max_reorth, double orth_zero_tol, 
	void **mv_ws, double *dbl_ws, struct OPS_ *ops);
void MultiVecOrthSetup_BinaryGramSchmidt(
	int block_size, int max_reorth, double orth_zero_tol, 
	void **mv_ws, double *dbl_ws, struct OPS_ *ops);

#endif  /* -- #ifndef _OPS_ORTH_H_ -- */


