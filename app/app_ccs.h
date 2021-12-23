/**
 *    @file  app_ccs.h
 *   @brief  app of ccs 
 *
 *  列压缩存储的稀疏矩阵的操作接口
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/11/13
 *      Revision:  none
 */
#ifndef  _APP_CCS_H_
#define  _APP_CCS_H_

#include	"ops.h"
#include	"app_lapack.h"

/* 第 i_row[i] 行, 第 j 列 元素非零, 为 data[i]
 * j_col[j] <= i < j_col[j+1] */
typedef struct CCSMAT_ {
	double *data ; 
	int    *i_row; int *j_col;
	int    nrows ; int ncols ;
} CCSMAT;
/* CCSMAT 对应的向量直接使用 LAPACKVEC */


void OPS_CCS_Set  (struct OPS_ *ops);
	
#endif  /* -- #ifndef _APP_CCS_H_ -- */
