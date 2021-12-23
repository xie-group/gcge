/**
 *    @file  app_ccs.h
 *   @brief  app of ccs 
 *
 *  ��ѹ���洢��ϡ�����Ĳ����ӿ�
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

/* �� i_row[i] ��, �� j �� Ԫ�ط���, Ϊ data[i]
 * j_col[j] <= i < j_col[j+1] */
typedef struct CCSMAT_ {
	double *data ; 
	int    *i_row; int *j_col;
	int    nrows ; int ncols ;
} CCSMAT;
/* CCSMAT ��Ӧ������ֱ��ʹ�� LAPACKVEC */


void OPS_CCS_Set  (struct OPS_ *ops);
	
#endif  /* -- #ifndef _APP_CCS_H_ -- */
