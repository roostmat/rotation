#define GAMMA_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "linalg.h"
#include "mesons.h"

complex_dble GAMMA0[16];
complex_dble GAMMA1[16];
complex_dble GAMMA2[16];
complex_dble GAMMA3[16];
complex_dble GAMMA5[16];
complex_dble G2[16][16];


void init_GAMMA(void)
{
    complex_dble G_tmp[16],G_tmp2[16];

    /* GAMMA0 */
    GAMMA0[0].re=0.0; GAMMA0[0].im=0.0;
    GAMMA0[1].re=0.0; GAMMA0[1].im=0.0;
    GAMMA0[2].re=-1.0; GAMMA0[2].im=0.0;
    GAMMA0[3].re=0.0; GAMMA0[3].im=0.0;
    GAMMA0[4].re=0.0; GAMMA0[4].im=0.0;
    GAMMA0[5].re=0.0; GAMMA0[5].im=0.0;
    GAMMA0[6].re=0.0; GAMMA0[6].im=0.0;
    GAMMA0[7].re=-1.0; GAMMA0[7].im=0.0;
    GAMMA0[8].re=-1.0; GAMMA0[8].im=0.0;
    GAMMA0[9].re=0.0; GAMMA0[9].im=0.0;
    GAMMA0[10].re=0.0; GAMMA0[10].im=0.0;
    GAMMA0[11].re=0.0; GAMMA0[11].im=0.0;
    GAMMA0[12].re=0.0; GAMMA0[12].im=0.0;
    GAMMA0[13].re=-1.0; GAMMA0[13].im=0.0;
    GAMMA0[14].re=0.0; GAMMA0[14].im=0.0;
    GAMMA0[15].re=0.0; GAMMA0[15].im=0.0;

    /* GAMMA1 */
    GAMMA1[0].re=0.0; GAMMA1[0].im=0.0;
    GAMMA1[1].re=0.0; GAMMA1[1].im=0.0;
    GAMMA1[2].re=0.0; GAMMA1[2].im=0.0;
    GAMMA1[3].re=0.0; GAMMA1[3].im=-1.0;
    GAMMA1[4].re=0.0; GAMMA1[4].im=0.0;
    GAMMA1[5].re=0.0; GAMMA1[5].im=0.0;
    GAMMA1[6].re=0.0; GAMMA1[6].im=-1.0;
    GAMMA1[7].re=0.0; GAMMA1[7].im=0.0;
    GAMMA1[8].re=0.0; GAMMA1[8].im=0.0;
    GAMMA1[9].re=0.0; GAMMA1[9].im=1.0;
    GAMMA1[10].re=0.0; GAMMA1[10].im=0.0;
    GAMMA1[11].re=0.0; GAMMA1[11].im=0.0;
    GAMMA1[12].re=0.0; GAMMA1[12].im=1.0;
    GAMMA1[13].re=0.0; GAMMA1[13].im=0.0;
    GAMMA1[14].re=0.0; GAMMA1[14].im=0.0;
    GAMMA1[15].re=0.0; GAMMA1[15].im=0.0;

    /* GAMMA2 */
    GAMMA2[0].re=0.0; GAMMA2[0].im=0.0;
    GAMMA2[1].re=0.0; GAMMA2[1].im=0.0;
    GAMMA2[2].re=0.0; GAMMA2[2].im=0.0;
    GAMMA2[3].re=-1.0; GAMMA2[3].im=0.0;
    GAMMA2[4].re=0.0; GAMMA2[4].im=0.0;
    GAMMA2[5].re=0.0; GAMMA2[5].im=0.0;
    GAMMA2[6].re=1.0; GAMMA2[6].im=0.0;
    GAMMA2[7].re=0.0; GAMMA2[7].im=0.0;
    GAMMA2[8].re=0.0; GAMMA2[8].im=0.0;
    GAMMA2[9].re=1.0; GAMMA2[9].im=0.0;
    GAMMA2[10].re=0.0; GAMMA2[10].im=0.0;
    GAMMA2[11].re=0.0; GAMMA2[11].im=0.0;
    GAMMA2[12].re=-1.0; GAMMA2[12].im=0.0;
    GAMMA2[13].re=0.0; GAMMA2[13].im=0.0;
    GAMMA2[14].re=0.0; GAMMA2[14].im=0.0;
    GAMMA2[15].re=0.0; GAMMA2[15].im=0.0;

    /* GAMMA3 */
    GAMMA3[0].re=0.0; GAMMA3[0].im=0.0;
    GAMMA3[1].re=0.0; GAMMA3[1].im=0.0;
    GAMMA3[2].re=0.0; GAMMA3[2].im=-1.0;
    GAMMA3[3].re=0.0; GAMMA3[3].im=0.0;
    GAMMA3[4].re=0.0; GAMMA3[4].im=0.0;
    GAMMA3[5].re=0.0; GAMMA3[5].im=0.0;
    GAMMA3[6].re=0.0; GAMMA3[6].im=0.0;
    GAMMA3[7].re=0.0; GAMMA3[7].im=1.0;
    GAMMA3[8].re=0.0; GAMMA3[8].im=1.0;
    GAMMA3[9].re=0.0; GAMMA3[9].im=0.0;
    GAMMA3[10].re=0.0; GAMMA3[10].im=0.0;
    GAMMA3[11].re=0.0; GAMMA3[11].im=0.0;
    GAMMA3[12].re=0.0; GAMMA3[12].im=0.0;
    GAMMA3[13].re=0.0; GAMMA3[13].im=-1.0;
    GAMMA3[14].re=0.0; GAMMA3[14].im=0.0;
    GAMMA3[15].re=0.0; GAMMA3[15].im=0.0;

    /* GAMMA5 */
    GAMMA5[0].re=1.0; GAMMA5[0].im=0.0;
    GAMMA5[1].re=0.0; GAMMA5[1].im=0.0;
    GAMMA5[2].re=0.0; GAMMA5[2].im=0.0;
    GAMMA5[3].re=0.0; GAMMA5[3].im=0.0;
    GAMMA5[4].re=0.0; GAMMA5[4].im=0.0;
    GAMMA5[5].re=1.0; GAMMA5[5].im=0.0;
    GAMMA5[6].re=0.0; GAMMA5[6].im=0.0;
    GAMMA5[7].re=0.0; GAMMA5[7].im=0.0;
    GAMMA5[8].re=0.0; GAMMA5[8].im=0.0;
    GAMMA5[9].re=0.0; GAMMA5[9].im=0.0;
    GAMMA5[10].re=-1.0; GAMMA5[10].im=0.0;
    GAMMA5[11].re=0.0; GAMMA5[11].im=0.0;
    GAMMA5[12].re=0.0; GAMMA5[12].im=0.0;
    GAMMA5[13].re=0.0; GAMMA5[13].im=0.0;
    GAMMA5[14].re=0.0; GAMMA5[14].im=0.0;
    GAMMA5[15].re=-1.0; GAMMA5[15].im=0.0;

    /* G2 */
    /* G2[0] */
    cmat_dag_dble(4,GAMMA0,G_tmp);
    cmat_mul_dble(4,GAMMA5,G_tmp,G2[0]);
    /* G2[1] */
    cmat_dag_dble(4,GAMMA1,G_tmp);
    cmat_mul_dble(4,GAMMA5,G_tmp,G2[1]);
    /* G2[2] */
    cmat_dag_dble(4,GAMMA2,G_tmp);
    cmat_mul_dble(4,GAMMA5,G_tmp,G2[2]);
    /* G2[3] */
    cmat_dag_dble(4,GAMMA3,G_tmp);
    cmat_mul_dble(4,GAMMA5,G_tmp,G2[3]);
    /* G2[4] */
    cmat_dag_dble(4,GAMMA5,G_tmp);
    cmat_mul_dble(4,GAMMA5,G_tmp,G2[4]);
    /* G2[5] */    
    cmat_dag_dble(4,GAMMA5,G2[5]);
    /* G2[6] */
    cmat_mul_dble(4,GAMMA0,GAMMA1,G_tmp);
    cmat_dag_dble(4,G_tmp,G_tmp2);
    cmat_mul_dble(4,GAMMA5,G_tmp2,G2[6]);
    /* G2[7] */
    cmat_mul_dble(4,GAMMA0,GAMMA2,G_tmp);
    cmat_dag_dble(4,G_tmp,G_tmp2);
    cmat_mul_dble(4,GAMMA5,G_tmp2,G2[7]);
    /* G2[8] */
    cmat_mul_dble(4,GAMMA0,GAMMA3,G_tmp);
    cmat_dag_dble(4,G_tmp,G_tmp2);
    cmat_mul_dble(4,GAMMA5,G_tmp2,G2[8]);
    /* G2[9] */
    cmat_mul_dble(4,GAMMA0,GAMMA5,G_tmp);
    cmat_dag_dble(4,G_tmp,G_tmp2);
    cmat_mul_dble(4,GAMMA5,G_tmp2,G2[9]);
    /* G2[10] */
    cmat_mul_dble(4,GAMMA1,GAMMA2,G_tmp);
    cmat_dag_dble(4,G_tmp,G_tmp2);
    cmat_mul_dble(4,GAMMA5,G_tmp2,G2[10]);
    /* G2[11] */
    cmat_mul_dble(4,GAMMA1,GAMMA3,G_tmp);
    cmat_dag_dble(4,G_tmp,G_tmp2);
    cmat_mul_dble(4,GAMMA5,G_tmp2,G2[11]);
    /* G2[12] */
    cmat_mul_dble(4,GAMMA1,GAMMA5,G_tmp);
    cmat_dag_dble(4,G_tmp,G_tmp2);
    cmat_mul_dble(4,GAMMA5,G_tmp2,G2[12]);
    /* G2[13] */
    cmat_mul_dble(4,GAMMA2,GAMMA3,G_tmp);
    cmat_dag_dble(4,G_tmp,G_tmp2);
    cmat_mul_dble(4,GAMMA5,G_tmp2,G2[13]);
    /* G2[14] */
    cmat_mul_dble(4,GAMMA2,GAMMA5,G_tmp);
    cmat_dag_dble(4,G_tmp,G_tmp2);
    cmat_mul_dble(4,GAMMA5,G_tmp2,G2[14]);
    /* G2[15] */
    cmat_mul_dble(4,GAMMA3,GAMMA5,G_tmp);
    cmat_dag_dble(4,G_tmp,G_tmp2);
    cmat_mul_dble(4,GAMMA5,G_tmp2,G2[15]);

}

