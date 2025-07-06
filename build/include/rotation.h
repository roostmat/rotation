#ifndef ROTATION_H
#define ROTATION_H

#ifndef SU3_H
#include "su3.h"
#endif



#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)



typedef struct
{
    int size;                        /* outlat[0]*outlat[1]*outlat[2]*outlat[3] */
    complex_dble *corr;              /* size: npcorr*VOLUME */
    complex_dble *corr_tmp;          /* size: npcorr*VOLUME */
    complex_dble *corr_out;          /* size: npcorr*VOLUME */
    int nc;
} corr_data;



/* ROTATION_C */
extern const int npcorr;
extern const int outlat[4];
extern const int pos;
extern const int bcon;
extern corr_data data;               /* data structure for the correlators */
extern void copy_corr_data(complex_dble *dest);



/* MPI_UTIL_C*/
extern const MPI_Datatype MPI_COMPLEX_DOUBLE;
extern void create_MPI_COMPLEX_DOUBLE(void);
extern void free_MPI_COMPLEX_DOUBLE(void);



/* SHIFT_C */
extern void shift_corr(complex_dble *corr,int *shift_vec);
extern void cleanup_shift(void);



/* PARALLEL_OUT_C */
extern void lex_global(int *x,int *ip,int *ix);
extern void set_up_parallel_out(void);
extern void parallel_write(char *filename,corr_data *data,int *srcs);

#endif