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
    complex_dble *corr;              /* size: npcorr*VOLUME */
    complex_dble *corr_tmp;          /* size: npcorr*VOLUME */
    int nc;                          /* configuration */
} corr_data_t;

typedef struct
{
    int outlat[4];                   /* output lattice dimensions */
    int size;                        /* outlat[0]*outlat[1]*outlat[2]*outlat[3] */
    int npcorr;                      /* number of point correlators */
} corr_data_parms_t;

typedef struct {
    int value;
    int index;
} IndexedValue_t;




/* ROTATION_C 
extern const int npcorr;
extern const int outlat[4];
extern const int pos;
extern const int bcon;
extern corr_data data;
extern void copy_corr_data(complex_dble *dest); */



/* ROTATION_UTIL_C*/
extern void lex_global(int *x,int *ip,int *ix);
extern void create_MPI_COMPLEX_DOUBLE(MPI_Datatype *mpi_complex_double);
extern void free_MPI_COMPLEX_DOUBLE(MPI_Datatype *mpi_complex_double);
extern void set_corr_data_parms(int outlat[4], int npcorr);
extern void get_outlat(int outlat[4]);
extern int get_size(void);
extern int get_npcorr(void);
extern void get_sorted_indices(int *array,int *indices,int length);
extern void sort_array_from_indices(int *array, int *indices, int length);
extern int lex_rank_to_ipr_global(int lex_rank);



/* SHIFT_C */
extern void shift_corr(complex_dble *corr,int *shift_vec);
extern void cleanup_shift(void);
extern void average_equiv(complex_dble *corr,int outlat[4],int bc);
extern void average_equiv2(complex_dble *corr);



/* PARALLEL_OUT_C */
extern void set_up_parallel_out(void);
extern void parallel_write(char *filename,corr_data_t *data,int *srcs);

#endif