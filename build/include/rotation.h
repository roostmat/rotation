/*******************************************************************************
*
* File rotation.h
*
* Copyright (C) 2025 Mattis Roost
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

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
    int nc;                          /* configuration */
    int npcorr;                      /* number of point correlators */
    complex_dble *corr;              /* size: npcorr*VOLUME */
    complex_dble *corr_tmp;          /* size: npcorr*VOLUME */
    int outlat[4];                   /* output lattice dimensions */
} corr_data_t;

typedef struct {
    int value;
    int index;
} IndexedValue_t;



/* ROTATION_UTIL_C*/
extern void lex_global(int *x,int *ip,int *ix);
extern void create_MPI_COMPLEX_DOUBLE(MPI_Datatype *mpi_complex_double);
extern void free_MPI_COMPLEX_DOUBLE(MPI_Datatype *mpi_complex_double);
extern void get_sorted_indices(int *array,int *indices,int length);
extern void sort_array_from_indices(int *array,int *indices,int length);
extern int lex_rank_to_ipr_global(int lex_rank);

/* SHIFT_C */
extern void shift_corr_tmp(corr_data_t *data,int *shift_vec);
extern void cleanup_shift(void);

/* PARALLEL_OUT_C */
extern void setup_parallel_out(corr_data_t *data);
extern void parallel_write(char *filename,corr_data_t *data,int *srcs,MPI_Offset skip);

#endif