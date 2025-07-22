#define MPI_UTIL_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "global.h" /* for NPROC0, NPROC1, NPROC2, NPROC3, L0, L1, L2, L3, VOLUME, ipt[] */
#include "lattice.h" /* for ipr_global */
#include "mpi.h"
#include "rotation.h"
#include "utils.h" /* for error, error_root */



static int data_parms_set=0;
static corr_data_parms_t data_parms={{0,0,0,0},0,0};



/* Rank and index functions */

/* Get rank and local index from global coords */
void lex_global(int *x,int *ip,int *ix)
{
   int x0,x1,x2,x3;
   int n[4];

   x0=safe_mod(x[0],NPROC0*L0);
   x1=safe_mod(x[1],NPROC1*L1);
   x2=safe_mod(x[2],NPROC2*L2);
   x3=safe_mod(x[3],NPROC3*L3);

   n[0]=x0/L0;
   n[1]=x1/L1;
   n[2]=x2/L2;
   n[3]=x3/L3;

   (*ip)=ipr_global(n);

   x0=x0%L0;
   x1=x1%L1;
   x2=x2%L2;
   x3=x3%L3;

   (*ix)=x3+x2*L3+x1*L2*L3+x0*L1*L2*L3;
}


/* Get rank from lexicographical rank index */
int lex_rank_to_ipr_global(int lex_rank)
{
    int rank_coords[4],rank;

    error((lex_rank<0)||(lex_rank>=NPROC),1,"lex_rank_to_ipr_global [rotation_utils.c]",
            "lex_rank %d is out of bounds [0, %d)", lex_rank, NPROC);

    rank_coords[0]=lex_rank/(NPROC1*NPROC2*NPROC3);
    rank_coords[1]=(lex_rank/(NPROC2*NPROC3))%NPROC1;
    rank_coords[2]=(lex_rank/NPROC3)%NPROC2;
    rank_coords[3]=lex_rank%NPROC3;

    rank=ipr_global(rank_coords);
    return rank;
}



/* Create MPI data type for complex_dble */
void create_MPI_COMPLEX_DOUBLE(MPI_Datatype *mpi_complex_double)
{
    int err_count=0;
    err_count=MPI_Type_contiguous(2,MPI_DOUBLE,mpi_complex_double);
    err_count+=MPI_Type_commit(mpi_complex_double);
    error(err_count!=MPI_SUCCESS,1,"create_MPI_COMPLEX_DOUBLE [rotation_utils.c]",
            "Failed to create MPI_COMPLEX_DOUBLE data type");
}



void free_MPI_COMPLEX_DOUBLE(MPI_Datatype *mpi_complex_double)
{
    int err_count=0;
    err_count=MPI_Type_free(mpi_complex_double);
    error(err_count!=MPI_SUCCESS,1,"free_MPI_COMPLEX_DOUBLE [rotation_utils.c]",
            "Failed to free MPI_COMPLEX_DOUBLE data type");
}



/* Set correlation data parameters */
void set_corr_data_parms(int outlat[4],int npcorr)
{
    data_parms.outlat[0] = outlat[0];
    data_parms.outlat[1] = outlat[1];
    data_parms.outlat[2] = outlat[2];
    data_parms.outlat[3] = outlat[3];
    data_parms.size = outlat[0]*outlat[1]*outlat[2]*outlat[3];
    data_parms.npcorr = npcorr;
    data_parms_set=1;
}



void get_outlat(int outlat[4])
{
    outlat[0] = data_parms.outlat[0];
    outlat[1] = data_parms.outlat[1];
    outlat[2] = data_parms.outlat[2];
    outlat[3] = data_parms.outlat[3];
}



int get_size(void)
{
    error(!data_parms_set,1,"get_size [rotation_utils.c]",
            "Data parameters not set. Call set_corr_data_parms first.");
    return data_parms.size;
}



int get_npcorr(void)
{
    error(!data_parms_set,1,"get_npcorr [rotation_utils.c]",
            "Data parameters not set. Call set_corr_data_parms first.");
    return data_parms.npcorr;
}



/* Argsort function */
static int compare(const void *a,const void *b)
{
    IndexedValue_t *ia=(IndexedValue_t *)a;
    IndexedValue_t *ib=(IndexedValue_t *)b;
    return ia->value-ib->value;
}



void get_sorted_indices(int *array,int *indices,int length)
{
    int i;
    IndexedValue_t *iv=malloc(length*sizeof(IndexedValue_t));

    for (i=0;i<length;i++)
    {
        iv[i].value=array[i];
        iv[i].index=i;
    }
    
    qsort(iv,length,sizeof(IndexedValue_t),compare);
    for (i=0;i<length;i++)
    {
        indices[i]=iv[i].index;
    }
    
    free(iv);
}



void sort_array_from_indices(int *array,int *indices,int length)
{
    int i,tmp[length];
    for (i=0;i<length;i++)
    {
        tmp[i]=array[indices[i]];
    }
    for (i=0;i<length;i++)
    {
        array[i]=tmp[i];
    }
}