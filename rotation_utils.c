#define MPI_UTIL_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "mpi.h"
#include "rotation.h"
#include "utils.h" /* for error, error_root */



static int data_parms_set=0;
static corr_data_parms_t data_parms={{0,0,0,0},0,0};



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



void set_corr_data_parms(int outlat[4], int npcorr)
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