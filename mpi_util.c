#define MPI_UTIL_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "mpi.h"
#include "rotation.h"
#include "utils.h" /* for error, error_root */



MPI_Datatype MPI_COMPLEX_DOUBLE;



void create_MPI_COMPLEX_DOUBLE(void)
{
    int err_count=0;
    err_count=MPI_Type_contiguous(2,MPI_DOUBLE,&MPI_COMPLEX_DOUBLE);
    err_count+=MPI_Type_commit(&MPI_COMPLEX_DOUBLE);
    error(err_count!=MPI_SUCCESS,1,"create_MPI_COMPLEX_DOUBLE",
            "Failed to create MPI_COMPLEX_DOUBLE data type");
}



void free_MPI_COMPLEX_DOUBLE(void)
{
    int err_count=0;
    err_count=MPI_Type_free(&MPI_COMPLEX_DOUBLE);
    error(err_count!=MPI_SUCCESS,1,"free_MPI_COMPLEX_DOUBLE",
            "Failed to free MPI_COMPLEX_DOUBLE data type");
}