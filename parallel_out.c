/*******************************************************************************
*
* File parallel_out.c
*
* Copyright (C) 2024, 2025 Mattis Roost
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs related to writing out correlation data.
*
*   void set_up_parallel_out(void)
*     Initializes the parallel output setup, including MPI data types and
*     and the correlation data structure.
*
*   void parallel_write(char *filename,corr_data_t *data,int *src,MPI_Offset skip)
*     Writes the correlation data stored in data.corr to a file in parallel using
*     MPI. The src parameter specifies the starting coordinates of the lattice
*     that is to be written out relative to the global lattice. The dimensions
*     of the lattice to be written out are specified by the outlat array. If
*     src is NULL, and outlat matches the global lattice dimensions, the full
*     lattice is written out. The skip parameter specifies the offset in bytes
*     to skip before writing the data, allowing for adding header information.
*
* See ./doc/rotation_input_sample.in on how to set the output lattice.
*
* To use parallel_write, the function set_up_parallel_out must be called first.
*
*******************************************************************************/

#define PARALLEL_OUT

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "endian.h" /* endianness */
#include "global.h" /* NPROC0, NPROC1, NPROC2, NPROC3, L0, L1, L2, L3, VOLUME, ipt[] */
#include "lattice.h" /* ipr_global */
#include "mpi.h"
#include "random.h" /* ranlxd */
#include "rotation.h" /* lex_to_coords */
#include "utils.h" /* error, error_root */



static int my_rank,endian,setup=0;
static int outlat[4]={-1,-1,-1,-1},size,npcorr; /* output lattice dimensions */
static MPI_Datatype MPI_COMPLEX_DOUBLE;
static int complex_double_size=0;



void setup_parallel_out(corr_data_t *data)
{
    if (setup==0)
    {
        int err_count;

        /* Initialize output lattice dimensions and number of point correlators */
        npcorr=data->npcorr;
        outlat[0]=data->outlat[0];
        outlat[1]=data->outlat[1];
        outlat[2]=data->outlat[2];
        outlat[3]=data->outlat[3];
        size=outlat[0]*outlat[1]*outlat[2]*outlat[3];

        create_MPI_COMPLEX_DOUBLE(&MPI_COMPLEX_DOUBLE);
        endian=endianness();
        /* Calculate data sizes */
        err_count=MPI_Type_size(MPI_COMPLEX_DOUBLE,&complex_double_size);
        error(err_count!=MPI_SUCCESS,1,"set_up_parallel_out [parallel_out.c]",
                "Failed to get size of MPI_COMPLEX_DOUBLE data type");
        /* Set my_rank */
        err_count=MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
        error(err_count!=MPI_SUCCESS,err_count,"set_up_parallel_out [parallel_out.c]",
                "Failed to get rank of current process");
        setup=1;
    }
}



void parallel_write(char *filename,corr_data_t *data,int *src,MPI_Offset skip)
{
    int ip,ix,*io,coords[4],start_coords[4];
    int ipcorr,t,x,y,z,npts;
    int *blocklengths,*displacements;
    int err_count=0;
    MPI_Datatype filetype;
    MPI_File fh;
    MPI_Aint lb,extent;

    error(setup!=1,1,"parallel_write [parallel_out.c]",
            "parallel_write called before set_up_parallel_out");

    io=malloc(npcorr*VOLUME*sizeof(int));
    error(io==NULL,1,"parallel_write [parallel_out.c]",
            "Failed to allocate memory for io array");

    /* Collect the data that each process contributes (0 <= npts < VOLUME)
       to data.corr_tmp in the correct order and find the displacements of the
       data points in the output file */
    if (src==NULL)
    {
        start_coords[0]=0;
        start_coords[1]=0;
        start_coords[2]=0;
        start_coords[3]=0;
    }
    else
    {
        start_coords[0]=src[0];
        start_coords[1]=src[1];
        start_coords[2]=src[2];
        start_coords[3]=src[3];
    }

    error_root((start_coords[0]+outlat[0])>N0||
               (start_coords[1]+outlat[1])>N1||
               (start_coords[2]+outlat[2])>N2||
               (start_coords[3]+outlat[3])>N3,1,
               "parallel_write [parallel_out.c]",
               "Output lattice dimensions exceed global lattice dimensions");

    npts=0;
    for (ipcorr=0;ipcorr<npcorr;ipcorr++)
    {
        for (t=0;t<outlat[0];t++)
        {
            coords[0]=(start_coords[0]+t)%N0;
            for (x=0;x<outlat[1];x++)
            {
                coords[1]=(start_coords[1]+x)%N1;
                for (y=0;y<outlat[2];y++)
                {
                    coords[2]=(start_coords[2]+y)%N2;
                    for (z=0;z<outlat[3];z++)
                    {
                        coords[3]=(start_coords[3]+z)%N3;
                        lex_global(coords,&ip,&ix);
                        if (my_rank==ip)
                        {
                            io[npts]=ipcorr*size+((t*outlat[1]+x)*outlat[2]+y)*outlat[3]+z;
                            (data->corr_tmp)[npts]=(data->corr)[ipcorr*VOLUME+ix];
                            npts++;
                        }
                    }
                }
            }
        }
    }

    /* Convert endianess to little endian */
    if (endian==BIG_ENDIAN)
    {
        bswap_double(npcorr*VOLUME*2,data->corr_tmp);
    }

    /* Create blocklenghts and displacements arrays for MPI_Type_indexed */
    if (npts>0)
    {
        blocklengths=malloc(npts*sizeof(int));
        displacements=malloc(npts*sizeof(int));
        error_loc((blocklengths==NULL)||(displacements==NULL),1,"parallel_write [parallel_out.c]",
                    "Failed to allocate memory for blocklengths and displacements arrays");

        for (t=0;t<npts;t++)
        {
            blocklengths[t]=1;
            displacements[t]=io[t];
        }
    }
    else
    {
        blocklengths=NULL;
        displacements=NULL;
    }

    /* Create custom MPI data type for each file view of each process */
    err_count=MPI_Type_indexed(npts,blocklengths,displacements,MPI_COMPLEX_DOUBLE,&filetype);
    err_count+=MPI_Type_commit(&filetype);
    error(err_count!=MPI_SUCCESS,1,"parallel_write [parallel_out.c]",
            "Failed to create custom MPI data type for file view");
    err_count=MPI_Type_get_extent(filetype,&lb,&extent);
    error(err_count!=MPI_SUCCESS,err_count,"parallel_write [parallel_out.c]",
            "Failed to get extent of custom MPI data type for file view");
    error((long)extent>npcorr*size*sizeof(complex_dble),1,"parallel_write [parallel_out.c]",
            "Extent of custom MPI data type for file view is too large");
    
    /* Open file */
    err_count=MPI_File_open(MPI_COMM_WORLD,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
    error(err_count!=MPI_SUCCESS,err_count,"parallel_write [parallel_out.c]",
            "Failed to open file for writing");

    /* Set file view */
    err_count=MPI_File_set_view(fh,skip,MPI_COMPLEX_DOUBLE,filetype,"native",MPI_INFO_NULL);
    error(err_count!=MPI_SUCCESS,err_count,"parallel_write",
            "Failed to set file view");

    /* Write data in parallel */
    err_count=MPI_File_write_all(fh,data->corr_tmp,npts,MPI_COMPLEX_DOUBLE,MPI_STATUSES_IGNORE);
    error(err_count!=MPI_SUCCESS,err_count,"parallel_write [parallel_out.c]",
            "Failed to write data to file");

    /* Ensure all operations are completed before closing the file */
    err_count=MPI_Barrier(MPI_COMM_WORLD);
    error(err_count != MPI_SUCCESS, err_count, "parallel_write [parallel_out.c]",
        "Failed at MPI_Barrier before closing file");

    /* Close the file */
    err_count=MPI_File_close(&fh);
    error(err_count!=MPI_SUCCESS,err_count,"parallel_write [parallel_out.c]",
            "Failed to close file");

    /* Free allocated memory */
    free(io);
    if (npts>0)
    {
        free(blocklengths);
        free(displacements);
    }
    err_count=MPI_Type_free(&filetype);
    error(err_count!=MPI_SUCCESS,err_count,"parallel_write [parallel_out.c]",
            "Failed to free custom MPI data type for file view");
}