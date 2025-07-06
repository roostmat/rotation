#define PARALLEL_OUT

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "endian.h" /* for endianness */
#include "global.h" /* for NPROC0, NPROC1, NPROC2, NPROC3, L0, L1, L2, L3, VOLUME, ipt[] */
#include "lattice.h" /* for ipr_global */
#include "mpi.h"
#include "random.h" /* for ranlxd */
#include "rotation.h" /* for lex_to_coords, source_pos */
#include "utils.h" /* for error, error_root */



static int my_rank,endian,setup=0;
static int int_size=0,complex_double_size=0;



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



void set_up_parallel_out(void)
{
    if (setup==0)
    {
        int err_count;
        create_MPI_COMPLEX_DOUBLE();
        endian=endianness();
        /* Calculate data sizes */
        err_count=MPI_Type_size(MPI_INT,&int_size);
        err_count+=MPI_Type_size(MPI_COMPLEX_DOUBLE,&complex_double_size);
        error(err_count!=MPI_SUCCESS,1,"parallel_write [parallel_out.c]",
                "Failed to get size of MPI_INT and MPI_COMPLEX_DOUBLE data types");
        /* Set my_rank */
        err_count=MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
        error(err_count!=MPI_SUCCESS,err_count,"parallel_write [parallel_out.c]",
                "Failed to get rank of current process");
        setup=1;
    }
}



void parallel_write(char *filename, corr_data *data, int *srcs)
{
    int ip,ix,*io,coords[4],start_coords[4];
    int ipcorr,t,x,y,z,npts;
    int *blocklengths,*displacements;
    int err_count=0;
    MPI_Datatype filetype;
    MPI_File fh;
    MPI_Offset disp;
    MPI_Aint lb,extent;

    error(setup!=1,1,"parallel_write [parallel_out.c]",
            "parallel_write called before set_up_parallel_out");
    io=malloc(npcorr*VOLUME*sizeof(int));
    error(io==NULL,1,"parallel_write [parallel_out.c]",
            "Failed to allocate memory for io array");

    /* Collect the data that each process contributes (0 <= npts < VOLUME)
       to data.corr_out in the correct order and find the displacements of the
       data points in the output file */
    npts=0;
    for (ipcorr=0;ipcorr<npcorr;ipcorr++)
    {
        if (pos==0)
        {
            start_coords[0]=srcs[4*ipcorr+0];
            start_coords[1]=srcs[4*ipcorr+1];
            start_coords[2]=srcs[4*ipcorr+2];
            start_coords[3]=srcs[4*ipcorr+3];
        }
        else
        {
            start_coords[0]=safe_mod(srcs[4*ipcorr+0]-(outlat[0]-1)/2,N0);
            start_coords[1]=safe_mod(srcs[4*ipcorr+1]-(outlat[1]-1)/2,N1);
            start_coords[2]=safe_mod(srcs[4*ipcorr+2]-(outlat[2]-1)/2,N2);
            start_coords[3]=safe_mod(srcs[4*ipcorr+3]-(outlat[3]-1)/2,N3);
        }

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
                            io[npts]=ipcorr*(data->size)+((t*outlat[1]+x)*outlat[2]+y)*outlat[3]+z;
                            (data->corr_out)[npts]=(data->corr)[ipcorr*VOLUME+ix];
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
        bswap_double(npcorr*VOLUME*2,data->corr_out);
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
    error((long)extent>npcorr*data->size*sizeof(complex_dble),1,"parallel_write [parallel_out.c]",
            "Extent of custom MPI data type for file view is too large");

    /* Open file */
    err_count=MPI_File_open(MPI_COMM_WORLD,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
    error(err_count!=MPI_SUCCESS,err_count,"parallel_write [parallel_out.c]",
            "Failed to open file for writing");

    /* Set file view */
    disp=4*npcorr*int_size; /* Skip the source coords at beginning of file */
    err_count=MPI_File_set_view(fh,disp,MPI_COMPLEX_DOUBLE,filetype,"native",MPI_INFO_NULL);
    error(err_count!=MPI_SUCCESS,err_count,"parallel_write",
            "Failed to set file view");

    /* Write data in parallel */
    err_count=MPI_File_write_all(fh,data->corr_out,npts,MPI_COMPLEX_DOUBLE,MPI_STATUSES_IGNORE);
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