/*******************************************************************************
 *
 * Copyright (C) 2024, 2025 Mattis Roost
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *******************************************************************************
 *
 * Test for parallel_write()
 * 
 * This program tests the parallel_write function by writing out
 * data from all processes to a file in parallel and then comparing
 * the result with a serial write from process 0 to ensure correctness.
 * 
 * COMPILATION:
 *   make parallel_out_test
 * 
 * USAGE:
 *   export OMP_NUM_THREADS=1
 *   mpirun -np <num_processes> ./parallel_out_test [OPTIONS]
 * 
 * OPTIONS:
 *  -outlat <t> <x> <y> <z>    Output lattice dimensions (required)
 *  -source <t> <x> <y> <z>    Source position coordinates (optional, default: 0 0 0 0)
 *  -npcorr <n>                Number of point correlators (optional, default: 1)
 * 
 *******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h> /* isalpha */
#include <math.h>
#include "global.h" /* NPROC0, NPROC1, NPROC2, NPROC3, L0, L1, L2, L3, VOLUME */
#include "lattice.h" /* geometry */
#include "mpi.h"
#include "random.h" /* start_ranlux */
#include "rotation.h" /* corr_data set_outlat alloc_ranks source_pos gather_data */
#include "utils.h" /* mpi_init error error_root safe_mod */





int main(int argc, char *argv[])
{
    corr_data_t data;

    int my_rank,endian,iw,num_bytes;
    int size,source[4];
    char test_file[]="parallel_write_test.dat", cmp_file[]="serial_write_cmp.dat";
    FILE *flog=NULL,*ftest=NULL,*fcmp=NULL;
    int err_count,int_size;
    MPI_Offset skip;

    int i,j,t,x,y,z,ipcorr,err,cmp_result;
    int index,loc_index;
    char *buf_test,*buf_cmp;
    size_t ir_test,ir_cmp;
    complex_dble data_pt;

    mpi_init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Parse command line arguments on rank 0 */
    if (my_rank==0)
    {
        /* Set default source and npcorr*/
        source[0]=0;
        source[1]=0;
        source[2]=0;
        source[3]=0;
        data.npcorr=1;

        /* Open log file */
        flog=freopen("parallel_out_test.log","w", stdout);
        error_root(flog==NULL,1,"main [parallel_out_test.c]",
                "Unable to open log file");

        for (i=1;i<argc;i++)
        {
            if (argv[i][0]=='-')
            {
                if (strcmp(argv[i],"-outlat")==0)
                {
                    error_root(i+4>argc,1,"main [parallel_out_test.c]",
                                "Too few arguments were given. outlat flag requires four positive integer arguments.");
                    for (j=0;j<4;j++)
                    {
                        i++;
                        data.outlat[j]=atoi(argv[i]);
                        error_root(data.outlat[j]==0,1,"main [parallel_out_test.c]",
                                "outlat flag requires four positive integer arguments");
                    }
                }
                else if (strcmp(argv[i],"-source")==0)
                {
                    error_root(i+4>argc,1,"main [parallel_out_test.c]",
                                "Too few arguments were given. source flag requires four positive integer arguments.");
                    for (j=0;j<4;j++)
                    {
                        i++;
                        source[j]=atoi(argv[i]);
                    }
                }
                else if (strcmp(argv[i],"-npcorr")==0)
                {
                    error_root(i+1>argc,1,"main [parallel_out_test.c]",
                                "Too few arguments were given. npcorr flag requires one positive integer argument.");
                    i++;
                    data.npcorr=atoi(argv[i]);
                    error_root(data.npcorr<=0,1,"main [parallel_out_test.c]",
                                "npcorr flag requires one positive integer argument");
                }
                else
                {
                    error_root(1,1,"main [parallel_out_test.c]",
                                "Unknown flag");
                }
            }
        }

        /* Check that lattice sizes are positive integers */
        if ((N0<1)||(N1<1)||(N2<1)||(N3<1)
            ||(L0<1)||(L1<1)||(L2<1)||(L3<1)
            ||(data.outlat[0]<1)||(data.outlat[1]<1)||(data.outlat[2]<1)||(data.outlat[3]<1))
        {
            error_root(1,1,"main [gather_data_test.c]",
                    "Lattice sizes must be positive integers");
        }
        /* Check that output lattice size is less than or equal to global lattice size */
        if ((data.outlat[0])>(N0)||(data.outlat[1])>(N1)
            ||(data.outlat[2])>(N2)||(data.outlat[3])>(N3))
        {
            error_root(1,1,"main [gather_data_test.c]",
                    "Output lattice size must be less than or equal to global lattice size");
        }
    }
    MPI_Bcast(&data.npcorr,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(data.outlat,4,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&source,4,MPI_INT,0,MPI_COMM_WORLD);

    /* Allocate data */
    size=data.outlat[0]*data.outlat[1]*data.outlat[2]*data.outlat[3];
    data.nc=1; /* irrelevant in this case */
    data.corr=malloc(data.npcorr*VOLUME*sizeof(complex_dble));
    data.corr_tmp=malloc(data.npcorr*VOLUME*sizeof(complex_dble));
    error((data.corr==NULL)||(data.corr_tmp==NULL),1,"main [gather_data_test.c]",
            "Unable to allocate data arrays");

    /* Allocate buffer */
    num_bytes=4*sizeof(int)+2*data.npcorr*size*sizeof(double);
    buf_test=malloc(num_bytes);
    buf_cmp=malloc(num_bytes);
    if (buf_test==NULL||buf_cmp==NULL)
    {
        error_root(1, 1, "main [parallel_out_test.c]", "Unable to allocate memory for buffers");
    }

    /* Set endianess */
    endian=endianness();

    /* Set up lattice geometry */
    geometry();
    
    /* Write local lattice on each process */
    for (ipcorr=0;ipcorr<data.npcorr;ipcorr++)
    {
        for (t=0;t<L0;t++)
        {
            for (x=0;x<L1;x++)
            {
                for (y=0;y<L2;y++)
                {
                    for (z=0;z<L3;z++)
                    {
                        loc_index=z+y*L3+x*L2*L3+t*L1*L2*L3;
                        index=ipcorr*N0*N1*N2*N3
                                +cpr[3]*L3+z
                                +(cpr[2]*L2+y)*N3
                                +(cpr[1]*L1+x)*N2*N3
                                +(cpr[0]*L0+t)*N1*N2*N3;
                        data.corr[ipcorr*VOLUME+loc_index].re=(double)index;
                        data.corr[ipcorr*VOLUME+loc_index].im=-(double)index;
                    }
                }
            }
        }
    }

    /* Write test settings to log file */
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank==0)
    {
        printf("PARALLEL_OUT TEST\n\n");
        printf("Global lattice: %d %d %d %d\n",
                N0,N1,N2,N3);
        printf("Local lattice: %d %d %d %d\n",L0,L1,L2,L3);
        printf("Output lattice: %d %d %d %d\n",
                data.outlat[0],data.outlat[1],data.outlat[2],data.outlat[3]);
        printf("Source position: %d %d %d %d\n",
                source[0],source[1],source[2],source[3]);
        printf("Number of point correlators: %d\n\n",data.npcorr);
        fflush(stdout);
    }

    /* Set up parallel out */
    setup_parallel_out(&data);

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank==0)
    {
        printf("setup_parallel_out() completed\n\n");
        fflush(stdout);
    }
    
    if (my_rank==0)
    {
        /* Check for old test files */
        error_root((fopen(test_file,"rb")!=NULL)||(fopen(cmp_file,"rb")!=NULL),
                    1,"main [parallel_out_test.c]",
                    "Attempt to overwrite old data files");

        ftest=fopen(test_file,"wb");
        error_root(ftest==NULL,1,"main [parallel_out_test.c]",
                    "Unable to open test file");

        /* Write source coordinates to file */
        iw=0;
        if (endian==BIG_ENDIAN)
        {
            bswap_int(4,source);
        }
        iw+=fwrite(source,sizeof(int),4,ftest);
        error_root(iw!=4,1,"main [parallel_out_test.c]",
                    "Incorrect write count");
        fclose(ftest);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank==0)
    {
        printf("Writing data to parallel_write_test.dat in parallel...\n");
        fflush(stdout);
    }
    /* Use parallel_write() to write the data to parallel_write_test.dat */
    err_count=MPI_Type_size(MPI_INT,&int_size);
    error(err_count!=MPI_SUCCESS,1,"write_data [parallel_out.c]",
            "Failed to get size of MPI_INT data type");
    skip=4*int_size; /* Skip the source coords at beginning of data file */
    parallel_write(test_file,&data,source,skip);

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank==0)
    {
        printf("...completed.\n\n");
        fflush(stdout);
    }

    /* Write serial_write_cmp.dat for comparison */
    if (my_rank==0)
    {
        printf("Writing data to serial_write_cmp.dat in serial...\n");
        fflush(stdout);

        fcmp=fopen(cmp_file,"wb");
        error_root(fcmp==NULL,1,"main [parallel_out_test.c]",
                    "Unable to open comparison file");

        iw=0;
        if (endian==BIG_ENDIAN)
        {
            bswap_int(4,source);
        }
        iw+=fwrite(source,sizeof(int),4,fcmp);
        error_root(iw!=4,1,"main [parallel_out_test.c]",
                    "Incorrect write count");

        iw=0;
        for (ipcorr=0;ipcorr<data.npcorr;ipcorr++)
        {
            for (t=0;t<data.outlat[0];t++)
            {
                for (x=0;x<data.outlat[1];x++)
                {
                    for (y=0;y<data.outlat[2];y++)
                    {
                        for (z=0;z<data.outlat[3];z++)
                        {
                            index=ipcorr*N0*N1*N2*N3
                                    +(source[3]+z)%(N3)
                                    +((source[2]+y)%(N2))*N3
                                    +((source[1]+x)%(N1))*N2*N3
                                    +((source[0]+t)%(N0))*N1*N2*N3;
                            data_pt.re=(double)index;
                            data_pt.im=-(double)index;
                            if (endian==BIG_ENDIAN)
                            {
                                bswap_double(2,&data_pt);
                            }
                            iw+=fwrite(&data_pt,sizeof(double),2,fcmp);
                        }
                    }
                }
            }
        }
        error_root(iw!=2*data.npcorr*size,1,"main [parallel_out_test.c]","Incorrect write count");
        fclose(fcmp);

        printf("...completed.\n\n");
        fflush(stdout);

        /* Read both files and compare */
        err=0;
        printf("Reading both files and comparing...\n");
        fflush(stdout);

        ftest=fopen(test_file,"rb");
        error_root(ftest==NULL,1,"main [parallel_out_test.c]",
                    "Unable to open test file");
        fcmp=fopen(cmp_file,"rb");
        error_root(fcmp==NULL,1,"main [parallel_out_test.c]",
                    "Unable to open comparison file");

        cmp_result=0;
        ir_test=fread(buf_test,1,num_bytes,ftest);
        ir_cmp=fread(buf_cmp,1,num_bytes,fcmp);
        if (fread(buf_test,1,1,ftest) > 1) { /* satisfies compiler */ }
        if (fread(buf_cmp,1,1,fcmp) > 1) { /* satisfies compiler */ }

        if ((ir_test==num_bytes)&&(ir_cmp==num_bytes)&&(feof(ftest))&&(feof(fcmp)))
        {
            cmp_result=memcmp(buf_test,buf_cmp,num_bytes);
            if (cmp_result!=0)
                err=1;
        }
        else
        {
            printf("Error: Files have unexpected sizes\n");
            printf("ir_test: %ld, ir_cmp: %ld\n",ir_test,ir_cmp);
            printf("feof(ftest): %d, feof(fcmp): %d\n",feof(ftest),feof(fcmp));
            err=1;
        }

        fclose(ftest);
        fclose(fcmp);

        if ((remove(test_file)!=0)||(remove(cmp_file)!=0))
        {
            error_root(1,1,"main [parallel_out_test.c]",
                        "Unable to remove test files");
        }
        printf("...completed.\n\n\n");
        fflush(stdout);
    }

    free(buf_test);
    free(buf_cmp);

    if (my_rank==0)
    {
        if (err==0)
        {
            printf("PARALLEL_OUT TEST COMPLETED SUCCESSFULLY\n");
        }
        else
        {
            printf("PARALLEL_OUT TEST FAILED\n\n");
            printf("COMPARE RESULT: %d\n", cmp_result);
        }
        fclose(flog);
    }
    
    MPI_Finalize();
    exit(0);
}