/*******************************************************************************
*
* File parallel_out_test.c
*
* Test program for the parallel_write function. The program assigns every point
* on each local lattice its lexicographical index with respect to the global
* lattice. Then, the parallel_write function is called and should produce a data
* file with the lexicographical indices of the points in the correct order.
* The program then reads the data file and checks if the lexicographical indices
* are in the correct order. The program is run for all possible combinations of
* boundary conditions and source positions.
*
* compile:  make parallel_out_test (global and local lattice dimensions are set
*           via the global.h file in ../../openQCD-2.4.1/include/)
* run:  ./parallel_out_test -O <outlat[0]> <outlat[1]> <outlat[2]> <outlat[3]>
*
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
#include "version.h"

#include <time.h> /* time */


corr_data data;
static int my_rank,endian,iw,num_bytes;
static int outlat[4];
static char test_file[]="parallel_write_test.dat", cmp_file[]="serial_write_cmp.dat";
static FILE *flog=NULL,*ftest=NULL,*fcmp=NULL;



void source_pos(int x0, int pos, int bcon, int *src_coords)
{
    int err_count=0;
    double rand[4];

    /* check if outlat has been set */
    error_root((outlat[0]==-1)||(outlat[1]==-1)||(outlat[2]==-1)||(outlat[3]==-1),1,
                "source_pos [parallel_out.c]","Output lattice not set");

    if (my_rank==0)
    {
        ranlxd(rand,4);

        if (pos==0)
        {
            if (bcon==0)
            {
                if (x0<0)
                    src_coords[0]=(N0-outlat[0])/2;
                else
                    src_coords[0]=x0;
            }
            else if (bcon==3)
            {
                src_coords[0]=(int)(rand[0]*N0);
            }
            else
            {
                error_root(1,1,"source_pos [parallel_out.c]",
                            "Unknown or unsupported boundary condition");
            }
        }
        else
        {
            if (bcon==0)
            {
                if (x0<0)
                    src_coords[0]=(N0-1)/2;
                else
                    src_coords[0]=x0;
            }
            else if (bcon==3)
            {
                src_coords[0]=(int)(rand[0]*N0);
            }
            else
            {
                error_root(1,1,"source_pos [parallel_out.c]",
                            "Unknown or unsupported boundary condition");
            }
        }
        src_coords[1]=(int)(rand[1]*N1);
        src_coords[2]=(int)(rand[2]*N2);
        src_coords[3]=(int)(rand[3]*N3);
    }
    err_count=MPI_Bcast(src_coords,4,MPI_INT,0,MPI_COMM_WORLD);
    error(err_count!=MPI_SUCCESS,1,"source_pos [parallel_out.c]",
            "Failed to broadcast source coordinates");
}



int main(int argc, char *argv[])
{
    int i,j,t,x,y,z,seed=-1,err,err_sum,cmp_result;
    int index,loc_index,pos,bc,start_coords[4],src_coords[4];
    char *buf_test,*buf_cmp;
    size_t ir_test,ir_cmp;
    complex_dble data_pt;

    mpi_init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    create_MPI_COMPLEX_DOUBLE();

    /* Parse command line arguments on rank 0 */
    if (my_rank==0)
    {
        /* Open log file */
        flog=freopen("parallel_out_test.log","w", stdout);
        error_root(flog==NULL,1,"main [parallel_out_test.c]",
                "Unable to open log file");

        for (i=1;i<argc;i++)
        {
            if (argv[i][0]=='-')
            {
                if (argv[i][1]=='O')
                {
                    error_root(i+4>argc,1,"main [parallel_out_test.c]",
                                "Too few arguments were given. O flag requires four positive integer arguments.");
                    for (j=0;j<4;j++)
                    {
                        i++;
                        outlat[j]=atoi(argv[i]);
                        error_root(outlat[j]==0,1,"main [parallel_out_test.c]",
                                "O flag requires four positive integer arguments");
                    }
                }
                else if (argv[i][1]=='s')
                {
                    error_root(i+1>argc,1,"main [parallel_out_test.c]",
                                "Too few arguments were given. s flag requires one positive integer argument.");
                    i++;
                    seed=atoi(argv[i]);
                    error_root(seed<1,1,"main [parallel_out_test.c]",
                                "Seed must be a positive integer");
                }
                else
                {
                    error_root(1,1,"main [parallel_out_test.c]",
                                "Unknown flag");
                }
            }
        }

        /* Check that lattice sizes are positive integers */
        if ((NPROC0*L0<1)||(NPROC1*L1<1)||(NPROC2*L2<1)||(NPROC3*L3<1)
            ||(L0<1)||(L1<1)||(L2<1)||(L3<1)
            ||(outlat[0]<1)||(outlat[1]<1)||(outlat[2]<1)||(outlat[3]<1))
        {
            error_root(1,1,"main [gather_data_test.c]",
                    "Lattice sizes must be positive integers");
        }
        /* Check that output lattice size is less than or equal to global lattice size */
        if ((outlat[0])>(NPROC0*L0)||(outlat[1])>(NPROC1*L1)
            ||(outlat[2])>(NPROC2*L2)||(outlat[3])>(NPROC3*L3))
        {
            error_root(1,1,"main [gather_data_test.c]",
                    "Output lattice size must be less than or equal to global lattice size");
        }
    }
    MPI_Bcast(outlat,4,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);

    /* Allocate data */
    data.size=outlat[0]*outlat[1]*outlat[2]*outlat[3];
    data.corr=malloc(VOLUME*sizeof(complex_dble));
    data.corr_out=malloc(VOLUME*sizeof(complex_dble));
    error((data.corr==NULL)||(data.corr_out==NULL),1,"main [gather_data_test.c]",
            "Unable to allocate data arrays");

    /* Allocate buffer */
    num_bytes=4*sizeof(int)+2*data.size*sizeof(double);
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

    /* Generate seed if not set via commandline */
    if (seed==-1)
    {
        srand(time(NULL));
        seed=rand()%10001;
    }

    start_ranlux(0,seed);
    
    /* Write local lattice on each process */
    for (t=0;t<L0;t++)
    {
        for (x=0;x<L1;x++)
        {
            for (y=0;y<L2;y++)
            {
                for (z=0;z<L3;z++)
                {
                    loc_index=z+y*L3+x*L2*L3+t*L1*L2*L3;
                    index=cpr[3]*L3+z
                            +(cpr[2]*L2+y)*NPROC3*L3
                            +(cpr[1]*L1+x)*NPROC2*L2*NPROC3*L3
                            +(cpr[0]*L0+t)*NPROC1*L1*NPROC2*L2*NPROC3*L3;
                    data.corr[loc_index].re=(double)index;
                    data.corr[loc_index].im=0.0;
                }
            }
        }
    }

    /* Write test settings to log file */
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank==0)
    {
        printf("PARALLEL_OUT TEST\n");
        printf("Global lattice: %d %d %d %d\n",
                NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
        printf("Local lattice: %d %d %d %d\n",L0,L1,L2,L3);
        printf("Output lattice: %d %d %d %d\n",
                outlat[0],outlat[1],outlat[2],outlat[3]);
        printf("Random seed: %d\n\n\n",seed);
    }

    /* Run test for all possible combinations of boundary conditions and source positions */
    err_sum=0;
    for (bc=0;bc<2;bc++)
    {
        if (bc==1)
            bc=3;

        for (pos=0;pos<2;pos++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (my_rank==0)
            {
                printf("TESTING\n");
                printf("Boundary condition: %d\n",bc);
                printf("Source positioning: %d\n",pos);
            }

            /* Set up parallel out */
            set_up_parallel_out(outlat, 1, pos, bc);

            /* Set source position */
            source_pos(-1, pos, bc, src_coords);
            if (my_rank==0)
            {
                printf("Source position: %d %d %d %d\n\n",
                        src_coords[0],src_coords[1],src_coords[2],src_coords[3]);
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
                    bswap_int(4,src_coords);
                }
                iw+=fwrite(src_coords,sizeof(int),4,ftest);
                error_root(iw!=4,1,"main [parallel_out_test.c]",
                            "Incorrect write count");
                fclose(ftest);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if (my_rank==0)
            {
                printf("writing data to parallel_write_test.dat in parallel...\n");
                fflush(stdout);
            }
            /* Use parallel_write() to write the data to parallel_write_test.dat */
            parallel_write(test_file,&data,src_coords);

            /* Write serial_write_cmp.dat for comparison */
            if (my_rank==0)
            {
                printf("writing data to serial_write_cmp.dat in serial...\n");
                fflush(stdout);

                fcmp=fopen(cmp_file,"wb");
                error_root(fcmp==NULL,1,"main [parallel_out_test.c]",
                            "Unable to open comparison file");

                if (pos==0)
                {
                    for (i=0;i<4;i++)
                        start_coords[i]=src_coords[i];
                }
                else
                {
                    start_coords[0]=safe_mod(src_coords[0]-(outlat[0]-1)/2,NPROC0*L0);
                    start_coords[1]=safe_mod(src_coords[1]-(outlat[1]-1)/2,NPROC1*L1);
                    start_coords[2]=safe_mod(src_coords[2]-(outlat[2]-1)/2,NPROC2*L2);
                    start_coords[3]=safe_mod(src_coords[3]-(outlat[3]-1)/2,NPROC3*L3);
                }

                iw=0;
                if (endian==BIG_ENDIAN)
                {
                    bswap_int(4,src_coords);
                }
                iw+=fwrite(src_coords,sizeof(int),4,fcmp);
                error_root(iw!=4,1,"main [parallel_out_test.c]",
                            "Incorrect write count");

                iw=0;
                data_pt.im=0.0;
                for (t=0;t<outlat[0];t++)
                {
                    for (x=0;x<outlat[1];x++)
                    {
                        for (y=0;y<outlat[2];y++)
                        {
                            for (z=0;z<outlat[3];z++)
                            {
                                index=(start_coords[3]+z)%(NPROC3*L3)
                                        +((start_coords[2]+y)%(NPROC2*L2))*NPROC3*L3
                                        +((start_coords[1]+x)%(NPROC1*L1))*NPROC2*L2*NPROC3*L3
                                        +((start_coords[0]+t)%(NPROC0*L0))*NPROC1*L1*NPROC2*L2*NPROC3*L3;
                                data_pt.re=(double)index;
                                if (endian==BIG_ENDIAN)
                                {
                                    bswap_double(2,&data_pt);
                                }
                                iw+=fwrite(&data_pt,sizeof(double),2,fcmp);
                            }
                        }
                    }
                }
                error_root(iw!=2*data.size,1,"main [parallel_out_test.c]","Incorrect write count");
                fclose(fcmp);

                /* Read both files and compare */
                err=0;
                printf("reading both files and comparing...\n");
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
                fread(buf_test,1,1,ftest);
                fread(buf_cmp,1,1,fcmp);

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

                if (err==0)
                {
                    printf("TEST PASSED\n\n\n");
                    fflush(stdout);
                }
                else
                {
                    printf("TEST FAILED\n\n\n");
                    fflush(stdout);
                    err_sum++;
                }
            }
        }
    }

    free(buf_test);
    free(buf_cmp);

    if (my_rank==0)
    {
        if (err_sum==0)
        {
            printf("PARALLEL_OUT TEST COMPLETED SUCCESSFULLY\n");
        }
        else
        {
            printf("PARALLEL_OUT TEST FAILED (%d/4 passed)\n",4-err_sum);
        }
        fclose(flog);
    }
    
    free_MPI_COMPLEX_DOUBLE();
    MPI_Finalize();
    exit(0);
}