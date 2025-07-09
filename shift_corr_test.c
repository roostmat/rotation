/**************************************************************************************
 *  shift_corr_test.c
 * 
 * Test program for the shift_corr module.
 * 
 * To run a test, compile this file using "make shift_corr_test"
 * and run it with the command line arguments:
 * -v v0 v1 v2 v3 [-npcorr n]
 * where v0, v1, v2, v3 are the shift values in each direction
 * and n is the number of correlators to test (default is 1).
 * 
 **************************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "endian.h" /* for endianness */
#include "global.h" /* for NPROC0, NPROC1, NPROC2, NPROC3, L0, L1, L2, L3, VOLUME, ipt[] */
#include "lattice.h" /* for ipr_global */
#include "mpi.h"
#include "rotation.h"
#include "utils.h" /* for error, error_root */



#define DIM 4



static int my_rank,lattice_setup=0;
static int dummy_outlat[4]={0,0,0,0}; /* Dummy output lattice for compatibility */
static complex_dble *corr,*corr_shifted;
static int shift_vec[4]={0,0,0,0},npcorr=-1;



static int global_index(int *coords)
{
    return coords[0]*N1*N2*N3+coords[1]*N2*N3+coords[2]*N3+coords[3];
}



static void setup_lattices(void)
{
    int t,x,y,z,n,i,index,shifted_index;
    int global_coords[DIM],shifted_global_coords[DIM];

    corr=malloc(npcorr*VOLUME*sizeof(complex_dble));
    error(corr==NULL,1,"setup_lattice [shift.c]",
            "Unable to allocate corr array");
    corr_shifted=malloc(npcorr*VOLUME*sizeof(complex_dble));
    error(corr_shifted==NULL,1,"setup_lattice [shift.c]",
            "Unable to allocate corr_shifted array");

    for (t=0;t<L0;t++)
    {
        for (x=0;x<L1;x++)
        {
            for (y=0;y<L2;y++)
            {
                for (z=0;z<L3;z++)
                {
                    i=z+y*L3+x*L2*L3+t*L1*L2*L3;
                    
                    global_coords[0]=cpr[0]*L0+t;
                    global_coords[1]=cpr[1]*L1+x;
                    global_coords[2]=cpr[2]*L2+y;
                    global_coords[3]=cpr[3]*L3+z;

                    shifted_global_coords[0]=safe_mod(global_coords[0]-shift_vec[0],N0);
                    shifted_global_coords[1]=safe_mod(global_coords[1]-shift_vec[1],N1);
                    shifted_global_coords[2]=safe_mod(global_coords[2]-shift_vec[2],N2);
                    shifted_global_coords[3]=safe_mod(global_coords[3]-shift_vec[3],N3);

                    index=global_index(global_coords);
                    shifted_index=global_index(shifted_global_coords);

                    for (n=0;n<npcorr;n++)
                    {
                        corr[n*VOLUME+i].re=index;
                        corr[n*VOLUME+i].im=index;
                        corr_shifted[n*VOLUME+i].re=shifted_index;
                        corr_shifted[n*VOLUME+i].im=shifted_index;
                    }
                }
            }
        }
    }
    lattice_setup=1;
}



static void compare_lattices(void)
{
    int i,errors=0,total_errors=0;

    for (i=0;i<npcorr*VOLUME;i++)
    {
        if ((corr[i].re!=corr_shifted[i].re)||(corr[i].im!=corr_shifted[i].im))
        {
            errors++;
        }
    }

    MPI_Allreduce(&errors,&total_errors,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

    if (my_rank==0)
    {
        if (total_errors>0)
        {
            printf("SHIFT FAILED:Total errors: %d\n",total_errors);
        }
        else
        {
            printf("SHIFT SUCCESSFUL!\n");
        }
    }
}



static void print_corr_slice(complex_dble *corr_array,int y_slice,int z_slice,int correlator,const char *array_name)
{
    int t,x,i;
    int global_coords[DIM];
    int rank,local_index;
    double row[N0];

    error(!lattice_setup,1,"print_corr_slice [shift.c]",
          "Lattice not initialized. Call setup_lattices() first.");

    if (my_rank==0)
    {
        printf("\n%s[correlator=%d, y=%d, z=%d]:\n",array_name,correlator,y_slice,z_slice);

        /* Print header with t-coordinates */
        printf("   t: ");
        for (t=0;t<N0;t++)
        {
            printf("%6d ",t);
        }
        printf("\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Print each x-row (reverse order for visual clarity) */
    for (x=N1-1;x>=0;x--)
    {
        for (i=0;i<N0;i++)
        {
            row[i]=0.0; /* Initialize row */
        }
        
        for (t=0;t<N0;t++)
        {
            /* Set global coordinates */
            global_coords[0]=t;
            global_coords[1]=x;
            global_coords[2]=y_slice;
            global_coords[3]=z_slice;

            /* Calculate which process this coordinate belongs to */
            /* and the local index */
            lex_global(global_coords,&rank,&local_index);
            
            /* print the real value */
            if (my_rank==rank)
            {
                row[t]=corr_array[(correlator-1)*VOLUME+local_index].re;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE,row,N0,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        
        if (my_rank==0)
        {
            printf("x=%2d: ",x);
            for (i=0;i<N0;i++)
            {
                printf("%6.0f ",row[i]);
            }
            printf("\n");
            fflush(stdout);
        }
    }
    if (my_rank==0)
    {
        printf("\n");
        fflush(stdout);
    }
}



int main(int argc,char *argv[])
{
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    /* Read command line input on rank 0 */
    if (my_rank==0)
    {
        if (argc<6)
        {
            if (my_rank==0)
            {
                printf("Usage: %s -v v0 v1 v2 v3 [-npcorr n]\n", argv[0]);
                fflush(stdout);
            }
        }
        else
        {
            shift_vec[0]=atoi(argv[2]);
            shift_vec[1]=atoi(argv[3]);
            shift_vec[2]=atoi(argv[4]);
            shift_vec[3]=atoi(argv[5]);

            if (argc==8 && strcmp(argv[6],"-npcorr")==0)
            {
                npcorr=atoi(argv[7]);
            }
            else
            {
                npcorr=1; /* Default value */
            }
        }
    }
    MPI_Bcast(shift_vec,4,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&npcorr,1,MPI_INT,0,MPI_COMM_WORLD);
    if (npcorr<1)
    {
        MPI_Finalize();
        return 1;
    }
    set_corr_data_parms(dummy_outlat,npcorr);

    if (my_rank==0)
    {
        printf("Shift vector: (%d, %d, %d, %d)\n",shift_vec[0],shift_vec[1],shift_vec[2],shift_vec[3]);
        printf("Number of correlators: %d\n\n",npcorr);
        fflush(stdout);
    }

    /* Set up lattice geometry */
    geometry();
    setup_lattices();

    print_corr_slice(corr,0,0,1,"Original Lattice");
    print_corr_slice(corr_shifted,0,0,1,"Control Lattice");

    shift_corr(corr,shift_vec);
    shift_corr(corr,shift_vec);
    shift_vec[0]=-shift_vec[0];
    shift_vec[1]=-shift_vec[1];
    shift_vec[2]=-shift_vec[2];
    shift_vec[3]=-shift_vec[3];
    shift_corr(corr,shift_vec);
    if (my_rank==0)
    {
        printf("Correlation functions shifted.\n");
        fflush(stdout);
    }

    print_corr_slice(corr,0,0,1,"Shifted Lattice");

    cleanup_shift();
    if (my_rank==0)
    {
        printf("Shift cleanup done.\n");
        fflush(stdout);
    }

    compare_lattices();
    free(corr);
    free(corr_shifted);
    MPI_Finalize();
    return 0;
}