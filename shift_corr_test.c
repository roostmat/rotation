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


#include "su3.h"
#include "random.h"
#include "flags.h"
#include "archive.h"
#include "sflds.h"
#include "linalg.h"
#include "dirac.h"
#include "sap.h"
#include "dfl.h"
#include "forces.h"
#include "version.h"
#include "uflds.h"
#include "mesons.h"



#define DIM 4



static int my_rank;
static complex_dble *corr,*corr_shifted;
static int shift_vec[4]={0,0,0,0},npcorr=-1;



int global_index(int *coords)
{
    return coords[0]*N1*N2*N3 + coords[1]*N2*N3 + coords[2]*N3 + coords[3];
}



void setup_lattices(void)
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
                    i=z+ y*L3 + x*L2*L3 + t*L1*L2*L3;
                    
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
}



void compare_lattices(void)
{
    int i,errors=0,total_errors=0;

    for (i=0;i<npcorr*VOLUME;i++)
    {
        if ((corr[i].re!=corr_shifted[i].re) || (corr[i].im!=corr_shifted[i].im))
        {
            errors++;
        }
    }

    MPI_Allreduce(&errors,&total_errors,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

    if (my_rank==0)
    {
        if (total_errors>0)
        {
            printf("SHIFT FAILED:Total errors: %d\n", total_errors);
        }
        else
        {
            printf("SHIFT SUCCESSFUL!\n");
        }
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
    MPI_Bcast(shift_vec, 4, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&npcorr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (npcorr<1)
    {
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank==0)
    {
        printf("Shift vector: (%d, %d, %d, %d)\n", shift_vec[0], shift_vec[1], shift_vec[2], shift_vec[3]);
        printf("Number of correlators: %d\n\n", npcorr);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /* Set up lattice geometry */
    geometry();
    if (my_rank==0)
    {
        printf("Geometry set up.\n");
        fflush(stdout);
    }

    setup_lattices();
    if (my_rank==0)
    {
        printf("Lattices set up.\n");
        fflush(stdout);
    }

    shift_corr(corr, shift_vec);
    if (my_rank==0)
    {
        printf("Correlation functions shifted.\n");
        fflush(stdout);
    }

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