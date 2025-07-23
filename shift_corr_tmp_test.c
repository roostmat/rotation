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
 * Test for shift_corr_tmp()
 * 
 * This program tests the shift_corr_tmp function by setting up a data structure
 * with data.corr_tmp and a comparison array corr_shifted that is shifted gloally
 * by a specified vector. It then shifts data.corr_tmp using shift_corr_tmp and
 * compares the result with corr_shifted to ensure correctness.
 * 
 * COMPILATION:
 *   make shift_corr_tmp_test
 * 
 * USAGE:
 *   export OMP_NUM_THREADS=1
 *   mpirun -np <num_processes> ./shift_corr_tmp_test [OPTIONS]
 * 
 * OPTIONS:
 *   -vec <t> <x> <y> <z>    Shift vector (default: 0 0 0 0)
 *   -npcorr <n>             Number of point correlators (default: 1)
 * 
 *******************************************************************************/

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



static corr_data_t data;


static int my_rank,lattice_setup=0;
static complex_dble *corr_shifted;
static int shift_vec[4]={0,0,0,0};



static int global_index(int *coords)
{
    return coords[0]*N1*N2*N3+coords[1]*N2*N3+coords[2]*N3+coords[3];
}



static void setup_lattices(void)
{
    int ipcorr,t,x,y,z,n,i,index,shifted_index;
    int global_coords[DIM],shifted_global_coords[DIM];

    data.corr_tmp=malloc(data.npcorr*VOLUME*sizeof(complex_dble));
    error(data.corr_tmp==NULL,1,"setup_lattice [shift.c]",
            "Unable to allocate corr array");
    corr_shifted=malloc(data.npcorr*VOLUME*sizeof(complex_dble));
    error(corr_shifted==NULL,1,"setup_lattice [shift.c]",
            "Unable to allocate corr_shifted array");

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

                        for (n=0;n<data.npcorr;n++)
                        {
                            data.corr_tmp[n*VOLUME+i].re=index;
                            data.corr_tmp[n*VOLUME+i].im=index;
                            corr_shifted[n*VOLUME+i].re=shifted_index;
                            corr_shifted[n*VOLUME+i].im=shifted_index;
                        }
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

    for (i=0;i<data.npcorr*VOLUME;i++)
    {
        if ((data.corr_tmp[i].re!=corr_shifted[i].re)||(data.corr_tmp[i].im!=corr_shifted[i].im))
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
    int i,j;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    /* Read command line input on rank 0 */
    if (my_rank==0)
    {
        for (i=1;i<argc;i++)
        {
            if (argv[i][0]=='-')
            {
                if (strcmp(argv[i],"-vec")==0)
                {
                    error_root(i+4>argc,1,"main [shift_corr_test.c]",
                                "Too few arguments were given. vec flag requires four integer arguments.");
                    for (j=0;j<4;j++)
                    {
                        i++;
                        shift_vec[j]=atoi(argv[i]);
                    }
                }
                else if (strcmp(argv[i],"-npcorr")==0)
                {
                    error_root(i+1>argc,1,"main [shift_corr_test.c]",
                                "Too few arguments were given. npcorr flag requires one positive integer argument.");
                    i++;
                    data.npcorr=atoi(argv[i]);
                    error_root(data.npcorr<=0,1,"main [shift_corr_test.c]",
                                "npcorr flag requires one positive integer argument");
                }
                else
                {
                    error_root(1,1,"main [shift_corr_test.c]",
                                "Unknown command line option: %s", argv[i]);
                }
            }
        }
    }
    MPI_Bcast(shift_vec,4,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&data.npcorr,1,MPI_INT,0,MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank==0)
    {
        printf("SHIFT_CORR_TMP TEST\n\n");
        printf("Global lattice: %d %d %d %d\n",
                N0,N1,N2,N3);
        printf("Local lattice: %d %d %d %d\n",L0,L1,L2,L3);
        printf("Shift vector: (%d, %d, %d, %d)\n",shift_vec[0],shift_vec[1],shift_vec[2],shift_vec[3]);
        printf("Number of point correlators: %d\n\n",data.npcorr);
        fflush(stdout);
    }

    /* Set up lattice geometry */
    geometry();
    setup_lattices();

    print_corr_slice(data.corr_tmp,0,0,1,"Original Lattice");
    print_corr_slice(corr_shifted,0,0,1,"Control Lattice");

    shift_corr_tmp(&data,shift_vec);
    shift_corr_tmp(&data,shift_vec);
    shift_vec[0]=-shift_vec[0];
    shift_vec[1]=-shift_vec[1];
    shift_vec[2]=-shift_vec[2];
    shift_vec[3]=-shift_vec[3];
    shift_corr_tmp(&data,shift_vec);
    if (my_rank==0)
    {
        printf("Correlation data shifted.\n");
        fflush(stdout);
    }

    print_corr_slice(data.corr_tmp,0,0,1,"Shifted Lattice");

    cleanup_shift();
    if (my_rank==0)
    {
        printf("Shift cleanup done.\n\n");
        fflush(stdout);
    }

    compare_lattices();
    free(data.corr_tmp);
    free(corr_shifted);
    MPI_Finalize();
    return 0;
}