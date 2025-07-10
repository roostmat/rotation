# define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "endian.h" /* for endianness */
#include "global.h" /* for NPROC0, NPROC1, NPROC2, NPROC3, L0, L1, L2, L3, VOLUME, ipt[] */
#include "lattice.h" /* for ipr_global */
#include "mpi.h"
#include "random.h"
#include "rotation.h"
#include "utils.h" /* for error, error_root */



static int seed,my_rank,lattice_setup=0;
static int bc,npcorr,fv=N0*N1*N2*N3;
static int dummy_outlat[4]={N0,N1,N2,N3}; /* Dummy output lattice for compatibility */
static complex_dble *global_corr,*averaged_global_corr;
static complex_dble *corr,*corr_cmp;


/*
static void setup_lattices(void)
{
    int i;
    double rand;

    Allocate memory for global and averaged global correlation arrays
    global_corr=malloc(npcorr*fv*sizeof(complex_dble));
    error(global_corr==NULL,1,"setup_lattices [average_test.c]",
            "Unable to allocate global_corr array");
    averaged_global_corr=malloc(npcorr*fv*sizeof(complex_dble));
    error(averaged_global_corr==NULL,1,"setup_lattices [average_test.c]",
            "Unable to allocate averaged_global_corr array");

    Allocate memory for local correlation arrays
    corr=malloc(npcorr*VOLUME*sizeof(complex_dble));
    error(corr==NULL,1,"setup_lattices [average_test.c]",
            "Unable to allocate corr array");
    corr_cmp=malloc(npcorr*VOLUME*sizeof(complex_dble));
    error(corr_cmp==NULL,1,"setup_lattices [average_test.c]",
            "Unable to allocate corr_cmp array");

    Populate global correlation array with random values
    for (i=0;i<fv;i++)
    {
        ranlxd(&rand,1);
        rand=floor(rand*100);
        global_corr[i].re=rand;
        global_corr[i].im=rand;
        averaged_global_corr[i].re=rand;
        averaged_global_corr[i].im=rand;
    }
    lattice_setup=1;
}*/

static void setup_lattices_simple(void)
{
    int i,partner_index,coords[4];

    /* Allocate memory for global and averaged global correlation arrays */
    global_corr=malloc(npcorr*fv*sizeof(complex_dble));
    error(global_corr==NULL,1,"setup_lattices [average_test.c]",
            "Unable to allocate global_corr array");
    averaged_global_corr=malloc(npcorr*fv*sizeof(complex_dble));
    error(averaged_global_corr==NULL,1,"setup_lattices [average_test.c]",
            "Unable to allocate averaged_global_corr array");

    /* Allocate memory for local correlation arrays */
    corr=malloc(npcorr*VOLUME*sizeof(complex_dble));
    error(corr==NULL,1,"setup_lattices [average_test.c]",
            "Unable to allocate corr array");
    corr_cmp=malloc(npcorr*VOLUME*sizeof(complex_dble));
    error(corr_cmp==NULL,1,"setup_lattices [average_test.c]",
            "Unable to allocate corr_cmp array");

    for (i=0;i<fv;i++)
    {
        if (bc==0)
        {
            coords[0]=i/(N1*N2*N3);
        }
        else if (bc==3)
        {
            coords[0]=safe_mod(-(i/(N1*N2*N3)),N0);
        }
        else
        {
            error_root(1,1,"setup_lattices [average_test.c]",
                    "Unsupported boundary condition %d", bc);
        }
        coords[1]=safe_mod(-((i/(N2*N3))%N1),N1);
        coords[2]=safe_mod(-((i/N3)%N2),N2);
        coords[3]=safe_mod(-(i%N3),N3);
        partner_index=coords[0]*N1*N2*N3+coords[1]*N2*N3
                            +coords[2]*N3+coords[3];

        if (i<partner_index)
        {
            global_corr[i].re=1.0;
            global_corr[i].im=1.0;
            averaged_global_corr[i].re=1.0;
            averaged_global_corr[i].im=1.0;
        }
        else if (i>partner_index)
        {
            global_corr[i].re=3.0;
            global_corr[i].im=3.0;
            averaged_global_corr[i].re=3.0;
            averaged_global_corr[i].im=3.0;
        }
        else
        {
            /* i == partner_index */
            global_corr[i].re=2.0;
            global_corr[i].im=2.0;
            averaged_global_corr[i].re=2.0;
            averaged_global_corr[i].im=2.0;
        }
    }
    lattice_setup=1;
}

static void global_average(int _outlat[4])
{
    int i,ipcorr,coords1[4],coords2[4],partner_index;

    for (i=0;i<fv;i++)
    {
        coords1[0]=i/(N1*N2*N3);
        coords1[1]=(i/(N2*N3))%N1;
        coords1[2]=(i/N3)%N2;
        coords1[3]=i%N3;

        if (bc==0)
        {
            coords2[0]=coords1[0];
        }
        else if (bc==3)
        {
            coords2[0]=(N0-coords1[0])%N0;
        }
        else
        {
            error_root(1,1,"average_equiv [shift_corr.c]",
                    "Unsupported boundary condition %d",bc);
        }
        coords2[1]=(N1-coords1[1])%N1;
        coords2[2]=(N2-coords1[2])%N2;
        coords2[3]=(N3-coords1[3])%N3;

        partner_index=coords2[0]*N1*N2*N3+coords2[1]*N2*N3+coords2[2]*N3+coords2[3];

        if ((i<partner_index)&&
            (((coords1[0]<_outlat[0])&&(coords1[1]<_outlat[1])&&
            (coords1[2]<_outlat[2])&&(coords1[3]<_outlat[3]))||
            ((coords2[0]<_outlat[0])&&(coords2[1]<_outlat[1])&&
            (coords2[2]<_outlat[2])&&(coords2[3]<_outlat[3]))))
        {
            for (ipcorr=0;ipcorr<npcorr;ipcorr++)
            {
                averaged_global_corr[ipcorr*fv+i].re=0.5*(global_corr[ipcorr*fv+i].re+global_corr[ipcorr*fv+partner_index].re);
                averaged_global_corr[ipcorr*fv+i].im=0.5*(global_corr[ipcorr*fv+i].im+global_corr[ipcorr*fv+partner_index].im);
                averaged_global_corr[ipcorr*fv+partner_index].re=averaged_global_corr[ipcorr*fv+i].re;
                averaged_global_corr[ipcorr*fv+partner_index].im=averaged_global_corr[ipcorr*fv+i].im;
            }
        }
    }
}

static void fill_local_lattice(complex_dble *_local_corr,complex_dble *_global_corr)
{
    int i,ipcorr,global_coords[4],global_index;
    for (i=0;i<VOLUME;i++)
    {
        global_coords[0]=cpr[0]*L0+i/(L1*L2*L3);
        global_coords[1]=cpr[1]*L1+(i/(L2*L3))%L1;
        global_coords[2]=cpr[2]*L2+(i/L3)%L2;
        global_coords[3]=cpr[3]*L3+i%L3;

        global_index=global_coords[0]*N1*N2*N3+global_coords[1]*N2*N3
                            +global_coords[2]*N3+global_coords[3];

        for (ipcorr=0;ipcorr<npcorr;ipcorr++)
        {
            _local_corr[ipcorr*VOLUME+i].re=_global_corr[ipcorr*fv+global_index].re;
            _local_corr[ipcorr*VOLUME+i].im=_global_corr[ipcorr*fv+global_index].im;
        }
    }
}

static void compare_local_lattices(void)
{
    int i,errors=0,total_errors=0;

    for (i=0;i<npcorr*VOLUME;i++)
    {
        if ((corr[i].re!=corr_cmp[i].re)||(corr[i].im!=corr_cmp[i].im))
        {
            errors++;
        }
    }

    MPI_Allreduce(&errors,&total_errors,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

    if (my_rank==0)
    {
        if (total_errors>0)
        {
            printf("AVERAGE FAILED:Total errors: %d\n",total_errors);
        }
        else
        {
            printf("AVERAGE SUCCESSFUL!\n");
        }
    }
}

static void print_corr_slice(complex_dble *corr_array,int y_slice,int z_slice,int correlator,const char *array_name)
{
    int t,x,i;
    int global_coords[4];
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
    double start_time,end_time,elapsed_time,min_time,max_time,avg_time;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    if (my_rank==0)
    {
        if (argc!=4)
        {
            printf("Usage: %s <seed> <npcorr> <bc>\n",argv[0]);
            printf("  seed: Non-negative integer seed for random number generation.\n");
            printf("  npcorr: Number of correlators (must be at least 1).\n");
            printf("  bc: Boundary condition (0 for periodic, 3 for anti-periodic).\n");
            error_root(1,1,"main [average_test.c]",
                    "Incorrect number of arguments. Expected 3, got %d.",argc-1);
        }
        else
        {
            seed=atoi(argv[1]);
            error_root(seed<0,1,"main [average_test.c]",
                    "Seed must be a non-negative integer. Given: %d",seed);
            npcorr=atoi(argv[2]);
            error_root(npcorr<1,1,"main [average_test.c]",
                    "Number of correlators must be at least 1. Given: %d",npcorr);
            bc=atoi(argv[3]);
            error_root((bc!=0)&&(bc!=3),1,"main [average_test.c]",
                    "Boundary condition must be 0 or 3. Given: %d",bc);
        }
    }
    MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&npcorr,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);

    set_corr_data_parms(dummy_outlat,npcorr);

    geometry();
    start_ranlux(1,seed);
    /* Set up lattice geometry */
    if (my_rank==0)
    {
        printf("Setting up the lattices with\n");
        printf("Process grid: %d x %d x %d x %d\n",NPROC0,NPROC1,NPROC2,NPROC3);
        printf("Lattice dimensions: %d x %d x %d x %d\n",L0,L1,L2,L3);
        printf("Total volume: %d\n",fv);
        printf("Boundary condition: %d\n",bc);
        printf("Number of correlators: %d\n",npcorr);
        printf("Seed=%d\n",seed);
        fflush(stdout);
    }
    setup_lattices_simple();
    global_average(dummy_outlat);

    fill_local_lattice(corr,global_corr);
    fill_local_lattice(corr_cmp,averaged_global_corr);

    print_corr_slice(corr,0,0,1,"Original Correlation Array");
    print_corr_slice(corr_cmp,0,0,1,"Comparison Correlation Array");

    if (my_rank==0)
    {
        printf("Performing the averaging operation...\n");
        fflush(stdout);
    }

    /* Start timing */
    start_time=MPI_Wtime();

    average_equiv(corr,dummy_outlat,bc);

    end_time=MPI_Wtime();
    elapsed_time=end_time-start_time;

    MPI_Reduce(&elapsed_time,&min_time,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&elapsed_time,&max_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&elapsed_time,&avg_time,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    avg_time/=NPROC0*NPROC1*NPROC2*NPROC3;

    if (my_rank==0)
    {
        printf("...done.\n");
        printf("Timing results for average_equiv():\n");
        printf("  Minimum time: %.6f seconds\n", min_time);
        printf("  Maximum time: %.6f seconds\n", max_time);
        printf("  Average time: %.6f seconds\n", avg_time);
        printf("  Time difference (max-min): %.6f seconds\n", max_time - min_time);
        fflush(stdout);
    }

    print_corr_slice(corr,0,0,1,"Averaged Correlation Array");

    compare_local_lattices();
    MPI_Finalize();
    return 0;
}
