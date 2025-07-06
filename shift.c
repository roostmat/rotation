#define SHIFT

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

#define DIM 4
#define MAX_NEIGHBORS 16



static int my_rank,setup=0,mpi_setup=0;
static int int_size=0,complex_double_size=0;
static complex_dble *corr_copy=NULL;

static int shift[DIM]={0},neighbor[DIM]={0},neighbor_long[MAX_NEIGHBORS]={0};
static int receive_bases[MAX_NEIGHBORS*DIM],send_bases[MAX_NEIGHBORS*DIM],
            base_spans[MAX_NEIGHBORS*DIM],block_volume[MAX_NEIGHBORS];
static MPI_Datatype receive_types[MAX_NEIGHBORS]={MPI_DATATYPE_NULL},
                    send_types[MAX_NEIGHBORS]={MPI_DATATYPE_NULL};
static int *receive_indices[MAX_NEIGHBORS]={NULL},
            *send_indices[MAX_NEIGHBORS]={NULL};



static void setup_shift(void)
{
    int err_count=0;

    if (setup==0)
    {
        err_count=MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
        error(err_count!=MPI_SUCCESS,err_count,"setup_shift [shift.c]",
                "Failed to get rank of current process");

        corr_copy=malloc(npcorr*VOLUME*sizeof(complex_dble));
        error(corr_copy==NULL,1,"setup_shift [shift.c]",
                "Unable to allocate corr_copy array");
        setup=1;
    }
}



static void calculate_send_receive_structure(void)
{
    int t,x,y,z,i;
    int local_shift[DIM],little_vec[DIM];

    local_shift[0]=safe_mod(shift[0],L0); /* local coords of base point (0,0,0,0) */
    local_shift[1]=safe_mod(shift[1],L1); /* of local lattice + shift*/
    local_shift[2]=safe_mod(shift[2],L2);
    local_shift[3]=safe_mod(shift[3],L3);

    little_vec[0]=L0-local_shift[0]; /* span of first send/receive block */
    little_vec[1]=L1-local_shift[1];
    little_vec[2]=L2-local_shift[2];
    little_vec[3]=L3-local_shift[3];

    neighbor[0]=!!local_shift[0]; /* Do we have to include the neighboring processes? */
    neighbor[1]=!!local_shift[1];
    neighbor[2]=!!local_shift[2];
    neighbor[3]=!!local_shift[3];

    for (t=0;t<neighbor[0]+1;t++)
    {
        for (x=0;x<neighbor[1]+1;x++)
        {
            for (y=0;y<neighbor[2]+1;y++)
            {
                for (z=0;z<neighbor[3]+1;z++)
                {
                    i=(z+y*2+x*4+t*8);

                    neighbor_long[i]=1;

                    send_bases[i*DIM]=(t==0)?0:little_vec[0];
                    send_bases[i*DIM+1]=(x==0)?0:little_vec[1];
                    send_bases[i*DIM+2]=(y==0)?0:little_vec[2];
                    send_bases[i*DIM+3]=(z==0)?0:little_vec[3];

                    base_spans[i*DIM]=(t==0)?little_vec[0]:local_shift[0];
                    base_spans[i*DIM+1]=(x==0)?little_vec[1]:local_shift[1];
                    base_spans[i*DIM+2]=(y==0)?little_vec[2]:local_shift[2];
                    base_spans[i*DIM+3]=(z==0)?little_vec[3]:local_shift[3];
                    block_volume[i]=base_spans[i*DIM]*base_spans[i*DIM+1]
                                        *base_spans[i*DIM+2]*base_spans[i*DIM+3];

                    receive_bases[i*DIM]=(t==0)?local_shift[0]:0;
                    receive_bases[i*DIM+1]=(x==0)?local_shift[1]:0;
                    receive_bases[i*DIM+2]=(y==0)?local_shift[2]:0;
                    receive_bases[i*DIM+3]=(z==0)?local_shift[3]:0;
                }
            }
        }
    }
}



static void create_MPI_types(void)
{
    int ts,xs,ys,zs,i,j,n;
    int receive_coords[DIM],send_coords[DIM];
    int receive_i,send_i;
    int err_count=0;

    for (i=0;i<MAX_NEIGHBORS;i++)
    {
        if (neighbor_long[i])
        {
            /* Allocate receive and send index arrays */
            receive_indices[i]=malloc(npcorr*block_volume[i]*sizeof(int));
            send_indices[i]=malloc(npcorr*block_volume[i]*sizeof(int));
            error((receive_indices[i]==NULL)||(send_indices[i]==NULL),1,"create_MPI_types [shift.c]",
                    "Failed to allocate receive or send index arrays");

            /* Calculate receive and send indices */
            for (ts=0;ts<base_spans[i*DIM];ts++)
            {
                for (xs=0;xs<base_spans[i*DIM+1];xs++)
                {
                    for (ys=0;ys<base_spans[i*DIM+2];ys++)
                    {
                        for (zs=0;zs<base_spans[i*DIM+3];zs++)
                        {
                            j=(zs+ys*base_spans[i*DIM+3]
                                +xs*base_spans[i*DIM+2]*base_spans[i*DIM+3]
                                +ts*base_spans[i*DIM+1]*base_spans[i*DIM+2]*base_spans[i*DIM+3]);

                            receive_coords[0]=receive_bases[i*DIM]+ts;
                            receive_coords[1]=receive_bases[i*DIM+1]+xs;
                            receive_coords[2]=receive_bases[i*DIM+2]+ys;
                            receive_coords[3]=receive_bases[i*DIM+3]+zs;

                            send_coords[0]=send_bases[i*DIM]+ts;
                            send_coords[1]=send_bases[i*DIM+1]+xs;
                            send_coords[2]=send_bases[i*DIM+2]+ys;
                            send_coords[3]=send_bases[i*DIM+3]+zs;

                            receive_i=receive_coords[3]+receive_coords[2]*L3
                                        +receive_coords[1]*L2*L3+receive_coords[0]*L1*L2*L3;
                            send_i=send_coords[3]+send_coords[2]*L3
                                    +send_coords[1]*L2*L3+send_coords[0]*L1*L2*L3;

                            for (n=0;n<npcorr;n++)
                            {
                                receive_indices[i][n*block_volume[i]+j]=n*VOLUME+receive_i;
                                send_indices[i][n*block_volume[i]+j]=n*VOLUME+send_i;
                            }
                        }
                    }
                }           
            }

            if ((send_bases[i*DIM]+shift[0]<0)||(send_bases[i*DIM]+shift[0]>=L0)||
                (send_bases[i*DIM+1]+shift[1]<0)||(send_bases[i*DIM+1]+shift[1]>=L1)||
                (send_bases[i*DIM+2]+shift[2]<0)||(send_bases[i*DIM+2]+shift[2]>=L2)||
                (send_bases[i*DIM+3]+shift[3]<0)||(send_bases[i*DIM+3]+shift[3]>=L3))
            {
                /* This data block needs to be send to another process */
                /* Create receive and send types */
                err_count=MPI_Type_indexed_block(npcorr*block_volume[i],1,receive_indices[i],
                                                    MPI_COMPLEX_DOUBLE,&receive_types[i]);
                err_count+=MPI_Type_indexed_block(npcorr*block_volume[i],1,send_indices[i],
                                                    MPI_COMPLEX_DOUBLE,&send_types[i]);
                error(err_count!=MPI_SUCCESS,1,"create_MPI_types [shift.c]",
                        "Failed to create MPI indexed block types");

                /* Commit receive and send types */
                err_count=MPI_Type_commit(&receive_types[i]);
                err_count+=MPI_Type_commit(&send_types[i]);
                error(err_count!=MPI_SUCCESS,1,"create_MPI_types [shift.c]",
                        "Failed to commit MPI indexed block types");
            }
        }
    }
    mpi_setup=1;
}



static void shift_data(complex_dble *corr)
{
    int t,x,y,z,i,j;
    int receive_rank_coords[DIM],send_rank_coords[DIM];
    int receive_rank,send_rank;
    int err_count=0;

    for (i=0;i<MAX_NEIGHBORS;i++)
    {
        if (neighbor_long[i])
        {
            receive_rank_coords[0]=safe_mod(cpr[0]*L0+receive_bases[i*DIM]-shift[0],N0)/L0;
            receive_rank_coords[1]=safe_mod(cpr[1]*L1+receive_bases[i*DIM+1]-shift[1],N1)/L1;
            receive_rank_coords[2]=safe_mod(cpr[2]*L2+receive_bases[i*DIM+2]-shift[2],N2)/L2;
            receive_rank_coords[3]=safe_mod(cpr[3]*L3+receive_bases[i*DIM+3]-shift[3],N3)/L3;
            receive_rank=ipr_global(receive_rank_coords);

            send_rank_coords[0]=safe_mod(cpr[0]*L0+send_bases[i*DIM]+shift[0],N0)/L0;
            send_rank_coords[1]=safe_mod(cpr[1]*L1+send_bases[i*DIM+1]+shift[1],N1)/L1;
            send_rank_coords[2]=safe_mod(cpr[2]*L2+send_bases[i*DIM+2]+shift[2],N2)/L2;
            send_rank_coords[3]=safe_mod(cpr[3]*L3+send_bases[i*DIM+3]+shift[3],N3)/L3;
            send_rank=ipr_global(send_rank_coords);

            if ((my_rank==receive_rank)&&(my_rank==send_rank))
            {
                /* No other process involved. We shift internally. */
                for (j=0;j<npcorr*block_volume[i];j++)
                {
                    corr[receive_indices[i][j]]=corr_copy[send_indices[i][j]];
                }
            }
            else
            {
                err_count=MPI_Sendrecv(corr_copy,npcorr*block_volume[i],send_types[i],
                                        send_rank,0,
                                        corr,npcorr*block_volume[i],receive_types[i],
                                        receive_rank,0,
                                        MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                error(err_count!=MPI_SUCCESS,1,"shift_data [shift.c]",
                        "Process with rank %d failed to send/receive data during shift", my_rank);
            }
        }
    }
}



void shift_corr(complex_dble *corr,int *shift_vec)
{
    int i;

    setup_shift();

    /* Save the current state of the corr array to corr_copy */
    memcpy(corr_copy,corr,npcorr*VOLUME*sizeof(complex_dble));

    /* Check if send/receive structure is already set up */
    if ((shift[0]!=shift_vec[0])||(shift[1]!=shift_vec[1])||
        (shift[2]!=shift_vec[2])||(shift[3]!=shift_vec[3]))
    {
        shift[0]=shift_vec[0];
        shift[1]=shift_vec[1];
        shift[2]=shift_vec[2];
        shift[3]=shift_vec[3];

        calculate_send_receive_structure();

        if (mpi_setup==1)
        {
            cleanup_MPI_resources();
        }
        create_MPI_types();
    }

    /* Perform the shift */
    shift_data(corr);
}



void cleanup_shift(void)
{
    int i;

    if (corr_copy!=NULL)
    {
        free(corr_copy);
        corr_copy=NULL;
    }
    cleanup_MPI_resources();
    setup=0;
}



static void cleanup_MPI_resources(void)
{
    int err_count=0,i;
    for (i=0;i<MAX_NEIGHBORS;i++)
    {
        if (receive_indices[i]!=NULL)
        {
            free(receive_indices[i]);
            receive_indices[i]=NULL;
            free(send_indices[i]);
            send_indices[i]=NULL;
        }
        if (receive_types[i]!=MPI_DATATYPE_NULL)
        {
            err_count=MPI_Type_free(&receive_types[i]);
            receive_types[i]=MPI_DATATYPE_NULL;
            err_count+=MPI_Type_free(&send_types[i]);
            send_types[i]=MPI_DATATYPE_NULL;

            error(err_count!=MPI_SUCCESS,err_count,"cleanup_mpi_resources [shift.c]",
                  "Failed to free MPI receive/send types");
        }
    }
    mpi_setup=0;
}