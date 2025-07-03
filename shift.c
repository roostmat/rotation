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



static int my_rank,local_lattice[DIM]={L0,L1,L2,L3};
static int int_size=0,complex_double_size=0;
static complex_dble *corr_copy;

static int shift[DIM]={0,0,0,0},neighbor[DIM],shift_setup=0;
static int receive_bases[MAX_NEIGHBORS*DIM],send_bases[MAX_NEIGHBORS*DIM],base_spans[MAX_NEIGHBORS*DIM];
static MPI_Datatype receive_types[MAX_NEIGHBORS],send_types[MAX_NEIGHBORS];



static void calculate_send_receive_structure(void)
{
    int index,t,x,y,z,i;
    int local_shift[DIM],little_vec[DIM];

    local_shift[0]=safe_mod(shift[0],L0);
    local_shift[1]=safe_mod(shift[1],L1);
    local_shift[2]=safe_mod(shift[2],L2);
    local_shift[3]=safe_mod(shift[3],L3);

    little_vec[0]=L0-1-local_shift[0];
    little_vec[1]=L1-1-local_shift[1];
    little_vec[2]=L2-1-local_shift[2];
    little_vec[3]=L3-1-local_shift[3];

    neighbor[0]=!!local_shift[0];
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
                    index=(z+y*2+x*4+t*8)*DIM;

                    send_bases[index]=(t==0)?0:little_vec[0]+1;
                    send_bases[index+1]=(x==0)?0:little_vec[1]+1;
                    send_bases[index+2]=(y==0)?0:little_vec[2]+1;
                    send_bases[index+3]=(z==0)?0:little_vec[3]+1;

                    base_spans[index]=(t==0)?little_vec[0]:L0-2-little_vec[0];
                    base_spans[index+1]=(x==0)?little_vec[1]:L1-2-little_vec[1];
                    base_spans[index+2]=(y==0)?little_vec[2]:L2-2-little_vec[2];
                    base_spans[index+3]=(z==0)?little_vec[3]:L3-2-little_vec[3];

                    receive_bases[index]=(t==0)?local_shift[0]:0;
                    receive_bases[index+1]=(x==0)?local_shift[1]:0;
                    receive_bases[index+2]=(y==0)?local_shift[2]:0;
                    receive_bases[index+3]=(z==0)?local_shift[3]:0;
                }
            }
        }
    }
}



static void create_MPI_types(void) /*CHECK HOW THIS FUNCTION CAN BE COMBINED WITH ipt[]!!!!*/
{                                  /*USE MPI_Type_create_indexed_block()*/
    int t,x,y,z,i;
    int index;

    if (npcorr>1)
    {
        MPI_Datatype receive_block,receive_blocks[npcorr],send_block,send_blocks[npcorr];
        MPI_Aint displacements[npcorr];
        int block_lengths[npcorr];

        for (i=0;i<npcorr;i++)
        {
            displacements[i]=i*VOLUME*sizeof(complex_dble);
            block_lengths[i]=1;
        }

        for (t=0;t<neighbor[0]+1;t++)
        {
            for (x=0;x<neighbor[1]+1;x++)
            {
                for (y=0;y<neighbor[2]+1;y++)
                {
                    for (z=0;z<neighbor[3]+1;z++)
                    {
                        index=(z+y*2+x*4+t*8);

                        /*Create receive type*/
                        MPI_Type_create_subarray(DIM,local_lattice,base_spans+(index*DIM),
                                receive_bases+(index*DIM),MPI_ORDER_FORTRAN,MPI_COMPLEX_DOUBLE,
                                &receive_block);
                        MPI_Type_commit(&receive_block);

                        /*Create send type*/
                        MPI_Type_create_subarray(DIM,local_lattice,base_spans+(index*DIM),
                                send_bases+(index*DIM),MPI_ORDER_FORTRAN,MPI_COMPLEX_DOUBLE,
                                &send_block);
                        MPI_Type_commit(&send_block);

                        for (i=0;i<npcorr;i++)
                        {
                            receive_blocks[i]=receive_block;
                            send_blocks[i]=send_block;
                        }

                        /*Create super receive and send types*/
                        MPI_Type_create_struct(npcorr,block_lengths,displacements,
                                receive_blocks,&receive_types[index]);
                        MPI_Type_create_struct(npcorr,block_lengths,displacements,
                                send_blocks,&send_types[index]);

                        /*Commit super receive and send types*/
                        MPI_Type_commit(&receive_types[index]);
                        MPI_Type_commit(&send_types[index]);
                    }
                }
            }
        }
    }
    else
    {
        for (t=0;t<neighbor[0]+1;t++)
        {
            for (x=0;x<neighbor[1]+1;x++)
            {
                for (y=0;y<neighbor[2]+1;y++)
                {
                    for (z=0;z<neighbor[3]+1;z++)
                    {
                        index=(z+y*2+x*4+t*8);

                        /*Create and commit receive types*/
                        MPI_Type_create_subarray(DIM,local_lattice,base_spans+(index*DIM),
                                receive_bases+(index*DIM),MPI_ORDER_FORTRAN,MPI_COMPLEX_DOUBLE,
                                &receive_types[index]);
                        MPI_Type_commit(&receive_types[index]);

                        /*Create and commit send types*/
                        MPI_Type_create_subarray(DIM,local_lattice,base_spans+(index*DIM),
                                send_bases+(index*DIM),MPI_ORDER_FORTRAN,MPI_COMPLEX_DOUBLE,
                                &send_types[index]);
                        MPI_Type_commit(&send_types[index]);
                    }
                }
            }
        }
    }
}



static void shift_data(void)
{
    int t,x,y,z,index;
    int receive_rank_coords[DIM],send_rank_coords[DIM];
    int receive_rank,send_rank;
    int err_count=0;

    for (t=0;t<neighbor[0]+1;t++)
    {
        for (x=0;x<neighbor[1]+1;x++)
        {
            for (y=0;y<neighbor[2]+1;y++)
            {
                for (z=0;z<neighbor[3]+1;z++)
                {
                    index=(z+y*2+x*4+t*8);

                    receive_rank_coords[0]=safe_mod(cpr[0]*L0+receive_bases[index*DIM]-shift[0],N0)/L0;
                    receive_rank_coords[1]=safe_mod(cpr[1]*L1+receive_bases[index*DIM+1]-shift[1],N1)/L1;
                    receive_rank_coords[2]=safe_mod(cpr[2]*L2+receive_bases[index*DIM+2]-shift[2],N2)/L2;
                    receive_rank_coords[3]=safe_mod(cpr[3]*L3+receive_bases[index*DIM+3]-shift[3],N3)/L3;
                    receive_rank=ipr_global(receive_rank_coords);

                    send_rank_coords[0]=safe_mod(cpr[0]*L0+send_bases[index*DIM]+shift[0],N0)/L0;
                    send_rank_coords[1]=safe_mod(cpr[1]*L1+send_bases[index*DIM+1]+shift[1],N1)/L1;
                    send_rank_coords[2]=safe_mod(cpr[2]*L2+send_bases[index*DIM+2]+shift[2],N2)/L2;
                    send_rank_coords[3]=safe_mod(cpr[3]*L3+send_bases[index*DIM+3]+shift[3],N3)/L3;
                    send_rank=ipr_global(send_rank_coords);

                    if ((my_rank==receive_rank)&&(my_rank==send_rank))
                    {
                        /*No other process involved. We shift internally.*/
                        
                    }
                }
            }
        }
    }
}



static void create_corr_copy(void)
{
    corr_copy=malloc(npcorr*VOLUME*sizeof(complex_dble));
    error(corr_copy==NULL,1,"create_corr_copy [shift.c]",
            "Unable to allocate corr_copy array");

    copy_corr_data(corr_copy);
}



static void free_corr_copy(void)
{
    if (corr_copy!=NULL)
    {
        free(corr_copy);
        corr_copy=NULL;
    }
}



