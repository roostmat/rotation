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
#include "rotation.h"
#include "utils.h" /* for error, error_root */

#define DIM 4
#define MAX_NEIGHBORS 16



static int my_rank,setup=0;
static int npcorr=-1;
static MPI_Datatype MPI_COMPLEX_DOUBLE;
static complex_dble *corr_copy=NULL;

static int shift[DIM]={0,0,0,0},neighbor_long[MAX_NEIGHBORS]={0};
static int receive_bases[MAX_NEIGHBORS*DIM],send_bases[MAX_NEIGHBORS*DIM],
            block_volume[MAX_NEIGHBORS];
static MPI_Datatype receive_types[MAX_NEIGHBORS]={MPI_DATATYPE_NULL},
                    send_types[MAX_NEIGHBORS]={MPI_DATATYPE_NULL};
static int *receive_indices[MAX_NEIGHBORS]={NULL},
            *send_indices[MAX_NEIGHBORS]={NULL};



static void init_send_receive_structure(void)
{
    int i;

    for (i=0;i<MAX_NEIGHBORS;i++)
    {
        receive_indices[i]=NULL;
        send_indices[i]=NULL;
        neighbor_long[i]=0;
        block_volume[i]=0;
        receive_bases[i*DIM]=0;
        receive_bases[i*DIM+1]=0;
        receive_bases[i*DIM+2]=0;
        receive_bases[i*DIM+3]=0;
        send_bases[i*DIM]=0;
        send_bases[i*DIM+1]=0;
        send_bases[i*DIM+2]=0;
        send_bases[i*DIM+3]=0;
        receive_types[i]=MPI_DATATYPE_NULL;
        send_types[i]=MPI_DATATYPE_NULL;
    }
}



static void setup_shift(void)
{
    int err_count=0;

    if (!setup)
    {
        npcorr=get_npcorr();
        create_MPI_COMPLEX_DOUBLE(&MPI_COMPLEX_DOUBLE);

        err_count=MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
        error(err_count!=MPI_SUCCESS,err_count,"setup_shift [shift_corr.c]",
                "Failed to get rank of current process");

        corr_copy=malloc(npcorr*VOLUME*sizeof(complex_dble));
        error(corr_copy==NULL,1,"setup_shift [shift_corr.c]",
                "Unable to allocate corr_copy array");

        init_send_receive_structure();
        setup=1;
    }
}



static void calculate_send_receive_structure(void)
{
    int t,x,y,z,i;
    int ts,xs,ys,zs,j,n;
    int local_shift[DIM],little_vec[DIM],neighbor[DIM]={0},base_spans[MAX_NEIGHBORS*DIM];
    int receive_coords[DIM],send_coords[DIM];
    int receive_i,send_i;
    int err_count=0;

    local_shift[0]=safe_mod(shift[0],L0); /* local coords of base point (0,0,0,0) */
    local_shift[1]=safe_mod(shift[1],L1); /* of local lattice + shift */
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
                    i=z+y*2+x*4+t*8; /* i is the index of the neighbor */

                    neighbor_long[i]=1;

                    /* Calculate send/receive bases and spans */
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

                    /* Allocate send/receive index arrays */
                    receive_indices[i]=malloc(npcorr*block_volume[i]*sizeof(int));
                    send_indices[i]=malloc(npcorr*block_volume[i]*sizeof(int));
                    error((receive_indices[i]==NULL)||(send_indices[i]==NULL),1,"create_MPI_types [shift_corr.c]",
                            "Failed to allocate receive or send index arrays");

                    /* Calculate send/receive indices */
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
                        /* Create send/receive MPI types */
                        err_count=MPI_Type_create_indexed_block(npcorr*block_volume[i],1,receive_indices[i],
                                                            MPI_COMPLEX_DOUBLE,&receive_types[i]);
                        err_count+=MPI_Type_create_indexed_block(npcorr*block_volume[i],1,send_indices[i],
                                                            MPI_COMPLEX_DOUBLE,&send_types[i]);
                        error(err_count!=MPI_SUCCESS,1,"create_MPI_types [shift_corr.c]",
                                "Failed to create MPI indexed block types");

                        /* Commit send/receive MPI types */
                        err_count=MPI_Type_commit(&receive_types[i]);
                        err_count+=MPI_Type_commit(&send_types[i]);
                        error(err_count!=MPI_SUCCESS,1,"create_MPI_types [shift_corr.c]",
                                "Failed to commit MPI indexed block types");
                    }
                }
            }
        }
    }
}




static void shift_data(complex_dble *corr)
{
    int i,j;
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
                    error(receive_indices[i][j]<0||receive_indices[i][j]>=npcorr*VOLUME,1,
                          "shift_data [shift_corr.c]",
                          "Invalid receive index %d for process %d", receive_indices[i][j], my_rank);
                    error(send_indices[i][j]<0||send_indices[i][j]>=npcorr*VOLUME,1,
                          "shift_data [shift_corr.c]",
                          "Invalid send index %d for process %d", send_indices[i][j], my_rank);
                    corr[receive_indices[i][j]]=corr_copy[send_indices[i][j]];
                }
            }
            else
            {
                err_count=MPI_Sendrecv(corr_copy,1,send_types[i],send_rank,0,
                                        corr,1,receive_types[i],receive_rank,0,
                                        MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                error(err_count!=MPI_SUCCESS,1,"shift_data [shift_corr.c]",
                        "Process with rank %d failed to send/receive data during shift", my_rank);
            }
        }
    }
}



static void cleanup_indices(void)
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

            error(err_count!=MPI_SUCCESS,err_count,"cleanup_MPI_types [shift_corr.c]",
                  "Failed to free MPI receive/send types");
        }
    }
}



void cleanup_shift(void)
{
    if (corr_copy!=NULL)
    {
        free(corr_copy);
        corr_copy=NULL;
    }
    cleanup_indices();
    setup=0;
}



void shift_corr(complex_dble *corr,int *shift_vec)
{
    setup_shift();

    /* Create copy of current state of corr */
    memcpy(corr_copy,corr,npcorr*VOLUME*sizeof(complex_dble));

    /* Check if send/receive structure is already set up */
    if ((shift[0]!=shift_vec[0])||(shift[1]!=shift_vec[1])||
        (shift[2]!=shift_vec[2])||(shift[3]!=shift_vec[3]))
    {
        shift[0]=shift_vec[0];
        shift[1]=shift_vec[1];
        shift[2]=shift_vec[2];
        shift[3]=shift_vec[3];

        cleanup_indices();
        calculate_send_receive_structure();
    }

    if ((shift_vec[0]!=0)||(shift_vec[1]!=0)||
        (shift_vec[2]!=0)||(shift_vec[3]!=0))
    {
        /* Perform the shift */
        shift_data(corr);
    }
}



static void average_2points(complex_dble *corr,int coords1[DIM],int coords2[DIM])
{
    int index1,index2;
    int rank1,rank2;
    complex_dble received_value;
    int err_count=0;

    lex_global(coords1,&rank1,&index1);
    lex_global(coords2,&rank2,&index2);

    if ((rank1==my_rank)&&(rank2==my_rank))
    {
        /* Both points are on the same process */
        corr[index1].re=0.5*(corr[index1].re + corr[index2].re);
        corr[index1].im=0.5*(corr[index1].im + corr[index2].im);
        corr[index2].re=corr[index1].re;
        corr[index2].im=corr[index1].im;
    }
    else if (rank1==my_rank)
    {
        err_count=MPI_Sendrecv(corr+index1,1,MPI_COMPLEX_DOUBLE,
                                rank2,index1,
                                &received_value,1,MPI_COMPLEX_DOUBLE,
                                rank2,index2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        error(err_count!=MPI_SUCCESS,1,"average_2points [shift_corr.c]",
                "Failed to send/receive data for averaging points (%d,%d,%d,%d) and (%d,%d,%d,%d)",
                coords1[0],coords1[1],coords1[2],coords1[3],
                coords2[0],coords2[1],coords2[2],coords2[3]);
        corr[index1].re=0.5*(corr[index1].re + received_value.re);
        corr[index1].im=0.5*(corr[index1].im + received_value.im);
    }
    else if (rank2==my_rank)
    {
        err_count=MPI_Sendrecv(corr+index2,1,MPI_COMPLEX_DOUBLE,
                                rank1,index2,
                                &received_value,1,MPI_COMPLEX_DOUBLE,
                                rank1,index1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        error(err_count!=MPI_SUCCESS,1,"average_2points [shift_corr.c]",
                "Failed to send/receive data for averaging points (%d,%d,%d,%d) and (%d,%d,%d,%d)",
                coords1[0],coords1[1],coords1[2],coords1[3],
                coords2[0],coords2[1],coords2[2],coords2[3]);
        corr[index2].re=0.5*(corr[index2].re + received_value.re);
        corr[index2].im=0.5*(corr[index2].im + received_value.im);
    }
}



void average_equiv(complex_dble *corr,int outlat[4])
{
    int t,x,y,z,i,j;
    int coords1[DIM],coords2[DIM];
    int small_volume;

    /* Average on-axis */
    coords1[1]=0;
    coords1[2]=0;
    coords1[3]=0;
    coords2[1]=0;
    coords2[2]=0;
    coords2[3]=0;
    for (t=1;t<((outlat[0])<((N0+1)/2)?(outlat[0]):((N0+1)/2));t++)
    {
        coords1[0]=t;
        coords2[0]=N0-t;
        average_2points(corr,coords1,coords2);
    }

    coords1[0]=0;
    coords1[2]=0;
    coords1[3]=0;
    coords2[0]=0;
    coords2[2]=0;
    coords2[3]=0;
    for (x=1;x<((outlat[1])<((N1+1)/2)?(outlat[1]):((N1+1)/2));x++)
    {
        coords1[1]=x;
        coords2[1]=N1-x;
        average_2points(corr,coords1,coords2);
    }

    coords1[0]=0;
    coords1[1]=0;
    coords1[3]=0;
    coords2[0]=0;
    coords2[1]=0;
    coords2[3]=0;
    for (y=1;y<((outlat[2])<((N2+1)/2)?(outlat[2]):((N2+1)/2));y++)
    {
        coords1[2]=y;
        coords2[2]=N2-y;
        average_2points(corr,coords1,coords2);
    }

    coords1[0]=0;
    coords1[1]=0;
    coords1[2]=0;
    coords2[0]=0;
    coords2[1]=0;
    coords2[2]=0;
    for (z=1;z<((outlat[3])<((N3+1)/2)?(outlat[3]):((N3+1)/2));z++)
    {
        coords1[3]=z;
        coords2[3]=N3-z;
        average_2points(corr,coords1,coords2);
    }

    /* Average off-axis */
    small_volume=(N0-1)*(N1-1)*(N2-1)*(N3-1);

    for (i=0;i<small_volume;i++)
    {
        j=small_volume-1-i;
        coords1[0]=i/((N1-1)*(N2-1)*(N3-1))+1;
        coords1[1]=(i/((N2-1)*(N3-1)))%((N1-1))+1;
        coords1[2]=(i/((N3-1)))%((N2-1))+1;
        coords1[3]=i%((N3-1))+1;
        coords2[0]=j/((N1-1)*(N2-1)*(N3-1))+1;
        coords2[1]=(j/((N2-1)*(N3-1)))%((N1-1))+1;
        coords2[2]=(j/((N3-1)))%((N2-1))+1;
        coords2[3]=j%((N3-1))+1;

        if (((coords1[0]<outlat[0])&&(coords1[1]<outlat[1])&&
             (coords1[2]<outlat[2])&&(coords1[3]<outlat[3]))||
            ((coords2[0]<outlat[0])&&(coords2[1]<outlat[1])&&
             (coords2[2]<outlat[2])&&(coords2[3]<outlat[3])))
        {
            average_2points(corr,coords1,coords2);
        }
    }
}



/*************************************************************
 * This function averages the correlator array corr
 * over all points that are equivalent in the sense of
 * C(x)=C(-x) for all x. It assumes that the source is located
 * at the origin (0,0,0,0).
 *************************************************************/
void average_equiv2(complex_dble *corr)
{
    int i,j;
    int coords[DIM],bc;
    int dest[VOLUME],partner[VOLUME],partner_index,partner_rank;
    int send_receive_rank[NPROC],rank_index[NPROC];
    int nsend=0,*send_count,*dest_source_rank;
    int **my_index,**remote_index,**sort_index;
    complex_dble **send_buffer,**receive_buffer;
    MPI_Request *requests;
    int err_count=0;

    error(corr==NULL,1,"average_equiv2 [shift_corr.c]",
            "corr array is NULL");

    err_count=MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    error(err_count!=MPI_SUCCESS,err_count,"setup_shift [shift_corr.c]",
            "Failed to get rank of current process");

    /* Get boundary conditions */
    bc=bc_type();

    /* Initialize dest and partner array to -1 */
    for (i=0;i<VOLUME;i++)
    {
        dest[i]=-1;
        partner[i]=-1;
    }

    /* Initialize send_receive_ranks to 0 */
    /* and rank_index to -1 */
    for (i=0;i<NPROC;i++)
    {
        send_receive_rank[i]=0;
        rank_index[i]=-1;
    }

    for (i=0;i<VOLUME;i++)
    {
        if (dest[i]==-1)
        {
            if (bc==0)
            {
                coords[0]=cpr[0]*L0+i/(L1*L2*L3);
            }
            else if (bc==3)
            {
                coords[0]=safe_mod(-(cpr[0]*L0+i/(L1*L2*L3)),N0);
            }
            else
            {
                error_root(1,"average_equiv2 [shift_corr.c]",
                        "Unsupported boundary condition %d", bc);
            }
            coords[1]=safe_mod(-(cpr[1]*L1+(i/(L2*L3))%L1),N1);
            coords[2]=safe_mod(-(cpr[2]*L2+(i/L3)%L2),N2);
            coords[3]=safe_mod(-(cpr[3]*L3+i%L3),N3);

            lex_global(coords,&partner_rank,&partner_index);

            if ((my_rank==partner_rank)&&(i!=partner_index))
            {
                /* Partner point is on the same process */
                dest[i]=my_rank;
                dest[partner_index]=my_rank;
                partner[i]=partner_index;
            }
            else if (my_rank!=partner_rank)
            {
                /* Partner point is on a different process */
                dest[i]=partner_rank;
                partner[i]=partner_index;
                send_receive_rank[partner_rank]++;
                if (send_receive_rank[partner_rank]==1)
                    nsend++;
            }
        }
    }
    if (nsend>0)
    {
        /* Allocate buffers and index arrays */
        send_buffer=malloc(nsend*sizeof(complex_dble*));
        error(send_buffer==NULL,1,"average_equiv2 [shift_corr.c]",
                "Failed to allocate send_buffer array");
        receive_buffer=malloc(nsend*sizeof(complex_dble*));
        error(receive_buffer==NULL,1,"average_equiv2 [shift_corr.c]",
                "Failed to allocate receive_buffer array");
        my_index=malloc(nsend*sizeof(int*));
        error(my_index==NULL,1,"average_equiv2 [shift_corr.c]",
                "Failed to allocate my_index array");
        remote_index=malloc(nsend*sizeof(int*));
        error(remote_index==NULL,1,"average_equiv2 [shift_corr.c]",
                "Failed to allocate remote_index array");
        sort_index=malloc(nsend*sizeof(int*));
        error(sort_index==NULL,1,"average_equiv2 [shift_corr.c]",
                "Failed to allocate sort_index array");
        dest_source_rank=malloc(nsend*sizeof(int));
        error(dest_source_rank==NULL,1,"average_equiv2 [shift_corr.c]",
                "Failed to allocate dest_source_rank array");
        send_count=malloc(nsend*sizeof(int));
        error(send_count==NULL,1,"average_equiv2 [shift_corr.c]",
                "Failed to allocate send_count array");
        requests=malloc(2*nsend*sizeof(MPI_Request));
        error(requests==NULL,1,"average_equiv2 [shift_corr.c]",
                "Failed to allocate requests array");

        j=0;
        for (i=0;i<NPROC;i++)
        {
            error(j>=nsend,1,"average_equiv2 [shift_corr.c]",
                    "More send_receive_rank entries than nsend (%d > %d)", j, nsend);
            if (send_receive_rank[i]>0)
            {
                send_buffer[j]=malloc(send_receive_rank[i]*sizeof(complex_dble));
                error(send_buffer[j]==NULL,1,"average_equiv2 [shift_corr.c]",
                        "Failed to allocate send_buffer[%d] array", i);
                receive_buffer[j]=malloc(send_receive_rank[i]*sizeof(complex_dble));
                error(receive_buffer[j]==NULL,1,"average_equiv2 [shift_corr.c]",
                        "Failed to allocate receive_buffer[%d] array", i);
                my_index[j]=malloc(send_receive_rank[i]*sizeof(int));
                error(my_index[j]==NULL,1,"average_equiv2 [shift_corr.c]",
                        "Failed to allocate my_index[%d] array", i);
                remote_index[j]=malloc(send_receive_rank[i]*sizeof(int));
                error(remote_index[j]==NULL,1,"average_equiv2 [shift_corr.c]",
                        "Failed to allocate remote_index[%d] array", i);
                sort_index[j]=malloc(send_receive_rank[i]*sizeof(int));
                error(sort_index[j]==NULL,1,"average_equiv2 [shift_corr.c]",
                        "Failed to allocate sort_index[%d] array", i);
                send_count[j]=0;
                /* Store index of given rank */
                rank_index[i]=j;
                /* Convert dest_source_rank from lexicographical to ipr_global */
                dest_source_rank[j]=i;
                j++;
            }
        }

        error(j!=nsend,1,"average_equiv2 [shift_corr.c]",
                "Number of send_receive_rank entries (%d) does not match nsend (%d)", j, nsend);

        /* Fill send buffers or average locally */
        for (i=0;i<VOLUME;i++)
        {
            if (dest[i]!=-1)
            {
                if ((dest[i]==my_rank)&&(i<partner[i]))
                {
                    /* Perform local averaging */
                    corr[i].re=0.5*(corr[i].re + corr[partner[i]].re);
                    corr[i].im=0.5*(corr[i].im + corr[partner[i]].im);
                    corr[partner[i]].re=corr[i].re;
                    corr[partner[i]].im=corr[i].im;
                }
                else if (dest[i]!=my_rank)
                {
                    /* Fill send buffer */
                    error(rank_index[dest[i]]==-1,1,"average_equiv2 [shift_corr.c]",
                            "Rank %d not found in rank_index array", dest[i]);

                    send_buffer[rank_index[dest[i]]][send_count[rank_index[dest[i]]]]=corr[i];
                    my_index[rank_index[dest[i]]][send_count[rank_index[dest[i]]]]=i;
                    remote_index[rank_index[dest[i]]][send_count[rank_index[dest[i]]]]=partner[i];
                    send_count[rank_index[dest[i]]]++;
                }
            }
        }

        /* Exchange data */
        MPI_Barrier(MPI_COMM_WORLD);
        for (i=0;i<nsend;i++)
        {
            err_count=MPI_Irecv(receive_buffer[i],send_count[i],MPI_COMPLEX_DOUBLE,
                                dest_source_rank[i],my_rank,MPI_COMM_WORLD,&requests[i]);
            error(err_count!=MPI_SUCCESS,1,"average_equiv2 [shift_corr.c]",
                    "Failed to post Irecv for averaging points from rank %d", dest_source_rank[i]);

            err_count=MPI_Isend(send_buffer[i],send_count[i],MPI_COMPLEX_DOUBLE,
                                dest_source_rank[i],my_rank,MPI_COMM_WORLD,&requests[nsend+i]);
            error(err_count!=MPI_SUCCESS,1,"average_equiv2 [shift_corr.c]",
                    "Failed to post Isend for averaging points to rank %d", dest_source_rank[i]);
        }
        err_count=MPI_Waitall(2*nsend,requests,MPI_STATUSES_IGNORE);
        error(err_count!=MPI_SUCCESS,1,"average_equiv2 [shift_corr.c]",
                "Failed to wait for all send/receive operations during averaging points");

        /* Average received data */
        for (i=0;i<nsend;i++)
        {
            /* Sort my index in the way remote_index would be sorted */
            get_sorted_indices(remote_index[i],sort_index[i],send_count[i]);
            sort_array_from_indices(my_index[i],sort_index[i],send_count[i]);

            /* Take average with received data */
            for (j=0;j<send_count[i];j++)
            {
                corr[my_index[i][j]].re=0.5*(corr[my_index[i][j]].re+
                                            receive_buffer[i][j].re);
                corr[my_index[i][j]].im=0.5*(corr[my_index[i][j]].im+
                                            receive_buffer[i][j].im);
            }
        }

        /* Cleanup */
        for (i=0;i<nsend;i++)
        {
            free(send_buffer[i]);
            free(receive_buffer[i]);
            free(my_index[i]);
            free(remote_index[i]);
            free(sort_index[i]);
        }
        free(send_buffer);
        free(receive_buffer);
        free(my_index);
        free(remote_index);
        free(sort_index);
        free(dest_source_rank);
        free(send_count);
        free(requests);
    }
}