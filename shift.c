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



static int my_rank,setup=0;
static int int_size=0,complex_double_size=0;
static complex_dble *corr_copy;

static int neighbor[4],remote_base_process_rank;
static int local_bases[64],remote_bases[64],base_spans[64];



void set_up_shift(int *shift)
{
    if (setup==0)
    {
        int err_count;
        int i,t,x,y,z,index;
        int remote_basepoint[4],remote_basepoint_local[4],process_coords[4]; /*What else comes here?*/ 
        int remote_base_process_coords[4];
        int little_vec[4],big_vec[4];

        error_root((shift[0]!=0)&&(bcon!=3),1,"set_up_shift [shift.c]",
            "Shifts in time are only permitted for periodic bc");

        /*** MPI setup ***/
        create_MPI_COMPLEX_DOUBLE();
        /* Calculate data sizes */
        err_count=MPI_Type_size(MPI_INT,&int_size);
        err_count+=MPI_Type_size(MPI_COMPLEX_DOUBLE,&complex_double_size);
        error(err_count!=MPI_SUCCESS,1,"parallel_write [parallel_out.c]",
                "Failed to get size of MPI_INT and MPI_COMPLEX_DOUBLE data types");
        /* Set my_rank */
        err_count=MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
        error(err_count!=MPI_SUCCESS,err_count,"parallel_write [parallel_out.c]",
                "Failed to get rank of current process");

        /*** Shift setup ***/
        /*Calculate the global coords of the base of the sending lattice*/
        remote_basepoint[0]=safe_mod(cpr[0]*L0-shift[0],N0); 
        remote_basepoint[1]=safe_mod(cpr[1]*L1-shift[1],N1); 
        remote_basepoint[2]=safe_mod(cpr[2]*L2-shift[2],N2);
        remote_basepoint[3]=safe_mod(cpr[3]*L3-shift[3],N3);

        /*Calculate remote base process coords and rank*/
        remote_base_process_coords[0]=remote_basepoint[0]/L0;
        remote_base_process_coords[1]=remote_basepoint[1]/L1;
        remote_base_process_coords[2]=remote_basepoint[2]/L2;
        remote_base_process_coords[3]=remote_basepoint[3]/L3;
        remote_base_process_rank=ipr_global(remote_base_process_coords);

        /*Calculate the local coords of the base of the sending lattice*/
        remote_basepoint_local[0]=remote_basepoint[0]%L0;
        remote_basepoint_local[1]=remote_basepoint[1]%L1;
        remote_basepoint_local[2]=remote_basepoint[2]%L2;
        remote_basepoint_local[3]=remote_basepoint[3]%L3;

        /*Check if neighboring processes are involved in the shift*/
        neighbor[0]=!!remote_basepoint_local[0];
        neighbor[1]=!!remote_basepoint_local[1];
        neighbor[2]=!!remote_basepoint_local[2];
        neighbor[3]=!!remote_basepoint_local[3];

        /*Calculate little vec and big vec*/
        little_vec[0]=L0-1-remote_basepoint_local[0];
        little_vec[1]=L1-1-remote_basepoint_local[1];
        little_vec[2]=L2-1-remote_basepoint_local[2];
        little_vec[3]=L3-1-remote_basepoint_local[3];
        big_vec[0]=safe_mod(remote_basepoint_local[0]-1,L0);
        big_vec[1]=safe_mod(remote_basepoint_local[1]-1,L1);
        big_vec[2]=safe_mod(remote_basepoint_local[2]-1,L2);
        big_vec[3]=safe_mod(remote_basepoint_local[3]-1,L3);

        /*Calculate local bases, remote bases, and base spans*/
        for (t=0;t<neighbor[0];t++)
        {
            for (x=0;x<neighbor[1];x++)
            {
                for (y=0;y<neighbor[2];y++)
                {
                    for (z=0;z<neighbor[3];z++)
                    {
                        index=4*(z+y*2+x*4+t*8);

                        remote_bases[index]=(t==1)?0:remote_basepoint_local[0];
                        remote_bases[index+1]=(x==1)?0:remote_basepoint_local[1];
                        remote_bases[index+2]=(y==1)?0:remote_basepoint_local[2];
                        remote_bases[index+3]=(z==1)?0:remote_basepoint_local[3];
                        
                        base_spans[index]=(t==1)?big_vec[0]:little_vec[0];
                        base_spans[index+1]=(x==1)?big_vec[1]:little_vec[1];
                        base_spans[index+2]=(y==1)?big_vec[2]: little_vec[2];
                        base_spans[index+3]=(z==1)?big_vec[3]:little_vec[3];

                        local_bases[index]=(t==1)?base_spans[index]+1:0;
                        local_bases[index+1]=(x==1)?base_spans[index+1]+1:0;
                        local_bases[index+2]=(y==1)?base_spans[index+2]+1:0;
                        local_bases[index+3]=(z==1)?base_spans[index+3]+1:0;
                    }
                }
            }
        }
        setup=1;
    }
}



void alloc_corr_copy(void)
{
    corr_copy=malloc(npcorr*VOLUME*sizeof(complex_dble));
    error(corr_copy==NULL,1,"alloc_data [rotation.c]",
            "Unable to allocate data arrays");
}