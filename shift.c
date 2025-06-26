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

static int *basepoint,*basepoint_shifted,*basepoint_shifted_local;



void set_up_shift(int *shift)
{
    if (setup==0)
    {
        if (npcorr!=0)
        {
            1;
        }
        else
        {
            error(1,1,"set_up_shift",
            "npcorr has not been set yet.");
        }
    }
}



void alloc_corr_copy(void)
{
    corr_copy=malloc(npcorr*VOLUME*sizeof(complex_dble));
    error(corr_copy==NULL,1,"alloc_data [rotation.c]",
            "Unable to allocate data arrays");
}