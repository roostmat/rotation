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
#include "rotation.h"
#include "utils.h" /* for error, error_root */




static complex_dble global_corr[N0*N1*N2*N3],averaged_global_corr[N0*N1*N2*N3];
static complex_dble corr[VOLUME],corr_cmp[VOLUME];
static int outlat[4];



static void setup_lattices(void)
{
    int i;
    for (i=0;i<N0*N1*N2*N3;i++)
    {
        global_corr[i].re=(double)i;
        global_corr[i].im=(double)i;
        averaged_global_corr[i].re=(double)i;
        averaged_global_corr[i].im=(double)i;
    }
}