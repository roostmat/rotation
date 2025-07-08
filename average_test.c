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