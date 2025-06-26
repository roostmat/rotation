/*******************************************************************************
*
* File mesons.c
*
* Copyright (C) 2024, 2025 Tomasz Korzec
*
* Based on mesons
* Copyright (C) 2013, 2014 Tomasz Korzec
*
* Based on openQCD, ms1 and ms4
* Copyright (C) 2012 Martin Luescher and Stefan Schaefer
*
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
* Computation of two-point meson correlators
*
* Syntax: rotation -i <input file>
*
* For usage instructions see the file README.rotation
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "archive.h"
#include "dfl.h"
#include "dirac.h"
#include "flags.h"
#include "forces.h"
#include "global.h"
#include "lattice.h"
#include "linalg.h"
#include "mesons.h"
#include "mpi.h"
#include "random.h"
#include "rotation.h"
#include "sap.h"
#include "sflds.h"
#include "uflds.h"
#include "utils.h"
#include "version.h"

#define MAX(n,m) \
    if ((n)<(m)) \
        (n)=(m)



static struct
{
    int nux0;            /* number of unique time slices */
    int *ux0;            /* array of unique time slices */
    int nmax;            /* max(nuprop) */
    int *nuprop;         /* number of unique propagators (iprop) for each unique time slice */
    int **uprop;         /* array of unique propagators (iprop) for each unique time slice */
    int **uprop_index;   /* iprop = uprop[ix0][uprop_index[ix0][iprop]] */
} proplist;


/**************************************************************************
 * isps: index of the solver for each propagator
 * props1: index of the first propagator for each correlator
 * props2: index of the second propagator for each correlator
 * type1: Dirac structure of the first propagator for each correlator
 * type2: Dirac structure of the second propagator for each correlator
 * x0s: x0 position of the source for each correlator
 * srcs: source coordinates for each correlator
***************************************************************************/
static corr_data data;                  /* data structure for the correlators */
int npcorr=-1;                          /* number of point correlators */
int outlat[4]={-1,-1,-1,-1};            /* output lattice dimensions */
int pos=-1;                             /* positioning of the source insided the output lattice*/
int bcon=-1;                            /* boundary conditions of the run */
static int my_rank,noexp,endian; /* append,norng; */
static int first,step,last;
static int level,seed,nprop;
static int *isps,*props1,*props2,*type1,*type2,*x0s,*srcs;
static double *kappas,*mus;


/* Paths and files */
static char log_dir[NAME_SIZE],loc_dir[NAME_SIZE];
static char cnfg_dir[NAME_SIZE],dat_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE],end_file[NAME_SIZE];
static char dat_file[NAME_SIZE] /*, dat_save[NAME_SIZE] */;
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char head_file[NAME_SIZE];
static char cnfg_file[NAME_SIZE],nbase[NAME_SIZE],outbase[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fend=NULL,*fdat=NULL;


static void maxn(int *n, int m)
{
    if ((*n)<m)
        (*n)=m;
}


static void alloc_data(void)
{
    data.corr=malloc(npcorr*VOLUME*sizeof(complex_dble));
    data.corr_out=malloc(npcorr*VOLUME*sizeof(complex_dble));
    error((data.corr==NULL)||(data.corr_out==NULL),1,"alloc_data [rotation.c]",
            "Unable to allocate data arrays");
}


/**************************************************************************
 * 
 * void write_head(void)
 *    Writes the header file for the run. The header file contains the
 *    number of point correlators (4 bytes), the output lattice dimensions
 *    (4*4 bytes), and for each point correlator Kappa1 (8 bytes), Kappa2
 *    (8 bytes), type1 (4 bytes), type2 (4 bytes), x0s (4 bytes).
 * 
 *************************************************************************/
static void write_head(void)
{
    if (my_rank==0)
    {
        stdint_t istd[1];
        int iw=0;
        int i;
        double dbl[1];
        
        fdat=fopen(head_file,"wb");
        error_root(fdat==NULL,1,"write_head [rotation.c]",
                "Unable to open head file");

        istd[0]=(stdint_t)(npcorr);
        if (endian==BIG_ENDIAN)
            bswap_int(1, istd);
        iw=fwrite(istd,sizeof(stdint_t),1,fdat);

        for (i=0;i<4;i++)
        {
            istd[0]=(stdint_t)(outlat[i]);
            if (endian==BIG_ENDIAN)
                bswap_int(1,istd);
            iw+=fwrite(istd,sizeof(stdint_t),1,fdat);
        }

        istd[0]=(stdint_t)(pos);
        if (endian==BIG_ENDIAN)
            bswap_int(1,istd);
        iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

        istd[0]=(stdint_t)(bcon);
        if (endian==BIG_ENDIAN)
            bswap_int(1,istd);
        iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

        error_root(iw!=7,1,"write_file_head [rotation.c]",
                "Incorrect write count");

        for (i=0;i<npcorr;i++)
        {
            dbl[0]=kappas[props1[i]];
            if (endian==BIG_ENDIAN)
                bswap_double(1, dbl);
            iw=fwrite(dbl,sizeof(double),1,fdat);

            dbl[0]=kappas[props2[i]];
            if (endian==BIG_ENDIAN)
                bswap_double(1,dbl);
            iw+=fwrite(dbl,sizeof(double),1,fdat);

            istd[0]=(stdint_t)(type1[i]);
            if (endian==BIG_ENDIAN)
                bswap_int(1,istd);
            iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

            istd[0]=(stdint_t)(type2[i]);
            if (endian==BIG_ENDIAN)
                bswap_int(1,istd);
            iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

            istd[0]=(stdint_t)(x0s[i]);
            if (endian==BIG_ENDIAN)
                bswap_int(1,istd);
            iw+=fwrite(istd,sizeof(stdint_t),1,fdat);

            error_root(iw!=5,1,"write_file_head [rotation.c]",
                        "Incorrect write count");
        }
        fclose(fdat);        
    }
}


/**************************************************************************
 * 
 * void write_data(void)
 *    Writes out the data file for the current configuration. The data file
 *    contains the source coordinates of each point (npcorr*4*4 bytes) and
 *    the correlator data.size*16 bytes.
 * 
 *************************************************************************/
static void write_data(void)
{
    int iw;

    sprintf(dat_file,"%s/%sn%d.rotation.dat",dat_dir,outbase,data.nc);
    if (my_rank==0)
    {
        error_root(fopen(dat_file,"r")!=NULL,1,"write_data [rotation.c]",
                "Attempt to overwrite old data file");
        fdat=fopen(dat_file,"wb");
        error_root(fdat==NULL,1,"write_data [rotation.c]",
                "Unable to open data file");

        /* Write the source coordinates of each pcorr at the beginning */
        if (endian==BIG_ENDIAN)
        {
            bswap_int(npcorr*4,srcs);
        }
        iw=fwrite(srcs,sizeof(int),npcorr*4,fdat);
        if (endian==BIG_ENDIAN)
        {
            bswap_int(npcorr*4,srcs);
        }
        error_root(iw!=npcorr*4,1,"write_data [rotation.c]",
                "Incorrect write count");
        fclose(fdat);
    }
    /* Write the point correlators */
    MPI_Barrier(MPI_COMM_WORLD);
    parallel_write(dat_file,&data,srcs);
}


static void read_dirs(void)
{
    if (my_rank==0)
    {
        find_section("Run name");
        read_line("name","%s",nbase);
        read_line_opt("output",nbase,"%s",outbase);

        find_section("Directories");
        read_line("log_dir","%s",log_dir);

        if (noexp)
        {
            read_line("loc_dir","%s",loc_dir);
            cnfg_dir[0]='\0';
        }
        else
        {
            read_line("cnfg_dir","%s",cnfg_dir);
            loc_dir[0]='\0';
        }

        read_line("dat_dir","%s",dat_dir);

        find_section("Configurations");
        read_line("first","%d",&first);
        read_line("last","%d",&last);
        read_line("step","%d",&step);

        find_section("Random number generator");
        read_line("level","%d",&level);
        read_line("seed","%d",&seed);

        error_root((last<first)||(step<1)||(((last-first)%step)!=0),1,
                    "read_dirs [rotation.c]","Improper configuration range");
    }

    MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Bcast(outbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

    MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Bcast(loc_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
    MPI_Bcast(dat_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

    MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

    MPI_Bcast(&level,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void setup_files(void)
{
    if (noexp)
        error_root(name_size("%s/%sn%d_%d",loc_dir,nbase,last,NPROC-1)>=NAME_SIZE,
                    1,"setup_files [rotation.c]","loc_dir name is too long");
    else
        error_root(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,
                    1,"setup_files [rotation.c]","cnfg_dir name is too long");

    check_dir_root(dat_dir);
    error_root(name_size("%s/%sn%d.rotation.dat~",dat_dir,outbase,last)>=NAME_SIZE,
                1,"setup_files [rotation.c]","dat_dir name is too long");

    check_dir_root(log_dir);
    error_root(name_size("%s/%s.rotation.log~",log_dir,outbase)>=NAME_SIZE,
                1,"setup_files [rotation.c]","log_dir name is too long");

    sprintf(log_file,"%s/%s.rotation.log",log_dir,outbase);
    sprintf(end_file,"%s/%s.rotation.end",log_dir,outbase);
    sprintf(par_file,"%s/%s.rotation.par",dat_dir,outbase);
    sprintf(head_file,"%s/%s.rotation.head",dat_dir,outbase);
    sprintf(log_save,"%s~",log_file);
    sprintf(par_save,"%s~",par_file);
}


static void read_run_parms(void)
{
    double csw,cF;
    char tmpstring[NAME_SIZE];
    char tmpstring2[NAME_SIZE];
    int iprop,ipcorr;

    if (my_rank==0)
    {
        find_section("Measurements");
        read_line("nprop","%d",&nprop);
        read_line("npcorr","%d",&npcorr);
        read_line("csw","%lf",&csw);
        read_line("cF","%lf",&cF);
        read_iprms("outlat",4,outlat);
        read_line("pos","%d",&pos);
        error_root(nprop<1,1,"read_run_parms [rotation.c]",
                "Specified nprop must be greater than 0");
        error_root(npcorr<1,1,"read_run_parms [rotation.c]",
                "Specified npcorr must be greater than 0");
        error_root((outlat[0]<=0)||(outlat[0]>N0)||
                    (outlat[1]<=0)||(outlat[1]>N1)||
                    (outlat[2]<=0)||(outlat[2]>N2)||
                    (outlat[3]<=0)||(outlat[3]>N3),1,
                    "read_run_parms [rotation.c]",
                    "Specified outlat out of range");
        error_root((pos!=0)&&(pos!=1),1,"read_run_parms [rotation.c]",
                    "Specified pos must be 0 or 1");
    }
    MPI_Bcast(&nprop,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&npcorr,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(outlat,4,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&pos,1,MPI_INT,0,MPI_COMM_WORLD);
    data.size=outlat[0]*outlat[1]*outlat[2]*outlat[3]; /* set the volume of the output lattice */

    kappas=malloc(nprop*sizeof(double));
    mus=malloc(nprop*sizeof(double));
    isps=malloc(nprop*sizeof(int));
    props1=malloc(npcorr*sizeof(int));
    props2=malloc(npcorr*sizeof(int));
    type1=malloc(npcorr*sizeof(int));
    type2=malloc(npcorr*sizeof(int));
    x0s=malloc(npcorr*sizeof(int));
    srcs=malloc(4*npcorr*sizeof(int));

    error((kappas==NULL)||(mus==NULL)||(isps==NULL)||(props1==NULL)||
            (props2==NULL)||(type1==NULL)||(type2==NULL)||(x0s==NULL)||
            (srcs==NULL),1,"read_run_parms [rotation.c]","Out of memory");

    if (my_rank==0)
    {
        for (iprop=0;iprop<nprop;iprop++)
        {
            sprintf(tmpstring,"Propagator %d",iprop);
            find_section(tmpstring);
            read_line("kappa","%lf",&kappas[iprop]);
            read_line("isp","%d",&isps[iprop]);
            mus[iprop]=0;
        }
        for (ipcorr=0;ipcorr<npcorr;ipcorr++)
        {
            sprintf(tmpstring,"Point correlator %d",ipcorr);
            find_section(tmpstring);
            read_line("iprop","%d %d",&props1[ipcorr],&props2[ipcorr]);
            error_root((props1[ipcorr]<0)||(props1[ipcorr]>=nprop)||
                (props2[ipcorr]<0)||(props2[ipcorr]>=nprop),1,
                "read_run_parms [rotation.c]","Propagator index out of range");
            read_line("type","%s %s",tmpstring,tmpstring2);
            type1[ipcorr]=-1;
            type2[ipcorr]=-1;

            if (strncmp(tmpstring,"1",1)==0)
                type1[ipcorr]=ONE_TYPE;
            else if (strncmp(tmpstring,"G0G1",4)==0)
                type1[ipcorr]=GAMMA0GAMMA1_TYPE;
            else if (strncmp(tmpstring,"G0G2",4)==0)
                type1[ipcorr]=GAMMA0GAMMA2_TYPE;
            else if (strncmp(tmpstring,"G0G3",4)==0)
                type1[ipcorr]=GAMMA0GAMMA3_TYPE;
            else if (strncmp(tmpstring,"G1G2",4)==0)
                type1[ipcorr]=GAMMA1GAMMA2_TYPE;
            else if (strncmp(tmpstring,"G1G3",4)==0)
                type1[ipcorr]=GAMMA1GAMMA3_TYPE;
            else if(strncmp(tmpstring,"G1G5",4)==0)
                type1[ipcorr]=GAMMA1GAMMA5_TYPE;
            else if(strncmp(tmpstring,"G2G3",4)==0)
                type1[ipcorr]=GAMMA2GAMMA3_TYPE;
            else if(strncmp(tmpstring,"G2G5",4)==0)
                type1[ipcorr]=GAMMA2GAMMA5_TYPE;
            else if(strncmp(tmpstring,"G3G5",4)==0)
                type1[ipcorr]=GAMMA3GAMMA5_TYPE;
            else if(strncmp(tmpstring,"G0",2)==0)
                type1[ipcorr]=GAMMA0_TYPE;
            else if(strncmp(tmpstring,"G1",2)==0)
                type1[ipcorr]=GAMMA1_TYPE;
            else if(strncmp(tmpstring,"G2",2)==0)
                type1[ipcorr]=GAMMA2_TYPE;
            else if(strncmp(tmpstring,"G3",2)==0)
                type1[ipcorr]=GAMMA3_TYPE;
            else if(strncmp(tmpstring,"G5",2)==0)
                type1[ipcorr]=GAMMA5_TYPE;
            
            if(strncmp(tmpstring2,"1",1)==0)
                type2[ipcorr]=ONE_TYPE;
            else if(strncmp(tmpstring2,"G0G1",4)==0)
                type2[ipcorr]=GAMMA0GAMMA1_TYPE;
            else if(strncmp(tmpstring2,"G0G2",4)==0)
                type2[ipcorr]=GAMMA0GAMMA2_TYPE;
            else if(strncmp(tmpstring2,"G0G3",4)==0)
                type2[ipcorr]=GAMMA0GAMMA3_TYPE;
            else if(strncmp(tmpstring2,"G0G5",4)==0)
                type2[ipcorr]=GAMMA0GAMMA5_TYPE;
            else if(strncmp(tmpstring2,"G1G2",4)==0)
                type2[ipcorr]=GAMMA1GAMMA2_TYPE;
            else if(strncmp(tmpstring2,"G1G3",4)==0)
                type2[ipcorr]=GAMMA1GAMMA3_TYPE;
            else if(strncmp(tmpstring2,"G1G5",4)==0)
                type2[ipcorr]=GAMMA1GAMMA5_TYPE;
            else if(strncmp(tmpstring2,"G2G3",4)==0)
                type2[ipcorr]=GAMMA2GAMMA3_TYPE;
            else if(strncmp(tmpstring2,"G2G5",4)==0)
                type2[ipcorr]=GAMMA2GAMMA5_TYPE;
            else if(strncmp(tmpstring2,"G3G5",4)==0)
                type2[ipcorr]=GAMMA3GAMMA5_TYPE;
            else if(strncmp(tmpstring2,"G0",2)==0)
                type2[ipcorr]=GAMMA0_TYPE;
            else if(strncmp(tmpstring2,"G1",2)==0)
                type2[ipcorr]=GAMMA1_TYPE;
            else if(strncmp(tmpstring2,"G2",2)==0)
                type2[ipcorr]=GAMMA2_TYPE;
            else if(strncmp(tmpstring2,"G3",2)==0)
                type2[ipcorr]=GAMMA3_TYPE;
            else if(strncmp(tmpstring2,"G5",2)==0)
                type2[ipcorr]=GAMMA5_TYPE;
            
            error_root((type1[ipcorr]==-1)||(type2[ipcorr]==-1),1,"read_run_parms [rotation.c]",
                        "Unknown or unsupported Dirac structure");
            
            read_line_opt("x0","-1","%d",&x0s[ipcorr]);
            error_root((x0s[ipcorr]<-1)||(x0s[ipcorr]>N0),1,
                        "read_run_parms [rotation.c]","Specified x0 out of range");
            if ((bcon==0)&&(x0s[ipcorr]!=-1))
            {
                error_root(((x0s[ipcorr]-pos*(outlat[0]-1)/2)<0)||
                            x0s[ipcorr]+outlat[0]/(1+pos)-1+pos>N0,1,
                            "read_run_parms [rotation.c]","Combination of x0 and pos out of range");
            }
        }
    }

    MPI_Bcast(kappas,nprop,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(mus,nprop,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(isps,nprop,MPI_INT,0,MPI_COMM_WORLD);

    MPI_Bcast(props1,npcorr,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(props2,npcorr,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(type1,npcorr,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(type2,npcorr,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(x0s,npcorr,MPI_INT,0,MPI_COMM_WORLD);

    set_lat_parms(0.0,1.0,nprop,kappas,0,csw);
    set_sw_parms(sea_quark_mass(0));

    write_lat_parms(fdat);
}


static void read_solvers(void)
{
    solver_parms_t sp;
    int i,j;
    int isap=0,idfl=0;

    for (i=0;i<nprop;i++)
    {
        j=isps[i];
        sp=solver_parms(j);
        if (sp.solver==SOLVERS)
        {
            read_solver_parms(j);
            sp=solver_parms(j);
            if (sp.solver==SAP_GCR)
                isap=1;
            if (sp.solver==DFL_SAP_GCR)
            {
                isap=1;
                idfl=1;
            }
        }
    }

    if (isap)
        read_sap_parms("SAP",0x1);

    if (idfl)
    {
        read_dfl_parms("Deflation subspace");
        read_dfl_pro_parms("Deflation projection");
        read_dfl_gen_parms("Deflation subspace generation");
    }
}


static void read_infile(int argc,char *argv[])
{
    int ifile;

    if (my_rank==0)
    {
        flog=freopen("STARTUP_ERROR","w",stdout);
        ifile=find_opt(argc,argv,"-i");
        endian=endianness();

        error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [rotation.c]",
                "Syntax: rotation -i <input file> [-noexp]");

        error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [rotation.c]",
                "Machine has unknown endianness");

        noexp=find_opt(argc,argv,"-noexp");

        fin=freopen(argv[ifile+1],"r",stdin);
        error_root(fin==NULL,1,"read_infile [rotation.c]",
                "Unable to open input file");
    }

    MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&noexp,1,MPI_INT,0,MPI_COMM_WORLD);

    read_dirs();
    setup_files();

    if (my_rank==0)
    {
        fdat=fopen(par_file,"wb");
        error_root(fdat==NULL,1,"read_infile [rotation.c]",
                    "Unable to open parameter file");
    }

    read_bc_parms("Boundary Terms",0x2);
    bcon=bc_type();
    read_run_parms();
    read_solvers();

    if (my_rank==0)
    {
        fclose(fin);
        fclose(fdat);
    }
}


static void check_files(void)
{
    if (my_rank==0)
    {
        fin=fopen(log_file,"r");
        error_root(fin!=NULL,1,"check_files [rotation.c]",
                    "Attempt to overwrite old *.log file");
        fdat=fopen(head_file,"r");
        error_root(fdat!=NULL,1,"check_files [rotation.c]",
                    "Attempt to overwrite old *.head file");
        fdat=fopen(head_file,"wb");
        error_root(fdat==NULL,1,"check_files [rotation.c]",
                    "Unable to open head file");
        fclose(fdat);
    }
}


static void print_info(void)
{
    int i,isap,idfl;
    long ip;
    lat_parms_t lat;

    if (my_rank==0)
    {
        ip=ftell(flog);
        fclose(flog);

        if (ip==0L)
            remove("STARTUP_ERROR");

        flog=freopen(log_file,"w",stdout);
        error_root(flog==NULL,1,"print_info [rotation.c]",
                "Unable to open log file");
        printf("\n");

        printf("Computation of meson point correlators\n");
        printf("--------------------------------\n\n");
        printf("cnfg   base name: %s\n",nbase);
        printf("output base name: %s\n\n",outbase);

        printf("openQCD version: %s, meson version: %s\n",openQCD_RELEASE,
                                                            mesons_RELEASE);
        if (endian==LITTLE_ENDIAN)
            printf("Machine is little endian\n");
        else
            printf("Machine is big endian\n");
        if (noexp)
            printf("Configurations are read in imported file format\n\n");
        else
            printf("Configurations are read in exported file format\n\n");

        printf("Random number generator:\n");
        printf("level = %d, seed = %d\n\n",level,seed);

        lat=lat_parms();

        printf("Measurements:\n");
        printf("nprop     = %i\n",nprop);
        printf("csw       = %.6f\n",lat.csw);
        if (bc_type()!=3)
        {
            bc_parms_t bc=bc_parms();
            printf("cF        = %.6f\n",bc.cF[0]);
        }
        printf("outlat    = %d %d %d %d\n",outlat[0],outlat[1],outlat[2],outlat[3]);
        printf("source positioning: %s\n\n",(pos==0)?"edge":"center");

        for (i=0; i<nprop; i++)
        {
            printf("Propagator %i:\n",i);
            printf("kappa  = %.6f\n",kappas[i]);
            printf("isp    = %i\n",isps[i]);
            printf("mu     = %.6f\n\n",mus[i]);
        }
        for (i=0; i<npcorr; i++)
        {
            printf("Point correlator %i:\n",i);
            printf("iprop  = %i %i\n",props1[i],props2[i]);
            printf("type   = %i %i\n",type1[i],type2[i]);
            printf("x0     = %i\n\n",x0s[i]);
        }
        print_solver_parms(&isap,&idfl);

        if (isap)
            print_sap_parms(0);

        if (idfl)
            print_dfl_parms(0);

        printf("Configurations no %d -> %d in steps of %d\n\n",
                first,last,step);
        fflush(flog);
    }
}


static void dfl_wsize(int *nws,int *nwv,int *nwvd)
{
    dfl_parms_t dp;
    dfl_pro_parms_t dpr;

    dp=dfl_parms();
    dpr=dfl_pro_parms();

    maxn(nws,dp.Ns+2);
    maxn(nwv,2*dpr.nmx_gcr+3);
    maxn(nwvd,2*dpr.nkv+4);
}




static void make_proplist(void)
{
    int i,j,ipcorr,new_flag;

    /* find the unique time slices */
    proplist.nux0=0;
    proplist.ux0=malloc(N0*sizeof(int));
    error(proplist.ux0==NULL,1,"make_proplist [rotation.c]",
          "Unable to allocate proplist arrays");

    for (ipcorr=0;ipcorr<npcorr;ipcorr++)
    {
        for (i=0;i<proplist.nux0;i++)
        {
            if (x0s[ipcorr]==proplist.ux0[i])
                break;
        }
        if (i==proplist.nux0)
        {
            proplist.ux0[i]=x0s[ipcorr];
            proplist.nux0++;
        }
    }
   
    proplist.nuprop=malloc(proplist.nux0*sizeof(int));
    proplist.uprop=malloc(proplist.nux0*sizeof(int*));
    proplist.uprop_index=malloc(proplist.nux0*sizeof(int*));
    error((proplist.nuprop==NULL)||(proplist.uprop==NULL)||(proplist.uprop_index==NULL)
          ,1,"make_proplist [rotation.c]","Out of memory");
    proplist.nmax=0;

    for (i=0;i<proplist.nux0;i++)
    {
        proplist.nuprop[i]=0;
        proplist.uprop[i]=malloc(nprop*sizeof(int));
        proplist.uprop_index[i]=malloc(nprop*sizeof(int));
        error((proplist.uprop[i]==NULL)||(proplist.uprop_index[i]==0)
                ,1,"make_proplist [rotation.c]","Out of memory");
        for (ipcorr=0;ipcorr<npcorr;ipcorr++)
        {
            if (x0s[ipcorr]==proplist.ux0[i])
            {
                new_flag=1;
                for (j=0;j<proplist.nuprop[i];j++)
                {
                    if (props1[ipcorr]==proplist.uprop[i][j])
                    {
                        new_flag=0;
                        break;
                    }
                }
                if (new_flag)
                {
                    proplist.uprop[i][proplist.nuprop[i]]=props1[ipcorr];
                    proplist.uprop_index[i][props1[ipcorr]]=proplist.nuprop[i];
                    proplist.nuprop[i]++;
                }
                new_flag=1;
                for (j=0;j<proplist.nuprop[i];j++)
                {
                    if (props2[ipcorr]==proplist.uprop[i][j])
                    {
                        new_flag=0;
                        break;
                    }
                }
                if (new_flag)
                {
                    proplist.uprop[i][proplist.nuprop[i]]=props2[ipcorr];
                    proplist.uprop_index[i][props2[ipcorr]]=proplist.nuprop[i];
                    proplist.nuprop[i]++;
                }
            }
        }
        if (proplist.nuprop[i]>proplist.nmax)
            proplist.nmax=proplist.nuprop[i];
    }
}


static void solver_wsize(int isp,int nsds,int np,int *nws,int *nwv,int *nwvd)
{
    solver_parms_t sp;

    sp=solver_parms(isp);

    if (sp.solver==CGNE)
        maxn(nws,nsds+11);
    else if (sp.solver==MSCG)
    {
        if (np>1)
            maxn(nws,nsds+2*np+6);
        else
            maxn(nws,nsds+10);
    }
    else if (sp.solver==SAP_GCR)
        maxn(nws,nsds+2*sp.nkv+5);
    else if (sp.solver==DFL_SAP_GCR)
    {
        maxn(nws,nsds+2*sp.nkv+6);
        dfl_wsize(nws,nwv,nwvd);
    }
}


static void wsize(int *nws,int *nwv,int *nwvd)
{
    int iprop;

    (*nws)=0;
    (*nwv)=0;
    (*nwvd)=0;

    for (iprop=0;iprop<nprop;iprop++)
        solver_wsize(isps[iprop],0,1,nws,nwv,nwvd);
    *nws+=2*(proplist.nmax*12+2);

}

/*******************************************************************************
 * 
 *   void source_pos(int x0, int *src_coords)
 *      Sets the source coordinates src_coords. The time slice x0 is only used
 *      if bcon=0. In that case, if x0<0, the source is placed at the center of
 *      the lattice if pos=1, or at (N0-outlat[0])/2 if pos=0. Otherwise, the
 *      the source is placed at x0. If bcon=3, the source is placed at random.
 * 
 ******************************************************************************/
void source_pos(int x0, int *src_coords)
{
    int err_count=0;
    double rand[4];

    /* check if outlat has been set */
    error_root((outlat[0]==-1)||(outlat[1]==-1)||(outlat[2]==-1)||(outlat[3]==-1),1,
                "source_pos [parallel_out.c]","Output lattice not set");

    if (my_rank==0)
    {
        ranlxd(rand,4);

        if (pos==0)
        {
            if (bcon==0)
            {
                if (x0<0)
                    src_coords[0]=(N0-outlat[0])/2;
                else
                    src_coords[0]=x0;
            }
            else if (bcon==3)
            {
                src_coords[0]=(int)(rand[0]*N0);
            }
            else
            {
                error_root(1,1,"source_pos [parallel_out.c]",
                            "Unknown or unsupported boundary condition");
            }
        }
        else
        {
            if (bcon==0)
            {
                if (x0<0)
                    src_coords[0]=(N0-1)/2;
                else
                    src_coords[0]=x0;
            }
            else if (bcon==3)
            {
                src_coords[0]=(int)(rand[0]*N0);
            }
            else
            {
                error_root(1,1,"source_pos [parallel_out.c]",
                            "Unknown or unsupported boundary condition");
            }
        }
        src_coords[1]=(int)(rand[1]*N1);
        src_coords[2]=(int)(rand[2]*N2);
        src_coords[3]=(int)(rand[3]*N3);
    }
    err_count=MPI_Bcast(src_coords,4,MPI_INT,0,MPI_COMM_WORLD);
    error(err_count!=MPI_SUCCESS,1,"source_pos [parallel_out.c]",
            "Failed to broadcast source coordinates");
}


/*******************************************************************************
 * 
 *   void point_source(spinor_dble *eta, int src_index, int cc)
 *     Sets the global spinor field eta to 1 at the local source index src_index
 *     in the local lattice residing on head_rank and spin-color index cc 
 *     (0:c1.c1, 1:c1.c2,..., 11:c4.c3) and to 0 elsewhere.
 * 
 ******************************************************************************/
static void point_source(spinor_dble *eta, int *src_coords, int cc)
{
    int ix,head_rank;

    /* Check if cc is valid */
    error_root((cc < 0 || cc > 11), 1, "point_source [rotation.c]",
                        "Invalid spin-color index");

    /* find head_rank and local source index ix */
    ipt_global(src_coords,&head_rank,&ix);

    /* set eta to zero */
    set_sd2zero(VOLUME, 0, eta);

    if (my_rank==head_rank)
    {
        switch (cc)
        {
            case 0:
                eta[ix].c1.c1.re=1.0;
                break;
            case 1:
                eta[ix].c1.c2.re=1.0;
                break;
            case 2:
                eta[ix].c1.c3.re=1.0;
                break;
            case 3:
                eta[ix].c2.c1.re=1.0;
                break;
            case 4:
                eta[ix].c2.c2.re=1.0;
                break;
            case 5:
                eta[ix].c2.c3.re=1.0;
                break;
            case 6:
                eta[ix].c3.c1.re=1.0;
                break;
            case 7:
                eta[ix].c3.c2.re=1.0;
                break;
            case 8:
                eta[ix].c3.c3.re=1.0;
                break;
            case 9:
                eta[ix].c4.c1.re=1.0;
                break;
            case 10:
                eta[ix].c4.c2.re=1.0;
                break;
            case 11:
                eta[ix].c4.c3.re=1.0;
                break;
            default:
                error_loc(1, 1, "point_source [rotation.c]",
                            "Invalid spin-color index");
        }
    }
}

/*******************************************************************************
 * 
 *   int choose_cc(int type, int cc, complex_dble *factor)
 *      For a given point source, the inverse Dirac operator is solved for all
 *      12 combinations of spin and color indices. The spin-color index is
 *      represented by a source vector of dimension 12 that is 1 at index cc
 *      and 0 elsewhere.
 *      For a given type of Gamma matrix and spin-color index cc, this function
 *      returns the spin-color index cc that is obtained after the spin-color
 *      source vector is multiplied by
 *      gamma_5 * gamma_0 * Gamma * gamma_0
 *      A possible factor of +-1 or +-i is stored in factor.
 * 
 ******************************************************************************/
static int choose_cc(int type, int cc, complex_dble *factor)
{
    int spin_in,spin_out=-1,color;

    spin_in=cc/3;
    color=cc%3;

    if (spin_in==0)
    {
        switch (type)
        {
            case GAMMA0_TYPE:
                spin_out=2;
                (*factor).re=1.0;
                (*factor).im=0.0;
                break;
            case GAMMA1_TYPE:
                spin_out=3;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA2_TYPE:
                spin_out=3;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA3_TYPE:
                spin_out=2;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA5_TYPE:
                spin_out=0;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case ONE_TYPE:
                spin_out=0;
                (*factor).re=1.0;
                (*factor).im=0.0;
                break;
            case GAMMA0GAMMA1_TYPE:
                spin_out=1;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA0GAMMA2_TYPE:
                spin_out=1;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA0GAMMA3_TYPE:
                spin_out=0;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA0GAMMA5_TYPE:
                spin_out=2;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA1GAMMA2_TYPE:
                spin_out=0;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA1GAMMA3_TYPE:
                spin_out=1;
                (*factor).re=1.0;
                (*factor).im=0.0;
                break;
            case GAMMA1GAMMA5_TYPE:
                spin_out=3;
                (*factor).re=0.0;
                (*factor).im=-1.0;
                break;
            case GAMMA2GAMMA3_TYPE:
                spin_out=1;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA2GAMMA5_TYPE:
                spin_out=3;
                (*factor).re=1.0;
                (*factor).im=0.0;
                break;
            case GAMMA3GAMMA5_TYPE:
                spin_out=2;
                (*factor).re=0.0;
                (*factor).im=-1.0;
                break;
        }
    }
    else if (spin_in==1)
    {
        switch (type)
        {
            case GAMMA0_TYPE:
                spin_out=3;
                (*factor).re=1.0;
                (*factor).im=0.0;
                break;
            case GAMMA1_TYPE:
                spin_out=2;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA2_TYPE:
                spin_out=2;
                (*factor).re=1.0;
                (*factor).im=0.0;
                break;
            case GAMMA3_TYPE:
                spin_out=3;
                (*factor).re=0.0;
                (*factor).im=-1.0;
                break;
            case GAMMA5_TYPE:
                spin_out=1;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case ONE_TYPE:
                spin_out=1;
                (*factor).re=1.0;
                (*factor).im=0.0;
                break;
            case GAMMA0GAMMA1_TYPE:
                spin_out=0;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA0GAMMA2_TYPE:
                spin_out=0;
                (*factor).re=1.0;
                (*factor).im=0.0;
                break;
            case GAMMA0GAMMA3_TYPE:
                spin_out=1;
                (*factor).re=0.0;
                (*factor).im=-1.0;
                break;
            case GAMMA0GAMMA5_TYPE:
                spin_out=3;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA1GAMMA2_TYPE:
                spin_out=1;
                (*factor).re=0.0;
                (*factor).im=-1.0;
                break;
            case GAMMA1GAMMA3_TYPE:
                spin_out=0;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA1GAMMA5_TYPE:
                spin_out=2;
                (*factor).re=0.0;
                (*factor).im=-1.0;
                break;
            case GAMMA2GAMMA3_TYPE:
                spin_out=0;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA2GAMMA5_TYPE:
                spin_out=2;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA3GAMMA5_TYPE:
                spin_out=3;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
        }
    }
    else if (spin_in==2)
    {
        switch (type)
        {
            case GAMMA0_TYPE:
                spin_out=0;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA1_TYPE:
                spin_out=1;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA2_TYPE:
                spin_out=1;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA3_TYPE:
                spin_out=0;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA5_TYPE:
                spin_out=2;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case ONE_TYPE:
                spin_out=2;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA0GAMMA1_TYPE:
                spin_out=3;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA0GAMMA2_TYPE:
                spin_out=3;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA0GAMMA3_TYPE:
                spin_out=2;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA0GAMMA5_TYPE:
                spin_out=0;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA1GAMMA2_TYPE:
                spin_out=2;
                (*factor).re=0.0;
                (*factor).im=-1.0;
                break;
            case GAMMA1GAMMA3_TYPE:
                spin_out=3;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA1GAMMA5_TYPE:
                spin_out=1;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA2GAMMA3_TYPE:
                spin_out=3;
                (*factor).re=0.0;
                (*factor).im=-1.0;
                break;
            case GAMMA2GAMMA5_TYPE:
                spin_out=1;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA3GAMMA5_TYPE:
                spin_out=0;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
        }
    }
    else if (spin_in==3)
    {
        switch (type)
        {
            case GAMMA0_TYPE:
                spin_out=1;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA1_TYPE:
                spin_out=0;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA2_TYPE:
                spin_out=0;
                (*factor).re=1.0;
                (*factor).im=0.0;
                break;
            case GAMMA3_TYPE:
                spin_out=1;
                (*factor).re=0.0;
                (*factor).im=-1.0;
                break;
            case GAMMA5_TYPE:
                spin_out=3;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case ONE_TYPE:
                spin_out=3;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA0GAMMA1_TYPE:
                spin_out=2;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA0GAMMA2_TYPE:
                spin_out=2;
                (*factor).re=1.0;
                (*factor).im=0.0;
                break;
            case GAMMA0GAMMA3_TYPE:
                spin_out=3;
                (*factor).re=0.0;
                (*factor).im=-1.0;
                break;
            case GAMMA0GAMMA5_TYPE:
                spin_out=1;
                (*factor).re=-1.0;
                (*factor).im=0.0;
                break;
            case GAMMA1GAMMA2_TYPE:
                spin_out=3;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA1GAMMA3_TYPE:
                spin_out=2;
                (*factor).re=1.0;
                (*factor).im=0.0;
                break;
            case GAMMA1GAMMA5_TYPE:
                spin_out=0;
                (*factor).re=0.0;
                (*factor).im=1.0;
                break;
            case GAMMA2GAMMA3_TYPE:
                spin_out=2;
                (*factor).re=0.0;
                (*factor).im=-1.0;
                break;
            case GAMMA2GAMMA5_TYPE:
                spin_out=0;
                (*factor).re=1.0;
                (*factor).im=0.0;
                break;
            case GAMMA3GAMMA5_TYPE:
                spin_out=1;
                (*factor).re=0.0;
                (*factor).im=-1.0;
                break;
        }
    }
    else
    {
        error_root(1,1,"choose_source [rotation.c]",
                "Unknown or unsupported spin index");
    }
    if (spin_out==-1)
        error_root(1,1,"choose_source [rotation.c]",
                "Unknown or unsupported type");

    return spin_out*3+color;
}


/* xi = Gamma * eta */
static void make_source(spinor_dble *eta, int type, spinor_dble *xi)
{
    switch (type)
    {
        case GAMMA0_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg0_dble(VOLUME,xi);
            break;
        case GAMMA1_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg1_dble(VOLUME,xi);
            break;
        case GAMMA2_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg2_dble(VOLUME,xi);
            break;
        case GAMMA3_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg3_dble(VOLUME,xi);
            break;
        case GAMMA5_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg5_dble(VOLUME_TRD,2,xi);
            break;
        case ONE_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            break;
        case GAMMA0GAMMA1_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg0g1_dble(VOLUME,xi);
            break;
        case GAMMA0GAMMA2_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg0g2_dble(VOLUME,xi);
            break;
        case GAMMA0GAMMA3_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg0g3_dble(VOLUME,xi);
            break;
        case GAMMA0GAMMA5_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg0g5_dble(VOLUME,xi);
            break;
        case GAMMA1GAMMA2_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg1g2_dble(VOLUME,xi);
            break;
        case GAMMA1GAMMA3_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg1g3_dble(VOLUME,xi);
            break;
        case GAMMA1GAMMA5_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg1g5_dble(VOLUME,xi);
            break;
        case GAMMA2GAMMA3_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg2g3_dble(VOLUME,xi);
            break;
        case GAMMA2GAMMA5_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg2g5_dble(VOLUME,xi);
            break;
        case GAMMA3GAMMA5_TYPE:
            assign_sd2sd(VOLUME_TRD,2,eta,xi);
            mulg3g5_dble(VOLUME,xi);
            break;
        default:
            error_root(1,1,"make_source [rotation.c]",
                    "Unknown or unsupported type");
    }
}


/* xi = gamma_5 eta */
static void make_sink(spinor_dble *eta, spinor_dble *xi)
{
    assign_sd2sd(VOLUME_TRD,2,eta,xi);
    mulg5_dble(VOLUME_TRD,2,xi);
}


static void solve_dirac(int prop, spinor_dble *eta, spinor_dble *psi,
                        int *status)
{
    solver_parms_t sp;
    sap_parms_t sap;
    int ifail[2];

    sp=solver_parms(isps[prop]);
    set_sw_parms(0.5/kappas[prop]-4.0);

    if (sp.solver==CGNE)
    {
        mulg5_dble(VOLUME_TRD,2,eta);

        tmcg(sp.nmx,sp.istop,sp.res,mus[prop],eta,eta,ifail,status);
        if (my_rank==0)
            printf("%i\n",status[0]);
        error_root(ifail[0]<0,1,"solve_dirac [rotation.c]",
                    "CGNE solver failed (ifail[0] = %d)",ifail[0]);

        Dw_dble(-mus[prop],eta,psi);
        mulg5_dble(VOLUME_TRD,2,psi);
    }
    else if (sp.solver==SAP_GCR)
    {
        sap=sap_parms();
        set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

        sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mus[prop],eta,psi,ifail,status);
        if (my_rank==0)
            printf("%i\n",status[0]);
        error_root(ifail[0]<0,1,"solve_dirac [rotation.c]",
                    "SAP_GCR solver failed (ifail[0] = %d)",ifail[0]);
    }
    else if (sp.solver==DFL_SAP_GCR)
    {
        sap=sap_parms();
        set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

        dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mus[prop],eta,psi,ifail,status);
        if (my_rank==0)
            printf("%3i %3i\n",status[0],status[1]);
        error_root((ifail[0]<0),1,
                    "solve_dirac [rotation.c]","DFL_SAP_GCR solver failed "
                    "(ifail[0] = %d)",ifail[0]);
    }
    else
        error_root(1,1,"solve_dirac [rotation.c]",
                    "Unknown or unsupported solver");
}


static void point_correlators(void)
{
    int i,iy,ix0,iprop,ipcorr,cc,stat[6],src_coords[4];
    spinor_dble *source,*sink,**solution,**wsd;
    complex_dble factor={0.0,0.0};
    complex_qflt tmp;

    wsd=reserve_wsd(12*proplist.nmax+2);
    error(wsd==NULL,1,"point_correlators [rotation.c]",
            "Unable to reserve working space");
    source=wsd[0];
    sink=wsd[1];
    solution=malloc(12*proplist.nmax*sizeof(spinor_dble*));
    error(solution==NULL,1,"point_correlators [rotation.c]",
            "Unable to allocate solution array");

    for (i=0;i<12*proplist.nmax;i++)
        solution[i]=wsd[i+2];

    for (i=0;i<npcorr*VOLUME;i++)
    {
        data.corr[i].re=0.0;
        data.corr[i].im=0.0;
        data.corr_out[i].re=0.0;
        data.corr_out[i].im=0.0;
    }

    for (ix0=0;ix0<proplist.nux0;ix0++)
    {
        /* calculate all solutions with ix0=proplist.ux0[ix0] */
        source_pos(proplist.ux0[ix0],src_coords);
        if (my_rank==0)    
            printf("Calculating all 12 Dirac inversions at source position [%d, %d, %d, %d] ...\n",
                    src_coords[0],src_coords[1],src_coords[2],src_coords[3]);
        for (iprop=0;iprop<proplist.nuprop[ix0];iprop++)
        {
            if (my_rank==0)
                printf("... and propagator %d.\nNo   Status\n",proplist.uprop[ix0][iprop]);
            for (cc=0;cc<12;cc++)
            {
                point_source(source,src_coords,cc);
                if (my_rank==0)
                    printf("%2d   ",(cc+1));
                MPI_Barrier(MPI_COMM_WORLD);
                solve_dirac(proplist.uprop[ix0][iprop],source,solution[12*iprop+cc],stat);
                if (my_rank==0)
                    fflush(stdout);
            }
        }
        for (ipcorr=0;ipcorr<npcorr;ipcorr++)
        {
            if (x0s[ipcorr]==proplist.ux0[ix0])
            {
                for (i=0;i<4;i++)
                    srcs[4*ipcorr+i]=src_coords[i];

                for (cc=0;cc<12;cc++)
                {   
                    make_source(solution[12*proplist.uprop_index[ix0][props1[ipcorr]]+cc],
                                type1[ipcorr],source);
                    make_sink(solution[12*proplist.uprop_index[ix0][props2[ipcorr]]
                                        +choose_cc(type2[ipcorr],cc,&factor)],sink);
                    if ((factor.re == 1.0)&&(factor.im == 0.0))
                    {
                        for (i=0;i<VOLUME;i++)
                        {
                            iy=ipt[i];
                            tmp=spinor_prod_dble(1,0,sink+iy,source+iy);
                            data.corr[ipcorr*VOLUME+i].re+=tmp.re.q[0];
                            data.corr[ipcorr*VOLUME+i].im+=tmp.im.q[0];
                        }
                    }
                    else if ((factor.re == -1.0)&&(factor.im == 0.0))
                    {
                        for (i=0;i<VOLUME;i++)
                        {
                            iy=ipt[i];
                            tmp=spinor_prod_dble(1,0,sink+iy,source+iy);
                            data.corr[ipcorr*VOLUME+i].re-=tmp.re.q[0];
                            data.corr[ipcorr*VOLUME+i].im-=tmp.im.q[0];
                        }
                    }
                    /* factors of +i and -i get an extra minus sign
                       due to complex conjugation of first element
                       in complex scalar product */
                    else if ((factor.re == 0.0)&&(factor.im == 1.0))
                    {
                        for (i=0;i<VOLUME;i++)
                        {
                            iy=ipt[i];
                            tmp=spinor_prod_dble(1,0,sink+iy,source+iy);
                            data.corr[ipcorr*VOLUME+i].re+=tmp.im.q[0];
                            data.corr[ipcorr*VOLUME+i].im-=tmp.re.q[0];
                        }
                    }
                    else if ((factor.re == 0.0)&&(factor.im == -1.0))
                    {
                        for (i=0;i<VOLUME;i++)
                        {
                            iy=ipt[i];
                            tmp=spinor_prod_dble(1,0,sink+iy,source+iy);
                            data.corr[ipcorr*VOLUME+i].re-=tmp.im.q[0];
                            data.corr[ipcorr*VOLUME+i].im+=tmp.re.q[0];
                        }
                    }
                    else
                    {
                        error(1,1,"point_correlators [rotation.c]",
                                "Unknown or unsupported factor");
                    }
                }
            }
        }
    }
    free(solution);
    release_wsd();
}


static void set_data(int nc)
{
    data.nc=nc;
    point_correlators();
}


static void init_rng(void)
{
    start_ranlux(level,seed);
}


static void check_endflag(int *iend)
{
   if (my_rank==0)
   {
      fend=fopen(end_file,"r");

      if (fend!=NULL)
      {
         fclose(fend);
         remove(end_file);
         (*iend)=1;
         printf("End flag set, run stopped\n\n");
      }
      else
         (*iend)=0;
   }

   MPI_Bcast(iend,1,MPI_INT,0,MPI_COMM_WORLD);
}


int main(int argc,char *argv[])
{
    int nc,iend,*status;
    int nws,nwv,nwvd;
    double wt1,wt2,wtavg;
    dfl_parms_t dfl;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    read_infile(argc,argv);
    check_machine();
    alloc_data();
    check_files();
    print_info();
    write_head();
    dfl=dfl_parms();

    geometry();
    init_rng();
    set_up_parallel_out(outlat,pos,bcon);

    make_proplist();
    wsize(&nws,&nwv,&nwvd);
    alloc_ws(nws);
    wsd_uses_ws();
    alloc_wv(nwv);
    alloc_wvd(nwvd);
    status=alloc_std_status();

    iend=0;
    wtavg=0.0;

    for (nc=first;(iend==0)&&(nc<=last);nc+=step)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        wt1=MPI_Wtime();

        if (my_rank==0)
            printf("Configuration no %d\n",nc);
        
        if (noexp)
        {
            sprintf(cnfg_file,"%s/%sn%d_%d",loc_dir,nbase,nc,my_rank);
            read_cnfg(cnfg_file);
        }
        else
        {
            sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,nc);
            import_cnfg(cnfg_file,0x0);
            set_ud_phase();
        }

        if (dfl.Ns)
        {
            int ifail;
            dfl_modes(&ifail,status);
            error_root(ifail<0,1,"main [rotation.c]",
                        "Deflation subspace generation failed (status = %d)",
                        ifail);

            if (my_rank==0)
                printf("Deflation subspace generation: status = %d\n",status[0]);
        }

        set_data(nc);
        write_data();

        MPI_Barrier(MPI_COMM_WORLD);
        wt2=MPI_Wtime();
        wtavg+=(wt2-wt1);

        if (my_rank==0)
        {
            printf("Configuration no %d fully processed in %.2e sec ",
                    nc,wt2-wt1);
            printf("(average = %.2e sec)\n\n",
                    wtavg/(double)((nc-first)/step+1));
        }
        check_endflag(&iend);

        if (my_rank==0)
        {
            fflush(flog);
            copy_file(log_file,log_save);
        }
    }

    if (my_rank==0)
    {
        fclose(flog);
    }

    MPI_Finalize();
    exit(0);
}