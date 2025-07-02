#ifndef ACCELINCLUDESNOGLIBH
#define ACCELINCLUDESNOGLIBH

#include "presto.h"
#include "accelsearch_cmd.h"

// ACCEL_USELEN must be less than 65536 since we
// use unsigned short ints to index our arrays...
//
// #define ACCEL_USELEN 32000 // This works up to zmax=300 to use 32K FFTs
// #define ACCEL_USELEN 15660 // This works up to zmax=300 to use 16K FFTs
//   The following is probably the best bet for general use given
//   current speeds of FFTs.  However, if you only need to search up
//   to zmax < 100, dropping to 4K FFTs is a few percent faster.  SMR 131110
#define ACCEL_USELEN 7470 // This works up to zmax=300 to use 8K FFTs
// #define ACCEL_USELEN 7960 // This works up to zmax=100 to use 8K FFTs
// #define ACCEL_USELEN 3850 // This works up to zmax=100 to use 4K FFTs
// #define ACCEL_USELEN 1820 // This works up to zmax=100 to use 2K FFTs

/* Stepsize in Fourier Freq */
#define ACCEL_NUMBETWEEN 2
/* Stepsize in Fourier Freq */
#define ACCEL_DR 0.5
/* Reciprocal of ACCEL_DR */
#define ACCEL_RDR 2
/* Stepsize in Fourier F-dot */
#define ACCEL_DZ 2
/* Reciprocal of ACCEL_DZ */
#define ACCEL_RDZ 0.5
/* Stepsize in Fourier F-dot-dot */
#define ACCEL_DW 20
/* Reciprocal of ACCEL_DW */
#define ACCEL_RDW 0.05
/* Closest candidates we will accept as independent */
#define ACCEL_CLOSEST_R 15.0
/* Padding for .dat file reading so that we don't SEGFAULT */
#define ACCEL_PADDING 2000

#define matrix_3d_index(x, y, z, y_len, z_len) ((x * y_len + y) * z_len + z)

typedef struct kernel
{
    int z;               /* The fourier f-dot of the kernel */
    int w;               /* The fourier f-dot-dot of the kernel */
    int fftlen;          /* Number of complex points in the kernel */
    int numgoodbins;     /* The number of good points you can get back */
    int numbetween;      /* Fourier freq resolution (2=interbin) */
    int kern_half_width; /* Half width (bins) of the raw kernel. */
    fcomplex *data;      /* The FFTd kernel itself */
} kernel;

typedef struct subharminfo
{
    int numharm;           /* The number of sub-harmonics */
    int harmnum;           /* The sub-harmonic number (fundamental = numharm) */
    int zmax;              /* The maximum Fourier f-dot for this harmonic */
    int wmax;              /* The maximum Fourier f-dot-dot for this harmonic */
    int numkern_zdim;      /* Number of kernels calculated in the z dimension */
    int numkern_wdim;      /* Number of kernels calculated in the w dimension */
    int numkern;           /* Total number of kernels in the vector */
    kernel **kern;         /* A 2D array of the kernels themselves, with dimensions of z and w */
    unsigned short *rinds; /* Table of lookup indices for Fourier Freqs: subharmonic r values corresponding to "fundamental" r values */
    unsigned short *zinds; /* Table of lookup indices for Fourier F-dots */
} subharminfo;

typedef struct ffdotpows_cu
{
    long long rlo;         /* Lowest Fourier freq present */
    int zlo;               /* Lowest Fourier f-dot present */
    int wlo;               /* Lowest Fourier f-dot-dot present */
    int numrs;             /* Number of Fourier freqs present */
    int numzs;             /* Number of Fourier f-dots present */
    int numws;             /* Number of Fourier f-dot-dots present */
    float *powers;         /* 3D Matrix of the powers */
    unsigned short *rinds; /* Table of lookup indices for Fourier Freqs */
    unsigned short *zinds; /* Table of lookup indices for Fourier f-dots */
} ffdotpows_cu;

#endif
