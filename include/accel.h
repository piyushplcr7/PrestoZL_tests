#ifndef ACCELH
#define ACCELH

#include <glib.h>
#include "presto.h"
#include "accelsearch_cmd.h"
#include "accel_includes_noglib.h"


typedef struct accelobs
{
    long long N;                /* Number of data points in observation */
    long long numbins;          /* Number of spectral bins in the file */
    long long lobin;            /* Lowest spectral bin present in the file */
    long long highestbin;       /* Highest spectral bin present in the file */
    int maxkernlen;             /* Maximum full width (in points, not Fourier bins) of corr kernels */
    int corr_uselen;            /* Number of good data points we will get from high-harm correlations */
    int fftlen;                 /* Length of short FFTs to us in search */
    int numharmstages;          /* Number of stages of harmonic summing */
    int numz;                   /* Number of f-dots searched */
    int numw;                   /* Number of f-dot-dots searched */
    int numbetween;             /* Highest fourier freq resolution (2=interbin) */
    int numzap;                 /* Number of birdies to zap */
    int dat_input;              /* The input file is a short time series */
    int mmap_file;              /* The file number if using MMAP */
    int inmem;                  /* True if we want to keep the full f-fdot plane in RAM */
    int norm_type;              /* 0 = old-style block median, 1 = local-means power norm */
    double dt;                  /* Data sample length (s) */
    double T;                   /* Total observation length */
    double rlo;                 /* Minimum fourier freq to search */
    double rhi;                 /* Maximum fourier freq to search */
    double dr;                  /* Stepsize in fourier freq (1/numbetween) */
    double zlo;                 /* Minimum fourier fdot to search */
    double zhi;                 /* Maximum fourier fdot to search */
    double dz;                  /* Stepsize in fourier fdot */
    double wlo;                 /* Minimum fourier f-dot-dot to search */
    double whi;                 /* Maximum fourier f-dot-dot to search */
    double dw;                  /* Stepsize in fourier f-dot-dot */
    double baryv;               /* Average barycentric velocity during observation */
    float nph;                  /* Freq 0 level if requested, 0 otherwise */
    float sigma;                /* Cutoff sigma to choose a candidate */
    float *powcut;              /* Cutoff powers to choose a cand (per harmsummed) */
    float *ffdotplane;          /* The full f-fdot-fdotdot plane if working in memory */
    double *lobins;             /* The low Fourier freq boundaries to zap (RFI) */
    double *hibins;             /* The high Fourier freq boundaries to zap (RFI) */
    long long *numindep;        /* Number of independent spectra (per harmsummed) */
    FILE *fftfile;              /* The FFT file that we are analyzing */
    FILE *workfile;             /* A text file with candidates as they are found */
    fcomplex *fft;              /* A pointer to the FFT for MMAPing or input time series */
    char *rootfilenm;           /* The root filename for associated files. */
    char *candnm;               /* The fourierprop save file for the fundamentals */
    char *accelnm;              /* The filename of the final candidates in text */
    char *workfilenm;           /* The filename of the working candidates in text */
    int use_harmonic_polishing; /* Should we force harmonics to be related */
} accelobs;

typedef struct accelcand
{
    float power;     /* Summed power level (normalized) */
    float sigma;     /* Equivalent sigma based on numindep (above) */
    int numharm;     /* Number of harmonics summed */
    double r;        /* Fourier freq of first harmonic */
    double z;        /* Fourier f-dot of first harmonic */
    double w;        /* Fourier f-dot-dot of first harmonic */
    double *pows;    /* Optimized powers for the harmonics */
    double *hirs;    /* Optimized freqs for the harmonics */
    double *hizs;    /* Optimized f-dots for the harmonics */
    double *hiws;    /* Optimized f-dot-dots for the harmonics */
    rderivs *derivs; /* An rderivs structure for each harmonic */
} accelcand;

typedef struct ffdotpows
{
    long long rlo;         /* Lowest Fourier freq present */
    int zlo;               /* Lowest Fourier f-dot present */
    int wlo;               /* Lowest Fourier f-dot-dot present */
    int numrs;             /* Number of Fourier freqs present */
    int numzs;             /* Number of Fourier f-dots present */
    int numws;             /* Number of Fourier f-dot-dots present */
    float ***powers;       /* 3D Matrix of the powers */
    unsigned short *rinds; /* Table of lookup indices for Fourier Freqs */
    unsigned short *zinds; /* Table of lookup indices for Fourier f-dots */
} ffdotpows;

/* accel_utils.c */
#ifdef __cplusplus
extern "C" {
#endif

subharminfo **create_subharminfos(accelobs *obs, Cmdline *cmd);
void free_subharminfos(accelobs *obs, subharminfo **shis);
void create_accelobs(accelobs *obs, infodata *idata,
                     Cmdline *cmd, int usemmap);
GSList *sort_accelcands(GSList *list);
GSList *eliminate_harmonics(GSList *cands, int *numcands);
void deredden(fcomplex *fft, int numamps);
void optimize_accelcand(accelcand *cand, accelobs *obs);
void output_fundamentals(fourierprops *props, GSList *list,
                         accelobs *obs, infodata *idata);
void output_harmonics(GSList *list, accelobs *obs, infodata *idata);


void free_accelcand(gpointer data, gpointer user_data);

void print_accelcand(gpointer data, gpointer user_data);
fcomplex *get_fourier_amplitudes(long long lobin, int numbins, accelobs *obs);
ffdotpows *subharm_fderivs_vol(int numharm, int harmnum,
                               double fullrlo, double fullrhi,
                               subharminfo *shi, accelobs *obs);

ffdotpows *copy_ffdotpows(ffdotpows *orig);
void fund_to_ffdotplane(ffdotpows *ffd, accelobs *obs);
void inmem_add_ffdotpows(ffdotpows *fundamental, accelobs *obs,
                         int numharm, int harmnum);
void fund_to_ffdotplane_trans(ffdotpows *ffd, accelobs *obs);
void inmem_add_ffdotpows_trans(ffdotpows *fundamental, accelobs *obs,
                               int numharm, int harmnum);
void free_ffdotpows(ffdotpows *ffd);
void add_ffdotpows_ptrs(ffdotpows *fundamental, ffdotpows *subharmonic,
                        int numharm, int harmnum);
void add_ffdotpows(ffdotpows *fundamental, ffdotpows *subharmonic,
                   int numharm, int harmnum);
GSList *search_ffdotpows(ffdotpows *ffdot, int numharm,
                         accelobs *obs, GSList *cands);
void free_accelobs(accelobs *obs);
int compare(const void *a, const void *b);


void accelsearch_CPU1(int argc, char *argv[], subharminfo ***subharminfs_ptr, accelobs *obs_ptr, infodata *idata_ptr, Cmdline **cmd_ptr);
int accelsearch_GPU(accelobs obs, subharminfo **subharminfs, GSList **cands_ptr, Cmdline *cmd);
void accelsearch_CPU2(GSList **cands, accelobs *obs, infodata *idata, Cmdline *cmd);
#ifdef __cplusplus
}
#endif

#endif