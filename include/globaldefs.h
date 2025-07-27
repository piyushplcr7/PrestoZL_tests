#ifndef GLOBALDEFSH
#define GLOBALDEFSH

typedef struct
{
    int harm_fract;
    int subharmonic_wlo;
    unsigned short *subharmonic_zinds;
    unsigned short *subharmonic_rinds;
    float *subharmonic_powers;
    int subharmonic_numzs;
    int subharmonic_numrs;
} SubharmonicMap;

typedef struct
{
    long long index;
    float pow;
    float sig;
} SearchValue;

size_t printGPUUsage();

extern size_t powers_len_total;

extern size_t fkern_size_bytes;
//extern size_t total_powers_size;
extern size_t total_powers_size_without_batchsize;
extern size_t proper_batch_size_global;

extern float* powers_dev_batch;
extern float* powers_dev_batch_all;


#endif