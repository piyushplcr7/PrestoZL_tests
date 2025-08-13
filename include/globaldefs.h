#ifndef GLOBALDEFSH
#define GLOBALDEFSH

#include "cuda_runtime.h"
#include "cuda_helper.h"
#include <nvtx3/nvToolsExt.h>
#include <pthread.h>
#include <stdbool.h>

typedef struct
{
    int harm_fract;
    int subharmonic_wlo;
    unsigned short *subharmonic_zinds;
    unsigned short *subharmonic_rinds;
    float *subharmonic_powers;
    int subharmonic_numzs;
    int subharmonic_numrs;
    int subharmonic_numrs_fixed;
} SubharmonicMap;

typedef struct
{
    long long index;
    float pow;
    float sig;
} SearchValue;

enum pre_fft_kernel_type {
	PRE_FFT_KERNEL_SHARED_MEM_C2,
	PRE_FFT_KERNEL_SHARED_MEM_C1,
	PRE_FFT_KERNEL_SKMF,
	PRE_FFT_KERNEL_MKSF,
	PRE_FFT_KERNEL_DEFAULT
 };

size_t printGPUUsage();

extern size_t powers_len_total;

extern size_t fkern_size_bytes;
//extern size_t total_powers_size;
extern size_t total_powers_size_without_batchsize;
extern size_t proper_batch_size_global;

extern float* powers_dev_batch;
extern float* powers_dev_batch_all;

extern pthread_t fftw_thread;
extern int max_threads;
extern fcomplex* fkern_cpu_global;
extern fcomplex* fkern_gpu_global;
extern cudaStream_t h2d_memcpy_stream; 
extern cudaStream_t h2d_stream_pdata_dev_batch_all;
extern int** offset_array_global;
extern int** batched_fft_offset_array;

extern pthread_t batched_fft_thread, mem_alloc_thread, mem_free_thread, candidate_refining_thread;

extern int** fftlens;
extern int** kernel_half_widths;
extern fcomplex* batched_fft_thread_buffer;
extern fftwf_plan** fft_plans;

#define NUM_BATCHES_IN_BUFFER 2

extern bool buffer_filled[];

extern pthread_mutex_t* mutex;
extern pthread_cond_t* cond;

extern double* map_startr_array;
extern double* map_lastr_array;

extern enum pre_fft_kernel_type pre_fft_kernel_choice;

#endif