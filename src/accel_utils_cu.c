/*
 * Copyright (c) 2024 Zhejiang Lab
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#include "accel.h"
#include "cuda_runtime.h"
#include "cuda_helper.h"
#include <sys/time.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include <nvtx3/nvToolsExt.h>

#define NEAREST_INT(x) (int)(x < 0 ? x - 0.5 : x + 0.5)

void do_fft_batch(int fftlen, int binoffset, ffdotpows_cu *ffdot_array, subharminfo *shi, fcomplex *pdata_array, int *idx_array,
    fcomplex *full_tmpdat_array, fcomplex *full_tmpout_array, int batch_size, fcomplex *fkern, cudaStream_t stream, cudaTextureObject_t texObj);

unsigned short **inds_array;

void init_inds_array(int size)
{
    inds_array = (unsigned short **)malloc(size * sizeof(unsigned short *));
    for (int i = 0; i < size; i++)
    {
        inds_array[i] = NULL;
    }
}

void free_inds_array()
{
    free(inds_array);
}

typedef struct
{
    long long index;
    float pow;
    float sig;
} SearchValue;

// Structures to store multiple subharmonics
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

// Comparison function for comparing the index field of two SearchValue structures
int compare(const void *a, const void *b)
{
    const SearchValue *valueA = (const SearchValue *)a;
    const SearchValue *valueB = (const SearchValue *)b;

    if (valueA->index < valueB->index)
    {
        return -1;
    }
    else if (valueA->index > valueB->index)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/* Return 2**n */
#define index_to_twon(n) (1 << n)

float ***gen_f3Darr_cu(long nhgts, long nrows, long ncols, cudaStream_t stream);

kernel **gen_kernmatrix_cu(int numz, int numw);

long long timeInMilliseconds(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (((long long)tv.tv_sec) * 1000) + (tv.tv_usec / 1000);
}

/* Return x such that 2**x = n */
static inline int twon_to_index(int n)
{
    int x = 0;

    while (n > 1)
    {
        n >>= 1;
        x++;
    }
    return x;
}

inline double calc_required_r(double harm_fract, double rfull)
/* Calculate the 'r' you need for subharmonic  */
/* harm_fract = harmnum / numharm if the       */
/* 'r' at the fundamental harmonic is 'rfull'. */
{
    return rint(ACCEL_RDR * rfull * harm_fract) * ACCEL_DR;
}

static inline int calc_required_z(double harm_fract, double zfull)
/* Calculate the 'z' you need for subharmonic  */
/* harm_fract = harmnum / numharm if the       */
/* 'z' at the fundamental harmonic is 'zfull'. */
{
    return NEAREST_INT(ACCEL_RDZ * zfull * harm_fract) * ACCEL_DZ;
}

static inline int calc_required_w(double harm_fract, double wfull)
/* Calculate the maximum 'w' needed for the given subharmonic  */
/* harm_fract = harmnum / numharm if the       */
/* 'w' at the fundamental harmonic is 'wfull'. */
{
    return NEAREST_INT(ACCEL_RDW * wfull * harm_fract) * ACCEL_DW;
}

static inline int index_from_r(double r, double lor)
/* Return an index for a Fourier Freq given an array that */
/* has stepsize ACCEL_DR and low freq 'lor'.              */
{
    return (int)((r - lor) * ACCEL_RDR + DBLCORRECT);
}

static inline int index_from_z(double z, double loz)
/* Return an index for a Fourier Fdot given an array that */
/* has stepsize ACCEL_DZ and low freq dot 'loz'.              */
{
    return (int)((z - loz) * ACCEL_RDZ + DBLCORRECT);
}

static inline int index_from_w(double w, double low)
/* Return an index for a Fourier Fdotdot given an array that */
/* has stepsize ACCEL_DW and low freq dotdot 'low'.              */
{
    return (int)((w - low) * ACCEL_RDW + DBLCORRECT);
}

static fcomplex *gen_cvect_cu(long length)
{
    fcomplex *v;

    CUDA_CHECK(cudaMallocManaged(&v, (size_t)(sizeof(fcomplex) * length), cudaMemAttachGlobal));

    return v;
}

static int calc_fftlen(int numharm, int harmnum, int max_zfull, int max_wfull, accelobs *obs)
/* The fft length needed to properly process a subharmonic */
{
    int bins_needed, end_effects;
    double harm_fract;

    harm_fract = (double)harmnum / (double)numharm;
    bins_needed = (int)ceil(obs->corr_uselen * harm_fract) + 2;
    end_effects = 2 * ACCEL_NUMBETWEEN *
                  w_resp_halfwidth(calc_required_z(harm_fract, max_zfull),
                                   calc_required_w(harm_fract, max_wfull), LOWACC);
    return next_good_fftlen(bins_needed + end_effects);
}

static void init_kernel(int z, int w, int fftlen, kernel *kern)
{
    int numkern;
    fcomplex *tempkern;

    kern->z = z;
    kern->w = w;
    kern->fftlen = fftlen;
    kern->numbetween = ACCEL_NUMBETWEEN;
    kern->kern_half_width = w_resp_halfwidth((double)z, (double)w, LOWACC);
    numkern = 2 * kern->numbetween * kern->kern_half_width;
    kern->numgoodbins = kern->fftlen - numkern;
    kern->data = gen_cvect_cu(kern->fftlen);
    tempkern = gen_w_response(0.0, kern->numbetween, kern->z, kern->w, numkern);
    place_complex_kernel(tempkern, numkern, kern->data, kern->fftlen);
    vect_free(tempkern);
    COMPLEXFFT(kern->data, kern->fftlen, -1);

    // pre fetch kern->data
    CUDA_CHECK(cudaMemPrefetchAsync(kern->data, (size_t)(sizeof(fcomplex) * kern->fftlen), 0, NULL));
}

static void init_subharminfo_cu(int numharm, int harmnum, int zmax, int wmax, subharminfo *shi, accelobs *obs)
/* Note:  'zmax' is the overall maximum 'z' in the search while
          'wmax' is the overall maximum 'w' in the search       */
{
    int ii, jj, fftlen;
    double harm_fract;

    harm_fract = (double)harmnum / (double)numharm;
    shi->numharm = numharm;
    shi->harmnum = harmnum;
    shi->zmax = calc_required_z(harm_fract, zmax);
    shi->wmax = calc_required_w(harm_fract, wmax);
    if (numharm > 1)
    {
        CUDA_CHECK(cudaMallocManaged(&(shi->rinds), (size_t)(obs->corr_uselen * sizeof(unsigned short)), cudaMemAttachGlobal));
        CUDA_CHECK(cudaMallocManaged(&(shi->zinds), (size_t)(obs->corr_uselen * sizeof(unsigned short)), cudaMemAttachGlobal));
    }
    if (numharm == 1 && harmnum == 1)
        fftlen = obs->fftlen;
    else
        fftlen = calc_fftlen(numharm, harmnum, zmax, wmax, obs);
    shi->numkern_zdim = (shi->zmax / ACCEL_DZ) * 2 + 1;
    shi->numkern_wdim = (shi->wmax / ACCEL_DW) * 2 + 1;
    shi->numkern = shi->numkern_zdim * shi->numkern_wdim;
    /* Allocate 2D array of kernels, with dimensions being z and w */
    shi->kern = gen_kernmatrix_cu(shi->numkern_zdim, shi->numkern_wdim);

    /* Actually append kernels to each array element */
    for (ii = 0; ii < shi->numkern_wdim; ii++)
    {
        for (jj = 0; jj < shi->numkern_zdim; jj++)
        {
            init_kernel(-shi->zmax + jj * ACCEL_DZ,
                        -shi->wmax + ii * ACCEL_DW, fftlen, &shi->kern[ii][jj]);
        }
    }
}

float *gen_f3Darr_flat(long nhgts, long nrows, long ncols, cudaStream_t main_stream)
{
    float *c;

    CUDA_CHECK(cudaMallocAsync(&c, (size_t)(nhgts * nrows * ncols * sizeof(float)), main_stream));

    return c;
}

void free_ffdotpows_cu(ffdotpows *ffd,
                       cudaStream_t sub_stream)
{
    CUDA_CHECK(cudaFreeAsync(ffd->powers, sub_stream));
    free(ffd);
}

void free_ffdotpows_cu_batch(ffdotpows_cu *ffd_array, int batch_size,
                             cudaStream_t sub_stream)
{
    CUDA_CHECK(cudaFreeAsync(ffd_array[0].powers, sub_stream));
    free(ffd_array);
}

void free_subharmonic_cu_batch(SubharmonicMap *ffd_array, int batch_size, int num_expand,
                               cudaStream_t sub_stream)
{
    for (int i = 0; i < num_expand; i++)
    {
        SubharmonicMap *ffd = &ffd_array[i * batch_size];
        CUDA_CHECK(cudaFreeAsync(ffd->subharmonic_powers, sub_stream));
        CUDA_CHECK(cudaFreeAsync(inds_array[i], sub_stream));
    }
}

void free_subharminfo_cu(subharminfo *shi)
{
    int ii, jj;

    for (ii = 0; ii < shi->numkern_wdim; ii++)
    {
        for (jj = 0; jj < shi->numkern_zdim; jj++)
        {
            CUDA_CHECK(cudaFree((&shi->kern[ii][jj])->data));
        }
    }

    if (shi->numharm > 1)
    {
        CUDA_CHECK(cudaFree(shi->rinds));
        CUDA_CHECK(cudaFree(shi->zinds));
    }

    CUDA_CHECK(cudaFree(shi->kern[0]));
    CUDA_CHECK(cudaFree(shi->kern));
}

void free_subharminfos_cu(accelobs *obs, subharminfo **shis)
{
    int ii, jj, harmtosum;

    /* Free the sub-harmonics */
    if (!obs->inmem)
    {
        for (ii = 1; ii < obs->numharmstages; ii++)
        {
            harmtosum = index_to_twon(ii);
            for (jj = 1; jj < harmtosum; jj += 2)
            {
                free_subharminfo_cu(&shis[ii][jj - 1]);
            }
            free(shis[ii]);
        }
    }

    /* Free the fundamental */
    free_subharminfo_cu(&shis[0][0]);
    free(shis[0]);
    /* Free the container */
    free(shis);
}

subharminfo **create_subharminfos_cu(accelobs *obs)
{
    double kern_ram_use = 0;
    int ii, jj, harmtosum, fftlen;
    subharminfo **shis;

    shis = (subharminfo **)malloc(obs->numharmstages * sizeof(subharminfo *));
    /* Prep the fundamental (actually, the highest harmonic) */
    shis[0] = (subharminfo *)malloc(2 * sizeof(subharminfo));
    init_subharminfo_cu(1, 1, (int)obs->zhi, (int)obs->whi, &shis[0][0], obs);
    fftlen = obs->fftlen;
    kern_ram_use += shis[0][0].numkern * fftlen * sizeof(fcomplex); // in Bytes
    if (obs->numw)
        printf("  Harm  1/1 : %5d kernels, %4d < z < %-4d and %5d < w < %-5d (%5d pt FFTs)\n",
               shis[0][0].numkern, -shis[0][0].zmax, shis[0][0].zmax,
               -shis[0][0].wmax, shis[0][0].wmax, fftlen);
    else
        printf("  Harm  1/1 : %5d kernels, %4d < z < %-4d (%d pt FFTs)\n",
               shis[0][0].numkern, -shis[0][0].zmax, shis[0][0].zmax, fftlen);
    /* Prep the sub-harmonics if needed */
    if (!obs->inmem)
    {
        for (ii = 1; ii < obs->numharmstages; ii++)
        {
            harmtosum = index_to_twon(ii);
            shis[ii] = (subharminfo *)malloc(harmtosum * sizeof(subharminfo));
            for (jj = 1; jj < harmtosum; jj += 2)
            {
                init_subharminfo_cu(harmtosum, jj, (int)obs->zhi,
                                    (int)obs->whi, &shis[ii][jj - 1], obs);
                fftlen = calc_fftlen(harmtosum, jj, (int)obs->zhi, (int)obs->whi, obs);
                kern_ram_use += shis[ii][jj - 1].numkern * fftlen * sizeof(fcomplex); // in Bytes
                if (obs->numw)
                    printf("  Harm %2d/%-2d: %5d kernels, %4d < z < %-4d and %5d < w < %-5d (%5d pt FFTs)\n",
                           jj, harmtosum, shis[ii][jj - 1].numkern,
                           -shis[ii][jj - 1].zmax, shis[ii][jj - 1].zmax,
                           -shis[ii][jj - 1].wmax, shis[ii][jj - 1].wmax, fftlen);
                else
                    printf("  Harm %2d/%-2d: %5d kernels, %4d < z < %-4d (%d pt FFTs)\n",
                           jj, harmtosum, shis[ii][jj - 1].numkern,
                           -shis[ii][jj - 1].zmax, shis[ii][jj - 1].zmax, fftlen);
            }
        }
    }
    printf("Total RAM used by correlation kernels:  %.3f GB\n", kern_ram_use / (1 << 30));
    return shis;
}

void deep_copy_ffdotpows_cpu2cu(ffdotpows_cu *ffdot, subharminfo *shi, int corr_uselen, unsigned short *inds, int b, int batch_size, cudaStream_t stream)
{
    // alloc and copy rinds
    ffdot->rinds = &inds[b * corr_uselen];
    CUDA_CHECK(cudaMemcpyAsync(ffdot->rinds, shi->rinds, corr_uselen * sizeof(unsigned short), cudaMemcpyHostToDevice, stream));
    // alloc and copy zinds
    ffdot->zinds = &inds[batch_size * corr_uselen + b * corr_uselen];
    CUDA_CHECK(cudaMemcpyAsync(ffdot->zinds, shi->zinds, corr_uselen * sizeof(unsigned short), cudaMemcpyHostToDevice, stream));
}

void deep_copy_ffdotpows_cpu2cu_modified(ffdotpows_cu *ffdot, unsigned short **rinds, 
                                         unsigned short **zinds, int corr_uselen, unsigned short *inds, 
                                         int b, int batch_size, cudaStream_t stream,
                                        int numharm, int harmnum)
{
    // alloc and copy rinds
    ffdot->rinds = &inds[b * corr_uselen];
    char message[256];
    sprintf(message, "cpy rinds (H2D) %d/%d, b = %d", harmnum, numharm, b);
    nvtxRangePush(message);
    //CUDA_CHECK(cudaMemcpyAsync(ffdot->rinds, rinds[b], corr_uselen * sizeof(unsigned short), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(&inds[b * corr_uselen], rinds[b], corr_uselen * sizeof(unsigned short), cudaMemcpyHostToDevice, stream));
    nvtxRangePop();
    //CUDA_CHECK(cudaStreamSynchronize(stream));
    // alloc and copy zinds
    ffdot->zinds = &inds[batch_size * corr_uselen + b * corr_uselen];
    sprintf(message, "cpy zinds (H2D) %d/%d, b = %d", harmnum, numharm, b);
    nvtxRangePush(message);
    //CUDA_CHECK(cudaMemcpyAsync(ffdot->zinds, zinds[b], corr_uselen * sizeof(unsigned short), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(&inds[batch_size * corr_uselen + b * corr_uselen], zinds[b], corr_uselen * sizeof(unsigned short), cudaMemcpyHostToDevice, stream));
    nvtxRangePop();
    //CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Function that copies all the kernels from host to device
// Two options: 
// 1. Use one main buffer and an alternative one to fill kernel data and move it to the gpu. 
//    The copy to the buffer is serialized but the movement to the gpu can be done concurrently.
//    Advantage: hiding latency of copying
// 2. Use several buffers which are filled in parallel. This is harder to achieve because of the 
//    nature of the data we have. If it can be done, we can do the copy operations which would be 
//    serialized.
//    Advantage: copy into a buffer in a parallel way.
fcomplex *fkern_host_to_dev_modified(subharminfo **subharminfs, int numharmstages, int **offset_array)
{
    size_t fkern_size = 0;
    int harm, harmtosum, stage;
    int numkern, fftlen;
    int offset_base = 0;
    size_t offset_tmp;
    fcomplex *fkern_cpu;
    fcomplex* fkern_cpu_buffer_pointers[2];
    cudaEvent_t buffer_copy_events[2];
    cudaEvent_t allocated_gpu_mem_event;
    CUDA_CHECK(cudaEventCreate(&buffer_copy_events[0]));
    CUDA_CHECK(cudaEventCreate(&buffer_copy_events[1]));
    CUDA_CHECK(cudaEventCreate(&allocated_gpu_mem_event));

    fcomplex *fkern_gpu;
    cudaStream_t kern_copy_streams[2];
    //cudaStream_t gpu_mem_allocation_stream;
    CUDA_CHECK(cudaStreamCreate(&kern_copy_streams[0]));
    CUDA_CHECK(cudaStreamCreate(&kern_copy_streams[1]));
    //CUDA_CHECK(cudaStreamCreate(&gpu_mem_allocation_stream));

    for (stage = 0; stage < numharmstages; stage++)
    {
        harmtosum = (stage == 0 ? 2 : 1 << stage);
        for (harm = 1; harm < harmtosum; harm += 2)
        {
            numkern = subharminfs[stage][harm - 1].numkern;
            fftlen = subharminfs[stage][harm - 1].kern[0][0].fftlen;
            offset_array[stage][harm - 1] = fkern_size;
            fkern_size = fkern_size + numkern * fftlen;
        }
    }


    CUDA_CHECK(cudaMallocAsync((void **)&fkern_gpu, sizeof(fcomplex) * fkern_size, kern_copy_streams[0]));
    CUDA_CHECK(cudaEventRecord(buffer_copy_events[0], kern_copy_streams[0]));

    //fkern_cpu = (fcomplex *)malloc(sizeof(fcomplex) * fkern_size);
    size_t sizeAllKernels = sizeof(fcomplex) * fkern_size;
    size_t trial_size_bytes = sizeAllKernels;
    trial_size_bytes = 1 << 30;

    cudaError_t mallocHostErr = cudaMallocHost((void **)&fkern_cpu_buffer_pointers[0], trial_size_bytes); 


    while (mallocHostErr != cudaSuccess) {
        printf("Problem with allocating pinned memory. Trying a smaller size\n");
        trial_size_bytes /= 2;
        mallocHostErr = cudaMallocHost((void **)&fkern_cpu_buffer_pointers[0], trial_size_bytes);
    }

    printf("Found a size that works: %ld\n", trial_size_bytes);

    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        printf("cudaGetLastError returned an error: %s after successfully finishing cudaMallocHost. Resetting the error state.\n", cudaGetErrorString(err1));
    }

    int data_written_at_once = 2 * sizeof(rawtype_part);
    assert(data_written_at_once == sizeof(fcomplex));

    if (trial_size_bytes%data_written_at_once != 0) {
        printf("trial size doesn't match with data written at once!\n");
        exit(1);
    }

    // Additional buffer
    CUDA_CHECK(cudaMallocHost((void **)&fkern_cpu_buffer_pointers[1], trial_size_bytes));
    printf("Allocated additional buffer\n");

    size_t accumulated = 0;
    int current_buffer_idx = 0;
    fkern_cpu = fkern_cpu_buffer_pointers[0];

    fcomplex *fkern_gpu_current = fkern_gpu;
    size_t trial_size_in_fcomplex = trial_size_bytes / sizeof(fcomplex);

    printf("trial size in fcomplex = %ld\n", trial_size_in_fcomplex);
    printf("Beginning to go over kernels\n");

    for (stage = 0; stage < numharmstages; stage++)
    {
        harmtosum = (stage == 0 ? 2 : 1 << stage);
        //printf("In stage %d, harmtosum = %d\n", stage, harmtosum);
        for (harm = 1; harm < harmtosum; harm += 2)
        {
            //printf("In harm = %d\n", harm);

            offset_base = offset_array[stage][harm - 1];

            fftlen = subharminfs[stage][harm - 1].kern[0][0].fftlen;
            numkern = subharminfs[stage][harm - 1].numkern;

            for (int ii = 0; ii < numkern; ii++)
            {
                //printf("In kernel no. ii = %d\n", ii);
                for (int jj = 0; jj < fftlen; jj++)
     
                {
                    //printf("In entry no. %d\n", jj);
                    offset_tmp = offset_base + fftlen * ii + jj;

                    // Filling the current buffer. Need to wait for the previous copy to be finished before modifying the buffer
                    if (accumulated == 0) {
                        nvtxRangePush("waiting for buffer copy event to be finished");
                        CUDA_CHECK(cudaEventSynchronize(buffer_copy_events[current_buffer_idx]));
                        nvtxRangePop();
                    }

                    //printf("offset_tmp = %d\n", offset_tmp);
                    fkern_cpu[offset_tmp%trial_size_in_fcomplex].r = subharminfs[stage][harm - 1].kern[0][ii].data[jj].r;
                    fkern_cpu[offset_tmp%trial_size_in_fcomplex].i = subharminfs[stage][harm - 1].kern[0][ii].data[jj].i;
                    //printf("written at the offset tmp / trial_size_in_fcomplex value mentioned = %ld\n", offset_tmp%trial_size_in_fcomplex);

                    accumulated += 2 * sizeof(rawtype_part);

                    if (accumulated == trial_size_bytes) {
                        //printf("beginning to write accumulated data at jj = %d\n",jj);
                        CUDA_CHECK(cudaMemcpyAsync(fkern_gpu_current, fkern_cpu, accumulated, cudaMemcpyHostToDevice, kern_copy_streams[current_buffer_idx]));
                        // Event corresponding to finished copy operation
                        CUDA_CHECK(cudaEventRecord(buffer_copy_events[current_buffer_idx], kern_copy_streams[current_buffer_idx]));
                        accumulated = 0;
                        fkern_gpu_current = fkern_gpu_current + trial_size_in_fcomplex;

                        // move the fkern_cpu pointer
                        current_buffer_idx = (current_buffer_idx + 1)%2;
                        fkern_cpu = fkern_cpu_buffer_pointers[current_buffer_idx];
                    }
                }
            }
        }
    }

    printf("Checking if there is a remaining unwritten chunk of data\n");
    if (accumulated > 0) {
        printf("Writing remaining chunk of data\n");
        CUDA_CHECK(cudaMemcpy(fkern_gpu_current, fkern_cpu, accumulated, cudaMemcpyHostToDevice));
    }
    printf("Moved everything to GPU\n");

    /* // Copy the data back to cpu for comparison
    fcomplex* fullSizeCpuBuffer = (fcomplex *)malloc(sizeof(fcomplex) * fkern_size);
    CUDA_CHECK(cudaMemcpy(fullSizeCpuBuffer, fkern_gpu, sizeof(fcomplex) * fkern_size, cudaMemcpyDeviceToHost));

    // Generating the data on CPU in a contiguous fashion
    fcomplex *data = (fcomplex *) malloc(sizeof(fcomplex) * fkern_size);
    for (stage = 0; stage < numharmstages; stage++)
    {
        harmtosum = (stage == 0 ? 2 : 1 << stage);
        for (harm = 1; harm < harmtosum; harm += 2)
        {

            offset_base = offset_array[stage][harm - 1];

            fftlen = subharminfs[stage][harm - 1].kern[0][0].fftlen;
            numkern = subharminfs[stage][harm - 1].numkern;

            for (int ii = 0; ii < numkern; ii++)
            {
                for (int jj = 0; jj < fftlen; jj++)
                {
                    offset_tmp = offset_base + fftlen * ii + jj;
                    data[offset_tmp].r = subharminfs[stage][harm - 1].kern[0][ii].data[jj].r;
                    data[offset_tmp].i = subharminfs[stage][harm - 1].kern[0][ii].data[jj].i;
                }
            }
        }
    }

    // Comparing the data
    for (int i = 0 ; i < fkern_size ; ++i) {
        if (fabs(data[i].r - fullSizeCpuBuffer[i].r) > 1e-6 ||  fabs(data[i].i - fullSizeCpuBuffer[i].i) > 1e-6) {
            printf("Mismatch at %d\n",i);
        }
    } */

    CUDA_CHECK(cudaFreeHost(fkern_cpu_buffer_pointers[0]));
    CUDA_CHECK(cudaFreeHost(fkern_cpu_buffer_pointers[1]));

    CUDA_CHECK(cudaEventDestroy(buffer_copy_events[0]));
    CUDA_CHECK(cudaEventDestroy(buffer_copy_events[1]));
    CUDA_CHECK(cudaStreamDestroy(kern_copy_streams[0]));
    CUDA_CHECK(cudaStreamDestroy(kern_copy_streams[1]));
    /* free(fullSizeCpuBuffer);
    free(data); */

    //exit(1);

    return fkern_gpu;
}

fcomplex *fkern_host_to_dev(subharminfo **subharminfs, int numharmstages, int **offset_array)
{
    int fkern_size = 0;
    int harm, harmtosum, stage;
    int numkern, fftlen;
    int offset_base = 0, offset_tmp;
    fcomplex *fkern_cpu;
    fcomplex *fkern_gpu;
    /* struct timespec start_cpu, end_cpu;
    float elapsed=0.123456789;
    // Create cuda events for timing 
    cudaEvent_t start,stop, start1, stop1;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1)); */

    for (stage = 0; stage < numharmstages; stage++)
    {
        harmtosum = (stage == 0 ? 2 : 1 << stage);
        for (harm = 1; harm < harmtosum; harm += 2)
        {
            numkern = subharminfs[stage][harm - 1].numkern;
            fftlen = subharminfs[stage][harm - 1].kern[0][0].fftlen;
            offset_array[stage][harm - 1] = fkern_size;
            fkern_size = fkern_size + numkern * fftlen;
        }
    }

    fkern_cpu = (fcomplex *)malloc(sizeof(fcomplex) * fkern_size);

    /* clock_gettime(CLOCK_MONOTONIC, &start_cpu);
    CUDA_CHECK(cudaEventRecord(start,0)); */
    CUDA_CHECK(cudaMalloc((void **)&fkern_gpu, sizeof(fcomplex) * fkern_size));
    double bytes_in_GB = (double)(1<<30);
    printf("size of all kernels = %f GB\n", sizeof(fcomplex) * fkern_size/bytes_in_GB); 
    /* CUDA_CHECK(cudaEventRecord(stop,0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop)); */

    /* printf("fkern_host_to_dev: cudaMalloc (CUDA) : %.9f seconds\n", elapsed/1000.0f);

    clock_gettime(CLOCK_MONOTONIC, &end_cpu);
    double elapsed_cpu = (end_cpu.tv_sec - start_cpu.tv_sec) +
                     (end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e9;

    printf("fkern_host_to_dev: cudaMalloc : %.9f seconds\n", elapsed_cpu); */

    for (stage = 0; stage < numharmstages; stage++)
    {
        harmtosum = (stage == 0 ? 2 : 1 << stage);
        for (harm = 1; harm < harmtosum; harm += 2)
        {

            offset_base = offset_array[stage][harm - 1];

            fftlen = subharminfs[stage][harm - 1].kern[0][0].fftlen;
            numkern = subharminfs[stage][harm - 1].numkern;

            for (int ii = 0; ii < numkern; ii++)
            {
                for (int jj = 0; jj < fftlen; jj++)
                {
                    offset_tmp = offset_base + fftlen * ii + jj;
                    fkern_cpu[offset_tmp].r = subharminfs[stage][harm - 1].kern[0][ii].data[jj].r;
                    fkern_cpu[offset_tmp].i = subharminfs[stage][harm - 1].kern[0][ii].data[jj].i;
                }
            }
        }
    }

    /* clock_gettime(CLOCK_MONOTONIC, &start_cpu);
    CUDA_CHECK(cudaEventRecord(start1,0)); */
    CUDA_CHECK(cudaMemcpy(fkern_gpu, fkern_cpu, sizeof(fcomplex) * fkern_size, cudaMemcpyHostToDevice));
    /* CUDA_CHECK(cudaEventRecord(stop1,0));
    CUDA_CHECK(cudaEventSynchronize(stop1));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start1, stop1));

    printf("fkern_host_to_dev: cudaMemcpy (CUDA): %.9f seconds\n", elapsed/1000.0f);

    clock_gettime(CLOCK_MONOTONIC, &end_cpu);
    elapsed_cpu = (end_cpu.tv_sec - start_cpu.tv_sec) +
                     (end_cpu.tv_nsec - start_cpu.tv_nsec) / 1e9;

    printf("fkern_host_to_dev: cudaMemcpy : %.9f seconds\n", elapsed_cpu); */

    free(fkern_cpu);

    return fkern_gpu;
}

// batch create subharmonics
size_t subharm_fderivs_vol_cu_batch(
    ffdotpows_cu *ffdot_array,
    int numharm,
    int harmnum,
    double *fullrlo_array, // store batch fullrlo
    double *fullrhi_array, // store batch fullrhi
    subharminfo *shi,      // store batch subharminfo
    accelobs *obs,         // store batch accelobs
    cudaStream_t stream,
    fcomplex *full_tmpdat_array,
    fcomplex *full_tmpout_array,
    int batch_size,
    fcomplex *fkern,
    int inds_idx,
    fcomplex* pdata_dev,
    unsigned short* rinds_all,
    unsigned short* zinds_all,
    cudaEvent_t rzinds_copy_finished,
    cudaStream_t some_stream,
    fcomplex* pdata_all_pinned,
    cudaEvent_t pdata_copy_finished,
    cudaStream_t pdata_stream) // size of batch
{
    // local variables needed
    int ii, numdata, fftlen, binoffset;
    float powargr, powargi;
    double harm_fract;
    binoffset = shi->kern[0][0].kern_half_width;

    int max_threads = omp_get_max_threads();

    // prepare pdata_dev
    //fcomplex *pdata_dev;
    
    fftlen = shi->kern[0][0].fftlen;
    numdata = fftlen / ACCEL_NUMBETWEEN;
    //CUDA_CHECK(cudaMallocAsync(&pdata_dev, (size_t)(sizeof(fcomplex) * fftlen * batch_size), stream));

    if (!(numharm == 1 && harmnum == 1))
    {
        // Size 2X to hold both rinds and zinds
        CUDA_CHECK(cudaMallocAsync(&(inds_array[inds_idx]), (size_t)(obs->corr_uselen * sizeof(unsigned short) * batch_size * 2), stream));
        //CUDA_CHECK(cudaMemsetAsync(inds_array[inds_idx], 0, (size_t)(obs->corr_uselen * sizeof(unsigned short) * batch_size * 2), stream));
    }

    // Create a common FFTW plan to be executed by all the threads
    fftwf_complex *dummy = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fftlen);
    if (!dummy) {
        fprintf(stderr, "Failed to allocate dummy buffer\n");
        exit(1);
    }

    // Create a single shared FFTW plan (1D, forward transform, in-place)
    fftwf_plan shared_plan = fftwf_plan_dft_1d(
        fftlen,      // FFT size
        dummy,         // input array
        dummy,         // output array (same as input for in-place)
        FFTW_FORWARD,  // direction
        FFTW_MEASURE   // planning strategy, computes optimal strategy. NEEDS TO BE CACHED!!!!!
    );

    harm_fract = (double)harmnum / (double)numharm;

    unsigned short** rinds = (unsigned short**) malloc(batch_size * sizeof(unsigned short*));
    unsigned short** zinds = (unsigned short**) malloc(batch_size * sizeof(unsigned short*));

    bool writtenfftouts = false;

    #define ALLOCATE_PDATA_ALL_ONCE

    #define RZINDS_PINNED_MEM

    //#define PINNED_PDATA_ALL

    #ifdef ALLOCATE_PDATA_ALL_ONCE

    #ifdef PINNED_PDATA_ALL
    fcomplex* pdata_all = pdata_all_pinned;
    #else
    //fcomplex* pdata_all = gen_cvect(fftlen * batch_size);
    // TEST:
    fcomplex* pdata_all =  pdata_all_pinned;
    #endif

    #endif

    unsigned short* inds_gpu = inds_array[inds_idx];

    // Wait for the copy from rinds_all to be finished from the previous invocation of this function
    // This ensures that we don't overwrite this only buffer while it is being copied to GPU
    nvtxRangePush("rzinds_copy_finished event synchronize");
    CUDA_CHECK(cudaEventSynchronize(rzinds_copy_finished));
    nvtxRangePop();

    #ifdef PINNED_PDATA_ALL
    // Similarly wait for copy from pdata_all to be finished from the prev invocation
    CUDA_CHECK(cudaEventSynchronize(pdata_copy_finished));
    #endif

    // loop through each batch
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch_size; b++)
    {
        double fullrlo = fullrlo_array[b];
        double fullrhi = fullrhi_array[b];

        //printf("fullrlo for b = %d : %f\n", b, fullrlo);

        // ffdot is thread local
        ffdotpows_cu *ffdot = &ffdot_array[b];
        /* Calculate and get the required amplitudes */
        
        double drlo = calc_required_r(harm_fract, fullrlo);
        double drhi = calc_required_r(harm_fract, fullrhi);
        ffdot->rlo = (long long)floor(drlo);
        ffdot->zlo = calc_required_z(harm_fract, obs->zlo);
        ffdot->wlo = calc_required_w(harm_fract, obs->wlo);

        /* Initialize the lookup indices */
        // Calculated for each chunk, but is it really necessary? can it be done just once?
        if (numharm > 1 && !obs->inmem)
        {
            
            // Allocate these arrays only if processing something other than fundamental!
            #ifdef RZINDS_PINNED_MEM
            // Instead of allocating, use the supplied pinned memory
            rinds[b] = rinds_all + b * obs->corr_uselen;
            zinds[b] = zinds_all + b * obs->corr_uselen;
            #else
            rinds[b] = (unsigned short*) malloc(obs->corr_uselen * sizeof(unsigned short));
            zinds[b] = (unsigned short*) malloc(obs->corr_uselen * sizeof(unsigned short));
            #endif

            for (ii = 0; ii < obs->corr_uselen; ii++)
            {
                double rr = fullrlo + ii * ACCEL_DR;
                double subr = calc_required_r(harm_fract, rr);
                //shi->rinds[ii] = index_from_r(subr, ffdot->rlo);
                rinds[b][ii] = index_from_r(subr, ffdot->rlo);
            }

            //printf("corr_uselen = %d, numz = %d\n", obs->corr_uselen, obs->numz);
            for (ii = 0; ii < obs->numz; ii++)
            {
                double zz = obs->zlo + ii * ACCEL_DZ;
                double subz = calc_required_z(harm_fract, zz);
                //shi->zinds[ii] = index_from_z(subz, ffdot->zlo);
                zinds[b][ii] = index_from_z(subz, ffdot->zlo);
            }
        }

        // Setting numrs value
        // The +1 below is important!
        ffdot->numrs = (int)((ceil(drhi) - floor(drlo)) * ACCEL_RDR + DBLCORRECT) + 1;
        if (numharm == 1 && harmnum == 1)
        {
            ffdot->numrs = obs->corr_uselen;
        }
        else
        {
            if (ffdot->numrs % ACCEL_RDR)
                ffdot->numrs = (ffdot->numrs / ACCEL_RDR + 1) * ACCEL_RDR;
        }

        // Copying rinds, zinds to device
        ffdot->numzs = shi->numkern_zdim;
        ffdot->numws = shi->numkern_wdim;
        if (numharm == 1 && harmnum == 1)
        {
            //ffdot->rinds = shi->rinds;
            //ffdot->zinds = shi->zinds;

            // Null pointers in the fundamental case! 
            ffdot->rinds = rinds[b];
            ffdot->zinds = zinds[b];
        }
        else
        {
            // Deep copy function never called for fundamental!
            //deep_copy_ffdotpows_cpu2cu(ffdot, shi, obs->corr_uselen, (inds_array[inds_idx]), b, batch_size, stream);

            // Commented out to do some work here outside the function and copy the whole thing at once
            //deep_copy_ffdotpows_cpu2cu_modified(ffdot, rinds, zinds, obs->corr_uselen, inds_array[inds_idx], b, batch_size, stream, numharm, harmnum);
            ffdot->rinds = &inds_gpu[b * obs->corr_uselen];
            ffdot->zinds = &inds_gpu[batch_size * obs->corr_uselen + b * obs->corr_uselen];
        }


        /* Determine the largest kernel halfwidth needed to analyze the current subharmonic */
        /* Verified numerically that, as long as we have symmetric z's and w's, */
        /* shi->kern[0][0].kern_half_width is the maximal halfwidth over the range of w's and z's */
        long long lobin = ffdot->rlo - binoffset;
        fcomplex *data = get_fourier_amplitudes(lobin, numdata, obs);
        // Create the output power array

        // Normalize the Fourier amplitudes
        if (obs->nph > 0.0)
        {
            //  Use freq 0 normalization if requested (i.e. photons)
            double norm = 1.0 / sqrt(obs->nph);
            for (ii = 0; ii < numdata; ii++)
            {
                data[ii].r *= norm;
                data[ii].i *= norm;
            }
        }
        else if (obs->norm_type == 0)
        {
            // default block median normalization
            float *powers = gen_fvect(numdata);
            for (ii = 0; ii < numdata; ii++)
                powers[ii] = POWER(data[ii].r, data[ii].i);
            double norm = 1.0 / sqrt(median(powers, numdata) / log(2.0));
            vect_free(powers);
            for (ii = 0; ii < numdata; ii++)
            {
                data[ii].r *= norm;
                data[ii].i *= norm;
            }
        }
        else
        {
            // optional running double-tophat local-power normalization
            float *powers, *loc_powers;
            powers = gen_fvect(numdata);
            for (ii = 0; ii < numdata; ii++)
            {
                powers[ii] = POWER(data[ii].r, data[ii].i);
            }
            loc_powers = corr_loc_pow(powers, numdata);
            for (ii = 0; ii < numdata; ii++)
            {
                float norm = invsqrtf(loc_powers[ii]);
                data[ii].r *= norm;
                data[ii].i *= norm;
            }
            vect_free(powers);
            vect_free(loc_powers);
        }

        // start writing on pdata only after the copy has finished
        nvtxRangePush("pdata_copy_finished event synchronize");
        CUDA_CHECK(cudaEventSynchronize(pdata_copy_finished));
        nvtxRangePop();
        // Prep, spread, and FFT the data
        #ifdef ALLOCATE_PDATA_ALL_ONCE
        fcomplex *pdata = &pdata_all[b*fftlen];
        #else
        fcomplex *pdata = gen_cvect(fftlen);
        #endif

        spread_no_pad(data, fftlen / ACCEL_NUMBETWEEN, pdata, fftlen, ACCEL_NUMBETWEEN);

        // Note COMPLEXFFT is not thread-safe because of wisdom caching
        if (max_threads == 1) {
            //fftwcallsimple(pdata, fftlen, -1);
            COMPLEXFFT(pdata, fftlen, -1);
            //fftwf_execute_dft(shared_plan, pdata, pdata);
        }
        else {

            fftwf_execute_dft(shared_plan, (fftwf_complex*)pdata, (fftwf_complex*)pdata);
        }

        // Writing the FFT data for comparison
        /* if (numharm == 8 && harmnum == 3 ) {
            printf("Writing sgjkzz output of size %ld for batch item %d\n",fftlen,b);
            char fftoutfilename[256];
            if (max_threads == 1) {
                sprintf(fftoutfilename, "COMPLEXFFT_%d",b);
            }
            else {
                sprintf(fftoutfilename, "fftw_execute_dft_%d",b);
            }
            FILE* fftoutfile = fopen(fftoutfilename, "wb");

            if (fftoutfile == NULL) {
                printf("Error in writing file!\n");
                exit(1);
            }

            size_t written = fwrite(pdata, sizeof(fcomplex), fftlen, fftoutfile);

            if (written != fftlen) {
                printf("Error in writing to file\n");
                exit(1);
            }
            writtenfftouts = true;
            fclose(fftoutfile);
        } */

        // copy pdata to pdata_dev, host to device
        #ifndef ALLOCATE_PDATA_ALL_ONCE
        nvtxRangePush("Copy pdata_dev chunk (H2D) ");
        CUDA_CHECK(cudaMemcpyAsync(&pdata_dev[b * fftlen], pdata, (size_t)(sizeof(fcomplex) * fftlen), cudaMemcpyHostToDevice, stream));
        nvtxRangePop();
        vect_free(pdata);
        #endif

        //printf("shiboo\n");
        vect_free(data);
        
        #ifndef RZINDS_PINNED_MEM
        if (numharm > 1) {
            free(rinds[b]);
            free(zinds[b]);
        }
        #endif
    }
    
    // Copy all indices at once as the buffer is continuous, holding both rinds and zinds
    if (!(numharm == 1 && harmnum == 1))
    {
        nvtxRangePush("copying rzinds (H2D)");
        // The event recorded has to be on the same stream as 
        CUDA_CHECK(cudaMemcpyAsync(inds_gpu, rinds_all, obs->corr_uselen * sizeof(unsigned short) * 2 * batch_size, cudaMemcpyHostToDevice, some_stream));
        nvtxRangePop();
        CUDA_CHECK(cudaEventRecord(rzinds_copy_finished, some_stream));
        //CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    //printf("Copy all the pdata at once instead of doing it in a loop over b! \n");
    // Copy all the pdata at once instead of doing it in a loop over b!
    #ifdef ALLOCATE_PDATA_ALL_ONCE
    char message[256];
    size_t pdata_all_size = (size_t) sizeof(fcomplex) * fftlen * batch_size;
    sprintf(message, "cpy pdata_all (H2D) %d/%d, size = %ld", harmnum, numharm, pdata_all_size);
    nvtxRangePush(message);
    #ifdef PINNED_PDATA_ALL
    CUDA_CHECK(cudaMemcpyAsync(pdata_dev, pdata_all, 
        pdata_all_size, 
        cudaMemcpyHostToDevice, pdata_stream));
    CUDA_CHECK(cudaEventRecord(pdata_copy_finished, pdata_stream));
    #else
    CUDA_CHECK(cudaMemcpyAsync(pdata_dev, pdata_all, 
        pdata_all_size, 
        cudaMemcpyHostToDevice, stream));
    // TEST:
    CUDA_CHECK(cudaEventRecord(pdata_copy_finished, stream));
    #endif
    nvtxRangePop();
    
    #ifndef PINNED_PDATA_ALL
    // TEST:
    //vect_free(pdata_all);
    #endif

    #endif
    /* if (writtenfftouts) {
        exit(1);
    } */

    // Creating a texture object to handle the FFT data
    struct cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    // 2 * fftlen is the width, batch_size is the height
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, 2 * fftlen, batch_size, cudaArrayDefault));
    CUDA_CHECK(cudaMemcpy2DToArrayAsync(cuArray,
                        0, 0,
                        pdata_dev,
                        2 * fftlen * sizeof(float),
                        2 * fftlen * sizeof(float),
                        batch_size,
                        cudaMemcpyDeviceToDevice,
                        stream));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));

    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode     = cudaFilterModePoint;  // no interpolation
    texDesc.readMode       = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    free(rinds);
    free(zinds);    

    size_t powers_len = 0;
    for (int b = 0; b < batch_size; b++)
    {
        ffdotpows_cu *ffdot = &ffdot_array[b];
        // The power sizes being summed should be the same
        // They come from the harmonic fraction, which is the same for all the entries in the ffdot 
        // array. The difference is that they correspond to the same subharmonic for a different 
        // chunk of the data
        powers_len += ffdot->numws * ffdot->numzs * ffdot->numrs;
    }

    /* ffdotpows_cu *ffdot_temp = &ffdot_array[0];
    size_t manual_powers_len = batch_size * ffdot_temp->numws * ffdot_temp->numzs * ffdot_temp->numrs;
    if (manual_powers_len != powers_len) {
        printf("Mismatch in powers_len computed manually\n");
        printf("powers_len = %ld, manual_powers_len = %ld\n", powers_len, manual_powers_len);
        for (int b = 0; b < batch_size; b++)
        {
            ffdotpows_cu *ffdot = &ffdot_array[b];
            printf("b:%d => numws = %ld, numzs = %ld, numaaroos = %ld\n", b, ffdot->numws, ffdot->numzs, ffdot->numrs);
        }
        exit(1);
    } */

    float *powers_dev_batch;
    nvtxRangePush("malloc powers_dev_batch ");
    CUDA_CHECK(cudaMallocAsync(&powers_dev_batch, (size_t)(powers_len * sizeof(float)), stream));
    nvtxRangePop();
    // TODO Free powers_dev_batch
    int idx = 0;
    int *idx_array = (int *)malloc(batch_size * sizeof(int));
    /* if (harmnum == 7 && numharm == 8) {
        printf("Inside subharm_fderivs_vol %d/%d\n", harmnum, numharm);
    } */
    for (int b = 0; b < batch_size; b++)
    {
        ffdotpows_cu *ffdot = &ffdot_array[b];
        ffdot->powers = &powers_dev_batch[idx];
        /* if (harmnum == 1 && numharm == 2) {
            printf("b: %d, powersize: %ld\n", b, ffdot->numws * ffdot->numzs * ffdot->numrs);
        } */
        //ffdot->powers_size = ffdot->numws * ffdot->numzs * ffdot->numrs;
        idx_array[b] = idx;
        idx += ffdot->numws * ffdot->numzs * ffdot->numrs;
    }
    //printf("\n");

    // Make GPU wait for pdata copy to be finished before running the kernels
    #ifdef PINNED_PDATA_ALL
    CUDA_CHECK(cudaStreamWaitEvent(stream, pdata_copy_finished, 0));
    #endif
    do_fft_batch(fftlen, binoffset, ffdot_array, shi, pdata_dev, idx_array, full_tmpdat_array, full_tmpout_array, batch_size, fkern, stream, texObj);
    //CUDA_CHECK(cudaFreeAsync(pdata_dev, stream));
    
    // Destroy the texture object
    CUDA_CHECK(cudaDestroyTextureObject(texObj));
    CUDA_CHECK(cudaFreeArray(cuArray));

    free(idx_array);
    return powers_len;
}

static accelcand *create_accelcand(float power, float sigma,
                                   int numharm, double r, double z, double w)
{
    accelcand *obj;

    obj = (accelcand *)malloc(sizeof(accelcand));
    obj->power = power;
    obj->sigma = sigma;
    obj->numharm = numharm;
    obj->r = r;
    obj->z = z;
    obj->w = w;
    obj->pows = NULL;
    obj->hirs = NULL;
    obj->hizs = NULL;
    obj->hiws = NULL;
    obj->derivs = NULL;
    return obj;
}

static GSList *insert_new_accelcand(GSList *list, float power, float sigma,
                                    int numharm, double rr, double zz, double ww, int *added)
/* Checks the current list to see if there is already */
/* a candidate within ACCEL_CLOSEST_R bins.  If not,  */
/* it adds it to the list in increasing freq order.   */
{
    GSList *tmp_list = list, *prev_list = NULL, *new_list;
    double prev_diff_r = ACCEL_CLOSEST_R + 1.0, next_diff_r;

    *added = 0;
    if (!list)
    {
        new_list = g_slist_alloc();
        new_list->data =
            (gpointer *)create_accelcand(power, sigma, numharm, rr, zz, ww);
        *added = 1;
        return new_list;
    }

    /* Find the correct position in the list for the candidate */

    while ((tmp_list->next) && (((accelcand *)(tmp_list->data))->r < rr))
    {
        prev_list = tmp_list;
        tmp_list = tmp_list->next;
    }
    next_diff_r = fabs(rr - ((accelcand *)(tmp_list->data))->r);
    if (prev_list)
        prev_diff_r = fabs(rr - ((accelcand *)(prev_list->data))->r);

    /* Similar candidate(s) is(are) present */

    if (prev_diff_r < ACCEL_CLOSEST_R)
    {
        /* Overwrite the prev cand */
        if (((accelcand *)(prev_list->data))->sigma < sigma)
        {
            free_accelcand(prev_list->data, NULL);
            prev_list->data = (gpointer *)create_accelcand(power, sigma,
                                                           numharm, rr, zz, ww);
            *added = 1;
        }
        if (next_diff_r < ACCEL_CLOSEST_R)
        {
            if (((accelcand *)(tmp_list->data))->sigma < sigma)
            {
                free_accelcand(tmp_list->data, NULL);
                if (*added)
                {
                    /* Remove the next cand */
                    list = g_slist_remove_link(list, tmp_list);
                    g_slist_free_1(tmp_list);
                }
                else
                {
                    /* Overwrite the next cand */
                    tmp_list->data = (gpointer *)create_accelcand(power, sigma,
                                                                  numharm, rr, zz, ww);
                    *added = 1;
                }
            }
        }
    }
    else if (next_diff_r < ACCEL_CLOSEST_R)
    {
        /* Overwrite the next cand */
        if (((accelcand *)(tmp_list->data))->sigma < sigma)
        {
            free_accelcand(tmp_list->data, NULL);
            tmp_list->data = (gpointer *)create_accelcand(power, sigma,
                                                          numharm, rr, zz, ww);
            *added = 1;
        }
    }
    else
    { /* This is a new candidate */
        new_list = g_slist_alloc();
        new_list->data =
            (gpointer *)create_accelcand(power, sigma, numharm, rr, zz, ww);
        *added = 1;
        if (!tmp_list->next &&
            (((accelcand *)(tmp_list->data))->r < (rr - ACCEL_CLOSEST_R)))
        {
            tmp_list->next = new_list;
            return list;
        }
        if (prev_list)
        {
            prev_list->next = new_list;
            new_list->next = tmp_list;
        }
        else
        {
            new_list->next = list;
            return new_list;
        }
    }
    return list;
}

static GSList *insert_new_accelcand_last(GSList *list, float power, float sigma,
                                         int numharm, double rr, double zz, double ww, int *added, GSList **tmp_list_ptr, GSList **prev_list_ptr)
/* Checks the current list to see if there is already */
/* a candidate within ACCEL_CLOSEST_R bins.  If not,  */
/* it adds it to the list in increasing freq order.   */
/* Record last insert position, search new cand from  */
/* head or last position.                             */
{
    GSList *new_list;
    double prev_diff_r = ACCEL_CLOSEST_R + 1.0, next_diff_r;

    *added = 0;
    if (!list)
    {
        *tmp_list_ptr = g_slist_alloc();
        (*tmp_list_ptr)->data =
            (gpointer *)create_accelcand(power, sigma, numharm, rr, zz, ww);
        *added = 1;
        return *tmp_list_ptr;
    }

    /* Initialize tmp_list and prev_list with the values from the pointers */
    GSList *tmp_list = *tmp_list_ptr;
    GSList *prev_list = *prev_list_ptr;

    /* Find the correct position in the list for the candidate */

    while ((tmp_list->next) && (((accelcand *)(tmp_list->data))->r < rr))
    {
        prev_list = tmp_list;
        tmp_list = tmp_list->next;
    }
    next_diff_r = fabs(rr - ((accelcand *)(tmp_list->data))->r);
    if (prev_list)
    {
        prev_diff_r = fabs(rr - ((accelcand *)(prev_list->data))->r);
    }

    /* Similar candidate(s) is(are) present */

    if (prev_diff_r < ACCEL_CLOSEST_R)
    {
        /* Overwrite the prev cand */
        if (((accelcand *)(prev_list->data))->sigma < sigma)
        {
            free_accelcand(prev_list->data, NULL);
            prev_list->data = (gpointer *)create_accelcand(power, sigma,
                                                           numharm, rr, zz, ww);
            *added = 1;
        }
        if (next_diff_r < ACCEL_CLOSEST_R)
        {
            if (((accelcand *)(tmp_list->data))->sigma < sigma)
            {
                free_accelcand(tmp_list->data, NULL);
                if (*added)
                {
                    /* Remove the next cand */
                    list = g_slist_remove_link(list, tmp_list);
                    g_slist_free_1(tmp_list);
                    tmp_list = prev_list->next;
                    if (!tmp_list)
                    {
                        tmp_list = list;
                        prev_list = NULL;
                    }
                }
                else
                {
                    /* Overwrite the next cand */
                    tmp_list->data = (gpointer *)create_accelcand(power, sigma,
                                                                  numharm, rr, zz, ww);
                    *added = 1;
                }
            }
        }
    }
    else if (next_diff_r < ACCEL_CLOSEST_R)
    {
        /* Overwrite the next cand */
        if (((accelcand *)(tmp_list->data))->sigma < sigma)
        {
            free_accelcand(tmp_list->data, NULL);
            tmp_list->data = (gpointer *)create_accelcand(power, sigma,
                                                          numharm, rr, zz, ww);
            *added = 1;
        }
    }
    else
    { /* This is a new candidate */
        new_list = g_slist_alloc();
        new_list->data =
            (gpointer *)create_accelcand(power, sigma, numharm, rr, zz, ww);
        *added = 1;
        if (!tmp_list->next &&
            (((accelcand *)(tmp_list->data))->r < (rr - ACCEL_CLOSEST_R)))
        {
            tmp_list->next = new_list;
            *tmp_list_ptr = tmp_list;
            *prev_list_ptr = prev_list;
            return list;
        }
        if (prev_list)
        {
            prev_list->next = new_list;
            new_list->next = tmp_list;
            *tmp_list_ptr = new_list;
            *prev_list_ptr = prev_list;
            return list;
        }
        else
        {
            new_list->next = list;
            *tmp_list_ptr = list;
            *prev_list_ptr = new_list;
            return new_list;
        }
    }
    *tmp_list_ptr = tmp_list;
    *prev_list_ptr = prev_list;
    return list;
}

GSList *insert_to_cands(
    int fundamental_numrs,
    int fundamental_numzs,
    int fundamental_numws,
    long long *fundamental_rlos,
    int fundamental_zlo,
    int fundamental_wlo,
    int proper_batch_size,
    double *numindeps,
    GSList *cands,
    SearchValue *search_results,
    unsigned long long int *search_num,
    long long single_batch_size,
    int numharmstages,
    cudaStream_t main_stream,
    cudaStream_t sub_stream)
{
    int ii, jj, kk;

    long long fundamental_size = fundamental_numrs * fundamental_numzs * fundamental_numws;

    if (search_num <= 0)
        return cands;

    // record last insert position
    GSList *tmp_list = cands;
    GSList *prev_list = NULL;
    double last_rr = 0;

    for (int hitnum = 0; hitnum < search_num; hitnum++)
    {
        int current_batch = search_results[hitnum].index / single_batch_size;
        long long in_batch_id = search_results[hitnum].index % single_batch_size;
        int b = in_batch_id / (numharmstages * fundamental_size);
        long long act_id = in_batch_id % (numharmstages * fundamental_size);
        int stage = act_id / fundamental_size;
        int nh = 1 << stage;
        long long numindep = numindeps[stage];
        int fund_id = current_batch * proper_batch_size + b;
        int index = act_id % fundamental_size;
        ii = (int)(index / (fundamental_numzs * fundamental_numrs));
        jj = (int)((index / fundamental_numrs) % fundamental_numzs);
        kk = (int)(index % fundamental_numrs);

        float pow, sig;
        double rr, zz, ww;
        int added = 0;

        pow = search_results[hitnum].pow;
        sig = search_results[hitnum].sig;
        rr = (fundamental_rlos[fund_id] + kk * (double)ACCEL_DR) / (double)nh;
        zz = (fundamental_zlo + jj * (double)ACCEL_DZ) / (double)nh;
        ww = (fundamental_wlo + ii * (double)ACCEL_DW) / (double)nh;

        /*If the current rr >= the previous rr, continue searching from the last position;
        otherwise, start searching again from the beginning of cands list.*/
        if (rr <= last_rr || tmp_list == prev_list)
        {
            tmp_list = cands;
            prev_list = NULL;
        }

        {
            cands = insert_new_accelcand_last(cands, pow, sig, nh,
                                              rr, zz, ww, &added, &tmp_list, &prev_list);
        }
        last_rr = rr;
    }

    free(search_results);

    return cands;
}
