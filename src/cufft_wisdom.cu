#include <cufft.h>
#include "accel_includes_noglib.h"
#include "cuda_runtime.h"
#include "cuda_helper.h"
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

static const char *_cudaGetErrorEnum(cufftResult error)
{
  switch (error)
  {
  case CUFFT_SUCCESS:
    return "CUFFT_SUCCESS";

  case CUFFT_INVALID_PLAN:
    return "CUFFT_INVALID_PLAN";

  case CUFFT_ALLOC_FAILED:
    return "CUFFT_ALLOC_FAILED";

  case CUFFT_INVALID_TYPE:
    return "CUFFT_INVALID_TYPE";

  case CUFFT_INVALID_VALUE:
    return "CUFFT_INVALID_VALUE";

  case CUFFT_INTERNAL_ERROR:
    return "CUFFT_INTERNAL_ERROR";

  case CUFFT_EXEC_FAILED:
    return "CUFFT_EXEC_FAILED";

  case CUFFT_SETUP_FAILED:
    return "CUFFT_SETUP_FAILED";

  case CUFFT_INVALID_SIZE:
    return "CUFFT_INVALID_SIZE";

  case CUFFT_UNALIGNED_DATA:
    return "CUFFT_UNALIGNED_DATA";
  }

  return "<unknown>";
}

#define CHECK_CUFFT_ERRORS(call)                                                   \
  {                                                                                \
    cufftResult_t err;                                                             \
    if ((err = (call)) != CUFFT_SUCCESS)                                           \
    {                                                                              \
      fprintf(stderr, "cuFFT error %d:%s at %s:%d\n", err, _cudaGetErrorEnum(err), \
              __FILE__, __LINE__);                                                 \
      exit(1);                                                                     \
    }                                                                              \
  }

#define MAX_BATCH_SIZE 16384
#define MIN_BATCH_SIZE 16
#define MAX_FFT_SIZE 70000

int create_cufft_wisdom()
{
  FILE *outfile = fopen("cufft_wisdom.txt", "w");
  if (outfile == NULL)
  {
    perror("Failed to open file");
    return EXIT_FAILURE;
  }

  // Buffer for writing the output lines
  char buffer[256];

  cudaStream_t test_stream;
  cudaEvent_t begin_ffts, end_ffts;
  CUDA_CHECK(cudaStreamCreate(&test_stream));
  CUDA_CHECK(cudaEventCreate(&begin_ffts));
  CUDA_CHECK(cudaEventCreate(&end_ffts));

  fcomplex *full_tmpdat_array;
  size_t array_size = MAX_BATCH_SIZE * MAX_FFT_SIZE * sizeof(fcomplex);
  printf("Allocation size: %f GB\n", (double)array_size / (1 << 30));
  CUDA_CHECK(cudaMallocAsync(&full_tmpdat_array, array_size, test_stream));

  int padlen[13] = {192, 288, 384, 540, 768, 1080, 1280, 2100, 4200, 5120,
                    7680, 8232, 10240};

  int fftlen = 64;

  while (fftlen <= MAX_FFT_SIZE)
  {
    int batch_size;

    if (fftlen > 1024)
    {
      batch_size = 2048;
    }
    else
    {
      batch_size = MAX_BATCH_SIZE;
    }
    float min_time = 1e20;
    int best_batch_size;

    printf("Testing FFT size: %d\n", fftlen);
    printf("-------------------------------\n");

    while (batch_size >= MIN_BATCH_SIZE)
    {
      // Number of batches of size batch_size
      int num_batches = MAX_BATCH_SIZE / batch_size;

      // Shared cufft plan for the batches
      cufftHandle cu_plan;
      int rank = 1;
      int n[1] = {fftlen};
      int istride = 1, idist = fftlen;
      int ostride = 1, odist = fftlen;
      int inembed[1] = {fftlen};
      int onembed[1] = {fftlen};

      CHECK_CUFFT_ERRORS(cufftPlanMany(&cu_plan, rank, n, inembed,
                                       istride, idist, onembed, ostride, odist,
                                       CUFFT_C2C, batch_size));

      CHECK_CUFFT_ERRORS(cufftSetStream(cu_plan, test_stream));

      // Begin timing FFTs
      CUDA_CHECK(cudaEventRecord(begin_ffts, test_stream));

      for (int b = 0; b < num_batches; ++b)
      {
        fcomplex *full_tmpdat =
            &full_tmpdat_array[b * (fftlen * batch_size)];

        CHECK_CUFFT_ERRORS(
            cufftExecC2C(cu_plan, (cufftComplex *)full_tmpdat,
                         (cufftComplex *)full_tmpdat, CUFFT_INVERSE));
      }

      CUDA_CHECK(cudaEventRecord(end_ffts, test_stream));
      CUDA_CHECK(cudaEventSynchronize(end_ffts));

      float elapsed;
      cudaEventElapsedTime(&elapsed, begin_ffts, end_ffts);

      printf("Batch size: %d, time: %f\n", batch_size, elapsed);

      if (elapsed < min_time)
      {
        best_batch_size = batch_size;
      }

      min_time = min_time < elapsed ? min_time : elapsed;

      batch_size /= 2;
    }

    printf("#################################################\n");
    printf("Found minimum time: %f for the best batch size: %d\n", min_time, best_batch_size);
    printf("#################################################\n\n");

    snprintf(buffer, sizeof(buffer), "fftlen: %d, batchsize: %d\n", fftlen, best_batch_size);
    fwrite(buffer, sizeof(char), strlen(buffer), outfile);

    fftlen *= 2;
  }

  fclose(outfile);
  return 0;
}

int main()
{
  return create_cufft_wisdom();
  
}
