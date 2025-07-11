#include <cufft.h>
#include "accel_includes_noglib.h"
#include "cuda_runtime.h"
#include "cuda_helper.h"
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>

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

extern "C"
{
  void testcufftbatchsize(int batch_size, fcomplex *full_tmpdat_array,
                          subharminfo **subharminfs, int stages)
  {

    cudaStream_t test_stream;
    cudaEvent_t begin_ffts, end_ffts;
    CUDA_CHECK(cudaStreamCreate(&test_stream));
    CUDA_CHECK(cudaEventCreate(&begin_ffts));
    CUDA_CHECK(cudaEventCreate(&end_ffts));

    for (int stage = 0; stage < stages; ++stage)
    {
      int harmtosum = 1 << stage;
      for (int jj = 1; jj < harmtosum; jj += 2)
      {
        int ws_len = subharminfs[stage][jj - 1].numkern_wdim;
        int zs_len = subharminfs[stage][jj - 1].numkern_zdim;
        int fftlen = subharminfs[stage][jj - 1].kern[0][0].fftlen;
        printf("\nPerforming tests for %d/%d: %d X %d FFTs of size %d\n", jj,
               harmtosum, ws_len, zs_len, fftlen);

        // Total no. of transforms
        size_t total_num_transforms = ws_len * zs_len * batch_size;

        cufftHandle cu_plan_original;

        //  Create CUFFT Plans
        int rank = 1;
        int n[1] = {fftlen};
        int istride = 1, idist = fftlen;
        int ostride = 1, odist = fftlen;
        int inembed[3] = {ws_len, zs_len, fftlen};
        int onembed[3] = {ws_len, zs_len, fftlen};

        CHECK_CUFFT_ERRORS(cufftPlanMany(&cu_plan_original, rank, n, inembed,
                                         istride, idist, onembed, ostride, odist,
                                         CUFFT_C2C, ws_len * zs_len));

        CHECK_CUFFT_ERRORS(cufftSetStream(cu_plan_original, test_stream));

        CUDA_CHECK(cudaEventRecord(begin_ffts, test_stream));
        // Execute the plan (original code)
        for (int b = 0; b < batch_size; b++)
        {
          fcomplex *full_tmpdat =
              &full_tmpdat_array[b * (fftlen * ws_len * zs_len)];

          CHECK_CUFFT_ERRORS(
              cufftExecC2C(cu_plan_original, (cufftComplex *)full_tmpdat,
                           (cufftComplex *)full_tmpdat, CUFFT_INVERSE));
        }
        CUDA_CHECK(cudaEventRecord(end_ffts, test_stream));
        CUDA_CHECK(cudaEventSynchronize(end_ffts));

        float elapsed;
        cudaEventElapsedTime(&elapsed, begin_ffts, end_ffts);
        printf("#############################################################\n");
        printf("Original: FFTs organized as %d X (%d X %d) take: %f\n", batch_size, ws_len * zs_len, fftlen, elapsed);

        int test_batch_size = 64;
        while (test_batch_size <= 4096 &&
               test_batch_size < total_num_transforms)
        {
          int num_test_batches = total_num_transforms / test_batch_size;
          int remaining_size = total_num_transforms % test_batch_size;

          // Creating plans
          cufftHandle cu_plan_main;
          cufftHandle cu_plan_leftover;

          int rank = 1;
          int n[1] = {fftlen};
          int istride = 1, idist = fftlen;
          int ostride = 1, odist = fftlen;
          int inembed_main[2] = {test_batch_size, fftlen};
          int onembed_main[2] = {test_batch_size, fftlen};

          int inembed_leftover[2] = {remaining_size, fftlen};
          int onembed_leftover[2] = {remaining_size, fftlen};

          CHECK_CUFFT_ERRORS(cufftPlanMany(&cu_plan_main, rank, n, inembed_main,
                                           istride, idist, onembed_main, ostride,
                                           odist, CUFFT_C2C, test_batch_size));

          CHECK_CUFFT_ERRORS(cufftPlanMany(
              &cu_plan_leftover, rank, n, inembed_leftover, istride, idist,
              onembed_leftover, ostride, odist, CUFFT_C2C, remaining_size));

          CUDA_CHECK(cudaEventRecord(begin_ffts, test_stream));
          // Execute the plan (original code)
          for (int b = 0; b < num_test_batches; b++)
          {
            fcomplex *full_tmpdat =
                &full_tmpdat_array[b * (fftlen * test_batch_size)];

            CHECK_CUFFT_ERRORS(
                cufftExecC2C(cu_plan_main, (cufftComplex *)full_tmpdat,
                             (cufftComplex *)full_tmpdat, CUFFT_INVERSE));
          }

          if (remaining_size > 0)
          {
            fcomplex *full_tmpdat =
                &full_tmpdat_array[num_test_batches * (fftlen * test_batch_size)];

            CHECK_CUFFT_ERRORS(
                cufftExecC2C(cu_plan_leftover, (cufftComplex *)full_tmpdat,
                             (cufftComplex *)full_tmpdat, CUFFT_INVERSE));
          }
          CUDA_CHECK(cudaEventRecord(end_ffts, test_stream));
          CUDA_CHECK(cudaEventSynchronize(end_ffts));

          float elapsed;
          cudaEventElapsedTime(&elapsed, begin_ffts, end_ffts);

          printf("FFTs organized as %d X (%d X %d) take: %f\n", num_test_batches, test_batch_size, fftlen, elapsed);

          test_batch_size *= 2;
        }

        // Putting all the FFTs together
        {
          cufftHandle cu_plan_combined;
          int rank = 1;
          int n[1] = {fftlen};
          int istride = 1, idist = fftlen;
          int ostride = 1, odist = fftlen;
          int inembed_combined[2] = {total_num_transforms, fftlen};
          int onembed_combined[2] = {total_num_transforms, fftlen};

          CHECK_CUFFT_ERRORS(cufftPlanMany(&cu_plan_combined, rank, n, inembed_combined,
                                           istride, idist, onembed_combined, ostride,
                                           odist, CUFFT_C2C, total_num_transforms));

          fcomplex *full_tmpdat =
              &full_tmpdat_array[0];

          CUDA_CHECK(cudaEventRecord(begin_ffts, test_stream));
          CHECK_CUFFT_ERRORS(
              cufftExecC2C(cu_plan_combined, (cufftComplex *)full_tmpdat,
                           (cufftComplex *)full_tmpdat, CUFFT_INVERSE));

          CUDA_CHECK(cudaEventRecord(end_ffts, test_stream));
          CUDA_CHECK(cudaEventSynchronize(end_ffts));

          float elapsed;
          cudaEventElapsedTime(&elapsed, begin_ffts, end_ffts);

          printf("FFTs organized as (%d X %d) take: %f\n", total_num_transforms, fftlen, elapsed);
        }
      }
    }
  }
}
