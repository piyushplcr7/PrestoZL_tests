#ifndef CPPDEFS
#define CPPDEFS

void fuse_add_search_batch(ffdotpows_cu *fundamentals,
                           SubharmonicMap *subhmap,
                           int stages,
                           int fundamental_num,
                           cudaStream_t stream,
                           SearchValue *search_results,
                           unsigned long long int *search_nums,
                           long long pre_size,
                           int proper_batch_size,
                           int max_searchnum,
                           int *too_large, float *stage_powers_device);
                           
#ifdef __cplusplus
extern "C" {
#endif

void free_subharmonic_cu_batch(SubharmonicMap *ffd_array, int batch_size, int num_expand,
                               cudaStream_t sub_stream);

void init_constant_device(int *subw_host, int subw_size, float *powcuts_host, int *numharms_host, double *numindeps_host, int numharmstages_size);
      
void free_ffdotpows_cu_batch(ffdotpows_cu *ffd_array, int batch_size,
                             cudaStream_t sub_stream);

size_t subharm_fderivs_vol_cu_batch(
    ffdotpows_cu *ffdot_array,
    int numharm,
    int harmnum,
    double *fullrlo_array,
    double *fullrhi_array,
    subharminfo *shi,
    accelobs *obs,
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
    fcomplex* pdata_all,
    cudaEvent_t pdata_copy_finished,
    cudaStream_t pdata_stream);

void init_inds_array(int size);
void clear_cache();
void sort_search_results(SearchValue *search_results, unsigned long long int search_num);

extern void zapbirds(double lobin, double hibin, FILE *fftfile, fcomplex *fft);
void set_openmp_numthreads(int numthreads);

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
    cudaStream_t sub_stream);

#ifdef __cplusplus
}
#endif

#endif