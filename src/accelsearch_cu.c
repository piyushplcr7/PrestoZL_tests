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
#include "time.h"

int main(int argc, char *argv[])
{
    int max_threads = omp_get_max_threads();
    printf("Max threads (likely from OMP_NUM_THREADS): %d\n", max_threads);
    struct timespec start, end;
    subharminfo **subharminfs;
    accelobs obs;
    infodata idata;
    GSList *cands = NULL;
    Cmdline *cmd;

    // Get starting time
    clock_gettime(CLOCK_MONOTONIC, &start);
    accelsearch_CPU1(argc, argv, &subharminfs, &obs, &idata, &cmd);
    // Get ending time
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate elapsed time in seconds and nanoseconds
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("accelsearch_CPU1: %.9f seconds\n", elapsed);

    clock_gettime(CLOCK_MONOTONIC, &start);
    int too_large = accelsearch_GPU(obs, subharminfs, &cands, cmd);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("accelsearch_GPU: %.9f seconds\n", elapsed);

    clock_gettime(CLOCK_MONOTONIC, &start);
    accelsearch_CPU2(&cands, &obs, &idata, cmd);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("accelsearch_CPU2: %.9f seconds\n", elapsed);

    return 0;
}
