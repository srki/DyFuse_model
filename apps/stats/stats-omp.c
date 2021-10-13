//
// Created by sm108 on 1/7/21.
//

#include <otx/atx.h>
#include <GraphBLAS.h>

#define TYPE_GrB GrB_FP64

#include "../util/GrB_util.h"
#include "../util/block_matrix.h"
#include "../util/reader_c.h"
#include "../util/timer.h"

void baseline(const char *name, GrB_Matrix M, GrB_Matrix A, GrB_Matrix B,
              unsigned transpose_A, unsigned transpose_B, size_t niter, size_t ncores,
              GrB_Type type, GrB_Semiring semiring, GrB_Monoid monoid) {
    double time_mxm, time_mxm_total = 0.0, time_reduce, time_reduce_total = 0;

    GrB_Descriptor desc;
    GrB_Index nrows;
    GrB_Index ncols;
    GrB_Matrix C;

    GrB_Index nflops = 0;
    GrB_Index nplanes;
    GrB_Index nnzA, nnzB, nnzC;

    OK(GrB_Descriptor_new(&desc));
    OK(GxB_Desc_set(desc, GrB_OUTP, GrB_REPLACE));
    if (transpose_A) { GxB_Desc_set(desc, GrB_INP0, GrB_TRAN); }
    if (transpose_B) { GxB_Desc_set(desc, GrB_INP1, GrB_TRAN); }
    OK(GxB_Desc_set(desc, GxB_DESCRIPTOR_NTHREADS, ncores));

    OK(GrB_Matrix_nrows(&nrows, A));
    OK(GrB_Matrix_ncols(&ncols, B));
    OK(GrB_Matrix_new(&C, type, nrows, ncols));


    OK(GrB_Matrix_ncols(&nplanes, A));
    OK(GrB_Matrix_nvals(&nnzA, A));
    OK(GrB_Matrix_nvals(&nnzB, B));

    for (size_t iter = 0; iter < niter; iter++) {
        GrB_Matrix_clear(C);

        timer_start();
        OK(GrB_mxm(C, M != NULL ? M : GrB_NULL, GrB_NULL, semiring, A, B, desc));
        time_mxm = timer_end();
        time_mxm_total += time_mxm;

        if (type == GrB_FP64) {
            double sum = 0.0;
            timer_start();
            OK(GrB_Matrix_reduce_FP64(&sum, GrB_NULL, GrB_PLUS_MONOID_FP64, C, GrB_NULL));
            time_reduce = timer_end();
            printf("LOG;baseline-iter;%s;%lf;%lf;%lf\n", name, time_mxm, time_reduce, sum);
            nflops = (GrB_Index) sum;
        } else if (type == GrB_UINT64) {
            uint64_t sum = 0;
            timer_start();
            OK(GrB_Matrix_reduce_UINT64(&sum, GrB_NULL, GrB_PLUS_MONOID_UINT64, C, GrB_NULL));
            time_reduce = timer_end();
            printf("LOG;baseline-iter;%s;%lf;%lf;%lu\n", name, time_mxm, time_reduce, sum);
            nflops = sum;
        }
        time_reduce_total += time_reduce;
    }


    OK(GrB_Matrix_nvals(&nnzC, C));
    printf("LOG;baseline-mean;%s;%zu;%zu;%zu;%zu;%zu;%zu;%zu;%lf; %lf\n",
           name, nrows, ncols, nplanes, nnzA, nnzB, nnzC, nflops,
           time_mxm_total / (double) niter, time_reduce_total / (double) niter);
    OK(GrB_Matrix_free(&C));
}

void omp_single_core(const char *name, GrB_Matrix *bC, GrB_Matrix *bM, GrB_Matrix *bA, GrB_Matrix *bB,
                     unsigned transpose_A, unsigned transpose_B, size_t niter, size_t ncores, size_t nblocks,
                     GrB_Type type, GrB_Semiring semiring, GrB_Monoid monoid) {
    double time_mxm_total = 0.0, time_reduce_total = 0.0, time_fused_total = 0.0;
    GrB_Descriptor desc;

    if (nblocks * nblocks < ncores) {
//        printf("LOG;omp_single_core;%s;%zu;%lf\n", name, nblocks, -1.0);
        return;
    }

    OK(GrB_Descriptor_new(&desc));
    OK(GxB_Desc_set(desc, GrB_OUTP, GrB_REPLACE));
    if (transpose_A) { GxB_Desc_set(desc, GrB_INP0, GrB_TRAN); }
    if (transpose_B) { GxB_Desc_set(desc, GrB_INP1, GrB_TRAN); }
    OK(GxB_Desc_set(desc, GxB_DESCRIPTOR_NTHREADS, 1));

    omp_set_num_threads(ncores);
    GxB_Global_Option_set(GxB_NTHREADS, 1);

    for (size_t iter = 0; iter < niter; iter++) {
        for (size_t l = 0; l < nblocks * nblocks; l++) { GrB_Matrix_clear(bC[l]); }

        /* mxm */
        timer_start();
#pragma omp parallel default(none) shared(nblocks, bM, bA, bB, bC, semiring, desc, stderr)
        {
#pragma omp single
            {
                for (size_t i = 0; i < nblocks; i++) {
                    for (size_t j = 0; j < nblocks; j++) {
#pragma omp task default(none) shared(nblocks, bM, bA, bB, bC, semiring, desc, stderr) firstprivate(i, j)
                        {
                            OK(GrB_mxm(bC[i * nblocks + j], bM != NULL ? bM[i * nblocks + j] : GrB_NULL,
                                       GrB_NULL, semiring, bA[i], bB[j], desc));
                        }
                    }
                }
            }
        }
        double time_mxm = timer_end();
        time_mxm_total += time_mxm;

        /* Reduction */
        timer_start();
        double sumReduceFP64 = 0.0;
        uint64_t sumReduceUI64 = 0;
        if (type == GrB_FP64) {
            for (size_t b = 0; b < nblocks * nblocks; b++) {
                OK(GrB_Matrix_reduce_FP64(&sumReduceFP64, GrB_PLUS_FP64, GrB_PLUS_MONOID_FP64, bC[b], GrB_NULL));
            }
        } else if (type == GrB_UINT64) {
            for (size_t b = 0; b < nblocks * nblocks; b++) {
                OK(GrB_Matrix_reduce_UINT64(&sumReduceUI64, GrB_PLUS_UINT64, GrB_PLUS_MONOID_UINT64, bC[b], GrB_NULL));
            }
        }
        double time_reduce = timer_end();
        time_reduce_total += time_reduce;

        for (size_t l = 0; l < nblocks * nblocks; l++) { GrB_Matrix_clear(bC[l]); }

        /* Fused */
        timer_start();
        double sumFusedFP64 = 0.0;
        uint64_t sumFusedUI64 = 0;
#pragma omp parallel default(none) shared(nblocks, bM, bA, bB, bC, semiring, desc, type, GrB_FP64, sumFusedFP64, GrB_PLUS_FP64, GrB_PLUS_MONOID_FP64, GrB_UINT64, sumFusedUI64, GrB_PLUS_UINT64, GrB_PLUS_MONOID_UINT64, stderr)
        {
#pragma omp single
            {
                for (size_t i = 0; i < nblocks; i++) {
                    for (size_t j = 0; j < nblocks; j++) {
#pragma omp task default(none) shared(nblocks, bM, bA, bB, bC, semiring, desc, type, GrB_FP64, sumFusedFP64, GrB_PLUS_FP64, GrB_PLUS_MONOID_FP64, GrB_UINT64, sumFusedUI64, GrB_PLUS_UINT64, GrB_PLUS_MONOID_UINT64, stderr) firstprivate(i, j)
                        {
                            OK(GrB_mxm(bC[i * nblocks + j], bM != NULL ? bM[i * nblocks + j] : GrB_NULL,
                                       GrB_NULL, semiring, bA[i], bB[j], desc));

                            if (type == GrB_FP64) {
                                double sum = 0.0;
                                OK(GrB_Matrix_reduce_FP64(&sum, GrB_PLUS_FP64, GrB_PLUS_MONOID_FP64,
                                                          bC[i * nblocks + j], GrB_NULL));
#pragma omp atomic
                                sumFusedFP64 += sum;
                            } else if (type == GrB_UINT64) {
                                uint64_t sum = 0;
                                OK(GrB_Matrix_reduce_UINT64(&sum, GrB_PLUS_UINT64, GrB_PLUS_MONOID_UINT64,
                                                            bC[i * nblocks + j], GrB_NULL));
#pragma omp atomic
                                sumFusedUI64 += sum;
                            }
                        }
                    }
                }
            }
        }
        double time_fused = timer_end();
        time_fused_total += time_fused;

        if (type == GrB_FP64) {
            printf("LOG;omp_single_core-iter;%s;%zu;%lf;%lf;%lf;%lf;%lf\n", name, nblocks,
                   time_mxm, time_reduce, time_fused, sumReduceFP64, sumFusedFP64);
        } else if (type == GrB_UINT64) {
            printf("LOG;omp_single_core-iter;%s;%zu;%lf;%lf;%lf;%lu;%lu\n", name, nblocks,
                   time_mxm, time_reduce, time_fused, sumReduceUI64, sumFusedUI64);
        }

    }
    printf("LOG;omp_single_core-mean;%s;%zu;%lf;%lf;%lf\n", name, nblocks, time_mxm_total / (double) niter,
           time_reduce_total / (double) niter, time_fused_total / (double) niter);
}

void omp_all_cores(const char *name, GrB_Matrix *bC, GrB_Matrix *bM, GrB_Matrix *bA, GrB_Matrix *bB,
                   unsigned transpose_A, unsigned transpose_B, size_t niter, size_t ncores, size_t nblocks,
                   GrB_Type type, GrB_Semiring semiring, GrB_Monoid monoid) {
    double time_mxm_total = 0.0, time_reduce_total = 0.0, time_fused_total = 0.0;

    GrB_Descriptor desc;

    OK(GrB_Descriptor_new(&desc));
    OK(GxB_Desc_set(desc, GrB_OUTP, GrB_REPLACE));
    if (transpose_A) { GxB_Desc_set(desc, GrB_INP0, GrB_TRAN); }
    if (transpose_B) { GxB_Desc_set(desc, GrB_INP1, GrB_TRAN); }
    OK(GxB_Desc_set(desc, GxB_DESCRIPTOR_NTHREADS, ncores));


    GxB_Global_Option_set(GxB_NTHREADS, ncores);

    for (size_t iter = 0; iter < niter; iter++) {
        for (size_t l = 0; l < nblocks * nblocks; l++) { GrB_Matrix_clear(bC[l]); }

        /* mxm */
        timer_start();
        for (size_t i = 0; i < nblocks; i++) {
            for (size_t j = 0; j < nblocks; j++) {
                OK(GrB_mxm(bC[i * nblocks + j], bM != NULL ? bM[i * nblocks + j] : GrB_NULL, GrB_NULL,
                           semiring, bA[i], bB[j], desc));
            }
        }
        double time_mxm = timer_end();
        time_mxm_total += time_mxm;

        /* Reduction */
        timer_start();
        double sumReduceFP64 = 0.0;
        uint64_t sumReduceUI64 = 0;
        if (type == GrB_FP64) {
            for (size_t b = 0; b < nblocks * nblocks; b++) {
                OK(GrB_Matrix_reduce_FP64(&sumReduceFP64, GrB_PLUS_FP64, GrB_PLUS_MONOID_FP64, bC[b], GrB_NULL));
            }
        } else if (type == GrB_UINT64) {
            for (size_t b = 0; b < nblocks * nblocks; b++) {
                OK(GrB_Matrix_reduce_UINT64(&sumReduceUI64, GrB_PLUS_UINT64, GrB_PLUS_MONOID_UINT64, bC[b], GrB_NULL));
            }
        }
        double time_reduce = timer_end();
        time_reduce_total += time_reduce;

        for (size_t l = 0; l < nblocks * nblocks; l++) { GrB_Matrix_clear(bC[l]); }

        /* fused */
        timer_start();
        double sumFusedFP64 = 0.0;
        uint64_t sumFusedUI64 = 0;
        for (size_t i = 0; i < nblocks; i++) {
            for (size_t j = 0; j < nblocks; j++) {
                OK(GrB_mxm(bC[i * nblocks + j], bM != NULL ? bM[i * nblocks + j] : GrB_NULL, GrB_NULL,
                           semiring, bA[i], bB[j], desc));

                if (type == GrB_FP64) {
                    double sum = 0.0;
                    OK(GrB_Matrix_reduce_FP64(&sum, GrB_PLUS_FP64, GrB_PLUS_MONOID_FP64,
                                              bC[i * nblocks + j], GrB_NULL));
                    sumFusedFP64 += sum;
                } else if (type == GrB_UINT64) {
                    uint64_t sum = 0;
                    OK(GrB_Matrix_reduce_UINT64(&sum, GrB_PLUS_UINT64, GrB_PLUS_MONOID_UINT64,
                                                bC[i * nblocks + j], GrB_NULL));
                    sumFusedUI64 += sum;
                }
            }
        }
        double time_fused = timer_end();
        time_fused_total += time_fused;

        if (type == GrB_FP64) {
            printf("LOG;omp_all_cores-iter;%s;%zu;%lf;%lf;%lf;%lf;%lf\n", name, nblocks,
                   time_mxm, time_reduce, time_fused, sumReduceFP64, sumFusedFP64);
        } else if (type == GrB_UINT64) {
            printf("LOG;omp_all_cores-iter;%s;%zu;%lf;%lf;%lf;%lu;%lu\n", name, nblocks,
                   time_mxm, time_reduce, time_fused, sumReduceUI64, sumFusedUI64);
        }
    }
    printf("LOG;omp_all_cores-mean;%s;%zu;%lf;%lf;%lf\n", name, nblocks, time_mxm_total / (double) niter,
           time_reduce_total / (double) niter, time_fused_total / (double) niter);
}

void omp(const char *name, GrB_Matrix *bC, GrB_Matrix *bM, GrB_Matrix *bA, GrB_Matrix *bB, unsigned transpose_A,
         unsigned transpose_B, size_t niter, size_t ncores, size_t nworkers_per_task, size_t nblocks, int use_task,
         GrB_Type type, GrB_Semiring semiring, GrB_Monoid monoid) {
    double time, time_total = 0;
    GrB_Descriptor desc;

    if (ncores % nworkers_per_task != 0) {
        printf("LOG;omp_single_error;%d\n", __LINE__);
        return;
    }

    if (nblocks * nblocks < ncores) {
//        printf("LOG;omp;%s;%zu;%lf\n", name, nblocks, -1.0);
        return;
    }

    OK(GrB_Descriptor_new(&desc));
    OK(GxB_Desc_set(desc, GrB_OUTP, GrB_REPLACE));
    if (transpose_A) { GxB_Desc_set(desc, GrB_INP0, GrB_TRAN); }
    if (transpose_B) { GxB_Desc_set(desc, GrB_INP1, GrB_TRAN); }
    OK(GxB_Desc_set(desc, GxB_DESCRIPTOR_NTHREADS, nworkers_per_task));

    const char *parallelism_type = use_task ? "task" : "for";

    for (size_t iter = 0; iter < niter; iter++) {
        for (size_t l = 0; l < nblocks * nblocks; l++) { GrB_Matrix_clear(bC[l]); }
        timer_start();

        if (use_task) {
#pragma omp parallel num_threads(ncores/nworkers_per_task) proc_bind(spread) default(none) shared(nworkers_per_task, nblocks, bA, bB, bC, bM, semiring, desc, stderr)
#pragma omp single
            for (size_t i = 0; i < nblocks; i++) {
                for (size_t j = 0; j < nblocks; j++) {
#pragma omp task default(none) shared(nworkers_per_task, nblocks, i, j, bA, bB, bC, bM, semiring, desc, stderr)
                    OK(GrB_mxm(bC[i * nblocks + j], bM != NULL ? bM[i * nblocks + j] : GrB_NULL, GrB_NULL,
                               semiring, bA[i],
                               bB[j], desc));
                }
            }
        } else {
#pragma omp parallel for num_threads(ncores/nworkers_per_task) proc_bind(spread) default(none) shared(nworkers_per_task, nblocks, bA, bB, bC, bM, semiring, desc, stderr)
            for (size_t l = 0; l < nblocks * nblocks; l++) {
                size_t i = l / nblocks;
                size_t j = l % nblocks;
                OK(GrB_mxm(bC[i * nblocks + j], bM != NULL ? bM[i * nblocks + j] : GrB_NULL, GrB_NULL,
                           semiring, bA[i],
                           bB[j], desc));
            }
        }

        time = timer_end();
        time_total += time;

        if (type == GrB_FP64) {
            double sum = 0.0;
            for (size_t b = 0; b < nblocks * nblocks; b++) {
                OK(GrB_Matrix_reduce_FP64(&sum, GrB_PLUS_FP64, GrB_PLUS_MONOID_FP64, bC[b], GrB_NULL));
            }
            printf("LOG;omp-%s-iter-iter;%s;%zu;%lf;%lf\n", parallelism_type, name, nblocks, time, sum);
        } else if (type == GrB_UINT64) {
            uint64_t sum = 0;
            for (size_t b = 0; b < nblocks * nblocks; b++) {
                OK(GrB_Matrix_reduce_UINT64(&sum, GrB_PLUS_UINT64, GrB_PLUS_MONOID_UINT64, bC[b], GrB_NULL));
            }
            printf("LOG;omp-%s-iter-iter;%s;%zu;%lf;%lu\n", parallelism_type, name, nblocks, time, sum);
        }
    }
    printf("LOG;omp-%s-mean;%s;%zu;%zu;%zu;%lf\n", parallelism_type, name, nblocks, ncores, nworkers_per_task,
           time_total / niter);
    OK(GrB_Descriptor_free(&desc));
}

void blocked(const char *name, GrB_Matrix M, GrB_Matrix A, GrB_Matrix B, unsigned transpose_A, unsigned transpose_B,
             size_t niter, size_t ncores, size_t nblocks, GrB_Type type, GrB_Semiring semiring, GrB_Monoid monoid,
             unsigned mode) {
    assert(!transpose_A);
    GrB_Matrix C;
    GrB_Matrix *bA, *bB, *bC, *bM;

    size_t nelem_tmp, nelem = 0;
    size_t ncache_lines_tmp, ncache_lines = 0;

    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t block_size_m, block_size_n, block_size_l;

    GrB_Matrix_nrows(&m, A);


    GrB_Matrix_ncols(&n, B);
    GrB_Matrix_ncols(&l, A);

    block_size_m = (m + nblocks - 1) / nblocks;
    block_size_n = (n + nblocks - 1) / nblocks;
    block_size_l = l;

    GrB_Matrix_new(&C, type, m, n);

    create_blocks_type(&bA, &bm, &bl, A, block_size_m, block_size_l, 0, type);

    if (transpose_B) {
        create_blocks_type(&bB, &bn, &bl, B, block_size_n, block_size_l, 0, type);
    } else {
        create_blocks_type(&bB, &bl, &bn, B, block_size_l, block_size_n, 0, type);
    }

    create_blocks_type(&bC, &bm, &bn, C, block_size_m, block_size_n, 0, type);

    if (M != GrB_NULL) {
        create_blocks_type(&bM, &bm, &bn, M, block_size_m, block_size_n, 0, type);
    } else {
        bM = NULL;
    }

    if (nblocks != bm || nblocks != bn || 1 != bl) {
        exit(5);
    }

    omp_set_num_threads(ncores);

    if (mode & 0x1) {
        omp_single_core(name, bC, bM, bA, bB, transpose_A, transpose_B, niter, ncores, nblocks, type, semiring, monoid);
    }

    if (mode & 0x2) {
        omp_all_cores(name, bC, bM, bA, bB, transpose_A, transpose_B, niter, ncores, nblocks, type, semiring, monoid);
    }

    if (mode & (0x4 | 0x8)) {
        omp_set_max_active_levels(3);

        if (mode & 0x4) {
            for (size_t i = 1; i <= ncores; i++) {
                if (ncores % i != 0) { continue; }
                omp(name, bC, bM, bA, bB, transpose_A, transpose_B, niter, ncores, i, nblocks,
                    0, type, semiring, monoid);
            }
        }

        if (mode & 0x8) {
            for (size_t i = 1; i <= ncores; i++) {
                if (ncores % i != 0) { continue; }
                omp(name, bC, bM, bA, bB, transpose_A, transpose_B, niter, ncores, i, nblocks,
                    1, type, semiring, monoid);
            }
        }
    }

    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    GrB_free(&C);
}

int main(int argc, char *argv[]) {
    char *A_path;
    uint64_t niter;
    int num_threads;
    size_t nblocks, min_blocks, max_blocks;
    unsigned mxm;
    unsigned mode;
    unsigned tranA, tranB;
    GrB_Matrix A, B, M;
    GrB_Type type;
    GrB_Semiring semiring;
    GrB_Monoid monoid;

    /* region read program arguments */
    if (arg_to_str_def(argc, argv, "-A", &A_path, INPUTS_DIR"/simple") != OTX_SUCCESS) { return 1; }
    if (arg_to_uint64_def(argc, argv, "--niter", &niter, 1) != OTX_SUCCESS) { return 2; }
    if (arg_to_int32_def(argc, argv, "-nt|--nthreads", &num_threads, 1) != OTX_SUCCESS) { return 3; }
    if (arg_to_uint64_def(argc, argv, "-minb|--minBlocks", &min_blocks, 1) != OTX_SUCCESS) { return 4; }
    if (arg_to_uint64_def(argc, argv, "-maxb|--maxBlocks", &max_blocks, 4) != OTX_SUCCESS) { return 5; }
    if (arg_to_uint32_def(argc, argv, "--mxm", &mxm, 0) != OTX_SUCCESS) { return 6; }
    if (arg_to_uint32_def(argc, argv, "--mode", &mode, 0) != OTX_SUCCESS) { return 7; }
    /* endregion */

    printf("%s; matrix: %s, mode %u; num iterations: %lu, num threads: %d; blocks: %lu-%lu\n", mxm ? "mxm" : "tricnt",
           get_file_name(A_path), mode, niter, num_threads, min_blocks, max_blocks);

    /* region init libraries */
    GrB_init(GrB_BLOCKING);
    GxB_Global_Option_set(GxB_NTHREADS, num_threads);
    /* endregion */

    /* region read matrix */
    timer_start();
    if (mxm) {
        if (read_Matrix_FP64(&A, A_path, false) != GrB_SUCCESS) { return 16; }
        GrB_Matrix_dup(&B, A);
        tranA = tranB = 0;
        M = NULL;
        type = GrB_FP64;
        semiring = GrB_PLUS_TIMES_SEMIRING_FP64;
        monoid = GrB_PLUS_MONOID_FP64;
    } else {
        readLU(&A, &B, A_path, 0, GrB_UINT64);
        GrB_Matrix_dup(&M, A);
        tranA = 0;
        tranB = 1;
        type = GrB_UINT64;
        semiring = GrB_PLUS_TIMES_SEMIRING_UINT64;
        monoid = GrB_PLUS_MONOID_UINT64;
    }

    printf("Read time: %lf\n", timer_end());
    /* endregion */

    baseline(get_file_name(A_path), M, A, B, tranA, tranB, niter, num_threads, type, semiring, monoid);

    nblocks = min_blocks;
    while (nblocks <= max_blocks) {
        blocked(get_file_name(A_path), M, A, B, tranA, tranB, niter,
                num_threads, nblocks, type, semiring, monoid, mode);

        if (nblocks == 1) { nblocks = 2; }
        else if (((nblocks - 1) & nblocks) == 0) { nblocks = nblocks * 3 / 2; }
        else { nblocks = ((nblocks - 1) & nblocks) * 2; }
    }

    baseline(get_file_name(A_path), M, A, B, tranA, tranB, niter, num_threads, type, semiring, monoid);

    /* region free resources and terminate libraries */
    GrB_Matrix_free(&A);
    GrB_Matrix_free(&B);
    GrB_finalize();
    /* endregion */

    return 0;
}