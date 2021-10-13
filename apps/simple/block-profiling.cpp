#include <otx/otx.h>
#include <Scheduler.h>
#include <algorithm>
#include <iomanip>
#include "../util/GrB_util.h"
#include "common.h"

extern "C" {
#include <GraphBLAS.h>
}

struct TaskArgs {
    GrB_Matrix C;
    GrB_Matrix M;
    GrB_Matrix A;
    GrB_Matrix B;
    GrB_Descriptor descriptor;
    int nthreads;

//    TaskInfo *taskInfo;
};

starpu_task *createTask(starpu_codelet *cl, GrB_Matrix C, GrB_Matrix M, GrB_Matrix A, GrB_Matrix B,
                        bool transposeA, bool transposeB, int nthreads) {

    auto task = starpu_task_create();
    task->cl = cl;

    auto args = static_cast<TaskArgs *>(malloc(sizeof(TaskArgs)));
    args->C = C;
    args->M = M;
    args->A = A;
    args->B = B;

    GrB_Descriptor desc;
    OK(GrB_Descriptor_new(&desc));
    OK(GxB_Desc_set(desc, GrB_OUTP, GrB_REPLACE));
    if (transposeA) { OK(GxB_Desc_set(desc, GrB_INP0, GrB_TRAN)); }
    if (transposeB) { OK(GxB_Desc_set(desc, GrB_INP1, GrB_TRAN)); }
    OK(GxB_Desc_set(desc, GxB_DESCRIPTOR_NTHREADS, 1));
    args->descriptor = desc;
    args->nthreads = nthreads;

    task->cl_arg = args;
    task->cl_arg_size = sizeof(TaskArgs);
    task->cl_arg_free = 1;

    return task;
}

void starpuBlocked(Scheduler &scheduler, GrB_Matrix M, GrB_Matrix A, GrB_Matrix B, bool transposeA, bool transposeB,
                   size_t nblocksM, size_t nblocksN, size_t nblocksL,
                   size_t witer, size_t niter, int ncores, const std::vector<int> &concurrencyClasses) {
    assert(nblocksL == 1);
    assert(ncores >= *std::max_element(concurrencyClasses.begin(), concurrencyClasses.end()));
    for (auto it : concurrencyClasses) { assert(ncores % it == 0); }

    // TODO: use witer

    GrB_Matrix *bC, *bM, *bA, *bB;
    createMxMBlocks(M, A, B, transposeA, transposeB, nblocksM, nblocksN, nblocksL, GrB_UINT64, &bC, &bM, &bA, &bB);

    // region init codelet
    starpu_codelet cl{};
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
        auto args = static_cast<TaskArgs *>(clArgs);
        OK(GxB_Desc_set(args->descriptor, GxB_DESCRIPTOR_NTHREADS, args->nthreads));
        GrB_Index nrows, ncols;
        GrB_Matrix_nrows(&nrows, args->C);
        GrB_Matrix_ncols(&ncols, args->C);
        GrB_Matrix_new(&args->C, GrB_UINT64, nrows, ncols);
        OK(GrB_mxm(args->C, args->M, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64, args->A, args->B, args->descriptor));
        OK(GrB_Matrix_free(&args->C));
    };
    cl.nbuffers = 0;
    // endregion

    auto *dupC = new GrB_Matrix[ncores];
    auto *dupM = new GrB_Matrix[ncores];
    auto *dupA = new GrB_Matrix[ncores];
    auto *dupB = new GrB_Matrix[ncores];

    for (int nthreads : concurrencyClasses) {
        int nworkers = ncores / nthreads;
        for (int i = 0; i < nworkers; i++) {
            scheduler.mergeWorkers(i * nthreads, nthreads);
        }
        scheduler.waitForAll();

        for (size_t i = 0; i < nblocksM; i++) {
            for (size_t j = 0; j < nblocksN; j++) {
                for (size_t k = 0; k < nblocksL; k++) {
                    size_t idxC = i * nblocksN + j;
                    size_t idxM = i * nblocksN + j;
                    size_t idxA = !transposeA ? i * nblocksL + k : k * nblocksL + i;
                    size_t idxB = !transposeB ? k * nblocksL + j : j * nblocksL + k;

                    // Create block duplicates
                    for (int w = 0; w < nworkers; w++) {
                        GrB_Matrix_dup(&dupC[w], bC[idxC]);
                        GrB_Matrix_dup(&dupM[w], bM[idxM]);
                        GrB_Matrix_dup(&dupA[w], bA[idxA]);
                        GrB_Matrix_dup(&dupB[w], bB[idxB]);
                    }

                    // Single task
                    double singleTaskTime = 0;
                    for (size_t iter = 0; iter < niter; iter++) {
                        scheduler.waitForAll();
                        timer_start();

                        scheduler.submitTask(createTask(&cl, dupC[0], dupM[0], dupA[0], dupB[0],
                                                        transposeA, transposeB, nthreads), 0, nthreads);

                        scheduler.waitForAll();
                        double iterTime = timer_end();
                        singleTaskTime += iterTime;
                    }

                    // Multi task
                    double multiTaskTime = 0;
                    for (size_t iter = 0; iter < niter; iter++) {
                        scheduler.waitForAll();
                        timer_start();

                        for (int w = 0; w < nworkers; w++) {
                            scheduler.submitTask(createTask(&cl, dupC[0], dupM[0], dupA[0], dupB[0],
                                                            transposeA, transposeB, nthreads), w * nthreads, nthreads);
                        }

                        scheduler.waitForAll();
                        double iterTime = timer_end();
//                        std::cout << nthreads << "," << i << "-" << j << "-" << k << "," << iterTime << std::endl;
                        multiTaskTime += iterTime;
                    }

                    // Multi task - copy
                    double multiTaskCopyTime = 0;
                    for (size_t iter = 0; iter < niter; iter++) {
                        scheduler.waitForAll();
                        timer_start();

                        for (int w = 0; w < nworkers; w++) {
                            scheduler.submitTask(createTask(&cl, dupC[w], dupM[w], dupA[w], dupB[w],
                                                            transposeA, transposeB, nthreads), w * nthreads, nthreads);
                        }

                        scheduler.waitForAll();
                        double iterTime = timer_end();
//                        std::cout << nthreads << "," << i << "-" << j << "-" << k << "," << iterTime << std::endl;
                        multiTaskCopyTime += iterTime;
                    }

                    // Free block duplicates
                    for (int w = 0; w < nworkers; w++) {
                        GrB_Matrix_free(&dupC[w]);
                        GrB_Matrix_free(&dupM[w]);
                        GrB_Matrix_free(&dupA[w]);
                        GrB_Matrix_free(&dupB[w]);
                    }

                    std::cout << "LOG-blocked;" << std::setw(3) << nthreads << ";"
                              << std::setw(10)
                              << (std::to_string(nblocksM) + "x" + std::to_string(nblocksN) + "x" +
                                  std::to_string(nblocksL)) << ";"
                              << std::setw(10) <<
                              (std::to_string(i) + "," + std::to_string(j) + "," + std::to_string(k)) << ";"
                              << std::setprecision(8) << std::setw(15) << singleTaskTime / niter << ";"
                              << std::setprecision(8) << std::setw(15) << multiTaskTime / niter << ";"
                              << std::setprecision(8) << std::setw(15) << multiTaskCopyTime / niter
                              << std::endl;

                }
            }
        }

        for (int i = 0; i < nworkers; i++) {
            scheduler.splitWorker(i * nthreads, nthreads, 1);
        }
        scheduler.waitForAll();
    }

    freeMxMBlocks(bC, bM, bA, bB, nblocksM, nblocksN, nblocksL);
}

int main(int argc, char *argv[]) {
    auto pathA = otx::argTo<std::string>(argc, argv, "-A", INPUTS_DIR"/simple");
    auto niter = otx::argTo<size_t>(argc, argv, "--niter", 1);
    auto witer = otx::argTo<size_t>(argc, argv, "--witer", 0);
    auto nthreads = otx::argTo<int>(argc, argv, {"--nthreads"}, 1);

    auto nblocks = otx::argTo<size_t>(argc, argv, {"--nblocks"}, 1);
    auto nblocksMin = otx::argTo<size_t>(argc, argv, {"--nblocksMin"}, nblocks);
    auto nblocksMax = otx::argTo<size_t>(argc, argv, {"--nblocksMax"}, nblocks);

    auto executeBaseline = otx::argTo<int>(argc, argv, {"--baseline"}, 0);
    auto executeBlockedBaseline = otx::argTo<int>(argc, argv, {"--blockedBaseline"}, 0);
    auto executeSpu = otx::argTo<int>(argc, argv, {"--starpu"}, 0);

    /* region init libraries */
    GrB_init(GrB_BLOCKING);
    GxB_Global_Option_set(static_cast<GxB_Option_Field>(GxB_NTHREADS), nthreads);
    /* endregion */

    // Read the input for triangle counting
    GrB_Matrix L, U, M;
    readLU(&L, &U, pathA.c_str(), false, GrB_UINT64);
    GrB_Matrix_dup(&M, L);

    // Baseline
    if (executeBaseline) {
        baselineTricnt(pathA.c_str(), L, L, U, witer, niter);
    }

    if (executeBlockedBaseline) {
        for (size_t b = nblocksMin; b <= nblocksMax; b++) {
            baselineBlockedTricnt(pathA.c_str(), L, L, U, witer, niter, b, b, 1);
        }
    }

    if (executeSpu) {
        // Init scheduler
        Scheduler scheduler;
        auto nworkers = scheduler.getNumWorkers();
        switch (nworkers) {
            case 12:
                scheduler.start({12, 6, 3, 1});
                break;
            case 64:
                scheduler.start({64, 32, 16, 8, 1});
                break;
            default:
                std::cerr << "Unsupported number of threads: " << nworkers << std::endl;
                exit(1);
        };

        for (size_t b = nblocksMin; b <= nblocksMax; b++) {
            starpuBlocked(scheduler, M, L, U, false, true, b, b, 1,
                          witer, niter, nthreads, scheduler.getConcurrencyClasses());
            std::cout << "LOG" << std::endl;
        }

        scheduler.stop();
    }

    GrB_Matrix_free(&M);
    GrB_Matrix_free(&L);
    GrB_Matrix_free(&U);

    GrB_finalize();
}

