#include <string>
#include <iostream>

#include <otx/otx.h>
#include <Scheduler.h>

#define TYPE_GrB GrB_BOOL

#include "../util/GrB_util.h"
#include "../util/timer.h"
#include "common.h"

extern "C" {
#include <GraphBLAS.h>
}

struct TaskInfo {
    starpu_task *task;
    size_t i;
    size_t j;
    size_t k;
    GrB_Matrix result;

    double mxmTime;

    TaskInfo(starpu_task *task, size_t i, size_t j, size_t k)
            : task(task), i(i), j(j), k(k), result(GrB_NULL) {}
};

struct TaskArgs {
    GrB_Matrix M;
    GrB_Matrix A;
    GrB_Matrix B;
    GrB_Descriptor descriptor;
    int nthreads;

    TaskInfo *taskInfo;
};

std::vector<TaskInfo *> createTasks(starpu_codelet *cl, GrB_Matrix *bM, GrB_Matrix *bA, GrB_Matrix *bB,
                                    bool transposeA, bool transposeB, size_t nblocksM, size_t nblocksN,
                                    size_t nblocksL) {
    std::vector<TaskInfo *> tasks;
    tasks.reserve(nblocksM * nblocksN * nblocksL);

    for (size_t i = 0; i < nblocksM; i++) {
        for (size_t j = 0; j < nblocksN; j++) {
            for (size_t k = 0; k < nblocksL; k++) {
                size_t idxC = i * nblocksN + j;
                size_t idxA = !transposeA ? i * nblocksL + k : k * nblocksL + i;
                size_t idxB = !transposeB ? k * nblocksL + j : j * nblocksL + k;

                auto task = starpu_task_create();
                task->cl = cl;

                auto args = static_cast<TaskArgs *>(malloc(sizeof(TaskArgs)));
                args->M = bM[idxC];
                args->A = bA[idxA];
                args->B = bB[idxB];

                GrB_Descriptor desc;
                OK(GrB_Descriptor_new(&desc));
                OK(GxB_Desc_set(desc, GrB_OUTP, GrB_REPLACE));
                if (transposeA) { OK(GxB_Desc_set(desc, GrB_INP0, GrB_TRAN)); }
                if (transposeB) { OK(GxB_Desc_set(desc, GrB_INP1, GrB_TRAN)); }
                OK(GxB_Desc_set(desc, GxB_DESCRIPTOR_NTHREADS, 1));
                args->descriptor = desc;
                args->nthreads = 1;

                task->cl_arg = args;
                task->cl_arg_size = sizeof(TaskArgs);
                task->cl_arg_free = 1;

                auto taskInfo = new TaskInfo{task, i, j, k};
                args->taskInfo = taskInfo;

                tasks.push_back(taskInfo);
            }
        }
    }

    return tasks;
}

void blockedSPU(const char *pathA, Scheduler &scheduler, GrB_Matrix M, GrB_Matrix A, GrB_Matrix B, bool transposeA,
                bool transposeB, size_t nblocksM, size_t nblocksN, size_t nblocksL,
                size_t witer, size_t niter, size_t nthreads, bool iterReconf) {
    assert(nblocksL == 1);

    GrB_Matrix *bA, *bB, *bM;
    createMxMBlocks(M, A, B, transposeA, transposeB, nblocksM, nblocksN, nblocksL, GrB_UINT64, nullptr, &bM, &bA, &bB);

    // region init codelet
    starpu_codelet cl{};
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
        auto args = static_cast<TaskArgs *>(clArgs);
        OK(GxB_Desc_set(args->descriptor, GxB_DESCRIPTOR_NTHREADS, args->nthreads));
        GrB_Index nrows, ncols;
        GrB_Matrix_nrows(&nrows, args->A);
        GrB_Matrix_nrows(&ncols, args->B); // transposed

        GrB_Matrix C;
        GrB_Matrix_new(&C, GrB_UINT64, nrows, ncols);

        timer_start();
        OK(GrB_mxm(C, args->M, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64, args->A, args->B, args->descriptor));
        args->taskInfo->result = C;
        args->taskInfo->mxmTime = timer_end();
    };
    cl.nbuffers = 0;
    // endregion

    if (!iterReconf) {
        scheduler.mergeWorkers(0, nthreads);
        scheduler.waitForAll();
    }

    double mxmTimeTotal = 0;
    double fullTimeTotal = 0;
    double grbKernelTimeTotal = 0;
    for (size_t i = 0; i < niter + witer; i++) {
        auto tasks = createTasks(&cl, bM, bA, bB, transposeA, transposeB, nblocksM, nblocksN, nblocksL);

        timer_start();
        if (iterReconf) {
            scheduler.mergeWorkers(0, nthreads);
            scheduler.waitForAll();
        }

        timer_start();
        for (auto taskInfo : tasks) {
            static_cast<TaskArgs *>(taskInfo->task->cl_arg)->nthreads = nthreads;
            scheduler.submitTask(taskInfo->task, 0, nthreads);
        }
        scheduler.waitForAll();
        double mxmTime = timer_end();

        if (iterReconf) {
            scheduler.splitWorker(0, nthreads, 1);
            scheduler.waitForAll();
        }
        double fullTime = timer_end();

        double grbKernelTime = 0;
        uint64_t result = 0;
        for (auto taskInfo : tasks) {
            auto blockC = taskInfo->result;
//            GrB_Matrix_reduce_UINT64(&result, GrB_PLUS_UINT64, GrB_PLUS_MONOID_UINT64, blockC, GrB_NULL);
            GrB_Matrix_free(&blockC);
            grbKernelTime += taskInfo->mxmTime;
        }

        if (i < witer) { continue; }

        grbKernelTimeTotal += grbKernelTime;
        mxmTimeTotal += mxmTime;
        fullTimeTotal += fullTime;

//        std::cout << "LOG-blocked-spu-iter,grb-blocked," << get_file_name(pathA) << ","
//                  << nblocksM << "," << nblocksN << "," << nblocksL << ","
//                  << grbKernelTime << "," << mxmTime << "," << fullTime << "," << result << std::endl;
    }

    if (!iterReconf) {
        scheduler.splitWorker(0, nthreads, 1);
        scheduler.waitForAll();
    }

    std::cout << "LOG-blocked-spu,grb-blocked," << get_file_name(pathA) << ","
              << nblocksM << "," << nblocksN << "," << nblocksL << ","
              << grbKernelTimeTotal / niter << "," << mxmTimeTotal / niter << "," << fullTimeTotal / niter
              << std::endl;

    freeMxMBlocks(nullptr, bM, bA, bB, nblocksM, nblocksN, nblocksL);
}

void staticScheduling(const char *pathA, Scheduler &scheduler, GrB_Matrix M, GrB_Matrix A, GrB_Matrix B,
                      size_t witer, size_t niter, int nthreads) {
//    assert(std::string(get_file_name(pathA)) == "circuit5M.mtx");
//    assert(std::string(get_file_name(pathA)) == "belgium_osm.mtx");

    GrB_Matrix *bA, *bB, *bM;
    createMxMBlocks(M, A, B, false, true, 5, 5, 1, GrB_UINT64, nullptr, &bM, &bA, &bB);

    // region init codelet
    starpu_codelet cl{};
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
        auto args = static_cast<TaskArgs *>(clArgs);
        OK(GxB_Desc_set(args->descriptor, GxB_DESCRIPTOR_NTHREADS, args->nthreads));
        GrB_Index nrows, ncols;
        GrB_Matrix_nrows(&nrows, args->A);
        GrB_Matrix_nrows(&ncols, args->B); // transposed

        GrB_Matrix C;
        GrB_Matrix_new(&C, GrB_UINT64, nrows, ncols);

        timer_start();
        OK(GrB_mxm(C, args->M, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64, args->A, args->B, args->descriptor));
        args->taskInfo->result = C;
        args->taskInfo->mxmTime = timer_end();
    };
    cl.nbuffers = 0;
    // endregion

    double totalTime = 0;
    for (size_t i = 0; i < niter + witer; i++) {
        auto tasks = createTasks(&cl, bM, bA, bB, false, true, 5, 5, 1);

        if (i >= witer) { timer_start(); }

        std::vector<TaskInfo *> full, quarter, single;
        for (auto task : tasks) {
            if (task->j == 0) {
                if (task->i == 0) { full.emplace_back(task); } else { quarter.emplace_back(task); }
            } else {
                single.push_back(task);
            }
        }
        tasks.clear();

        // Submit whole node tasks
        {
            if (i + 1 == niter + witer) { timer_start(); }
            scheduler.mergeWorkers(0, nthreads);
            if (i + 1 == niter + witer) {
                scheduler.waitForAll();
                std::cout << ";" << timer_end();
            }

            if (i + 1 == niter + witer) { timer_start(); }
            for (auto taskInfo : full) {
                static_cast<TaskArgs *>(taskInfo->task->cl_arg)->nthreads = nthreads;
                scheduler.submitTask(taskInfo->task, 0, nthreads);
            }
            if (i + 1 == niter + witer) {
                scheduler.waitForAll();
                std::cout << ";" << timer_end();
            }
        }

        // Submit quarter node
        {
            if (i + 1 == niter + witer) { timer_start(); }
            scheduler.splitWorker(0, 4, nthreads / 4);

            if (i + 1 == niter + witer) {
                scheduler.waitForAll();
                std::cout << ";" << timer_end();
            }

            if (i + 1 == niter + witer) { timer_start(); }
            int nextWorker = 0;
            for (auto taskInfo : quarter) {
                static_cast<TaskArgs *>(taskInfo->task->cl_arg)->nthreads = nthreads / 4;
                scheduler.submitTask(taskInfo->task, nextWorker * (nthreads / 4), nthreads / 4);
                nextWorker++;
                nextWorker %= 4;
            }
            if (i + 1 == niter + witer) {
                scheduler.waitForAll();
                std::cout << ";" << timer_end();
            }
        }

        // Submit single core tasks
        {
            if (i + 1 == niter + witer) { timer_start(); }
            for (int w = 0; w < 4; w++) {
                scheduler.splitWorker(w * (nthreads / 4), nthreads / 4, 1);
            }

            if (i + 1 == niter + witer) {
                scheduler.waitForAll();
                std::cout << ";" << timer_end();
            }

            if (i + 1 == niter + witer) { timer_start(); }
            int nextWorker = 0;
            for (auto taskInfo : single) {
                static_cast<TaskArgs *>(taskInfo->task->cl_arg)->nthreads = 1;
                scheduler.submitTask(taskInfo->task, nextWorker, 1);
                nextWorker %= nthreads;
            }
            if (i + 1 == niter + witer) {
                scheduler.waitForAll();
                std::cout << ";" << timer_end();
            }
        }
        if (i + 1 == niter + witer) { std::cout << std::endl; }

        scheduler.waitForAll();

        if (i + 1 == niter + witer) {
            std::cout << nthreads << " threads: ";
            for (auto task : full) { std::cout << task->mxmTime << ", "; }
            std::cout << std::endl;

            std::cout << nthreads / 4 << " threads: ";
            for (auto task : quarter) { std::cout << task->mxmTime << ", "; }
            std::cout << std::endl;

            std::cout << 1 << " thread: ";
            for (auto task : single) { std::cout << task->mxmTime << ", "; }
            std::cout << std::endl;
        }

        if (i >= witer) { totalTime += timer_end(); }
    }

    std::cout << "LOG-blocked-spu,static," << get_file_name(pathA) << ","
              << 5 << "," << 5 << "," << 1 << ","
              << totalTime / niter
              << std::endl;

    freeMxMBlocks(nullptr, bM, bA, bB, 5, 5, 1);
}

int main(int argc, char *argv[]) {
    auto pathA = otx::argTo<std::string>(argc, argv, "-A", INPUTS_DIR"/simple");
    auto niter = otx::argTo<size_t>(argc, argv, "--niter", 1);
    auto witer = otx::argTo<size_t>(argc, argv, "--witer", 0);
    auto nthreads = otx::argTo<size_t>(argc, argv, {"--nthreads"}, 1);

    auto nblocks = otx::argTo<size_t>(argc, argv, {"--nblocks"}, 1);
    auto nblocksMin = otx::argTo<size_t>(argc, argv, {"--nblocksMin"}, nblocks);
    auto nblocksMax = otx::argTo<size_t>(argc, argv, {"--nblocksMax"}, nblocks);

    auto executeBaseline = otx::argTo<int>(argc, argv, {"--baseline"}, 0);
    auto executeBaselineBlocked = otx::argTo<int>(argc, argv, {"--baselineBlocked"}, 0);
    auto executeSpu = otx::argTo<int>(argc, argv, {"--starpu"}, 0);
    auto executeStaticSched = otx::argTo<int>(argc, argv, {"--static"}, 0);

    std::cout << "iter: " << witer << " + " << niter << "; "
              << "nthreads: " << nthreads << "; "
              << "blocks: " << nblocksMin << "-" << nblocksMax << "; "
              << "baseline, baselineBlocked, StarPU: " << executeBaseline << "/" << executeBaselineBlocked
              << "/" << executeSpu << "/" << executeStaticSched
              << std::endl;


    /* region init libraries */
    GrB_init(GrB_BLOCKING);
    GxB_Global_Option_set(static_cast<GxB_Option_Field>(GxB_NTHREADS), nthreads);
    /* endregion */

    GrB_Matrix L, U, M;
    readLU(&L, &U, pathA.c_str(), false, GrB_UINT64);
    GrB_Matrix_dup(&M, L);
    if (executeBaseline) {
        baselineTricnt(pathA.c_str(), L, L, U, witer, niter);
        baselineTricnt(pathA.c_str(), M, L, U, witer, niter);
    }

    if (executeBaselineBlocked) {
        for (size_t b = nblocksMin; b <= nblocksMax; b++) {
            baselineBlockedTricnt(pathA.c_str(), L, L, U, witer, niter, b, b, 1);
        }
    }

    if (executeSpu || executeStaticSched) {
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

        for (int i = 0; i < 3; i++) {
            if (executeSpu) {
                for (size_t b = nblocksMin; b <= nblocksMax; b++) {
                    blockedSPU(pathA.c_str(), scheduler, L, L, U, false, true, b, b, 1, witer, niter, nworkers, true);
                }

                for (size_t b = nblocksMin; b <= nblocksMax; b++) {
                    blockedSPU(pathA.c_str(), scheduler, L, L, U, false, true, b, b, 1, witer, niter, nworkers, false);
                }
            }

            if (executeStaticSched) {
                staticScheduling(pathA.c_str(), scheduler, L, L, U, witer, niter, nworkers);
            }
        }

        scheduler.stop();
    }

    if (executeBaseline) {
        baselineTricnt(pathA.c_str(), L, L, U, witer, niter);
        baselineTricnt(pathA.c_str(), M, L, U, witer, niter);
    }

    if (executeBaselineBlocked) {
        for (size_t b = nblocksMin; b <= nblocksMax; b++) {
            baselineBlockedTricnt(pathA.c_str(), L, L, U, witer, niter, b, b, 1);
        }
    }


    GrB_finalize();

    return 0;
}
