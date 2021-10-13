#include <iostream>
#include <algorithm>
#include <chrono>
#include <otx/otx.h>
#include <iomanip>
#include "Scheduler.h"

struct TaskArgs {
    int nthreads;
};

const size_t CNT_IDX_MULT = 64;
size_t cnt[STARPU_MAXCPUS * CNT_IDX_MULT];

void resetCnt() {
    for (int i = 0; i < STARPU_MAXCPUS; i++) {
        cnt[i * CNT_IDX_MULT] = 0;
    }
}

void verifyCnt(size_t expected, int nworkers) {
    for (int i = 0; i < nworkers; i++) {
        size_t val = cnt[i * CNT_IDX_MULT];
        if (val == expected) { continue; }
        std::cerr << "Worker's " << std::setw(3) << i << " cnt is different than expected: "
                  << val << " - " << expected << std::endl;
    }
}

void userTask_cpu([[maybe_unused]] void *buffers[], void *cl_arg) {
    auto args = static_cast<TaskArgs *>(cl_arg);

//    std::cout << args->nthreads << std::endl;

#pragma omp parallel default(none) num_threads(args->nthreads) shared(cnt)
    {
        int cpuid = sched_getcpu();
        cnt[cpuid * CNT_IDX_MULT]++;
    }
}

starpu_codelet userTaskCl;
TaskArgs args[STARPU_MAXCPUS];

void initUserCl() {
    starpu_codelet_init(&userTaskCl);
    userTaskCl.cpu_funcs[0] = userTask_cpu;
    userTaskCl.nbuffers = 0;

    for (int i = 0; i < STARPU_MAXCPUS; i++) {
        args[i].nthreads = i;
    }
}

starpu_task *createTask(int nthreads) {
    auto task = starpu_task_create();

    task->cl = &userTaskCl;
    task->cl_arg = &args[nthreads];
    task->cl_arg_size = sizeof(TaskArgs);
    task->cl_arg_free = 0;

    return task;
}

void test(Scheduler &scheduler, const std::vector<int> &concurrencyClasses, size_t niter, size_t ntasks) {
    auto numWorkers = scheduler.getNumWorkers();

    std::stringstream name;
    name << concurrencyClasses.back();
    for (auto it : concurrencyClasses) {
        name << ">" << it;
    }

    if (ntasks > 0) { resetCnt(); }

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < niter; iter++) {
        int prevNumThreads = 0;
        for (auto it = concurrencyClasses.begin(); it != concurrencyClasses.end(); it++) {
            auto numThreads = *it;
            if (it == concurrencyClasses.begin()) {
                scheduler.mergeWorkers(0, numThreads);
            } else {
                for (int w = 0; w < numWorkers; w += prevNumThreads) {
                    scheduler.splitWorker(w, prevNumThreads / numThreads, numThreads);
                }
            }

            prevNumThreads = numThreads;

            for (size_t i = 0; i < ntasks; i++) {
                for (int w = 0; w < numWorkers; w += numThreads) {
                    scheduler.submitTask(createTask(numThreads), w, numThreads);
                }
            }
        }
    }

    scheduler.waitForAll();
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "LOG," << name.str() << "," << niter << "," << ntasks << "," <<
              time << "," << concurrencyClasses.size() << std::endl;
    if (ntasks > 0) { verifyCnt(ntasks * niter * concurrencyClasses.size(), scheduler.getNumWorkers()); }
}

int main(int argc, char *argv[]) {
    auto niter = otx::argTo<size_t>(argc, argv, "--niter", 1);

    Scheduler scheduler;
    initUserCl();

    auto numWorkers = scheduler.getNumWorkers();
    switch (numWorkers) {
        case 12:
            scheduler.start({12, 6, 3, 1});
            break;
        case 64:
            scheduler.start({64, 32, 16, 8, 4, 2, 1});
            break;
        default:
            std::cerr << "Unknown configuration: " << numWorkers << " workers" << std::endl;
            break;
    }

    auto concurrencyClasses = std::vector<int>(scheduler.getConcurrencyClasses());
    std::sort(concurrencyClasses.begin(), concurrencyClasses.end(), std::greater<>());

    auto minThreads = *std::min_element(concurrencyClasses.begin(), concurrencyClasses.end());
    auto maxThreads = *std::max_element(concurrencyClasses.begin(), concurrencyClasses.end());
    assert(minThreads == 1);
    assert(maxThreads == numWorkers);

    std::cout << "Min threads: " << minThreads << "; Max threads: " << maxThreads << std::endl;

    test(scheduler, {maxThreads, minThreads}, niter, 0);
    test(scheduler, {maxThreads, minThreads}, niter, 1);
    test(scheduler, {maxThreads, minThreads}, 1, niter);
    test(scheduler, concurrencyClasses, niter, 0);
    test(scheduler, concurrencyClasses, niter, 1);
    test(scheduler, concurrencyClasses, 1, niter);

    if (numWorkers == 64) {
        test(scheduler, {64, 16, 4, 1}, niter, 0);
        test(scheduler, {64, 16, 4, 1}, niter, 1);
        test(scheduler, {64, 16, 4, 1}, 1, niter);
    }

    scheduler.stop();
}