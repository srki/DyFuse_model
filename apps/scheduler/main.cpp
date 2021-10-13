#include <iostream>
#include <chrono>
#include "Scheduler.h"


int expDelay(int n) {
    return n > 1 ? expDelay(n - 1) + expDelay(n - 1) : 1;
}

struct TaskArgs {
    int nthreads;
};

starpu_codelet userTaskCl;

void userTask_cpu(void *buffers[], void *cl_arg) {
    auto args = static_cast<TaskArgs *>(cl_arg);

//    std::cout << args->nthreads << std::endl;

#pragma omp parallel default(none) num_threads(args->nthreads)
    {
//        expDelay(30);
    }
}

void initUserCl() {
    starpu_codelet_init(&userTaskCl);
    userTaskCl.cpu_funcs[0] = userTask_cpu;
    userTaskCl.nbuffers = 0;
}

void submitUserTasks(Scheduler &scheduler, int workerId, int nthreads, int ntasks) {
    for (int i = 0; i < ntasks; i++) {
        auto task = starpu_task_create();

        auto args = static_cast<TaskArgs *>(std::malloc(sizeof(TaskArgs)));
        args->nthreads = nthreads;

        task->cl = &userTaskCl;
        task->cl_arg_free = 1;
        task->cl_arg = args;

        scheduler.submitTask(task, workerId, nthreads);
    }
}

int main() {
    const int OPTION = 3;
    initUserCl();

    if (OPTION == 1) {
        Scheduler scheduler{ };
        scheduler.start({1});
        scheduler.stop();
        return 0;
    }

    if (OPTION == 2) {
        Scheduler scheduler{};

        scheduler.start({4, 2, 1});

        for (int i = 0; i < 2000; i++) {
            scheduler
                    .mergeWorkers(0, 4)
                    .splitWorker(0, {1, 1, 1, 1})
                    .mergeWorkers(0, 4)
                    .splitWorker(0, {1, 1, 1, 1});

            scheduler
                    .mergeWorkers(0, 2)
                    .mergeWorkers(2, 2)
                    .splitWorker(0, {1, 1})
                    .splitWorker(2, {1, 1});

            scheduler
                    .mergeWorkers(0, 2)
                    .mergeWorkers(2, 2)
                    .splitWorker(0, {1, 1, 1, 1});
        }

        scheduler.stop();

        return 0;
    }

    if (OPTION == 3) {
        Scheduler scheduler{};

        auto start = std::chrono::high_resolution_clock::now();
        scheduler.start({12, 6, 3, 1});

        for (int i = 0; i < 10000; i++) {
            submitUserTasks(scheduler, 6, 1, 1);
            submitUserTasks(scheduler, 7, 1, 1);

            scheduler
                    .mergeWorkers(0, 3)
                    .mergeWorkers(3, 3)
                    .mergeWorkers(6, 3)
                    .mergeWorkers(9, 3);

            submitUserTasks(scheduler, 9, 3, 1);

            scheduler
                    .mergeWorkers(0, 6)
                    .mergeWorkers(6, 6);

            submitUserTasks(scheduler, 0, 6, 1);

            scheduler
                    .mergeWorkers(0, 12);

            scheduler
                    .splitWorker(0, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

        }

        scheduler.stop();
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

        return 0;
    }
}