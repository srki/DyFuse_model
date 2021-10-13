#ifndef GRB_FUSION_CONCURRENCY_CONTROL_COMPONENT_H
#define GRB_FUSION_CONCURRENCY_CONTROL_COMPONENT_H

#include <vector>
#include <mutex>
#include <queue>
#include <starpu_sched_component.h>

struct ConcurrencyControlArgs {
    int numWorkers;
};

struct ConcurrencyControlData {
    std::vector<std::mutex> workerQueueLocks{};
    std::vector<std::queue<starpu_task *>> workerQueues{};
    std::vector<uint8_t> workerStates{};

    explicit ConcurrencyControlData(int numWorkers) : workerQueueLocks(numWorkers), workerQueues(numWorkers),
                                                      workerStates(numWorkers, 1) {}
};

starpu_sched_component *createConcurrencyControlComponent(struct starpu_sched_tree *tree, void *arg);


#endif //GRB_FUSION_CONCURRENCY_CONTROL_COMPONENT_H
