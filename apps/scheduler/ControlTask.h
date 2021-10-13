#ifndef GRB_FUSION_CONTROL_TASK_H
#define GRB_FUSION_CONTROL_TASK_H

#include <cstdint>
#include <atomic>
#include "ConcurrencyControlComponent.h"

struct ControlTaskArgs {
    unsigned ctxId;
    int workerId;
    int numWorkersToUpdate;
    bool resume;
    int *workersToUpdate;
    std::atomic_size_t *signalVar;
    ConcurrencyControlData *concurrencyControlData;
};

void controlTask_cpu(void *buffers[], void *clArgs);

struct SignalTaskArgs {
    std::atomic_size_t *signalVar;
    ConcurrencyControlData *concurrencyControlData;
    int workerId;

#ifdef SCHEDULDER_TREE
    int logicalId;
    int treeDegree;
    int treeSize;
    int numInternalNodes;
    std::atomic_size_t *signalVars;
#endif
};

void signalTask_cpu(void *buffers[], void *clArgs);

#endif //GRB_FUSION_CONTROL_TASK_H
