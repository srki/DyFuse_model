#include <cassert>
#include <iostream>
#include <sstream>
#include <starpu.h>
#include "ControlTask.h"

static int getNumChildren(int nodeId, int numInternal, int size, int degree) {
    int leftMost = nodeId;
    int dist = 1;
    while (leftMost * degree + 1 < size) {
        leftMost = leftMost * degree + 1;
        dist *= degree;
    }

    dist /= degree;

    int numChildren = leftMost >= numInternal ? std::min((size - 1 - leftMost + dist) / dist, degree) : 0;

    return numChildren;
}

void controlTask_cpu(void *buffers[], void *clArgs) {
    auto args = static_cast<ControlTaskArgs *>(clArgs);

    if (args->resume) {
        auto *data = static_cast<ConcurrencyControlData *>(args->concurrencyControlData);
        std::lock_guard lck(data->workerQueueLocks[args->workerId]);

        for (int i = 0; i < args->numWorkersToUpdate; i++) {
            int workerId = args->workersToUpdate[i];
            assert(workerId != starpu_worker_get_id());

            assert(data->workerStates[workerId] == 0);
            data->workerStates[workerId] = 1;
        }
    }

    for (int i = 0; i < args->numWorkersToUpdate; i++) {
        int workerId = args->workersToUpdate[i];

        if (args->resume) {
//            std::stringstream ss;
//            ss << "Resumed: " << workerId << "; Current: " << starpu_worker_get_id() << std::endl;
//            std::cout << ss.str();

            starpu_sched_ctx_unblock_workers_in_parallel_range(args->ctxId, workerId, 1, 0);
        } else {
            auto *data = static_cast<ConcurrencyControlData *>(args->concurrencyControlData);
            assert(data->workerStates[workerId] == 0);
//            std::stringstream ss;
//            ss << "Suspend: " << workerId << "; Current: " << starpu_worker_get_id() << std::endl;
//            std::cout << ss.str();

            starpu_sched_ctx_block_workers_in_parallel_range(args->ctxId, workerId, 1, 0);
        }
    }
}

void signalTask_cpu(void *buffers[], void *clArgs) {
    auto args = static_cast<SignalTaskArgs *>(clArgs);
#ifdef SCHEDULDER_TREE
    const int treeDegree = args->treeDegree;

    int parent = (args->logicalId - 1) / treeDegree;
    while (true) {
        std::stringstream ss;
        auto cnt = args->signalVars[parent].fetch_add(1) + 1;

        if (cnt != getNumChildren(parent, args->numInternalNodes, args->treeSize, args->treeDegree)) { break; }

        if (parent == 0) {
            args->signalVar->store(0);
            break;
        }
        parent = (parent - 1) / treeDegree;
    }
#else
    auto cnt = args->signalVar->fetch_add(-1) - 1;
#endif

}