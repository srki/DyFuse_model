#ifndef GRB_FUSION_SCHEDULER_H
#define GRB_FUSION_SCHEDULER_H

#include <vector>
#include <starpu.h>
#include <limits>

class Scheduler {
    enum class WorkerState : uint8_t {
        RUNNING = 1,
        SUSPENDED = 2
    };

public:
    Scheduler();

    [[deprecated]]
    Scheduler(int nworkers, std::vector<int> concurrencyClasses);

    [[nodiscard]] int getNumWorkers() const { return _nworkers; };

    [[nodiscard]] const std::vector<int> &getConcurrencyClasses() { return _concurrencyClasses; }

    [[deprecated]]
    Scheduler &start();

    Scheduler &start(std::vector<int> concurrencyClasses);

    Scheduler &stop();

    Scheduler &submitTask(starpu_task *task, int workerId, int nthreads);

    Scheduler &splitWorker(int workerId, const std::vector<int> &nthreads);

    Scheduler &splitWorker(int workerId, int nworkers, int nthreadsPerWorker);

    Scheduler &mergeWorkers(int workerId, int nthreads);

    Scheduler &waitForAll();

#ifdef SCHEDULDER_TREE
    void setTreeDegree(int treeDegree) { _treeDegree = treeDegree; }
#endif

private:
    void initCodelets();

    void setConcurrencyClasses(std::vector<int> &&concurrencyClasses);

    void pinThreads();

    bool isValidClass(int nthreads);

private:
    int _nworkers;
    std::vector<WorkerState> _workerStates;
    std::vector<int> _concurrencyClasses;

    unsigned _ctxId{0}; // TODO

    starpu_codelet _controlTaskCl{};
    starpu_codelet _signalTaskCl{};

#ifdef SCHEDULDER_TREE
    int _treeDegree{std::numeric_limits<int>::max() / 2};
#endif
};

#endif //GRB_FUSION_SCHEDULER_H
