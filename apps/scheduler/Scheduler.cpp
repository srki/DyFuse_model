#include "Scheduler.h"

#include <omp.h>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include "SchedulingPolicy.h"
#include "ControlTask.h"
#include "PinThreadsTask.h"
#include "SchedulerArgs.h"

//TODO: move
template<class T>
static T ceili(T a, T d) {
    static_assert(std::is_integral_v<T>);
    return (a + d - T(1)) / d;
}

static size_t get_internal_nodes_num(size_t numNodes, size_t treeDegree) {
    return ceili(numNodes - 1, treeDegree);
}

static int getNumChildren(int nodeId, int numInternal, int size, int degree) {
    int leftMost = nodeId;
    int dist = 1;
    while (leftMost * degree + 1 < size) {
        leftMost = leftMost * degree + 1;
        dist *= degree;
    }

    dist /= degree;

    int numChildren = leftMost >= numInternal ? std::min((size - 1 - leftMost + dist) / dist, degree) : 0;

    std::cout << "nodeId: " << std::setw(2) << nodeId << "; numChildren:" << std::setw(2) << numChildren
              << "; leftMost: " << std::setw(2) << leftMost << "; dist: " << std::setw(2) << dist << std::endl;

    return numChildren;
}

Scheduler::Scheduler() {
    initCodelets();

    starpu_conf conf{};
    starpu_conf_init(&conf);
    conf.sched_policy = &schedulingPolicy;
    STARPU_CHECK_RETURN_VALUE(starpu_init(&conf), "starpu_init");

    _nworkers = static_cast<int>(starpu_worker_get_count());
    _workerStates.resize(_nworkers, WorkerState::RUNNING);
}

Scheduler &Scheduler::start(std::vector<int> concurrencyClasses) {
    setConcurrencyClasses(std::move(concurrencyClasses));
    pinThreads();
    return *this;
}

Scheduler::Scheduler(int nworkers, std::vector<int> concurrencyClasses)
        : _nworkers(nworkers), _workerStates(nworkers, WorkerState::RUNNING) {
    setConcurrencyClasses(std::move(concurrencyClasses));
    initCodelets();
}

Scheduler &Scheduler::start() {
    starpu_conf conf{};
    starpu_conf_init(&conf);
    conf.sched_policy = &schedulingPolicy;
    STARPU_CHECK_RETURN_VALUE(starpu_init(&conf), "starpu_init");

    pinThreads();

    return *this;
}

Scheduler &Scheduler::stop() {
    starpu_task_wait_for_all();
    starpu_shutdown();
    return *this;
}

Scheduler &Scheduler::submitTask(starpu_task *task, int workerId, int nthreads) {
    assert(std::binary_search(_concurrencyClasses.begin(), _concurrencyClasses.end(), nthreads, std::greater<>{}));

    assert(_workerStates[workerId] == WorkerState::RUNNING);
    for (int i = 1; i < nthreads; i++) { assert(_workerStates[workerId + i] == WorkerState::SUSPENDED); }
    assert(workerId + nthreads >= _nworkers || _workerStates[workerId + nthreads] == WorkerState::RUNNING);

    assert(task->sched_data == nullptr);
    auto schedulerArgs = static_cast<SchedulerArgs *>(std::malloc(sizeof(SchedulerArgs)));
    schedulerArgs->workerId = workerId;
    schedulerArgs->taskType = SchedulerArgs::EnumTaskType::UserTask;

    task->sched_data = schedulerArgs;

    starpu_task_submit_to_ctx(task, _ctxId);

    return *this;
}

Scheduler &Scheduler::splitWorker(int workerId, const std::vector<int> &nthreads) {
    int totalThreads = 0;
    for (auto &nt : nthreads) {
        assert(isValidClass(nt));
        totalThreads += nt;
    }
    assert(workerId + totalThreads <= _nworkers);
    assert(workerId % totalThreads == 0);
    assert(_workerStates[workerId] == WorkerState::RUNNING);
//        for (int i = 1; i < totalThreads; i++) { assert(_workerStates[workerId + i] == 0); }

    // Create a list of workers that should be resumed
    std::vector<int> toResume;
    std::vector<int> toSignal;
    int next = workerId;
    for (auto it = nthreads.begin(); it != nthreads.end() - 1; it++) {
        next += *it;
        if (_workerStates[next] == WorkerState::RUNNING) {
            toSignal.push_back(next);
        } else {
            assert(_workerStates[next] = WorkerState::SUSPENDED);
            _workerStates[next] = WorkerState::RUNNING;
            toResume.push_back(next);
        }
    }

//  auto signalVar = !toSignal.empty() ? new std::atomic_size_t{toSignal.size()} : nullptr;
    auto signalVar = new std::atomic_size_t{toSignal.size()};

    {
        // Create resume task
        auto argsSize = sizeof(ControlTaskArgs) + toResume.size() * sizeof(int);
        auto *args = static_cast<std::byte *>( std::malloc(argsSize));

        auto toResumeIds = reinterpret_cast<int *>(args + sizeof(ControlTaskArgs));
        std::copy(toResume.begin(), toResume.end(), toResumeIds);


        // Fill control task args and submit the task
        auto &data = *reinterpret_cast<ControlTaskArgs *>(args);
        data.ctxId = _ctxId;
        data.workerId = workerId;
        data.numWorkersToUpdate = static_cast<int>(toResume.size());
        data.resume = true;
        data.workersToUpdate = toResumeIds;
        data.signalVar = signalVar;

        auto schedulerArgs = static_cast<SchedulerArgs *>(std::malloc(sizeof(SchedulerArgs)));
        schedulerArgs->workerId = workerId;
        schedulerArgs->taskType = SchedulerArgs::EnumTaskType::ResumeTasks;

        auto task = starpu_task_create();
        task->cl = &_controlTaskCl;
        task->cl_arg = args;
        task->cl_arg_size = argsSize;
        task->cl_arg_free = 1;
        task->sched_data = schedulerArgs;

        starpu_task_submit_to_ctx(task, _ctxId);
    }

    // Create and submit signal tasks
    for (auto signalTaskId : toSignal) {
        auto args = static_cast<SignalTaskArgs *>(std::malloc(sizeof(SignalTaskArgs)));
        args->signalVar = signalVar;
        args->workerId = signalTaskId;

        auto schedulerArgs = static_cast<SchedulerArgs *>(std::malloc(sizeof(SchedulerArgs)));
        schedulerArgs->workerId = signalTaskId;
        schedulerArgs->taskType = SchedulerArgs::EnumTaskType::SignalTaskSplit;

        auto task = starpu_task_create();
        task->cl = &_signalTaskCl;
        task->cl_arg = args;
        task->cl_arg_size = sizeof(SignalTaskArgs);
        task->cl_arg_free = 1;
        task->sched_data = schedulerArgs;

        starpu_task_submit_to_ctx(task, _ctxId);
    }

    return *this;
}

Scheduler &Scheduler::splitWorker(int workerId, int nworkers, int nthreadsPerWorker) {
    return splitWorker(workerId, std::vector<int>(nworkers, nthreadsPerWorker));
}

Scheduler &Scheduler::mergeWorkers(int workerId, int nthreads) {
    assert(nthreads >= 1);
    assert(workerId % nthreads == 0);
    assert(isValidClass(nthreads));

    // Create a list of workers that should be suspended
    std::vector<int> toSuspend;
    for (int i = 1; i < nthreads; i++) {
        if (_workerStates[workerId + i] == WorkerState::SUSPENDED) { continue; }
        _workerStates[workerId + i] = WorkerState::SUSPENDED;
        toSuspend.push_back(workerId + i);
    }

    auto signalVar = new std::atomic_size_t{toSuspend.size()};

#ifdef SCHEDULDER_TREE
    int idOffset = 0; //  Id of the first leaf (number of internal nodes)
    int levelNodes = 1;
    int leafNodes = static_cast<int>(toSuspend.size());
    while (levelNodes < leafNodes) {
        idOffset += levelNodes;
        levelNodes *= _treeDegree;
    }

    auto signalVars = new std::atomic_size_t[idOffset];
    for (int i = 0; i < idOffset; i++) { signalVars[i].store(0); }
#endif

    // Create suspend task
    {
        auto argsSize = sizeof(ControlTaskArgs) + toSuspend.size() * sizeof(int);
        auto *args = static_cast<std::byte *>(std::malloc(argsSize));

        auto toSuspendIds = reinterpret_cast<int *>(args + sizeof(ControlTaskArgs));
        for (int i = 0; i < toSuspend.size(); i++) {
            toSuspendIds[i] = toSuspend[i];
        }

        // Fill control task args and submit the task
        auto &data = *reinterpret_cast<ControlTaskArgs *>(args);
        data.ctxId = _ctxId;
        data.workerId = workerId;
        data.numWorkersToUpdate = static_cast<int>(toSuspend.size());
        data.resume = false;
        data.workersToUpdate = toSuspendIds;
        data.signalVar = signalVar;

        auto schedulerArgs = static_cast<SchedulerArgs *>(std::malloc(sizeof(SchedulerArgs)));
        schedulerArgs->workerId = workerId;
        schedulerArgs->taskType = SchedulerArgs::EnumTaskType::SuspendTask;

        auto task = starpu_task_create();
        task->cl = &_controlTaskCl;
        task->cl_arg = args;
        task->cl_arg_size = argsSize;
        task->cl_arg_free = 1;
        task->sched_data = schedulerArgs;

        starpu_task_submit_to_ctx(task, _ctxId);
    }

    for (int i = 0; i < toSuspend.size(); ++i) {
        int suspendWorkerId = toSuspend[i];

        // Create SignalTasksArgs and submit the task
        auto args = static_cast<SignalTaskArgs *>(std::malloc(sizeof(SignalTaskArgs)));
        args->signalVar = signalVar;
        args->workerId = suspendWorkerId;

#ifdef SCHEDULDER_TREE
        args->signalVars = signalVars;
        args->treeDegree = _treeDegree;
        args->logicalId = idOffset + i;
        args->treeSize = idOffset + toSuspend.size();
        args->numInternalNodes = idOffset;
#endif

        auto schedulerArgs = static_cast<SchedulerArgs *>(std::malloc(sizeof(SchedulerArgs)));
        schedulerArgs->workerId = suspendWorkerId;
        schedulerArgs->taskType = SchedulerArgs::EnumTaskType::SignalTaskMerge;

        auto task = starpu_task_create();
        task->cl = &_signalTaskCl;
        task->cl_arg = args;
        task->cl_arg_size = sizeof(SignalTaskArgs);
        task->cl_arg_free = 1;
        task->sched_data = schedulerArgs;

        starpu_task_submit_to_ctx(task, _ctxId);
    }

    return *this;
}

Scheduler &Scheduler::waitForAll() {
    starpu_task_wait_for_all();
    return *this;
}

void Scheduler::initCodelets() {
    starpu_codelet_init(&_controlTaskCl);
    _controlTaskCl.cpu_funcs[0] = controlTask_cpu;
    _controlTaskCl.nbuffers = 0;

    starpu_codelet_init(&_signalTaskCl);
    _signalTaskCl.cpu_funcs[0] = signalTask_cpu;
    _controlTaskCl.nbuffers = 0;
}

void Scheduler::pinThreads() {
    // region init codelet
    struct starpu_codelet clPinning{};
    starpu_codelet_init(&clPinning);
    clPinning.cpu_funcs[0] = pinThreads_cpu;
    clPinning.nbuffers = 0;

    // Pin threads
    for (int i = 0; i < _nworkers; i++) {
        for (const int nthreads : _concurrencyClasses) {
            if (i % nthreads != 0) { continue; }

            starpu_task *task = starpu_task_create();
            task->cl = &clPinning;

            auto args = static_cast<PinThreadsTaskArgs *>(std::malloc(sizeof(PinThreadsTaskArgs)));
            args->start = i;
            args->stride = 1;
            args->size = nthreads;

            task->cl_arg = args;
            task->cl_arg_size = sizeof(PinThreadsTaskArgs);
            task->cl_arg_free = 1;

            auto schedulerArgs = static_cast<SchedulerArgs *>(std::malloc(sizeof(SchedulerArgs)));
            schedulerArgs->workerId = i;
            schedulerArgs->taskType = SchedulerArgs::EnumTaskType::PinTask;
            task->sched_data = schedulerArgs;

            STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit_to_ctx");
            break;
        }
        starpu_task_wait_for_all();
    }

    starpu_task_wait_for_all();
}


bool Scheduler::isValidClass(int nthreads) {
    return std::binary_search(_concurrencyClasses.begin(), _concurrencyClasses.end(), nthreads, std::greater<>{});
}

void Scheduler::setConcurrencyClasses(std::vector<int> &&concurrencyClasses) {
    _concurrencyClasses = std::move(concurrencyClasses);
    std::sort(_concurrencyClasses.begin(), _concurrencyClasses.end(), std::greater<>{});

    assert(_nworkers % _concurrencyClasses.front() == 0);
    for (size_t i = 0; i < _concurrencyClasses.size() - 1; ++i) {
        assert(_concurrencyClasses[i] != _concurrencyClasses[i + 1]);
        assert(_concurrencyClasses[i] % _concurrencyClasses[i + 1] == 0);
    }
}
