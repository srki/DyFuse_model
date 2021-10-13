#include "PinThreadsTask.h"

#include <omp.h>
#include <thread>
#include <iostream>
#include <sstream>
#include <cassert>

static void pinThreads(int start, int stride, int size) {
    omp_set_num_threads(size);

#pragma omp parallel num_threads(size) default(none) shared(start, stride, std::cout)
    {
        int cpuid = start + omp_get_thread_num() * stride;
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpuid, &mask);
        int result = sched_setaffinity(0, sizeof(mask), &mask);

        std::stringstream ss;
        ss << "Pinning " << omp_get_thread_num() + 1 << "/" << omp_get_num_threads() << " to " << cpuid << std::endl;
//        std::cout << ss.str();
    }
}

static bool checkIfPinned(int start, int stride, int size) {
    size_t pinned = 0;

#pragma omp parallel default(none) shared(start, stride, std::cout, std::cerr) reduction(+:pinned)
    {
        int cpuid = sched_getcpu();
        int expectedCpuid = start + omp_get_thread_num() * stride;

        if (cpuid != expectedCpuid) {
            std::cerr << "Expected cpuid: " << expectedCpuid << "; observed cpuid:" << cpuid << std::endl;
        }
        assert(cpuid == expectedCpuid);

        if (cpuid == expectedCpuid) { pinned++; }
    }

    return size == pinned;
}

void pinThreads_cpu(void *[], void *clArgs) {
    auto args = static_cast<PinThreadsTaskArgs *>(clArgs);
    pinThreads(args->start, args->stride, args->size);
    checkIfPinned(args->start, args->stride, args->size);
}

void checkIfPinned_cpu(void *[], void *clArgs) {
    auto args = static_cast<PinThreadsTaskArgs *>(clArgs);
    checkIfPinned(args->start, args->stride, args->size);
}