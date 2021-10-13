#ifndef GRB_FUSION_PIN_THREADS_TASK_H
#define GRB_FUSION_PIN_THREADS_TASK_H

struct PinThreadsTaskArgs {
    int start;
    int stride;
    int size;
};

void pinThreads_cpu(void *[], void *clArgs);

void checkIfPinned_cpu(void *[], void *clArgs);

#endif //GRB_FUSION_PIN_THREADS_TASK_H
