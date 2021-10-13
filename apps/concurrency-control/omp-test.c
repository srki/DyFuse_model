#define _GNU_SOURCE
#include <sched.h>

#include <stdio.h>
#include <omp.h>

#define DELAY 32

size_t delay(size_t log_delay) {
    return log_delay > 1 ? delay(log_delay - 1) + delay(log_delay - 1) : 2;
}

int main() {
    omp_set_nested(1);
    omp_set_num_threads(12);

#pragma omp parallel num_threads(4) default(none)
    {
        delay(DELAY);
        if (sched_getcpu() == 0 || sched_getcpu() == 9) { delay(32); }
#pragma omp parallel num_threads(3) default(none)
        {
            printf("Thread %d/%d; Team %d/%d; CPU %u\n", omp_get_thread_num(), omp_get_num_threads(),
                   omp_get_team_num(), omp_get_num_teams(), sched_getcpu());
            delay(DELAY);
        }
    }

    return 0;
}