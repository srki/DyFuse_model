#ifndef GRB_FUSION_TIMER_H
#define GRB_FUSION_TIMER_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <time.h>

static unsigned long long int get_time_ns() {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return 1000000000ULL * (unsigned long long) time.tv_sec + (unsigned long long) time.tv_nsec;
}

#define MAX_TIMERS 10

static double timer(bool start) {
    static size_t current_timer = 0;
    static long long unsigned start_time[MAX_TIMERS];

    assert(current_timer < MAX_TIMERS);

    if (start) {
        start_time[current_timer++] = get_time_ns();
    } else {
        return (double) (get_time_ns() - start_time[--current_timer]) / 1e9;
    }

    return 0;
}

static void timer_start() { timer(true); }

static double timer_end() { return timer(false); }

#endif //GRB_FUSION_TIMER_H
