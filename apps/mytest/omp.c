#include <omp.h>
#include <sched.h>
#include <stdio.h>


unsigned long tacc_rdtscp(int *core, int *node){
    unsigned long a,d,c;
    __asm__ volatile("rdtscp": "=a"(a),"=d"(d),"=c"(c));
    *node=(c & 0xFFF000) >> 12;
    *core= c & 0xFFF;
    return ((unsigned long)a) | (((unsigned long)d)<<32);;
}

int main(){
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int cpuid,nodeid;
        tacc_rdtscp(&cpuid,&nodeid);

        printf("Thread %3d is running on CPU %3d NUMA NODE %3d\n", tid, cpuid,nodeid);
        fflush(stdout);

    }
    return 0;
}