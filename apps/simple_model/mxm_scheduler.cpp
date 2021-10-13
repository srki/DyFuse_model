
#include <otx/otx.h>
#include <mio/mio.h>
#include <starpu.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <bits/stdc++.h>

using namespace std;

namespace fs = std::filesystem;

extern "C" {
#include <GraphBLAS.h>
}

#define TYPE_GrB GrB_BOOL
#define MAX_THREAD 32
#define MAX_BLOCKS_IN_OUTPUT 256*256// 8x8: 1D=8x1, 2D=4x4
#include "../util/GrB_util.h"
#include "../util/get_feature.h"
#include "../util/block_matrix.h"
#include "../util/timer.h"
#include "../scheduler/Scheduler.h"
#include "../util/block_matrix_generic.h"
#include "../util/common.h"
#ifndef _USE_LIKWID
#define _USE_LIKWID
#endif
#ifdef _USE_LIKWID
#ifdef LIKWID_PERFMON
    #include <likwid.h>
#else
    #define LIKWID_MARKER_INIT
    #define LIKWID_MARKER_THREADINIT
    #define LIKWID_MARKER_SWITCH
    #define LIKWID_MARKER_REGISTER(regionTag)
    #define LIKWID_MARKER_START(regionTag)
    #define LIKWID_MARKER_STOP(regionTag)
    #define LIKWID_MARKER_CLOSE
    #define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
    #endif
#endif
struct TaskInfo {
    starpu_task *task;
    size_t i;
    size_t j;
    size_t k;
    GrB_Matrix result;

    double mxmTime;

    TaskInfo(starpu_task *task, size_t i, size_t j, size_t k)
            : task(task), i(i), j(j), k(k), result(GrB_NULL) {}
};

struct TaskArgs {
    GrB_Matrix M;
    GrB_Matrix A;
    GrB_Matrix B;
    GrB_Descriptor descriptor;
    int nthreads;

    TaskInfo *taskInfo;
};

std::vector<TaskInfo *> createTasks(starpu_codelet *cl, GrB_Matrix *bM, GrB_Matrix *bA, GrB_Matrix *bB,
                                    bool transposeA, bool transposeB, size_t nblocksM, size_t nblocksN,
                                    size_t nblocksL, int* nthreads) {
    std::vector<TaskInfo *> tasks;
    tasks.reserve(nblocksM * nblocksN * nblocksL);

    for (size_t i = 0; i < nblocksM; i++) {
        for (size_t j = 0; j < nblocksN; j++) {
            for (size_t k = 0; k < nblocksL; k++) {
                size_t idxC = i * nblocksN + j;
                size_t idxA = !transposeA ? i * nblocksL + k : k * nblocksL + i;
                size_t idxB = !transposeB ? k * nblocksL + j : j * nblocksL + k;

                auto task = starpu_task_create();
                task->cl = cl;

                auto args = static_cast<TaskArgs *>(malloc(sizeof(TaskArgs)));
                args->M = bM[idxC];
                args->A = bA[idxA];
                args->B = bB[idxB];

                GrB_Descriptor desc;
                OK(GrB_Descriptor_new(&desc));
                OK(GxB_Desc_set(desc, GrB_OUTP, GrB_REPLACE));
                if (transposeA) { OK(GxB_Desc_set(desc, GrB_INP0, GrB_TRAN)); }
                if (transposeB) { OK(GxB_Desc_set(desc, GrB_INP1, GrB_TRAN)); }
                OK(GxB_Desc_set(desc, GxB_DESCRIPTOR_NTHREADS, 1));
                args->descriptor = desc;
                args->nthreads = nthreads[idxC];
                //cout << "bid=" << idxC << ", nthreads=" << nthreads[idxC] << endl;
                //cout.flush();
                task->cl_arg = args;
                task->cl_arg_size = sizeof(TaskArgs);
                task->cl_arg_free = 1;

                auto taskInfo = new TaskInfo{task, i, j, k};
                args->taskInfo = taskInfo;

                tasks.push_back(taskInfo);
            }
        }
    }

    return tasks;
}



void test_scheduler_tricnt(string name, GrB_Matrix A, GrB_Matrix B, GrB_Matrix M,
                           size_t nblocksM, size_t nblocksN, size_t nblocksL,
                           size_t niter, double* featuresM,
                           vector<double> &report_time,
                           double* init_time,
                           Scheduler *scheduler,
                           int runscheduler, int rungrb, int runomp,
                           bool transposeA, bool transposeB) {


    int block_id,block_thread;
    vector<int> order;
    int nthreads[16384]={0};
    //string a="new_spgemm_for_tri_spgemm_mask_"+name+".csv";
    string a="new_tri_" + name + "_1013.csv";
    //cout << a  << "," << nblocksM << "," << nblocksN << "," << nblocksL << endl;
    //cout.flush();
    FILE* fp = fopen(a.c_str(),"r");
    while (fscanf(fp, "%d,%d", &block_id, &block_thread ) == 2) {
        //cout << block_id << "," << block_thread << endl;
        //cout.flush();
        order.push_back(block_id);
        if (block_id == 99999) {
            //printf("It's ok, wait...\n");
            //fflush(stdout);
            continue;
        }
        nthreads[block_id]=block_thread;
        //printf("%d,%d\n",block_id, nthreads[block_id]);
        //fflush(stdout);
    }

    fclose(fp);
    if ((nblocksM==1) && (nthreads[0]==MAX_THREAD)){
        cout << name << " uses the same config with baseline." << endl;
        cout.flush();
        return ;
    }

    GrB_Matrix *bA, *bB, *bM;
    createMxMBlocks(M, A, B, transposeA, transposeB, nblocksM, nblocksN, nblocksL, GrB_UINT64, nullptr, &bM, &bA, &bB);
    // region init codelet
    starpu_codelet cl{};
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
        auto args = static_cast<TaskArgs *>(clArgs);
        OK(GxB_Desc_set(args->descriptor, GxB_DESCRIPTOR_NTHREADS, args->nthreads));
        GrB_Index nrows, ncols;
        GrB_Matrix_nrows(&nrows, args->A);
        GrB_Matrix_nrows(&ncols, args->B); // transposed

        GrB_Matrix C;
        GrB_Matrix_new(&C, GrB_UINT64, nrows, ncols);

        timer_start();
        OK(GrB_mxm(C, args->M, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64, args->A, args->B, args->descriptor));
        args->taskInfo->result = C;
        args->taskInfo->mxmTime = timer_end();
    };
    cl.nbuffers = 0;

    int skip = 0;
    int myworker = 0;
    /*
    int nthreads[4]={16,1,0,16};
    vector<int> order(4, 0);
    order[0]=0;
    order[1]=3;
    order[2]=999;
    order[3]=1;
     */

    
    
    //printf("myblock_id.size=%d\n",order.size());
    //fflush(stdout);
    if (nthreads[order[0]]< scheduler->getNumWorkers() ) {
        int ii=0;
        scheduler->splitWorker(ii * nthreads[order[0]], scheduler->getNumWorkers() / nthreads[order[0]], nthreads[order[0]]);
        //printf("1-splitWorker %d into %d Q with %d threads\n", ii, scheduler->getNumWorkers()/ nthreads[order[0]],
        //       nthreads[order[0]]);
        //fflush(stdout);
    }

#ifdef _USE_LIKWID
    LIKWID_MARKER_START("scheduler");
#endif
    for (int iter = 0; iter < niter + skip; iter++) {
        if (iter >= skip) timer_start();
        auto tasks = createTasks(&cl, bM, bA, bB, transposeA, transposeB, nblocksM, nblocksN, nblocksL, nthreads);
        int nextWorker = 0;
        int old_threads= nthreads[order[0]];
        for (int taskid=0; taskid<order.size();taskid++) {

            //cout << "blockTask:" << taskid << "/" << order.size() << "," << order[taskid]<< endl;
            //cout.flush();
            if (order[taskid] == 99999) {
                //cout << "wait for all" << endl;
                //cout.flush();
                scheduler->waitForAll();
                int split_iter=scheduler->getNumWorkers()/old_threads;
                for (int ii = 0; ii < split_iter; ii++) {
                    scheduler->splitWorker(ii * old_threads, old_threads / nthreads[order[taskid+1]], nthreads[order[taskid+1]]);
                    nextWorker=0;
                    //printf("2-splitWorker %d into %d Q with %d threads\n", ii, old_threads / nthreads[order[taskid+1]], nthreads[order[taskid+1]]);
                    //fflush(stdout);
                }
            }else{
                scheduler->submitTask(tasks[order[taskid]]->task, nextWorker * (nthreads[order[taskid]]), nthreads[order[taskid]]);
                //cout << "iter=" << iter << ", submit task to Q "<< nextWorker* (nthreads[order[taskid]]) << ", nthreads=" << nthreads[order[taskid]] << endl;
                //cout.flush();
                old_threads=nthreads[order[taskid]];
                nextWorker++;
                if (nextWorker >= MAX_THREAD/old_threads) nextWorker=0;
            }

        }
        scheduler->waitForAll();
        if (iter >= skip) {
            report_time[0] += timer_end();
            cout << "spu, iter=" << iter << "," << report_time[0] << endl;
            cout.flush();
        }

    }
#ifdef _USE_LIKWID
    LIKWID_MARKER_STOP("scheduler");
    LIKWID_MARKER_CLOSE;
#endif
    freeMxMBlocks(nullptr, bM, bA, bB, nblocksM, nblocksN, nblocksL);

}

void test_grbbase_tricnt(const char *name, GrB_Matrix A, GrB_Matrix B, GrB_Matrix M,
                         size_t nblocksM, size_t nblocksN, size_t nblocksL,
                         size_t niter, double* featuresM,
                         vector<double> &report_time,
                         double* init_time,
                         Scheduler *scheduler,
                         int runscheduler, int rungrb, int runomp,
                         bool transposeA, bool transposeB){
    GrB_Matrix *bA, *bB, *bM;
    createMxMBlocks(M, A, B, transposeA, transposeB, nblocksM, nblocksN, nblocksL, GrB_UINT64, nullptr, &bM, &bA, &bB);

    // region init codelet
    starpu_codelet cl{};
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
        auto args = static_cast<TaskArgs *>(clArgs);
        OK(GxB_Desc_set(args->descriptor, GxB_DESCRIPTOR_NTHREADS, args->nthreads));
        GrB_Index nrows, ncols;
        GrB_Matrix_nrows(&nrows, args->A);
        GrB_Matrix_nrows(&ncols, args->B); // transposed

        GrB_Matrix C;
        GrB_Matrix_new(&C, GrB_UINT64, nrows, ncols);

        timer_start();
        OK(GrB_mxm(C, args->M, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64, args->A, args->B, args->descriptor));
        args->taskInfo->result = C;
        args->taskInfo->mxmTime = timer_end();
    };
    cl.nbuffers = 0;

    int skip = 0;
    int myworker = 0;
    int nthreads[1]={32};
    vector<int> order(1, 0);
    order[0]=0;
    int firstblock=1;

#ifdef _USE_LIKWID
    LIKWID_MARKER_START("grb");
#endif
    for (int iter = 0; iter < niter + skip; iter++) {
        if (iter >= skip) timer_start();
        auto tasks = createTasks(&cl, bM, bA, bB, transposeA, transposeB, nblocksM, nblocksN, nblocksL, nthreads);

        if (firstblock==1)
            if (nthreads[order[0]]< scheduler->getNumWorkers() ) {
                int ii=0;
                scheduler->splitWorker(ii * nthreads[order[0]], scheduler->getNumWorkers() / nthreads[order[0]], nthreads[order[0]]);
                //printf("1-splitWorker %d into %d Q with %d threads\n", ii, scheduler->getNumWorkers()/ nthreads[order[0]],
                //       nthreads[order[0]]);
                //fflush(stdout);
                firstblock=0;
            }
        int nextWorker = 0;
        int old_threads= nthreads[order[0]];
        for (int taskid=0; taskid<order.size();taskid++) {

            //cout << "blockTask:" << taskid << "/" << order.size() << "," << order[taskid]<< endl;
            //cout.flush();
            if (order[taskid] == 99999) {
                //cout << "wait for all" << endl;
                //cout.flush();
                scheduler->waitForAll();
            }else if ((taskid >0 ) && (nthreads[order[taskid]] != old_threads)){
                int split_iter=scheduler->getNumWorkers()/old_threads;
                for (int ii = 0; ii < split_iter; ii++) {
                    scheduler->splitWorker(ii * old_threads, old_threads / nthreads[order[taskid]], nthreads[order[taskid]]);
                    //printf("2-splitWorker %d into %d Q with %d threads\n", ii, old_threads / nthreads[order[taskid]], nthreads[order[taskid]]);
                    //fflush(stdout);
                }
            }else{
                scheduler->submitTask(tasks[taskid]->task, nextWorker * (nthreads[order[taskid]]), nthreads[order[taskid]]);
                //cout << "iter=" << iter << ", submit task to Q "<< nextWorker* (nthreads[order[taskid]]) << ", nthreads=" << nthreads[order[taskid]] << endl;
                //cout.flush();
                nextWorker++;
                old_threads=nthreads[order[taskid]];

            }
        }
        scheduler->waitForAll();
        if (iter >= skip) {
            report_time[0] += timer_end();
            cout << "grb, iter=" << iter << "," << report_time[0] << endl;
            cout.flush();
        }

    }
#ifdef _USE_LIKWID
    LIKWID_MARKER_STOP("grb");
    LIKWID_MARKER_CLOSE;
#endif
    freeMxMBlocks(nullptr, bM, bA, bB, nblocksM, nblocksN, nblocksL);
    //printf("Done grb\n");
    //fflush(stdout);
}

double init_time[15]={0.0};
int main(int argc, char *argv[]) {
    auto niter = otx::argTo<size_t>(argc, argv, {"-niter"}, 3);
    auto opt = otx::argTo<int>(argc, argv, "-o", 0);
    auto runscheduler = otx::argTo<int>(argc, argv, {"-runscheduler"}, 0);
    auto rungrb = otx::argTo<int>(argc, argv, {"-rungrb"}, 0);
    auto pathA = otx::argTo<std::string>(argc, argv, "-A", INPUTS_DIR"/simple");
#ifdef _USE_LIKWID
    LIKWID_MARKER_INIT;

#pragma omp parallel
    {
        //LIKWID_MARKER_THREADINIT;
        LIKWID_MARKER_REGISTER("grb");
        LIKWID_MARKER_REGISTER("scheduler");
    }
#endif


    OK(GrB_init(GrB_NONBLOCKING))


    int runomp = 0;
    Scheduler scheduler;
    auto numWorkers = scheduler.getNumWorkers();

    cout << "numWorkers = " << numWorkers << endl;
    cout.flush();

    if (numWorkers == 128) {
        scheduler.start({128, 64, 32, 16, 8, 4, 1});
    } else if (numWorkers == 64) {
        scheduler.start({64, 32, 16, 8, 1});
    } else if (numWorkers == 32) {
        scheduler.start({32, 16, 8, 4, 1});
    } else {
        cout << "numWorkers = " << numWorkers << endl;
        cout.flush();
    }

    int new_matrix = 0;
    int model_nthreads;

    //for(auto& p: fs::recursive_directory_iterator("/home/nanding/train_matrix/")) {
    //for(auto& p: fs::recursive_directory_iterator("/home/nanding/big_matrix/")) {
    //for(auto& p: fs::recursive_directory_iterator("/home/nanding/refer_matrix/")) {
    //for(auto& p: fs::recursive_directory_iterator("/project/projectdirs/m2956/nanding/myprojects/dyfuse/matrix/profile_matrix")) {
    //for(auto& p: fs::recursive_directory_iterator("/project/projectdirs/m2956/nanding/myprojects/dyfuse/matrix/srdan_matrix")){
    //for(auto& p: fs::recursive_directory_iterator("/home/nanding/test_matrix_quick/")) {
    int nblocks = 1;
    string a = pathA.c_str();
    std::replace(a.begin(), a.end(), '/', ' ');
    stringstream ss(a);
    string word;
    while (ss >> word) {};
if (runscheduler==1){

    /* try to open file to read */
    string name = "new_tri_block_" + word + "_1013.csv";
    ifstream ifile;
    ifile.open(name);
    if (ifile) {
        cout << "file exists" << endl;

        int r, c;
        FILE *fp = fopen(name.c_str(), "r");

        while (fscanf(fp, "%d,%d", &r, &c) == 2) {
            nblocks = c;
        }

        fclose(fp);
        cout << "Testing:" << word << "," << nblocks << '\n';
        cout.flush();

    } else {
        cout << name << " file doesn't exist" << endl;
        return 0;
    }
}

        GrB_Matrix L;
        GrB_Matrix U;
        GrB_Matrix M;
        GrB_Index nrows;
        GrB_Index ncols;


        timer_start();
        readLU(&L, &U, pathA.c_str(), 0, GrB_UINT64);
        init_time[0] += timer_end();

        GrB_Matrix_dup(&M, L);
        GrB_Matrix_nrows(&nrows, L);
        GrB_Matrix_ncols(&ncols, U);
        //printf("end read LU, nrows=%ld,ncols=%ld\n",nrows,ncols);
        //fflush(stdout);
        // 1D
        if (runscheduler==1) {
            //cout << "nblocks=" << nblocks << endl;
            //cout.flush();
            double *featuresM;
            if (!(featuresM = (double *) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double)))) ABORT(
                    "Malloc fails for featuresM[].");
            vector<double> report_time(MAX_BLOCKS_IN_OUTPUT, 0);
            test_scheduler_tricnt(word, L, U, L, nblocks, nblocks, 1,
                                  niter, featuresM,
                                  report_time,
                                  init_time,
                                  &scheduler, runscheduler, rungrb, runomp,
                                  false, true
            );


            //myfile << "omp," << p.path().c_str() << "," << nblocks << "," << gbase_time[nblocks][0]/niter << endl;
            //myfile.flush();
            free(featuresM);
        }


        if (rungrb==1) {
            double *featuresM;
            int nblocks=1;
            if (!(featuresM = (double *) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double)))) ABORT(
                    "Malloc fails for featuresM[].");
            vector<double> report_time(MAX_BLOCKS_IN_OUTPUT, 0);
            test_grbbase_tricnt(pathA.c_str(), L, L, U, nblocks, nblocks, 1,
                                niter, featuresM,
                                report_time,
                                init_time,
                                &scheduler, runscheduler, rungrb, runomp,
                                false, true);
        }
        GrB_Matrix_free(&L);
        GrB_Matrix_free(&U);


    //}

    scheduler.stop();
#ifdef _USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
    OK(GrB_finalize());
    return 0;
}
