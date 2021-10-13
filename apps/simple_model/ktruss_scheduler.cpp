#include <starpu.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <otx/otx.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <algorithm>

using namespace std;
namespace fs = std::filesystem;

extern "C" {
#include <GraphBLAS.h>
}

#define TYPE_GrB GrB_BOOL
#define MAX_THREAD 64
#define MAX_BLOCKS_IN_OUTPUT 128*128 // 8x8: 1D=8x1, 2D=4x4
#include "../util/GrB_util.h"
#include "../util/get_feature.h"

#include "../util/block_matrix_feature.h"
#include "../util/timer.h"

#include "../scheduler/Scheduler.h"
#include "../util/reader_c.h"

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


void test_grbbase_spgemm(string name, GrB_Matrix A, size_t k,
                         size_t nblocksI, size_t nblocksJ, size_t nblocksK,
                         size_t niter, size_t skip, double* featuresA, double* featuresB, double* featuresC, double* featuresM,
                         vector<double> &report_time,
                         double* init_time,
                         Scheduler *scheduler,
                         int runscheduler, int rungrb, int runomp,
                         bool transposeA, bool transposeB) {

    cout << "Enter...baseline," << name << endl;
    cout.flush();

    GrB_Index nrows;
    GrB_Index nnz_cur, nnz_last;
    GxB_Scalar s = NULL;
    GrB_Matrix C = NULL, T = NULL;

    GxB_Scalar_new(&s, GrB_UINT64);
    GxB_Scalar_setElement_UINT64(s, (uint64_t) k - 2);
    GrB_Matrix_nrows(&nrows, A);

    T = A;
    GrB_Matrix_nvals(&nnz_cur, A);
    int myiter=0;
    do {
        nnz_last = nnz_cur;
        GrB_Matrix_new(&C, GrB_UINT64, nrows, nrows);
        GrB_mxm(C, T, NULL, GxB_PLUS_PAIR_UINT64, T, T, GrB_DESC_S);

        GxB_Matrix_select(C, NULL, NULL, GxB_GE_THUNK, C, s, GrB_NULL);
        if (A != T) { GrB_Matrix_free(&T); }
        T = C;
        GrB_Index mask_rows, mask_cols, mask_nnz;
        GrB_Matrix_nrows(&mask_rows, T);
        GrB_Matrix_ncols(&mask_cols, T);
        GrB_Matrix_nvals(&mask_nnz, T);
        printf("mask %d: %lu,%lu,%lu\n",myiter,mask_rows,mask_cols,mask_nnz);
        fflush(stdout);
        GrB_Matrix_nvals(&nnz_cur, C);
        myiter++;
    } while (nnz_last != nnz_cur);
    GrB_Matrix_free(&T);

    printf("Done, k=%lu, nnz_last=%lu\n", k,nnz_last);
}

double init_time[15]={0.0};
int main(int argc, char *argv[]) {
    auto niter = otx::argTo<size_t>(argc, argv, {"-niter"}, 1);
    auto nskip = otx::argTo<size_t>(argc, argv, {"-nskip"}, 0);
    auto opt = otx::argTo<int>(argc, argv, "-o", 0);
    auto runscheduler = otx::argTo<int>(argc, argv, {"-runscheduler"}, 0);
    auto rungrb = otx::argTo<int>(argc, argv, {"-rungrb"}, 0);
    auto pathA = otx::argTo<std::string>(argc, argv, "-A", INPUTS_DIR"/simple");
    auto k = otx::argTo<int>(argc, argv, {"-k"}, 3);
    assert(k >= 3);
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

    GrB_Matrix A;

    timer_start();
    if (read_Matrix_FP64(&A, pathA.c_str(), 0) != GrB_SUCCESS) { return 1; }
    init_time[0] += timer_end();


    /* Remove A's diagonal */
    GxB_Scalar s;
    GxB_Scalar_new(&s, GrB_UINT64);
    GxB_Scalar_setElement_UINT64(s, (uint64_t) 0);
    GxB_Matrix_select(A, GrB_NULL, GrB_NULL, GxB_OFFDIAG, A, s, GrB_NULL);

    /* Convert A to matrix to symmetric 0-1 matrix */
    GrB_Index nrows;
    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_assign_INT32(A, A, GrB_NULL, 1, GrB_ALL, nrows, GrB_ALL, nrows, GrB_DESC_S);
    GrB_Matrix_eWiseAdd_BinaryOp(A, GrB_NULL, GrB_NULL, GrB_TIMES_UINT64, A, A, GrB_DESC_T1);


    if (rungrb==1) {
        double *featuresA,*featuresB,*featuresC, *featuresM;
        if (!(featuresA = (double *) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))))
        ABORT("Malloc fails for featuresA[].");
        if (!(featuresB = (double *) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))))
        ABORT("Malloc fails for featuresB[].");
        if (!(featuresC = (double *) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))))
        ABORT("Malloc fails for featuresC[].");
        if (!(featuresM = (double *) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))))
        ABORT("Malloc fails for featuresM[].");
        int nblocks=1;

        vector<double> report_time(MAX_BLOCKS_IN_OUTPUT, 0);
        test_grbbase_spgemm(word, A, k, nblocks, 1, 1,
                            niter,nskip, featuresA, featuresB,featuresC,featuresM,
                            report_time,
                            init_time,
                            &scheduler, runscheduler, rungrb, runomp,
                            false, true);
        free(featuresA);
        free(featuresB);
        free(featuresC);
        free(featuresM);
    }

    GrB_Matrix_free(&A);


    //}

    scheduler.stop();
#ifdef _USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
    OK(GrB_finalize());
    return 0;
}
