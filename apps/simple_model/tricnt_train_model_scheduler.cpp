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
#include "../util/block_matrix.h"
#include "../util/timer.h"
#include "../scheduler/Scheduler.h"

struct Arg {
    GrB_Matrix A;
    GrB_Matrix B;
    GrB_Matrix C;
    GrB_Matrix M;
    int numThreads;

    Arg() = default;

    Arg(GrB_Matrix a, GrB_Matrix b, GrB_Matrix c, GrB_Matrix m, int numThreads)
            : A(a), B(b), C(c), M(m), numThreads(numThreads) {}
};

void thread_affinity(int myname, int nthreads) {
    int tid;
#pragma omp parallel shared(myname, nthreads) private(tid) default(none)
    {
        tid=omp_get_thread_num();
        if (tid<nthreads) {
            printf("(%d) Hello world from thread %d of %d running on cpu %2d!\n",
                   myname,
                   omp_get_thread_num(),
                   omp_get_num_threads(),
                   sched_getcpu());
        }
    }
}

//Run a simple file


void test_scheduler_tricnt(const char *name, GrB_Matrix A, GrB_Matrix B, GrB_Matrix M, size_t nblocks_m, size_t nblocks_n, size_t nblocks_l,
               size_t niter, double* featuresM,  vector<vector<double>> &report_time,  vector<vector<double>> &gbase_time,
               double* init_time,Scheduler *scheduler, int count, int runscheduler, int rungrb, int runomp, int mtxcount) {
    GrB_Matrix C;
    GrB_Matrix *bA, *bB, *bC, *bM;
    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t block_size_m, block_size_n, block_size_l;
    uint64_t sum;
    //printf("enter test_scheduler\n");
    //fflush(stdout);

    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_nrows(&n, B);
    GrB_Matrix_ncols(&l, A);
    block_size_m = (m + nblocks_m - 1) / nblocks_m;
    block_size_n = (n + nblocks_n - 1) / nblocks_n;
    block_size_l = (l + nblocks_l - 1) / nblocks_l;

    GrB_Matrix_new(&C, GrB_UINT64, m, n);

    //printf("before create_blocks,block_size_m,n,l=%ld,%ld,%ld\n",block_size_m,block_size_n,block_size_l);
    //fflush(stdout);
    timer_start();
    create_blocks(&bA, &bm, &bl, A, block_size_m, block_size_l, 0);
    create_blocks(&bB, &bn, &bl, B, block_size_n, block_size_l, 0);
    create_blocks(&bC, &bm, &bn, C, block_size_m, block_size_n, 0);
    //printf("end create_blocks\n");
    //fflush(stdout);
    create_blocks_features(&bM, &bm, &bn, M, block_size_m, block_size_n, 0, featuresM);
    //printf("end create_blocks_features\n");
    //fflush(stdout);
    init_time[1] += timer_end();

    if (nblocks_m != bm || nblocks_n != bn || nblocks_l != bl) {
        exit(5);
    }

    starpu_codelet cl{};
    if (runscheduler == 1) {
        starpu_codelet_init(&cl);
        cl.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
            auto arg = (Arg *) clArgs;

            GrB_Descriptor desc;
            GrB_Descriptor_new(&desc);
            GrB_Descriptor_set(desc, GrB_INP1, GrB_TRAN);
            GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE);
            GrB_Descriptor_set(desc, GxB_DESCRIPTOR_NTHREADS, static_cast<GrB_Desc_Value>(arg->numThreads));
            GrB_mxm(arg->C, arg->M, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64, arg->A, arg->B, desc);
        };
        cl.nbuffers = 0;
    }
    int nthreads[5] = {64, 32, 16, 8, 1};
    //int nthreads[1] = {32};
    //int nthreads[5] = {64, 32, 16,8, 1};
    //cout << "Total Qs = " << sizeof(nthreads)/sizeof(nthreads[0]) << endl;
    //cout.flush();

    int merge_iter, split_iter;
    auto args = new Arg[bm * bn * bl];

    int skip = 2;
    for (int tid = 0; tid < sizeof(nthreads)/sizeof(nthreads[0]); tid++) {
        if (runscheduler == 1) {
            if (tid == 0) {
                cout << "nthreads=" << nthreads[tid] << "," << MAX_THREAD << ",merge workers: 1 -> 64" << endl;
                cout.flush();
                timer_start();
                merge_iter = MAX_THREAD / nthreads[tid];
                scheduler->mergeWorkers(0, nthreads[tid]);
                init_time[2 + tid] += timer_end();
            } else {
                cout << "nthreads=" << nthreads[tid] << "," << MAX_THREAD << ",split workers: " << nthreads[tid - 1]
                     << "->"
                     << nthreads[tid] << endl;
                cout.flush();
                timer_start();
                split_iter = MAX_THREAD / nthreads[tid - 1];
                for (int ii = 0; ii < split_iter; ii++) {
                    scheduler->splitWorker(ii * nthreads[tid], nthreads[tid - 1] / nthreads[tid], nthreads[tid]);
                    //printf("splitWorker %d into %d Q with %d threads\n", ii, nthreads[tid - 1] / nthreads[tid],
                    //       nthreads[tid]);
                    //fflush(stdout);
                }
                init_time[2 + tid] += timer_end();
            }

            int myworker = 0;
            for (size_t i = 0; i < bm; i++) {
                for (size_t j = 0; j < bn; j++) {
                    for (size_t k = 0; k < bl; k++) {

                        for (int iter = 0; iter < niter + skip; iter++) {
                            starpu_task_wait_for_all();
                            if (iter >= skip) timer_start();

                            int myidx = i * bn + j * bl + k;
                            sum = 0;
                            int bidA = i * bl + k;
                            int bidB = j * bl + k;
                            int bidM = i * bn + j;
                            auto task = starpu_task_create();
                            task->cl = &cl;
                            task->cl_arg = &(args[i * bn * bl + j * bl + k] = Arg(bA[i * bl + k], bB[j * bl + k],
                                                                                  bC[i * bn + j], bM[i * bn + j],
                                                                                  nthreads[tid]));
                            task->cl_arg_size = sizeof(Arg);
                            task->cl_arg_free = 0;
                            //cout << "MAX_THREAD=" <<MAX_THREAD << ",nthreads=" << nthreads[tid] << ", val=" << keep_busy_q << endl;
                            //cout.flush();

                            scheduler->submitTask(task, myworker * nthreads[tid], nthreads[tid]);
                            //cout << "iter=" << i << ", submit task to Q, nthreads=" << nthreads[tid] << ",workerID="
                            //     << myworker * nthreads[tid] << endl;
                            //cout.flush();
                            starpu_task_wait_for_all();
                            if (iter >= skip) {
                                report_time[myidx][tid] += timer_end();
                                cout << "spu: " << nthreads[tid] << ", iter=" << iter << "," << report_time[myidx][tid] << endl;
                                cout.flush();
                            }
                            starpu_task_wait_for_all();
                        } // iter over blocks
                    }
                }
            }// iter over blockls
        }
        if (rungrb==1){
            skip=0;
            cout << "Running Grb... threads=" << nthreads[tid] << endl;
            cout.flush();
            GxB_Global_Option_set(static_cast<GxB_Option_Field>(GxB_NTHREADS), nthreads[tid]);
            for (size_t i = 0; i < bm; i++) {
                for (size_t j = 0; j < bn; j++) {
                    for (size_t k = 0; k < bl; k++) {
                        int myidx = i * bn + j * bl + k;
                        for (int iter = 0; iter < niter + skip; iter++) {
                            sum = 0;
                            if (iter >= skip) timer_start();
                            int bidA = i * bl + k;
                            int bidB = j * bl + k;
                            int bidM = i * bn + j;
                            OK(GrB_mxm(bC[i * bn + j], bM[i * bn + j], GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64,
                                       bA[i * bl + k], bB[j * bl + k], GrB_DESC_RT1));
                            //GrB_Matrix_reduce_UINT64(&sum, TYPE_PLUS, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
                            if (iter >= skip) {
                                gbase_time[myidx][tid] += timer_end();
                                //cout << "grb: " << nthreads[tid] << ", iter=" << iter << "," << gbase_time[myidx][tid] << endl;
                                //cout.flush();
                            }
                            //printf("end submit task\n");
                            //fflush(stdout);
                        }// iter over blocks

                    }
                }
            }
        }

        if (runomp == 1) {
            cout << "TEST OMP...." << endl;
            cout.flush();
            GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, 1);
            int tid = 0;
            int skip = 2;
            for (int iter = 0; iter < niter + skip; iter++) {
                if (iter >= skip) timer_start();
#pragma omp parallel for collapse(3)
                for (size_t i = 0; i < bm; i++) {
                    for (size_t j = 0; j < bn; j++) {
                        for (size_t k = 0; k < bl; k++) {
                            int bidA = i * bl + k;
                            int bidB = j * bl + k;
                            int bidM = i * bn + j;
                            OK(GrB_mxm(bC[i * bn + j], bM[i * bn + j], GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64,
                                       bA[i * bl + k], bB[j * bl + k], GrB_DESC_RT1));
                            //GrB_Matrix_reduce_UINT32(&sum, TYPE_PLUS, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
                            //printf("end submit task\n");
                            //fflush(stdout);
                        }// iter over blocks
                    }
                }
                if (iter >= skip) {
                    gbase_time[nblocks_m][0] += timer_end();
                    cout << "omp, iter=" << iter << "," << ",nblocks=" << nblocks_m << "," << gbase_time[nblocks_m][0]
                         << endl;
                    cout.flush();
                }
            } // iter
        }
    }

    std::string rankfile("srdan_1_train");
    rankfile += "_";
    rankfile += std::to_string(count);
    cout << "dumping file name is: " << rankfile << endl;
    cout.flush();

    std::ofstream myfile;
    myfile.open (rankfile,std::ofstream::out | std::ofstream::app);

    //for (size_t i = 0; i < bm; i++) {
    //    for (size_t j = 0; j < bn; j++) {
    //        for (size_t k = 0; k < bl; k++) {
    //            int myidx=i*bn + j*bl + k;
    //            myfile << name <<", blocks," << nblocks_m  << ",id," << myidx << "," << i << "-" << j << "-"<< k
    //                   << ", nrows," << featuresM[myidx*NUM_FEATURE+1]
    //                   << ", ncols," << featuresM[myidx*NUM_FEATURE+2]
    //                   << ", nnz," << featuresM[myidx*NUM_FEATURE+3];
    //            myfile.flush();
    //            for (int tid=0;tid<sizeof(nthreads)/sizeof(nthreads[0]);tid++){
    //                myfile << "," <<  nthreads[tid] << "," << gbase_time[myidx][tid]/niter;
    //                myfile.flush();
    //            }
    //            myfile << endl;
    //            myfile.flush();
    //        }
    //    }
    //}
    if ((mtxcount==0) && (bm==1)) {
        myfile << "Matrix,readLU,createblocks1D, blocks,nrows,ncols,nnz";
        for (int tid = 0; tid < sizeof(nthreads) / sizeof(nthreads[0]); tid++) {
            myfile << ",scheduler" << nthreads[tid] << ",spu" << nthreads[tid] << ",grb" << nthreads[tid] ;
        }
        myfile << endl;
        myfile.flush();
    }
    for (size_t i = 0; i < bm; i++) {
        for (size_t j = 0; j < bn; j++) {
            for (size_t k = 0; k < bl; k++) {
                int myidx=i*bn + j*bl + k;
                myfile << name
                      << "," << init_time[0]
                      << "," << init_time[1]
                      << "," << nblocks_m
                      << "," << featuresM[myidx*NUM_FEATURE+1]
                      << "," << featuresM[myidx*NUM_FEATURE+2]
                      << "," << featuresM[myidx*NUM_FEATURE+3];
                myfile.flush();
                for (int tid=0;tid<sizeof(nthreads)/sizeof(nthreads[0]);tid++){
                    if (tid==0) {
                        myfile << "," << init_time[2+tid]
                               << "," << report_time[myidx][tid]/niter
                               << "," << gbase_time[myidx][tid]/niter;
                        myfile.flush();
                    }else{
                        myfile << "," << init_time[2+tid]
                               << "," << report_time[myidx][tid]/niter
                               << "," << gbase_time[myidx][tid]/niter;
                        myfile.flush();
                    }
                }
                myfile << endl;
                myfile.flush();
            }
        }
    }

    myfile.close();
    //delete[] args;
    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    free_blocks(bM, bm, bn);
    GrB_Matrix_free(&C);
    printf("Done\n");
    fflush(stdout);

}

double init_time[100]={0.0};
int main(int argc, char *argv[]) {
    auto niter = otx::argTo<size_t>(argc, argv, {"-niter"}, 3);
    auto opt = otx::argTo<int>(argc, argv, "-o", 0);
    OK(GrB_init(GrB_NONBLOCKING))


    Scheduler scheduler{};
    auto numWorkers = scheduler.getNumWorkers();
    if (numWorkers == 128)
        scheduler.start({128, 64, 32, 16, 8, 4, 1});
    else if (numWorkers == 64)
        scheduler.start({64, 32, 16, 1});
    else
        scheduler.start({12, 6, 3, 1});
    //printf("after  Scheduler scheduler{}\n");
    //fflush(stdout);
    // init: all 1 thread q

    //scheduler.start();
    //printf("after  scheduler.start();\n");
    //fflush(stdout);
    int count=0;
    int noblocks=0;
    int model_nthreads;

    std::string rankfile("model_train_0");

    int tmp = 1;
    int flag = 1;
    while (flag) {
        //cout << "This time file is " << rankfile << "," << tmp << endl;
        //cout.flush();
        ifstream ifile;
        ifile.open(rankfile);
        if (ifile) {
            //cout << rankfile << ",File exists" << endl;
            //cout.flush();

            ifile.close();
            rankfile = "model_train_";
            rankfile += std::to_string(tmp);

            //cout << "new file:" << rankfile << endl;
            //cout.flush();
            tmp = tmp + 1;
        } else {
            //cout << rankfile << ",File does not exists" << endl;
            //cout.flush();
            break;
        }
    }

    int final_rankfile_num;
    final_rankfile_num = tmp-1;
    cout << "output file num: " << final_rankfile_num << endl;
    cout.flush();

    //rankfile = "model_train_"+std::to_string(final_rankfile_num);
    //std::ofstream myfile;
    //myfile.open (rankfile,std::ofstream::out | std::ofstream::app);


    //for(auto& p: fs::recursive_directory_iterator("/home/nanding/train_matrix/")) {
    //for(auto& p: fs::recursive_directory_iterator("/home/nanding/big_matrix/")) {
    //for(auto& p: fs::recursive_directory_iterator("/home/nanding/refer_matrix/")) {
    for(auto& p: fs::recursive_directory_iterator("/home/nanding/srdan_matrix_1/")) {
    //for(auto& p: fs::recursive_directory_iterator("/home/nanding/test_matrix_quick/")) {
    //for(auto& p: fs::recursive_directory_iterator("/project/projectdirs/m2956/nanding/myprojects/dyfuse/matrix/profile_matrix")) {
            cout << p.path().c_str() << '\n';
            cout.flush();

            GrB_Matrix L;
            GrB_Matrix U;
            GrB_Matrix M;
            GrB_Index nrows;
            GrB_Index ncols;
            // maximum: have 8x8 blocks in C.

            vector<vector<double>> report_time; // readLU time, create block time, mxm kernel time
            vector<vector<double>> gbase_time;


            timer_start();
            readLU(&L, &U, p.path().c_str(), 0, GrB_UINT64);
            init_time[0] += timer_end();

            GrB_Matrix_dup(&M, L);
            GrB_Matrix_nrows(&nrows, L);
            GrB_Matrix_ncols(&ncols, U);
            //printf("end read LU, nrows=%ld,ncols=%ld\n",nrows,ncols);
            //fflush(stdout);
            // 1D
            int runscheduler=0, runomp=0, rungrb=0;
            int nblocks=1;
            while (nblocks<=128) {
                cout << "nblocks=" << nblocks << endl;
                cout.flush();
                double *featuresM;
                if (!(featuresM = (double *) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))))
                    ABORT("Malloc fails for featuresM[].");
                report_time.resize(MAX_BLOCKS_IN_OUTPUT, vector<double>(MAX_THREAD, 0.0));
                gbase_time.resize(MAX_BLOCKS_IN_OUTPUT, vector<double>(MAX_THREAD, 0.0));
                test_scheduler_tricnt(p.path().c_str() , L, U, L, nblocks,nblocks, 1, niter, featuresM, report_time,
                                      gbase_time, init_time, &scheduler, final_rankfile_num,runscheduler,rungrb,runomp,count);
                //myfile << "omp," << p.path().c_str() << "," << nblocks << "," << gbase_time[nblocks][0]/niter << endl;
                //myfile.flush();
                free(featuresM);
                nblocks=nblocks*2;

            } // while
            GrB_Matrix_free(&L);
            GrB_Matrix_free(&U);
            count=count+1;
            cout << "finish " << count << " matrix" << endl;
            cout.flush();
            char cmd[100];
            strcpy(cmd,"mv ");
            strcat(cmd,p.path().c_str());
            strcat(cmd," ");
            strcat(cmd, "/home/nanding/tri_finish_srdan_matrix/");
            system(cmd);
            ////// 2D
            //nblocks=1;
            //while (nblocks<=2) {
            //    cout << "nblocks=" << nblocks << endl;
            //    cout.flush();
            //    double *featuresM;
            //    if (!(featuresM = (double *) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))))
            //        ABORT("Malloc fails for featuresM[].");
            //    test_scheduler_2d(p.path().c_str() , L, U, L, nblocks,nblocks, nblocks, niter, featuresM, report_time,
            //                      gbase_time, init_time, &scheduler, final_rankfile_num);
            //    starpu_task_wait_for_all();
            //    free(featuresM);
            //    nblocks=nblocks+1;
            //} // while

            //if (count==1) exit(0);
        }

    //cout << "Stop scheduler" << endl;
    //scheduler.stop();
    //cout << "Stop scheduler already" << endl;
    //cout.flush();
    OK(GrB_finalize());
    return 0;
}