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
#include "../util/common_feature.h"
#include "../util/block_matrix_generic_feature.h"
#include "../util/block_matrix.h"
#include "../util/timer.h"
#include "../scheduler/Scheduler.h"
#include "../util/reader_cxx.h"
#include "../util/reader_c.h"
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
    GrB_Matrix C;
    GrB_Descriptor descriptor;
    int nthreads;

    TaskInfo *taskInfo;
};

std::vector<TaskInfo *> createTasks(starpu_codelet *cl, GrB_Matrix *bM, GrB_Matrix *bA, GrB_Matrix *bB,GrB_Matrix *bC,
                                    bool transposeA, bool transposeB, size_t nblocksM, size_t nblocksN,
                                    size_t nblocksL, int* nthreads) {
    std::vector<TaskInfo *> tasks;
    tasks.reserve(nblocksM * nblocksN * nblocksL);

    for (size_t i = 0; i < nblocksM; i++) {
        for (size_t j = 0; j < nblocksN; j++) {
            for (size_t k = 0; k < nblocksL; k++) {
                size_t idxC = i * nblocksN + j;
                size_t idxA = i * nblocksL + k;;
                size_t idxB = j * nblocksL + k;

                auto task = starpu_task_create();
                task->cl = cl;

                auto args = static_cast<TaskArgs *>(malloc(sizeof(TaskArgs)));
                args->M = bM[idxC];
                args->C = bC[idxC];
                args->A = bA[idxA];
                args->B = bB[idxB];

                GrB_Descriptor desc;
                OK(GrB_Descriptor_new(&desc));
                OK(GxB_Desc_set(desc, GrB_OUTP, GrB_REPLACE));
                if (transposeA) { OK(GxB_Desc_set(desc, GrB_INP0, GrB_TRAN)); }
                if (transposeB) { OK(GxB_Desc_set(desc, GrB_INP1, GrB_TRAN)); }
                OK(GxB_Desc_set(desc, GrB_MASK,  GrB_STRUCTURE));
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


void test_scheduler_spgemm(const char *name, GrB_Matrix A, GrB_Matrix B, GrB_Matrix M,
                           size_t nblocksM, size_t nblocksN, size_t nblocksL,
                           size_t niter, double* featuresA, double* featuresB, double* featuresC, double* featuresM,
                           vector<vector<double>> &report_time,  vector<vector<double>> &gbase_time,
                           double* init_time,Scheduler *scheduler, int count, int runscheduler, int rungrb, int runomp, int mtxcount) {
    GrB_Matrix *bA, *bB, *bM,*bC;
    createMxMBlocks_feature(M, A, B, false, true, nblocksM, nblocksN, nblocksL,
                            GrB_UINT64, nullptr, &bM, &bA, &bB,
                            featuresA,featuresB,featuresC,featuresM);
    int nthreads[5] = {64, 32, 16, 8, 1};


    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t block_size_m, block_size_n, block_size_l;
    GrB_Matrix C;

    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_nrows(&n, B);
    GrB_Matrix_ncols(&l, A);
    block_size_m = (m + nblocksM - 1) / nblocksM;
    block_size_n = (n + nblocksN - 1) / nblocksN;
    block_size_l = (l + nblocksL - 1) / nblocksL;

    GrB_Matrix_new(&C, GrB_UINT64, m, n);
    create_blocks(&bC, &bm, &bn, C, block_size_m, block_size_n, 0);

    // region init codelet
    starpu_codelet cl{};
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
        auto args = static_cast<TaskArgs *>(clArgs);
        OK(GxB_Desc_set(args->descriptor, GxB_DESCRIPTOR_NTHREADS, args->nthreads));
        //GrB_Index Anrows, Ancols;
        //GrB_Matrix_nrows(&Anrows, args->A);
        //GrB_Matrix_ncols(&Ancols, args->A);
        //GrB_Index Bnrows, Bncols;
        //GrB_Matrix_nrows(&Bnrows, args->B);
        //GrB_Matrix_ncols(&Bncols, args->B); // transposed
        //GrB_Index Mnrows, Mncols;
        //GrB_Matrix_nrows(&Mnrows, args->M);
        //GrB_Matrix_ncols(&Mncols, args->M); // transposed
        //printf("incodelet: A: %zu,%zu, B: %zu,%zu, M: %zu, %zu\n", Anrows, Ancols, Bnrows, Bncols, Mnrows, Mncols);
        //fflush(stdout);

        //GrB_Matrix C;
        //GrB_Matrix_new(&C, GrB_UINT64, nrows, ncols);

        //timer_start();
        OK(GrB_mxm(args->C, args->M, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64, args->A, args->B, args->descriptor));
        args->taskInfo->result = args->C;
        //args->taskInfo->mxmTime = timer_end();
    };
    cl.nbuffers = 0;

    int merge_iter, split_iter;

    int skip = 0;
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
            auto tasks = createTasks(&cl, bM, bA, bB, bC, false, true, nblocksM, nblocksN, nblocksL, nthreads);
            for (int taskid = 0; taskid < nblocksM * nblocksM; taskid++) {
                for (int iter = 0; iter < niter + skip; iter++) {
                    if (iter >= skip) timer_start();
                    scheduler->submitTask(tasks[taskid]->task, 0, nthreads[taskid]);
                    //cout << "iter=" << iter << ", submit task to Q "<< nextWorker* (nthreads[order[taskid]]) << ", nthreads=" << nthreads[order[taskid]] << endl;
                    //cout.flush();
                    starpu_task_wait_for_all();
                    if (iter >= skip) {
                        report_time[taskid][tid] += timer_end();
                        //cout << "spu: " << nthreads[tid] << ", iter=" << iter << "," << report_time[taskid][tid]
                        //     << endl;
                        //cout.flush();
                    }
                }
                //cout << "spu: " << nthreads[tid] << ", time" << report_time[taskid][tid]/niter << endl;
                //cout.flush();
            }
        }

        if (rungrb == 1) {
            skip = 0;
            cout << "Running Grb... threads=" << nthreads[tid] << endl;
            cout.flush();
            GxB_Global_Option_set(static_cast<GxB_Option_Field>(GxB_NTHREADS), nthreads[tid]);
            //FILE *ff = fopen ("fprint.txt", "w") ;
            for (size_t i = 0; i < nblocksM; i++) {
                for (size_t j = 0; j < nblocksN; j++) {
                    for (size_t k = 0; k < nblocksL; k++) {
                        int myidx = i * nblocksN + j * nblocksL + k;
                        for (int iter = 0; iter < niter + skip; iter++) {
                            if (iter >= skip) timer_start();
                            int bidA = i * nblocksL + k;
                            int bidB = j * nblocksL + k;
                            int bidM = i * nblocksN + j;
                            OK(GrB_mxm(bC[i * nblocksN + j], bM[i * nblocksN + j], GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64,
                                       bA[i * nblocksL + k], bB[j * nblocksL + k], GrB_DESC_RST1));
                            //OK(GrB_mxm(bC[i * nblocksN + j], bM[i * nblocksN + j], GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64,
                            //           bA[i * nblocksL + k], bB[j * nblocksL + k], GrB_DESC_RT1));
                            //GrB_Descriptor desc;
                            //OK(GrB_Descriptor_new(&desc));
                            //OK(GxB_Desc_get(desc, GrB_MASK));
                            //OK(GxB_Descriptor_fprint (desc, "GrB_DESC_RST1", GxB_COMPLETE,ff ));
                            //GrB_Matrix_reduce_UINT64(&sum, TYPE_PLUS, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
                            if (iter >= skip) {
                                gbase_time[myidx][tid] += timer_end();
                                //cout << "grb: " << nthreads[tid] << ", iter=" << iter << "," << gbase_time[myidx][tid] << endl;
                                //cout.flush();
                            }
                            //printf("end submit task\n");
                            //fflush(stdout);
                        }// iter over blocks
                        //cout << "grb: " << nthreads[tid] << ", time" << report_time[myidx][tid]/niter << endl;
                        //cout.flush();

                    }
                }
            }
        }
        }
    std::string rankfile("spgemm_maskA_feature_");
    rankfile += "_";
    rankfile += std::to_string(count);
    cout << "dumping file name is: " << rankfile << endl;
    cout.flush();

    std::ofstream myfile;
    myfile.open (rankfile,std::ofstream::out | std::ofstream::app);


    if ((mtxcount==0) && (nblocksM==1)) {
        myfile << "Matrix,readLU,createblocks1D, blocks,nrows,ncols,nnz,Anrows,Ancols,Annz,Bnrows,Bncols,Bnnz";
        for (int tid = 0; tid < sizeof(nthreads) / sizeof(nthreads[0]); tid++) {
            myfile << ",scheduler" << nthreads[tid] << ",spu" << nthreads[tid] << ",grb" << nthreads[tid] ;
        }
        myfile << endl;
        myfile.flush();
    }
    for (size_t i = 0; i < nblocksM; i++) {
        for (size_t j = 0; j < nblocksN; j++) {
            for (size_t k = 0; k < nblocksL; k++) {
                int myidx=i*nblocksN + j*nblocksL + k;
                myfile << name
                       << "," << init_time[0]
                       << "," << init_time[1]
                       << "," << nblocksM
                       << "," << featuresM[myidx*NUM_FEATURE+1]
                       << "," << featuresM[myidx*NUM_FEATURE+2]
                       << "," << featuresM[myidx*NUM_FEATURE+3]
                       << "," << featuresA[myidx*NUM_FEATURE+1]
                       << "," << featuresA[myidx*NUM_FEATURE+2]
                       << "," << featuresA[myidx*NUM_FEATURE+3]
                       << "," << featuresB[myidx*NUM_FEATURE+1]
                       << "," << featuresB[myidx*NUM_FEATURE+2]
                       << "," << featuresB[myidx*NUM_FEATURE+3];
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
    freeMxMBlocks(nullptr, bM, bA, bB, nblocksM, nblocksN, nblocksL);
    printf("Done\n");
    fflush(stdout);

}

double init_time[100]={0.0};
int main(int argc, char *argv[]) {
    auto niter = otx::argTo<size_t>(argc, argv, {"-niter"}, 3);
    auto opt = otx::argTo<int>(argc, argv, "-o", 0);
    //auto runscheduler = otx::argTo<int>(argc, argv, {"-runscheduler"}, 0);
    //auto rungrb = otx::argTo<int>(argc, argv, {"-rungrb"}, 0);
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

    std::string rankfile("spgemm_maskA_feature_0");

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
            rankfile = "spgemm_maskA_feature_";
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
    for(auto& p: fs::recursive_directory_iterator("/home/nanding/srdan_matrix")) {
    //for(auto& p: fs::recursive_directory_iterator("/home/nanding/test_matrix_quick/")) {
        //for(auto& p: fs::recursive_directory_iterator("/project/projectdirs/m2956/nanding/myprojects/dyfuse/matrix/profile_matrix")) {
        cout << p.path().c_str() << '\n';
        cout.flush();

        std::string matrixfile("matrix_name");
        std::ofstream mymatrixfile;
        mymatrixfile.open (matrixfile,std::ofstream::out | std::ofstream::app);
        mymatrixfile << p.path().c_str() << endl;
        mymatrixfile.flush();

        GrB_Matrix A;
        GrB_Matrix B;
        GrB_Matrix M;

        timer_start();
        if (read_Matrix_FP64(&A, p.path().c_str(), 0) != GrB_SUCCESS) { return 1; }
        init_time[0] += timer_end();
        cast_matrix(&A, GrB_UINT64, GxB_ONE_UINT64);
        OK(GrB_Matrix_dup(&B, A));
        OK(GrB_Matrix_dup(&M, A));

        vector<vector<double>> report_time; // readLU time, create block time, mxm kernel time
        vector<vector<double>> gbase_time;

        //printf("end read LU, nrows=%ld,ncols=%ld\n",nrows,ncols);
        //fflush(stdout);
        // 1D

        int nblocks=1, runscheduler=0, rungrb=0;
        while (nblocks<=128) {
            cout << "nblocks=" << nblocks << endl;
            cout.flush();
            double *featuresA,*featuresB,*featuresC, *featuresM;
            if (!(featuresA = (double *) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))))
                ABORT("Malloc fails for featuresA[].");
            if (!(featuresB = (double *) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))))
                ABORT("Malloc fails for featuresB[].");
            if (!(featuresC = (double *) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))))
                ABORT("Malloc fails for featuresC[].");
            if (!(featuresM = (double *) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))))
                ABORT("Malloc fails for featuresM[].");
            report_time.resize(MAX_BLOCKS_IN_OUTPUT, vector<double>(MAX_THREAD, 0.0));
            gbase_time.resize(MAX_BLOCKS_IN_OUTPUT, vector<double>(MAX_THREAD, 0.0));
            test_scheduler_spgemm(p.path().c_str() , A, B, M, nblocks,nblocks, 1, niter, featuresA, featuresB,featuresC,featuresM, report_time,
                                  gbase_time, init_time, &scheduler, final_rankfile_num,runscheduler,rungrb,0,count);
            //myfile << "omp," << p.path().c_str() << "," << nblocks << "," << gbase_time[nblocks][0]/niter << endl;
            //myfile.flush();
            free(featuresA);
            free(featuresB);
            free(featuresM);
            nblocks=nblocks*2;

        } // while
        GrB_Matrix_free(&A);
        GrB_Matrix_free(&B);
        count=count+1;
        cout << "finish " << count << " matrix" << endl;
        cout.flush();
        char cmd[100];
        strcpy(cmd,"mv ");
        strcat(cmd,p.path().c_str());
        strcat(cmd," ");
        strcat(cmd, "/home/nanding/spgemm_maskA_srdan_matrix/");
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