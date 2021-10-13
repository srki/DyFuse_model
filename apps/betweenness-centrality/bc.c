/* https://people.eecs.berkeley.edu/~aydin/GABB17.pdf */

#include <GraphBLAS.h>
#include <otx/otx.h>
#include "../util/reader_c.h"
#include "../util/timer.h"

GrB_Info BC_update(GrB_Vector *delta, GrB_Matrix A, GrB_Index *s, GrB_Index nsver) {
    GrB_Index n;
    GrB_Matrix_nrows(&n, A);                // n = # of vertices in graph
    GrB_Vector_new(delta, GrB_FP32, n);     // Vector<float> delta(n)

    GrB_Monoid Int32Add;
    GrB_Monoid_new(&Int32Add, GrB_PLUS_INT32, 0);   // Monoid <int32_t,+,0>
    GrB_Semiring Int32AddMul;
    GrB_Semiring_new(&Int32AddMul, Int32Add, GrB_TIMES_INT32);     // Semiring <int32_t,int32_t,int32_t,+,*,0>

    GrB_Descriptor desc_tsr;        // Descriptor for BFS phase mxm
    GrB_Descriptor_new(&desc_tsr);
    GrB_Descriptor_set(desc_tsr, GrB_INP0, GrB_TRAN);
    GrB_Descriptor_set(desc_tsr, GrB_MASK, GrB_SCMP);
    GrB_Descriptor_set(desc_tsr, GrB_OUTP, GrB_REPLACE);

    // index and value arrays needed to build numsp
    GrB_Index *i_nsver = (GrB_Index *) malloc(sizeof(GrB_Index) * nsver);
    int32_t *ones = (int32_t *) malloc(sizeof(int32_t) * nsver);
    for (int i = 0; i < nsver; ++i) {
        i_nsver[i] = i;
        ones[i] = 1;
    }

    GrB_Matrix numsp;
    GrB_Matrix_new(&numsp, GrB_INT32, n, nsver);
    GrB_Matrix_build(numsp, s, i_nsver, ones, nsver, GrB_PLUS_INT32);
    free(i_nsver);
    free(ones);

    GrB_Matrix frontier;
    GrB_Matrix_new(&frontier, GrB_INT32, n, nsver);
    GrB_extract(frontier, numsp, GrB_NULL, A, GrB_ALL, n, s, nsver, GrB_DESC_RCT0);

    GrB_Matrix *sigmas = (GrB_Matrix *) malloc(sizeof(GrB_Matrix) * n);
    int32_t d = 0;          // BFS level number
    GrB_Index nvals = 0;   // nvals == 0 when BFS phase is complete

    do {
        // sigmas[d](:, s_ = d^th level frontier from source vertex s
        GrB_Matrix_new(&(sigmas[d]), GrB_BOOL, n, nsver);
        GrB_apply(sigmas[d], GrB_NULL, GrB_NULL, GrB_IDENTITY_BOOL, frontier,
                  GrB_NULL);    // sigmas[d](:,:) = (Boolean frontier)
        GrB_eWiseAdd(numsp, GrB_NULL, GrB_NULL, Int32Add, numsp, frontier,
                     GrB_NULL);    // numsp += frontier (accum path counts))
        GrB_mxm(frontier, numsp, GrB_NULL, Int32AddMul, A, frontier,
                desc_tsr); // f<!<numsp> = A' +.* f (update frontier)
        GrB_Matrix_nvals(&nvals, frontier); // number of nodes in frontier at this level
        d++;
    } while (nvals);
    printf("BFS: %d\n", d);

    GrB_Monoid FP32Add;
    GrB_Monoid_new(&FP32Add, GrB_PLUS_FP32, 0.0f);
    GrB_Monoid FP32Mul;
    GrB_Monoid_new(&FP32Mul, GrB_TIMES_FP32, 1.0f);
    GrB_Semiring FP32AddMul;
    GrB_Semiring_new(&FP32AddMul, FP32Add, GrB_TIMES_FP32);

    GrB_Matrix nspinv;
    GrB_Matrix_new(&nspinv, GrB_FP32, n, nsver);
    GrB_apply(nspinv, GrB_NULL, GrB_NULL, GrB_MINV_FP32, numsp, GrB_NULL);

    GrB_Matrix bcu;
    GrB_Matrix_new(&bcu, GrB_FP32, n, nsver);
    GrB_assign(bcu, GrB_NULL, GrB_NULL, 1.0f, GrB_ALL, n, GrB_ALL, nsver, GrB_NULL);

    GrB_Descriptor desc_r;
    GrB_Descriptor_new(&desc_r);
    GrB_Descriptor_set(desc_r, GrB_OUTP, GrB_REPLACE);

    GrB_Matrix w;
    GrB_Matrix_new(&w, GrB_FP32, n, nsver);
    for (int i = d - 1; i > 0; i--) {
        GrB_eWiseMult(w, sigmas[i], GrB_NULL, FP32Mul, bcu, nspinv, desc_r);
        GrB_mxm(w, sigmas[i - 1], GrB_NULL, FP32AddMul, A, w, desc_r);
        GrB_eWiseMult(bcu, GrB_NULL, GrB_PLUS_FP32, FP32Mul, w, numsp, GrB_NULL);
    }
    GrB_assign(*delta, GrB_NULL, GrB_NULL, -(float) nsver, GrB_ALL, n, GrB_NULL);
    GrB_reduce(*delta, GrB_NULL, GrB_PLUS_FP32, GrB_PLUS_FP32, bcu, GrB_NULL);

    for (int i = 0; i < d; i++) {
        GrB_free(&sigmas[i]);
    }
    free(sigmas);

    GrB_free(&frontier);
    GrB_free(&numsp);
    GrB_free(&nspinv);
    GrB_free(&w);
    GrB_free(&bcu);
    GrB_free(&desc_tsr);
    GrB_free(&desc_r);
    GrB_free(&Int32AddMul);
    GrB_free(&Int32Add);
    GrB_free(&FP32AddMul);
    GrB_free(&FP32Add);
    GrB_free(&FP32Mul);

    return GrB_SUCCESS;
}

int main(int argc, char *argv[]) {
    int numThreads;
    char *matrix_path;
    uint64_t nbatch;
    uint64_t niter;
    GrB_Vector delta;
    GrB_Matrix A;
    GrB_Index nrows;
    GrB_Index *s;
    GrB_Index nsver;

    /* Init and set the number of threads */
    GrB_init(GrB_NONBLOCKING);
    if (arg_to_int32_def(argc, argv, "-nt|--numThreads", &numThreads, 1) != OTX_SUCCESS) { return 1; }
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, numThreads);

    /* Read the matrix */
    if (arg_to_str_def(argc, argv, "-g|--graph", &matrix_path, INPUTS_DIR"/simple") != OTX_SUCCESS) { return 1; }
    if (read_Matrix_INT32(&A, matrix_path, true) != GrB_SUCCESS) { return 1; }

    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_assign_INT32(A, A, GrB_NULL, 1, GrB_ALL, nrows, GrB_ALL, nrows, GrB_DESC_S);

    /* Parse the remaining arguments */
    if (arg_to_uint64_def(argc, argv, "--nbatch", &nbatch, 1) != OTX_SUCCESS) { return -1; }
    if (arg_to_uint64_def(argc, argv, "--niter", &niter, 1) != OTX_SUCCESS) { return -1; }
    if (arg_to_uint64_def(argc, argv, "-nv|--nsver", &nsver, 1) != OTX_SUCCESS) { return 1; }
    s = malloc(nsver * sizeof(GrB_Index));


    for (uint64_t batch = 0; batch < nbatch; batch++) {
        /* Create batch indices */
        for (GrB_Index i = 0; i < nsver; i++) {
            s[i] = batch * nsver + i;
        }

        /* Execute betweenness centrality for the iterations */
        for (uint64_t iter = 0; iter < niter; iter++) {
            unsigned long long start = get_time_ns();
            BC_update(&delta, A, s, nsver);
            unsigned long long end = get_time_ns();
            printf("Execution time: %lf\n", (end - start) / 1e9);

            /* Verification */
//            GxB_print(delta, GxB_COMPLETE);
            float sum;
            GrB_reduce(&sum, GrB_NULL, GrB_PLUS_MONOID_FP32, delta, GrB_NULL);
            printf("%f\n", sum);
        }
    }

    GrB_finalize();

    return 0;
}

