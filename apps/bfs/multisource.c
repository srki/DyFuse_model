#include <GraphBLAS.h>
#include <otx/otx.h>
#include "../util/reader_c.h"
#include "../util/timer.h"
#include "../util/GrB_util.h"

GrB_Info bfs(GrB_Matrix A, GrB_Index *s, GrB_Index nsver, int32_t *depth) {
    GrB_Index n;
    GrB_Matrix_nrows(&n, A);                // n = # of vertices in graph

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

    int32_t d = 0;          // BFS level number
    GrB_Index nvals = 0;   // nvals == 0 when BFS phase is complete

    do {
        GrB_eWiseAdd(numsp, GrB_NULL, GrB_NULL, GrB_PLUS_INT32, numsp, frontier, GrB_NULL);
        GrB_mxm(frontier, numsp, GrB_NULL, GxB_PLUS_TIMES_INT32, A, frontier, GrB_DESC_RCT0);
        GrB_Matrix_nvals(&nvals, frontier);
        d++;
    } while (nvals);

    *depth = d;

    GrB_free(&frontier);
    GrB_free(&numsp);

    return GrB_SUCCESS;
}

int main(int argc, char *argv[]) {
    int numThreads;
    char *matrix_path;
    uint64_t nbatch;
    uint64_t witer;
    uint64_t niter;
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
    if (arg_to_uint64_def(argc, argv, "--witer", &witer, 1) != OTX_SUCCESS) { return -1; }
    if (arg_to_uint64_def(argc, argv, "--niter", &niter, 1) != OTX_SUCCESS) { return -1; }
    if (arg_to_uint64_def(argc, argv, "-nv|--nsver", &nsver, 1) != OTX_SUCCESS) { return 1; }
    s = malloc(nsver * sizeof(GrB_Index));

    for (uint64_t batch = 0; batch < nbatch; batch++) {
        /* Create batch indices */
        for (GrB_Index i = 0; i < nsver; i++) {
            s[i] = batch * nsver + i;
        }

        /* Execute betweenness centrality for the iterations */
        double totalBFS = 0;

        for (uint64_t i = 0; i < niter + witer; i++) {
            int32_t depth;

            unsigned long long start = get_time_ns();
            bfs(A, s, nsver, &depth);
            unsigned long long end = get_time_ns();
            if (i < witer) { continue; }

            totalBFS += (double) (end - start) / 1e9;
            printf("LOG-iter;%s;1;%lu;%lf;0.0;%d\n", get_file_name(matrix_path), nsver, (double) (end - start) / 1e9,
                   depth);
        }

        printf("LOG-iter;%s;1;%lu;%lf;%lf\n", get_file_name(matrix_path), nsver, totalBFS / (double) niter, 0.0);

    }

    GrB_finalize();

    return 0;
}

