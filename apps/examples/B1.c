/* Copied from http://people.eecs.berkeley.edu/~aydin/GraphBLAS_API_C_v13.pdf and modified. */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <otx/otx.h>
#include "GraphBLAS.h"
#include "../util/reader_c.h"
#include "../util/timer.h"


/** Given a boolean n x n adjacency  matrix A and a source  vertex s, performs a BFS traversal
  * of the graph and sets v[i] to the level in which vertex i is  visited  (v[s] == 1)
  * If i is  not reachable from s, then v [ i ] = 0.  ( Vector v  should  be empty on input.) */
GrB_Info BFS(GrB_Vector *v, GrB_Matrix A, GrB_Index s) {
    GrB_Index n;
    GrB_Matrix_nrows(&n, A);    // n = # of  rows  of A

    GrB_Vector_new(v, GrB_INT32, n);    // Vector<int32t> v(n)

    GrB_Vector q;   // vertices visited in each level
    GrB_Vector_new(&q, GrB_BOOL, n);  // Vector<bool>q (n)
    GrB_Vector_setElement(q, (bool) true, s); // q[s] = true , false everywhere else

    /*
     * BFS traversal and label the vertices.
     */
    int32_t d = 0;      // d = level in BFS traversal
    bool succ = false;  // succ == true when some successor found
    do {
        ++d;   // next level (start with 1)
        GrB_assign(*v, q, GrB_NULL, d, GrB_ALL, n, GrB_NULL); // v[q] = d
        // q [!v] = q ||.&& A; finds all the unvisited successors from current q
        GrB_vxm(q, *v, GrB_NULL, GxB_LOR_LAND_BOOL, q, A, GrB_DESC_RC);
        GrB_reduce(&succ, GrB_NULL, GxB_LOR_BOOL_MONOID, q, GrB_NULL); // succ =||(q)
    } while (succ); // if there is no successor in q, we are done.
    GrB_free(&q);// q vector no longer  needed
    return GrB_SUCCESS;
}

int main(int argc, char *argv[]) {
    int numThreads;
    char *matrix_path;
    GrB_Vector v;
    GrB_Matrix A;
    GrB_Index n;
    size_t nsver;
    int sum;

    /* Init and set the number of threads */
    GrB_init(GrB_NONBLOCKING);
    if (arg_to_int32_def(argc, argv, "-nt|--numThreads", &numThreads, 1) != OTX_SUCCESS) { return 1;}
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, numThreads);

    if (arg_to_str_def(argc, argv, "-g|--graph", &matrix_path, INPUTS_DIR"/simple") != OTX_SUCCESS) { return 1; }
    if (read_Matrix_BOOL(&A, matrix_path, false) != GrB_SUCCESS) { return 1; }

    /* Execute BFS from random sources */
    if (arg_to_uint64_def(argc, argv, "-nv|--nsver", &nsver, 1) != OTX_SUCCESS) { return 1; }
    srand(0);
    GrB_Matrix_nrows(&n, A);


    unsigned long long start = get_time_ns();
    for (size_t i = 0; i < nsver; i++) {
        size_t source = rand() % n;
        BFS(&v, A, source);
//        GxB_print(v, GxB_COMPLETE);
        GrB_reduce(&sum, GrB_NULL, GrB_PLUS_MONOID_INT32, v, GrB_NULL);
        printf("Source: %zu; sum: %d\n", source, sum);
    }
    unsigned long long end = get_time_ns();
    printf("Execution time: %lf\n", (end - start) / 1e9);

    GrB_free(&v);
    GrB_free(&A);

    GrB_finalize();

    return 0;
}
