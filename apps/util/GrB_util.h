#ifndef GRB_FUSION_GRB_UTIL_H
#define GRB_FUSION_GRB_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <GraphBLAS.h>

#ifdef __cplusplus
}
#endif


#include <mio/mio.h>

static const char *GrB_info_to_str(GrB_Info info) {
    switch (info) {
        case GrB_SUCCESS:
            return "Success";

        case GrB_NO_VALUE:
            return "NoValue";

        case GrB_UNINITIALIZED_OBJECT:
            return "Uninitialized object";
        case GrB_INVALID_OBJECT:
            return "Invalid object";
        case GrB_NULL_POINTER:
            return "NULL pointer";
        case GrB_INVALID_VALUE:
            return "Invalid value";
        case GrB_INVALID_INDEX:
            return "Invalid index";

        case GrB_DOMAIN_MISMATCH:
            return "Domain mismatch";
        case GrB_DIMENSION_MISMATCH:
            return "Dimension mismatch";
        case GrB_OUTPUT_NOT_EMPTY:
            return "Output not empty";
        case GrB_OUT_OF_MEMORY:
            return "Out of memory";
        case GrB_INSUFFICIENT_SPACE:
            return "Insufficient space";
        case GrB_INDEX_OUT_OF_BOUNDS:
            return "Index out of bounds";
        case GrB_PANIC:
            return "PANIC";

        default:
            return "Unknown error";
    }
}

#define OK(A) do {GrB_Info inf = A; if (inf != GrB_SUCCESS) { fprintf(stderr, "Error \"%s\", line: %d \n", GrB_info_to_str(inf), __LINE__); } } while(0);
//#define BURBLE(_CODE) do { GxB_set(GxB_BURBLE, true); _CODE; GxB_set(GxB_BURBLE, false); } while(0);
#define BURBLE(_CODE) _CODE;

static const char *get_file_name(const char *path) {
    size_t len = strlen(path);
    while (path[len] != '/') { len--; }
    return path + len + 1;
}

static void cast_matrix(GrB_Matrix *A, GrB_Type type, GrB_UnaryOp cast_op) {
    GrB_Index nrows, ncols;
    GrB_Matrix_nrows(&nrows, *A);
    GrB_Matrix_ncols(&ncols, *A);

    GrB_Matrix tmpA;
    GrB_Matrix_new(&tmpA, type, nrows, ncols);
    GrB_Matrix_apply(tmpA, GrB_NULL, GrB_NULL, cast_op, *A, GrB_NULL);

    GrB_Matrix_free(A);
    *A = tmpA;
}

void readLU(GrB_Matrix *L_out, GrB_Matrix *U_out, const char *path, bool shuffle, GrB_Type type) {
    int nrows, ncols, nnz;
    int *I, *J;
    double *X;
    int Lcnt = 0, Ucnt = 0;
    GrB_Index *LI, *LJ, *UI, *UJ;
    uint64_t *LX, *UX;

    mio_read_tuples_FP64(path, &nrows, &ncols, &nnz, &I, &J, &X, shuffle);
    for (int i = 0; i < nnz; i++) {
        if (J[i] > I[i]) {
            Ucnt++;
        } else if (J[i] < I[i]) {
            Lcnt++;
        }
    }

    LI = (GrB_Index *) malloc(nnz * sizeof(GrB_Index));
    LJ = (GrB_Index *) malloc(nnz * sizeof(GrB_Index));
    LX = (uint64_t *) malloc(nnz * sizeof(uint64_t));

    UI = (GrB_Index *) malloc(nnz * sizeof(GrB_Index));
    UJ = (GrB_Index *) malloc(nnz * sizeof(GrB_Index));
    UX = (uint64_t *) malloc(nnz * sizeof(uint64_t));

    if (LI == NULL || LJ == NULL || LX == NULL || UI == NULL || UJ == NULL || UX == NULL) {
        fprintf(stderr, "Not enough memory.");
        exit(-1);
    }

    Ucnt = 0;
    Lcnt = 0;
    for (int i = 0; i < nnz; i++) {
        if (J[i] > I[i]) {
            UI[Ucnt] = I[i];
            UJ[Ucnt] = J[i];
            UX[Ucnt] = 1;
            Ucnt++;

            LI[Lcnt] = J[i];
            LJ[Lcnt] = I[i];
            LX[Lcnt] = 1;
            Lcnt++;
        } else if (J[i] < I[i]) {
            LI[Lcnt] = I[i];
            LJ[Lcnt] = J[i];
            LX[Lcnt] = 1;
            Lcnt++;

            UI[Ucnt] = J[i];
            UJ[Ucnt] = I[i];
            UX[Ucnt] = 1;
            Ucnt++;
        }
    }

    OK(GrB_Matrix_new(L_out, type, nrows, ncols));
    OK(GrB_Matrix_new(U_out, type, nrows, ncols));

    OK(GrB_Matrix_build_UINT64(*L_out, LI, LJ, LX, Lcnt, GrB_FIRST_UINT64));
    OK(GrB_Matrix_build_UINT64(*U_out, UI, UJ, UX, Ucnt, GrB_FIRST_UINT64));

    free(LI);
    free(LJ);
    free(LX);
    free(UI);
    free(UJ);
    free(UX);

    mio_free_tuples(I, J, X);
}

#endif //GRB_FUSION_GRB_UTIL_H
