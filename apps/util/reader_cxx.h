/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_READER_CXX_H
#define GRB_FUSION_READER_CXX_H

#include <grb/grb.h>
#include <random>
#include <mio/mio.h>

#define SWITCH_APPEND_TYPE(NAME)                                                \
do {                                                                            \
    if      constexpr (std::is_same_v<bool,     T>) { return NAME ## _BOOL;   } \
    else if constexpr (std::is_same_v<int8_t,   T>) { return NAME ## _INT8;   } \
    else if constexpr (std::is_same_v<uint8_t,  T>) { return NAME ## _UINT8;  } \
    else if constexpr (std::is_same_v<int16_t,  T>) { return NAME ## _INT16;  } \
    else if constexpr (std::is_same_v<uint16_t, T>) { return NAME ## _UINT16; } \
    else if constexpr (std::is_same_v<int32_t,  T>) { return NAME ## _INT32;  } \
    else if constexpr (std::is_same_v<uint32_t, T>) { return NAME ## _UINT32; } \
    else if constexpr (std::is_same_v<int64_t,  T>) { return NAME ## _INT64;  } \
    else if constexpr (std::is_same_v<uint64_t, T>) { return NAME ## _UINT64; } \
    else if constexpr (std::is_same_v<float,    T>) { return NAME ## _FP32;   } \
    else if constexpr (std::is_same_v<double,   T>) { return NAME ## _FP64;   } \
} while (0)

template<class T>
constexpr inline auto getReadTuplesFunc() {
    SWITCH_APPEND_TYPE(mio_read_tuples);
}

template<class T>
constexpr inline auto getGrbPlus() {
    SWITCH_APPEND_TYPE(GrB_PLUS);
}

template<class ScalarT>
auto readMatrix(const char *path, bool shuffle = false, size_t nrowsPerBlock = 0, size_t ncolsPerBlock = 0) {
    int nrows, ncols, nz;
    int *I, *J;
    ScalarT *X;

    getReadTuplesFunc<ScalarT>()(path, &nrows, &ncols, &nz, &I, &J, &X, shuffle);

    assert(nrows == ncols);

    auto I_grb = std::make_unique<uint64_t[]>(nz);
    auto J_grb = std::make_unique<uint64_t[]>(nz);
    auto X_grb = std::make_unique<ScalarT[]>(nz);

    for (int i = 0; i < nz; i++) {
        I_grb[i] = I[i];
        J_grb[i] = J[i];
        X_grb[i] = X[i];
    }

    grb::Matrix<ScalarT> A{static_cast<size_t>(nrows), static_cast<size_t>(ncols), nrowsPerBlock, ncolsPerBlock};

    try {
        A.build(I_grb.get(), J_grb.get(), X_grb.get(), nz, getGrbPlus<ScalarT>());
    } catch (std::exception &e) {
        std::cerr << "Cannot create the matrix!" << std::endl;
        exit(1);
    }

    mio_free_tuples(I, J, X);

    return A;
}

template<class ScalarT>
auto readMatrix(const std::string &path, bool shuffle = false, size_t nrowsPerBlock = 0, size_t ncolsPerBlock = 0) {
    return readMatrix<ScalarT>(path.c_str(), shuffle, nrowsPerBlock, ncolsPerBlock);
}


#endif //GRB_FUSION_READER_CXX_H
