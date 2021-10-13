#ifndef GRB_FUSION_UTIL_H
#define GRB_FUSION_UTIL_H

#include "GraphBLASImpl.h"
#include <exception>

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


namespace grb::detail {

    void grbTry(GrB_Info info) {
        if (info != GrB_SUCCESS && info != GrB_NO_VALUE) {
            throw std::exception{};
        }
    }

    /* @formatter:off */
    template<class T>
    constexpr bool isSupported = std::is_same_v<bool,     T> ||
                                 std::is_same_v<int8_t,   T> ||
                                 std::is_same_v<uint8_t,  T> ||
                                 std::is_same_v<int16_t,  T> ||
                                 std::is_same_v<uint16_t, T> ||
                                 std::is_same_v<int32_t,  T> ||
                                 std::is_same_v<uint32_t, T> ||
                                 std::is_same_v<int64_t,  T> ||
                                 std::is_same_v<uint64_t, T> ||
                                 std::is_same_v<float,    T> ||
                                 std::is_same_v<double,   T>;
    /* @formatter:on */

    template<class T>
    struct CInterface {
        static_assert(isSupported<T>, "Type not supported.");

        static constexpr auto type() { SWITCH_APPEND_TYPE(GrB); }

        static constexpr auto plus() { SWITCH_APPEND_TYPE(GrB_PLUS); }

    private:
        /* @formatter:off */
        static constexpr auto getScalarSetElement()     { SWITCH_APPEND_TYPE(GxB_Scalar_setElement); }

        static constexpr auto getScalarExtractElement() { SWITCH_APPEND_TYPE(GxB_Scalar_extractElement); }


        static constexpr auto getVectorBuild()          { SWITCH_APPEND_TYPE(GrB_Vector_build); }

        static constexpr auto getVectorSetElement()     { SWITCH_APPEND_TYPE(GrB_Vector_setElement); }

        static constexpr auto getVectorExtractElement() { SWITCH_APPEND_TYPE(GrB_Vector_extractElement); }

        static constexpr auto getVectorExtractTuples()  { SWITCH_APPEND_TYPE(GrB_Vector_extractTuples); }


        static constexpr auto getMatrixBuild()          { SWITCH_APPEND_TYPE(GrB_Matrix_build); }

        static constexpr auto getMatrixSetElement()     { SWITCH_APPEND_TYPE(GrB_Matrix_setElement); }

        static constexpr auto getMatrixExtractElement() { SWITCH_APPEND_TYPE(GrB_Matrix_extractElement); }

        static constexpr auto getMatrixExtractTuples()  { SWITCH_APPEND_TYPE(GrB_Matrix_extractTuples); }


        static constexpr auto getVectorAssignValue()    { SWITCH_APPEND_TYPE(GrB_Vector_assign); }

        static constexpr auto getMatrixAssignValue()    { SWITCH_APPEND_TYPE(GrB_Matrix_assign); }


        static constexpr auto getVectorReduceScalar()   { SWITCH_APPEND_TYPE(GrB_Vector_reduce); }

        static constexpr auto getMatrixReduceScalar()   { SWITCH_APPEND_TYPE(GrB_Matrix_reduce); }

        /* @formatter:on */

    public:

        /* @formatter:off */
        static constexpr auto scalarSetElement      = getScalarSetElement();
        static constexpr auto scalarExtractElement  = getScalarExtractElement();

        static constexpr auto vectorBuild           = getVectorBuild();
        static constexpr auto vectorSetElement      = getVectorSetElement();
        static constexpr auto vectorExtractElement  = getVectorExtractElement();
        static constexpr auto vectorExtractTuples   = getVectorExtractTuples();

        static constexpr auto matrixBuild           = getMatrixBuild();
        static constexpr auto matrixSetElement      = getMatrixSetElement();
        static constexpr auto matrixExtractElement  = getMatrixExtractElement();
        static constexpr auto matrixExtractTuples   = getMatrixExtractTuples();

        static constexpr auto vectorAssignValue     = getVectorAssignValue();
        static constexpr auto matrixAssignValue     = getMatrixAssignValue();

        static constexpr auto vectorReduceScalar    = getVectorReduceScalar();
        static constexpr auto matrixReduceScalar    = getMatrixReduceScalar();
        /* @formatter:on */

    };

}

#endif //GRB_FUSION_UTIL_H
