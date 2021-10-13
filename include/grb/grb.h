/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_GRB_H
#define GRB_FUSION_GRB_H

#include <starpu.h>

#include <grb/util/GraphVizUtil.h>
#include <grb/util/GraphBLASImpl.h>

#include <grb/grb-operations/ops.h>

#include <grb/objects/Matrix.h>
#include <grb/objects/Vector.h>
#include <grb/objects/traits.h>
#include <grb/context/Context.h>


namespace grb {
    namespace detail {
        template<class ScalarT, class IndexT>
        detail::VectorImpl<ScalarT, IndexT> &getVectorImpl(Vector<ScalarT, IndexT> &v) {
            return v.getImpl();
        }

        detail::VectorImpl<void, void> &getVectorImpl(std::nullptr_t _) {
            return detail::VectorImpl<void, void>::getInstance();
        }

        template<class ScalarT, class IndexT>
        detail::MatrixImpl<ScalarT, IndexT> &getMatrixImpl(Matrix<ScalarT, IndexT> &v) {
            return v.getImpl();
        }

        detail::MatrixImpl<void, void> &getMatrixImpl(std::nullptr_t _) {
            return detail::MatrixImpl<void, void>::getInstance();
        }
    }

    inline void init() {
        STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "starpu_init");
        detail::grbTry(GrB_init(GrB_BLOCKING));
//        GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, 1);
    }

    inline void finalize() {
        util::printGraphVizEnv();
        detail::Context::getDefaultContext().wait();
        detail::Context::getDefaultContext().releaseOps();
        GrB_finalize();
        starpu_shutdown();
    }

    inline std::nullptr_t null = nullptr;

    /* region operations */

    /* region mxm, vxm, mxv */

    template<class CMatrixT, class MaskT, class AMatrixT, class BMatrixT>
    inline void mxm(CMatrixT &C, MaskT &mask, GrB_BinaryOp const &accum, GrB_Semiring const &op,
                    AMatrixT &A, BMatrixT &B, GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                new detail::OpMxM{detail::getMatrixImpl(C), detail::getMatrixImpl(mask), accum, op,
                                  detail::getMatrixImpl(A), detail::getMatrixImpl(B), desc});
    }

    template<class WVectorT, class MaskT, class UVectorT, class AMatrixT>
    inline void vxm(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, GrB_Semiring op, UVectorT &u, AMatrixT &A,
                    GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                new detail::OpVxM(detail::getVectorImpl(w), detail::getVectorImpl(mask), accum, op,
                                  detail::getVectorImpl(u), detail::getMatrixImpl(A), desc));
    }

    template<class WVectorT, class MaskT, class AMatrixT, class UVectorT>
    inline void mxv(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, GrB_Semiring op, AMatrixT &A, UVectorT &u,
                    GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                new detail::OpMxV(detail::getVectorImpl(w), detail::getVectorImpl(mask), accum, op,
                                  detail::getMatrixImpl(A), detail::getVectorImpl(u), desc));
    }

    /* endregion */

    /* region element-wise ADD and MULTIPLY */

    template<class WVectorT, class MaskT, class OpT, class UVectorT, class VVectorT,
            std::enable_if_t<detail::IsVector_v<WVectorT>, int> = 0,
            std::enable_if_t<
                    detail::IsMatrix_v<MaskT> || std::is_same_v<std::remove_const_t<MaskT>, std::nullptr_t>, int> = 0,
            std::enable_if_t<detail::IsVector_v<UVectorT>, int> = 0,
            std::enable_if_t<detail::IsVector_v<VVectorT>, int> = 0>
    inline void eWiseAdd(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, OpT op, UVectorT &u, VVectorT &v,
                         GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                detail::newOpElemWiseVector<detail::OpElemWiseType::ADD>
                        (detail::getVectorImpl(w), detail::getVectorImpl(mask), accum, op,
                         detail::getVectorImpl(u), detail::getVectorImpl(v), desc));
    }

    template<class CMatrixT, class MaskT, class OpT, class AMatrixT, class BMatrixT,
            std::enable_if_t<detail::IsMatrix_v<CMatrixT>, int> = 0,
            std::enable_if_t<
                    detail::IsMatrix_v<MaskT> || std::is_same_v<std::remove_const_t<MaskT>, std::nullptr_t>, int> = 0,
            std::enable_if_t<detail::IsMatrix_v<AMatrixT>, int> = 0,
            std::enable_if_t<detail::IsMatrix_v<BMatrixT>, int> = 0>
    inline void eWiseAdd(CMatrixT &C, MaskT &mask, GrB_BinaryOp accum, OpT op, AMatrixT &A, BMatrixT &B,
                         GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                detail::newOpElemWiseMatrix<detail::OpElemWiseType::ADD>
                        (detail::getMatrixImpl(C), detail::getMatrixImpl(mask), accum, op,
                         detail::getMatrixImpl(A), detail::getMatrixImpl(B), desc));
    }

    template<class WVectorT, class MaskT, class OpT, class UVectorT, class VVectorT,
            std::enable_if_t<detail::IsVector_v<WVectorT>, int> = 0,
            std::enable_if_t<
                    detail::IsMatrix_v<MaskT> || std::is_same_v<std::remove_const_t<MaskT>, std::nullptr_t>, int> = 0,
            std::enable_if_t<detail::IsVector_v<UVectorT>, int> = 0,
            std::enable_if_t<detail::IsVector_v<VVectorT>, int> = 0>
    inline void eWiseMult(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, OpT op, UVectorT &u, VVectorT &v,
                          GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                detail::newOpElemWiseVector<detail::OpElemWiseType::MULTIPLY>
                        (detail::getVectorImpl(w), detail::getVectorImpl(mask), accum, op,
                         detail::getVectorImpl(u), detail::getVectorImpl(v), desc));
    }

    template<class CMatrixT, class MaskT, class OpT, class AMatrixT, class BMatrixT,
            std::enable_if_t<detail::IsMatrix_v<CMatrixT>, int> = 0,
            std::enable_if_t<
                    detail::IsMatrix_v<MaskT> || std::is_same_v<std::remove_const_t<MaskT>, std::nullptr_t>, int> = 0,
            std::enable_if_t<detail::IsMatrix_v<AMatrixT>, int> = 0,
            std::enable_if_t<detail::IsMatrix_v<BMatrixT>, int> = 0>
    inline void eWiseMult(CMatrixT &C, MaskT &mask, GrB_BinaryOp accum, OpT op, AMatrixT &A, BMatrixT &B,
                          GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                detail::newOpElemWiseMatrix<detail::OpElemWiseType::MULTIPLY>
                        (detail::getMatrixImpl(C), detail::getMatrixImpl(mask), accum, op,
                         detail::getMatrixImpl(A), detail::getMatrixImpl(B), desc));
    }

    /* endregion */

    /* region extract */

    template<class CMatrix, class MaskT, class AMatrixT, class RowIndexT, class NRowIndicesT,
            class ColIndexT, class NColIndicesT>
    inline void extract(CMatrix &C, MaskT &mask, GrB_BinaryOp accum, AMatrixT &A, const RowIndexT *rowIndices,
                        NRowIndicesT nrows, const ColIndexT *colIndices, NColIndicesT ncols, GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                new detail::OpExtractMatrix(detail::getMatrixImpl(C), detail::getMatrixImpl(mask), accum,
                                            detail::getMatrixImpl(A), rowIndices, nrows, colIndices, ncols, desc));
    }

    /* endregion */

    /* region assign */

    template<class WVectorT, class MaskT, class ValueT, class IndexT, class NIndicesT>
    inline void assign(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, const ValueT &val, IndexT *indices,
                       NIndicesT nindices, GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                new detail::OpAssignVectorValue{detail::getVectorImpl(w), detail::getVectorImpl(mask),
                                                accum, val, indices, nindices, desc});
    }

    template<class CMatrixT, class MaskT, class ValueT, class IndexT, class NIndicesT>
    inline void assign(CMatrixT &C, MaskT &mask, GrB_BinaryOp accum, const ValueT &val, IndexT *rowIndices,
                       NIndicesT nrows, IndexT *colIndices, NIndicesT ncols, GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                new detail::OpAssignMatrixValue{detail::getMatrixImpl(C), detail::getMatrixImpl(mask),
                                                accum, val, rowIndices, nrows, colIndices, ncols, desc});
    }

    /* endregion */

    /* region apply */

    template<class CMatrixT, class MaskT, class AMatrixT,
            std::enable_if_t<detail::IsMatrix_v<CMatrixT>, int> = 0,
            std::enable_if_t<
                    detail::IsMatrix_v<MaskT> || std::is_same_v<std::remove_const_t<MaskT>, std::nullptr_t>, int> = 0,
            std::enable_if_t<detail::IsMatrix_v<AMatrixT>, int> = 0>
    inline void apply(CMatrixT &C, MaskT &mask, GrB_BinaryOp accum, GrB_UnaryOp op, AMatrixT &A, GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                new detail::OpApplyMatrix{detail::getMatrixImpl(C), detail::getMatrixImpl(mask),
                                          accum, op, detail::getMatrixImpl(A), desc});
    }

    template<class WVectorT, class MaskT, class UVectorT,
            std::enable_if_t<detail::IsVector_v<WVectorT>, int> = 0,
            std::enable_if_t<
                    detail::IsVector_v<MaskT> || std::is_same_v<std::remove_const_t<MaskT>, std::nullptr_t>, int> = 0,
            std::enable_if_t<detail::IsVector_v<UVectorT>, int> = 0>
    inline void apply(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, GrB_UnaryOp op, UVectorT &u, GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                new detail::OpApplyVector{detail::getVectorImpl(w), detail::getVectorImpl(mask),
                                          accum, op, detail::getVectorImpl(u), desc});
    }

    /* endregion */

    /* region select */

    template<class CMatrixT, class MaskT, class AMatrixT, class ScalarT,
            std::enable_if_t<detail::IsMatrix_v<CMatrixT>, int> = 0,
            std::enable_if_t<
                    detail::IsMatrix_v<MaskT> || std::is_same_v<std::remove_const_t<MaskT>, std::nullptr_t>, int> = 0,
            std::enable_if_t<detail::IsMatrix_v<AMatrixT>, int> = 0>
    inline void select(CMatrixT &C, MaskT &mask, GrB_BinaryOp accum, GxB_SelectOp op, AMatrixT &A, ScalarT thunk,
                       GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                new detail::OpSelectMatrix{detail::getMatrixImpl(C), detail::getMatrixImpl(mask), accum, op,
                                           detail::getMatrixImpl(A), thunk, desc}
        );
    }

    /* endregion */

    /* region reduce */

    template<class WVectorT, class MaskT, class OpT, class AMatrixT>
    inline void reduce(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, OpT op, AMatrixT &A, GrB_Descriptor desc) {
        detail::Context::getDefaultContext().addOperation(
                new detail::OpReduceMatrixVector(detail::getVectorImpl(w), detail::getVectorImpl(mask),
                                                 accum, op, detail::getMatrixImpl(A), desc));
    }

    template<class ValT, class AMatrixT, std::enable_if_t<detail::IsMatrix<AMatrixT>::value, int> = 0>
    [[nodiscard]] inline ValT reduce(GrB_BinaryOp accum, GrB_Monoid op, AMatrixT &A, GrB_Descriptor desc) {
        ValT val;
        detail::Context::getDefaultContext().addOperationAndWait(
                new detail::OpReduceMatrixScalar{val, accum, op, detail::getMatrixImpl(A), desc});
        return val;
    }

    template<class ValT, class UVectorT, std::enable_if_t<detail::IsVector<UVectorT>::value, int> = 0>
    [[nodiscard]] inline ValT reduce(GrB_BinaryOp accum, GrB_Monoid op, UVectorT &u, GrB_Descriptor desc) {
        ValT val;
        detail::Context::getDefaultContext().addOperationAndWait(
                new detail::OpReduceVectorScalar{val, accum, op, detail::getVectorImpl(u), desc});
        return val;
    }

    /* endregion */

    /* region wait */

    inline void wait() {
        detail::Context::getDefaultContext().wait();
    }

    /* endregion */

}

#endif //GRB_FUSION_GRB_H
