/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_TRAITS_H
#define GRB_FUSION_TRAITS_H

#include <type_traits>

#include <grb/objects/Vector.h>
#include <grb/objects/VectorImpl.h>
#include <grb/objects/VectorData.h>

#include <grb/objects/Matrix.h>
#include <grb/objects/MatrixImpl.h>
#include <grb/objects/MatrixData.h>

namespace grb::detail {

    template<class, template<class, class...> class>
    struct IsInstance : public std::false_type {
    };

    template<class...Ts, template<class, class...> class U>
    struct IsInstance<U<Ts...>, U> : public std::true_type {
    };

    template<class T>
    struct IsMatrix : IsInstance<T, Matrix> {
    };

    template<class T>
    inline constexpr bool IsMatrix_v = IsMatrix<T>::value;

    template<class T>
    struct IsVector : IsInstance<T, Vector> {
    };

    template<class T>
    inline constexpr bool IsVector_v = IsVector<T>::value;

    template<class T>
    struct IsMatrixImpl : IsInstance<T, MatrixImpl> {
    };

    template<class T>
    inline constexpr bool IsMatrixImpl_v = IsMatrixImpl<T>::value;

    template<class T>
    struct IsVectorImpl : IsInstance<T, VectorImpl> {
    };

    template<class T>
    inline constexpr bool IsVectorImpl_v = IsVectorImpl<T>::value;

    template<class T>
    struct VectorInfo : std::false_type {
    };

    /* @formatter:off */
    template<class ScalarT, class IndexT>
    struct VectorInfo<Vector < ScalarT, IndexT>> : std::true_type {
        using scalar_t = ScalarT;
        using index_t = IndexT;
    };

    /* region MatrixInfo */

    template<class T>
    struct MatrixInfo : std::false_type {
    };

    template<class ScalarT, class IndexT>
    struct MatrixInfo<Matrix<ScalarT, IndexT>> : std::true_type {
        using scalar_t = ScalarT;
        using index_t = IndexT;
    };

    template<class ScalarT, class IndexT>
    struct MatrixInfo<MatrixImpl<ScalarT, IndexT>> : std::true_type {
        using scalar_t = ScalarT;
        using index_t = IndexT;
    };

    /* endregion */

    template<class U>
    struct Impl {
        using type = void;
    };

    template<>
    struct Impl<std::nullptr_t> {
        using type = std::nullptr_t ;
    };

    template<class ScalarT, class IndexT>
    struct Impl<Vector<ScalarT, IndexT>> {
        using type = VectorImpl<ScalarT, IndexT>;
    };

    template<class ScalarT, class IndexT>
    struct Impl<Matrix<ScalarT, IndexT>> {
        using type = MatrixImpl<ScalarT, IndexT>;
    };

    template<class T>
    using Impl_t = typename Impl<T>::type;

    /* region ImplPtr */

    template<class U>
    struct ImplPtr {
        using type = void;
    };

    template<>
    struct ImplPtr<std::nullptr_t> {
        using type = std::nullptr_t;
    };

    template<>
    struct ImplPtr<const std::nullptr_t> {
        using type = std::nullptr_t;
    };

    template<class ScalarT, class IndexT>
    struct ImplPtr<Vector<ScalarT, IndexT>> {
        using type = std::shared_ptr<VectorImpl<ScalarT, IndexT>>;
    };

    template<class ScalarT, class IndexT>
    struct ImplPtr<Matrix<ScalarT, IndexT>> {
        using type = std::shared_ptr<MatrixImpl<ScalarT, IndexT>>;
    };

    template<class T>
    using ImplPtr_t = typename ImplPtr<T>::type;

    /* endregion */

    /* region ImplPtr */

    template<class U>
    struct DataPtr {
        using type = void;
    };

    template<>

    struct [[deprecated]]DataPtr<std::nullptr_t> {
        using type = std::nullptr_t;
    };

    template<>
    struct [[deprecated]]DataPtr<const std::nullptr_t> {
        using type = std::nullptr_t;
    };

    template<class ScalarT, class IndexT>
    struct DataPtr<VectorImpl<ScalarT, IndexT>> {
        using type = std::shared_ptr<VectorData<ScalarT, IndexT>>;
    };

    template<class ScalarT, class IndexT>
    struct DataPtr<MatrixImpl<ScalarT, IndexT>> {
        using type = std::shared_ptr<MatrixData<ScalarT, IndexT>>;
    };

    template<class ScalarT, class IndexT>
    struct DataPtr<Vector<ScalarT, IndexT>> {
        using type = std::shared_ptr<VectorData<ScalarT, IndexT>>;
    };

    template<class ScalarT, class IndexT>
    struct DataPtr<Matrix<ScalarT, IndexT>> {
        using type = std::shared_ptr<MatrixData<ScalarT, IndexT>>;
    };

    template<>
    struct DataPtr<VectorImpl<void, void>> {
        using type = VectorData<void, void>;
    };

    template<>
    struct DataPtr<MatrixImpl<void, void>> {
        using type = MatrixData<void, void>;
    };

    template<class T>
    using DataPtr_t = typename DataPtr<T>::type;

    /* endregion */

    template <class T>
    struct HasData {};

    template <class ScalarT, class IndexT>
    struct HasData<VectorImpl<ScalarT, IndexT>> : std::true_type {};

    template <class ScalarT, class IndexT>
    struct HasData<MatrixImpl<ScalarT, IndexT>> : std::true_type {};

    template <>
    struct HasData<VectorImpl<void, void>> : std::false_type {};

    template <>
    struct HasData<MatrixImpl<void, void>> : std::false_type {};

    template <class T>
    inline constexpr bool HasData_v = HasData<T>::value;

    /* @formatter:on */

}


#endif //GRB_FUSION_TRAITS_H
