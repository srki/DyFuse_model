/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_VECTOR_H
#define GRB_FUSION_VECTOR_H

#include <memory>
#include <grb/objects/VectorImpl.h>

namespace grb {

    template<class ScalarT, class IndexT = uint64_t>
    class Vector {
        using Impl = detail::VectorImpl<ScalarT, IndexT>;

    public:
        explicit Vector(IndexT nsize, size_t blockSize = 0) : _impl(nsize, blockSize) {}

        Vector() : Vector(0) {}

        Vector(const Vector &other) : _impl{other._impl} {}

        Vector(Vector &&other) noexcept {
            *this = std::move(other);
        }

        Vector &operator=(const Vector &other) {
            if (this == &other) { return *this; }
            _impl = other._impl;
            return *this;
        }

        Vector &operator=(Vector &&other) noexcept {
            if (this == &other) { return *this; }
            _impl = std::move(other._impl);
            return *this;
        }

        /* region public API */

        void resize(IndexT nsize) {
            _impl.resize(nsize);
        }

        void clear() {
            _impl.clear();
        }

        [[nodiscard]] IndexT size() {
            return _impl.size();
        }

        [[nodiscard]] IndexT nvals() {
            return _impl.nvals();
        }

        template<class AccumT>
        void build(const IndexT *indices, const ScalarT *values, GrB_Index n, AccumT const &accum) {
            _impl.build(indices, values, n, accum);
        }

        void setElement(ScalarT val, IndexT index) {
            _impl.setElement(val, index);
        }

        void removeElement(IndexT index) {
            _impl.removeElement(index);
        }

        [[nodiscard]] ScalarT extractElement(IndexT index) {
            return _impl.extractElement(index);
        }

        void extractTuples(IndexT *indices, ScalarT *values, IndexT *n) {
            _impl.extractTuples(indices, values, n);
        }

        /* endregion */

        /* region API extensions */

        void wait() {
            _impl.wait();
        }

        void print() {
            _impl.print();
        }

        auto &getImpl() {
            return _impl;
        }

        const auto &getImpl() const {
            return _impl;
        }

        void block(size_t blockSize) {
            _impl.block(blockSize);
        }

        /* endregion */

    private:
        Impl _impl;

    public:
        Vector(IndexT nsize, const std::string &name, IndexT blockSize = 0) : Vector(nsize, blockSize) {
            setName(name);
        }

        void setName(const std::string &name) {
            _impl.setName(name);
        }

    };

    template <class ScalarT1, class IndexT1, class ScalarT2, class IndexT2>
    bool operator==(const Vector<ScalarT1, IndexT1> &lhs, const Vector<ScalarT2, IndexT2> &rhs) {
        return lhs.getImpl() == rhs.getImpl();
    }

}

#endif //GRB_FUSION_MATRIX_H
