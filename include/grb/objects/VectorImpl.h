/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_VECTOR_IMPL_H
#define GRB_FUSION_VECTOR_IMPL_H

#include <cassert>

#include <starpu.h>

#include <grb/objects/VectorData.h>
#include <grb/context/Context.h>
#include <grb/context/Operation.h>
#include <grb/context/OpInternalClone.h>
#include <grb/grb-operations/OpCtor.h>
#include <grb/util/Util.h>
#include <grb/util/GraphBLASImpl.h>

namespace grb::detail {

    template<class ScalarT, class IndexT>
    class VectorImpl {
        static_assert(std::is_same_v<IndexT, GrB_Index>, "IndexT must be same as GrB_Index.");

    public:
        VectorImpl() = default;

        explicit VectorImpl(IndexT nsize, IndexT blockSize)
                : _data{nsize > 0 ? std::make_shared<VectorData<ScalarT, IndexT>>(nsize, blockSize) : nullptr},
                  _op{new OpCtor<false>{}} {
            Context::getDefaultContext().addOperation(_op);
        }

        VectorImpl(const VectorImpl &other) : _data{other._data}, _op{other._op}, _name{other._name} {};

        VectorImpl(VectorImpl &&other) noexcept {
            *this = std::move(other);
        };

        VectorImpl &operator=(const VectorImpl &other) {
            if (this == &other) { return *this; }
            _data = other._data;
            _op = other._op;
            _name = other._name;
            return *this;
        };

        VectorImpl &operator=(VectorImpl &&other) noexcept {
            if (this == &other) { return *this; }
            release();
            swap(*this, other);
            return *this;
        };

        ~VectorImpl() {
            release();
        }

        friend void swap(VectorImpl &lhs, VectorImpl &rhs) {
            std::swap(lhs._data, rhs._data);
            std::swap(lhs._op, rhs._op);
            std::swap(lhs._name, rhs._name);
        }

        void release() {
            _op = nullptr;
            _data = nullptr;
            _name.clear();
        }

        /* region Public API */

        void resize(IndexT nsize) {
            std::cerr << "Not implemented: " << __PRETTY_FUNCTION__ << std::endl;
            exit(1);
        }

        void clear() {
            /* TODO: check whether _data was not modified or if there are pending ops */
            clone(false);
        }

        [[nodiscard]] IndexT size() {
            return _data->getSize();
        }

        [[nodiscard]] IndexT nvals() {
            Context::getDefaultContext().wait();
            return _data->nvals();
        }

        void build(const IndexT *indices, const ScalarT *values, GrB_Index n, GrB_BinaryOp accum) {
            _data->build(indices, values, n, accum);
        }

        void setElement(ScalarT val, IndexT index) {
            Context::getDefaultContext().wait(); // TODO: maybe create an Operation class instead
            _data->setElement(val, index);
        }

        void removeElement(IndexT index) {
            Context::getDefaultContext().wait(); // TODO: maybe create an Operation class instead
            _data->removeElement(index);
        }

        [[nodiscard]] ScalarT extractElement(IndexT index) {
            Context::getDefaultContext().wait();
            return _data->extractElement(index)();
        }

        void extractTuples(IndexT *indices, ScalarT *values, IndexT *n) {
            Context::getDefaultContext().wait();
            assert(*n >= this->nvals());
            _data->extractTuples(indices, values, n);
        }

        /* endregion */

        /* region Extended API */

        void wait() {
            _op->wait();
        }

        void print() {
            Context::getDefaultContext().wait();

            auto nvals = this->nvals();
            auto I = std::make_unique<IndexT[]>(nvals);
            auto X = std::make_unique<ScalarT[]>(nvals);
            extractTuples(I.get(), X.get(), &nvals);

            GrB_Vector t;
            GrB_Vector_new(&t, CInterface<ScalarT>::type(), size());
            grbTry(CInterface<ScalarT>::vectorBuild(t, I.get(), X.get(), nvals, CInterface<ScalarT>::plus()));

            GxB_Vector_fprint(t, _name.c_str(), GxB_COMPLETE, stdout);
            GrB_Vector_free(&t);
        }

        void block(IndexT blockSize) {
            wait();

            auto size = _data->getSize();
            auto nvals = _data->nvals();
            auto I = std::make_unique<IndexT[]>(nvals);
            auto X = std::make_unique<ScalarT[]>(nvals);

            _data->extractTuples(I.get(), X.get(), &nvals);
            _data.reset(new VectorData<ScalarT, IndexT>{size, blockSize});
            _data->build(I.get(), X.get(), nvals, CInterface<ScalarT>::plus());
        }

        /* endregion */

        /* region Internal API */

        [[nodiscard]] Operation *getOp() { return _op; }

        [[nodiscard]] std::shared_ptr<VectorData<ScalarT, IndexT>> &getData() { return _data; }

        void setData(VectorData<ScalarT, IndexT> *data) { _data.reset(data); };

        void setOp(Operation *op) { _op = op; }

        void setName(const std::string &name) {
            _name = name;
            if (_op->getType() == OperationType::VECTOR_OP) {
                _op->setName(name);
            }
        }

        friend bool operator==(const VectorImpl &lhs, const VectorImpl &rhs) {
            auto eq = lhs._data.get() == rhs._data.get();

            if (eq) { assert(lhs._op == rhs._op && lhs._name == rhs._name); }
            else { assert(lhs._op != rhs._op); }

            return eq;
        }

        void clone(bool fullClone) {
            _data.reset(_data->clone(fullClone));
            _op = new OpInternalClone{_name};
        }

        /* endregion */

    private:
        std::shared_ptr<VectorData<ScalarT, IndexT>> _data;
        Operation *_op{nullptr};
        std::string _name{};
    };


    template<>
    class VectorImpl<void, void> {
    public:
        [[nodiscard]] static VectorImpl &getInstance() {
            const static VectorImpl instance;
            return const_cast<VectorImpl &>(instance);
        }

        [[nodiscard]] Operation *getOp() const { return nullptr; }

        [[nodiscard]] VectorData<void, void> getData() const { return VectorData<void, void>(); }

        friend bool operator==(const VectorImpl &lhs, const VectorImpl &rhs) = delete;

    };


    template <class ScalarT1, class IndexT1, class ScalarT2, class IndexT2>
    bool operator==(const VectorImpl<ScalarT1, IndexT1> &lhs, const VectorImpl<ScalarT2, IndexT2> &rhs) {
        static_assert(!std::is_same_v<ScalarT1, ScalarT2> || !std::is_same_v<IndexT1, IndexT2>);
        return false;
    }
}

#endif //GRB_FUSION_VECTOR_IMPL_H
