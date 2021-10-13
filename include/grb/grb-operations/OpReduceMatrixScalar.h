/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_REDUCE_MATRIX_SCALAR_H
#define GRB_FUSION_OP_REDUCE_MATRIX_SCALAR_H

#include <starpu.h>

namespace grb::detail {

    template<class ValT, class AMatrixT>
    class OpReduceMatrixScalar : public Operation {
    public:
        OpReduceMatrixScalar(ValT &val, GrB_BinaryOp accum, GrB_Monoid op, AMatrixT &A, GrB_Descriptor desc)
                : Operation{OperationType::REDUCE_MATRIX_SCALAR}, _mA{A.getData()}, _args{&val, accum, op, desc},
                  _numBlocks{_mA->getNumBlocks()} {
            initDependencies(A);
            initCodelet();
            initBlockArgs();
            initTasks();
        }

        void initTasks() {
            for (size_t i = 0; i < _mA->getNumBlocks(); i++) {
                addTask(createTask(&_codelet, _blockArgs[i], _mA, i));
            }
        }

    protected:
        bool wait() override {
            if (!Operation::wait()) { return false; }

            GrB_Vector v;
            GrB_Vector_new(&v, CInterface<ValT>::type(), _numBlocks);
            for (size_t i = 0; i < _numBlocks; i++) {
                CInterface<ValT>::vectorSetElement(v, _tmpValues[i], i);
            }

            auto[val, accum, op, desc] = _args;
            CInterface<ValT>::vectorReduceScalar(val, accum, op, v, desc);
            GrB_Vector_free(&v);

            return true;
        }


        void release() override {
            _mA.reset();
        }


    private:
        DataPtr_t <AMatrixT> _mA;
        std::tuple<ValT *, GrB_BinaryOp, GrB_Monoid, GrB_Descriptor> _args;

        size_t _numBlocks;
        std::unique_ptr<decltype(_args)[]> _blockArgs;
        std::unique_ptr<ValT[]> _tmpValues;

        starpu_codelet _codelet;

        void initDependencies(AMatrixT &A) {
            addInputDependency(A.getOp(), DependencyType::READ);
        }

        void initCodelet() {
            starpu_codelet_init(&_codelet);
            _codelet.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto A = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto[val, accum, op, desc] = *static_cast<decltype(_args) *>(clArgs);

                grbTry(CInterface<ValT>::matrixReduceScalar(val, accum, op, A, desc));
            };
            _codelet.nbuffers = 1;
            _codelet.modes[0] = STARPU_R; // A
        }

        void initBlockArgs() {
            _blockArgs = std::make_unique<decltype(_args)[]>(_numBlocks);
            _tmpValues = std::make_unique<ValT[]>(_numBlocks);

            for (size_t i = 0; i < _numBlocks; i++) {
                std::get<0>(_blockArgs[i]) = &_tmpValues[i];
                std::get<1>(_blockArgs[i]) = std::get<1>(_args);
                std::get<2>(_blockArgs[i]) = std::get<2>(_args);
                std::get<3>(_blockArgs[i]) = std::get<3>(_args);
            }
        }
    };

}

#endif //GRB_FUSION_OP_REDUCE_MATRIX_SCALAR_H
