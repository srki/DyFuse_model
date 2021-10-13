/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_REDUCE_MATRIX_VECTOR_H
#define GRB_FUSION_OP_REDUCE_MATRIX_VECTOR_H

#include <starpu.h>


namespace grb::detail {
    template<class WVectorT, class MaskT, class OpT, class AMatrixT>
    class OpReduceMatrixVector : public Operation {
        static_assert(std::is_same_v<OpT, GrB_Monoid> ||
                      std::is_same_v<OpT, GrB_BinaryOp>);
    public:
        OpReduceMatrixVector(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, OpT op, AMatrixT &A,
                             GrB_Descriptor desc)
                : Operation{OperationType::REDUCE_MATRIX_VECTOR}, _w{w.getData()}, _mask{mask.getData()},
                  _mA{A.getData()}, _args{accum, op, desc} {
            initDependencies(w, mask, A);
            initCodelet();
            initTasks();
        }

        void initTasks() {
            /* Check whether A is transposed */
            GrB_Desc_Value transposed;
            grbTry(GxB_Descriptor_get(&transposed, std::get<2>(_args), GrB_INP0));

            if (transposed == GrB_TRAN) {
                for (size_t i = 0; i < _mA->getNumRowsBlocked(); i++) {
                    for (size_t j = 0; j < _mA->getNumColsBlocked(); j++) {
                        auto codelet = &(i == 0 ? _codeletA : _codeletB);
                        addTask(createTask(codelet, _args, _w, j, _mA, i * _mA->getNumColsBlocked() + j, _mask,
                                           j));
                    }
                }
            } else {
                for (size_t i = 0; i < _mA->getNumRowsBlocked(); i++) {
                    for (size_t j = 0; j < _mA->getNumColsBlocked(); j++) {
                        auto codelet = &(j == 0 ? _codeletA : _codeletB);
                        addTask(createTask(codelet, _args, _w, i, _mA, i * _mA->getNumColsBlocked() + j, _mask,
                                           i));
                    }
                }
            }

        }


        void release() override {
            _w.reset();
            _mask.reset();
            _mA.reset();
        }


    private:
        const static bool USE_MASK = !std::is_same_v<MaskT, VectorImpl < void, void>>
        ;

        DataPtr_t <WVectorT> _w;
        DataPtr_t <MaskT> _mask;
        DataPtr_t <AMatrixT> _mA;

        std::tuple<GrB_BinaryOp, OpT, GrB_Descriptor> _args;

        starpu_codelet _codeletA;
        starpu_codelet _codeletB;

        void initDependencies(WVectorT &w, MaskT &maskT, AMatrixT &A) {
            addInputDependency(A.getOp(), DependencyType::READ);

            for (const auto dep : w.getOp()->getOutputDependencies()) {
                if (dep.second == DependencyType::REUSE) { continue; }
                addInputDependency(dep.first, DependencyType::REUSE);
            }

            addInputDependency(w.getOp(), DependencyType::WRITE);
            w.setOp(this);
        }

        void initCodelet() {
            starpu_codelet_init(&_codeletA);
            _codeletA.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto w = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto A = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[1]);
                auto mask = USE_MASK ? *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[2]) : GrB_NULL;


                GrB_Index r, c, s;
                GrB_Matrix_nrows(&r, A);
                GrB_Matrix_ncols(&c, A);
                GrB_Vector_size(&s, w);

                auto[accum, op, desc] = *static_cast<decltype(_args) *>(clArgs);
                if constexpr (std::is_same_v<OpT, GrB_Monoid>) {
                    grbTry(GrB_Matrix_reduce_Monoid(w, mask, accum, op, A, desc));
                } else if constexpr (std::is_same_v<OpT, GrB_BinaryOp>) {
                    grbTry(GrB_Matrix_reduce_BinaryOp(w, mask, accum, op, A, desc));
                }
            };
            _codeletA.nbuffers = 2 + (USE_MASK ? 1 : 0);
            _codeletA.modes[0] = std::get<0>(_args) == GrB_NULL ? STARPU_W : STARPU_RW; // w
            _codeletA.modes[1] = STARPU_R; // A
            _codeletA.modes[2] = STARPU_R; // mask

            starpu_codelet_init(&_codeletB);
            _codeletB.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto w = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto A = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[1]);
                GrB_Vector mask = GrB_NULL;
                if (USE_MASK) {
                    mask = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[2]);
                }

                auto[accum, op, desc] = *static_cast<decltype(_args) *>(clArgs);
                if constexpr (std::is_same_v<OpT, GrB_Monoid>) {
                    GxB_Monoid_operator(&accum, op);
                    grbTry(GrB_Matrix_reduce_Monoid(w, mask, accum, op, A, desc));
                } else if constexpr (std::is_same_v<OpT, GrB_BinaryOp>) {
                    accum = op;
                    grbTry(GrB_Matrix_reduce_BinaryOp(w, mask, accum, op, A, desc));
                }
            };
            _codeletB.nbuffers = 2 + (USE_MASK ? 1 : 0);
            _codeletB.modes[0] = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE); // w
            _codeletB.modes[1] = STARPU_R; // A
            _codeletB.modes[2] = STARPU_R; // mask
        }
    };
}
#endif //GRB_FUSION_OP_REDUCE_MATRIX_VECTOR_H
