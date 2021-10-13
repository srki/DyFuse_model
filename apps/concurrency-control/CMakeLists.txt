find_package(OpenMP REQUIRED)

add_executable(spu-ctx spu-ctx.c)
#target_link_libraries(spu-ctx PUBLIC mio otx-cpp OpenMP::OpenMP_C)
#target_use_gblas(spu-ctx false)
target_use_starpu(spu-ctx)
#target_compile_definitions(mxm-spu-variable-threads PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(spu-dummy spu-dummy.cpp)
target_link_libraries(spu-dummy PUBLIC OpenMP::OpenMP_CXX)
target_use_starpu(spu-dummy)

add_executable(spu-ctx-tricnt spu-ctx-tricnt.cpp)
target_link_libraries(spu-ctx-tricnt PUBLIC mio otx-cpp OpenMP::OpenMP_C)
target_use_starpu(spu-ctx-tricnt)
target_use_gblas(spu-ctx-tricnt 0)
target_compile_definitions(spu-ctx-tricnt PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(omp-test omp-test.c)
target_link_libraries(omp-test PUBLIC OpenMP::OpenMP_C)

add_executable(omp-spu-test omp-spu-test.c)
target_link_libraries(omp-spu-test PUBLIC OpenMP::OpenMP_C)
target_use_starpu(omp-spu-test)

add_executable(spu-clusters spu-clusters.c)
target_link_libraries(spu-clusters PUBLIC OpenMP::OpenMP_C)
target_use_starpu(spu-clusters)

add_executable(spu-par-task spu-par-task.c)
target_link_libraries(spu-par-task PUBLIC OpenMP::OpenMP_C)
target_use_starpu(spu-par-task)