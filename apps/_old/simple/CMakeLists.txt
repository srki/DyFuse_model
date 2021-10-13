add_executable(simple-mxm-cxx ../_old/simple/mxm.cpp)
target_link_libraries(simple-mxm-cxx grb mio otx-cpp)
target_compile_definitions(simple-mxm-cxx PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(simple-mxm-c ../_old/simple/mxm.c)
target_link_libraries(simple-mxm-c PUBLIC mio otx-c)
target_use_gblas(simple-mxm-c false)
target_compile_definitions(simple-mxm-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

find_package(OpenMP REQUIRED)

set(SCHEDULER_SRC
        ../scheduler/Scheduler.cpp
        ../scheduler/ConcurrencyControlComponent.cpp
        ../scheduler/PinThreadsTask.cpp
        ../scheduler/ControlTask.cpp
        ../scheduler/SchedulingPolicy.cpp)

add_executable(simple-mxm_scheduler mxm_scheduler.cpp ${SCHEDULER_SRC})
target_link_libraries(simple-mxm_scheduler PUBLIC grb mio otx-cpp OpenMP::OpenMP_CXX)
target_use_starpu(simple-mxm_scheduler)
target_use_gblas(simple-mxm_scheduler 1)
target_compile_definitions(simple-mxm_scheduler PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(mxm-perf-c ../_old/simple/mxm-perf.c)
target_link_libraries(mxm-perf-c PUBLIC mio otx-c OpenMP::OpenMP_C)
target_use_gblas(mxm-perf-c false)
target_compile_definitions(mxm-perf-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(mxm-stats-c ../_old/simple/mxm-stats.c)
target_link_libraries(mxm-stats-c PUBLIC mio otx-c OpenMP::OpenMP_C)
target_use_gblas(mxm-stats-c false)
target_compile_definitions(mxm-stats-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(mxm-replace-c ../_old/simple/mxm-replace.c)
target_link_libraries(mxm-replace-c PUBLIC mio otx-c OpenMP::OpenMP_C)
target_use_gblas(mxm-replace-c false)
target_compile_definitions(mxm-replace-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(mxm-stats-omp-c ../_old/simple/mxm-stats-omp.c)
target_link_libraries(mxm-stats-omp-c PUBLIC mio otx-c OpenMP::OpenMP_C)
target_use_gblas(mxm-stats-omp-c true)
target_compile_definitions(mxm-stats-omp-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(mxm-individual-blocks_model mxm-individual-blocks_model.c)
target_link_libraries(mxm-individual-blocks_model PUBLIC mio otx-c OpenMP::OpenMP_CXX)
target_use_gblas(mxm-individual-blocks_model 1)
target_compile_definitions(mxm-individual-blocks_model PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(mxm-individual-blocks ../_old/simple/mxm-individual-blocks.c)
target_link_libraries(mxm-individual-blocks PUBLIC mio otx-c OpenMP::OpenMP_C)
target_use_gblas(mxm-individual-blocks false)
target_compile_definitions(mxm-individual-blocks PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(mxm-individual-blocks-2 ../_old/simple/mxm-individual-blocks-2.c)
target_link_libraries(mxm-individual-blocks-2 PUBLIC mio otx-c OpenMP::OpenMP_C)
target_use_gblas(mxm-individual-blocks-2 false)
target_compile_definitions(mxm-individual-blocks-2 PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(mxm-blocked_v ../_old/simple/mxm-blocked_v-test.c)
target_link_libraries(mxm-blocked_v PUBLIC mio otx-c OpenMP::OpenMP_C)
target_use_gblas(mxm-blocked_v false)
target_compile_definitions(mxm-blocked_v PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(mxm-spu-variable-threads ../_old/simple/mxm-spu-variable-threads.cpp)
target_link_libraries(mxm-spu-variable-threads PUBLIC mio otx-cpp OpenMP::OpenMP_CXX)
target_use_gblas(mxm-spu-variable-threads true)
target_use_starpu(mxm-spu-variable-threads)
target_compile_definitions(mxm-spu-variable-threads PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(mxm-hpctoolkit ../_old/simple/mxm-hpctoolkit.c)
target_link_libraries(mxm-hpctoolkit PUBLIC mio otx-c OpenMP::OpenMP_C)
target_use_gblas(mxm-hpctoolkit false)
target_compile_definitions(mxm-hpctoolkit PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(mxm-hpctoolkit-blocked ../_old/simple/mxm-hpctoolkit-blocked.c)
target_link_libraries(mxm-hpctoolkit-blocked PUBLIC mio otx-c OpenMP::OpenMP_C)
target_use_gblas(mxm-hpctoolkit-blocked false)
target_compile_definitions(mxm-hpctoolkit-blocked PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(mxm-hpctoolkit-blocked-omp ../_old/simple/mxm-hpctoolkit-blocked-omp.c)
target_link_libraries(mxm-hpctoolkit-blocked-omp PUBLIC mio otx-c OpenMP::OpenMP_C)
target_use_gblas(mxm-hpctoolkit-blocked-omp true)
target_compile_definitions(mxm-hpctoolkit-blocked-omp PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

if (HPCTOOLKIT_HOME)
    target_compile_definitions(mxm-hpctoolkit PRIVATE HPCTOOLKIT)
    target_include_directories(mxm-hpctoolkit PRIVATE ${HPCTOOLKIT_HOME}/include)
    target_link_libraries(mxm-hpctoolkit PRIVATE ${HPCTOOLKIT_HOME}/lib/hpctoolkit/libhpctoolkit.so)

    target_compile_definitions(mxm-hpctoolkit-blocked PRIVATE HPCTOOLKIT)
    target_include_directories(mxm-hpctoolkit-blocked PRIVATE ${HPCTOOLKIT_HOME}/include)
    target_link_libraries(mxm-hpctoolkit-blocked PRIVATE ${HPCTOOLKIT_HOME}/lib/hpctoolkit/libhpctoolkit.so)

    target_compile_definitions(mxm-hpctoolkit-blocked-omp PRIVATE HPCTOOLKIT)
    target_include_directories(mxm-hpctoolkit-blocked-omp PRIVATE ${HPCTOOLKIT_HOME}/include)
    target_link_libraries(mxm-hpctoolkit-blocked-omp PRIVATE ${HPCTOOLKIT_HOME}/lib/hpctoolkit/libhpctoolkit.so)
else ()
    message(WARNING "HPCTOOLKIT_HOME is not defined.")
endif ()
