find_package(OpenMP REQUIRED)

set(SCHEDULER_SRC
        ../scheduler/Scheduler.cpp
        ../scheduler/ConcurrencyControlComponent.cpp
        ../scheduler/PinThreadsTask.cpp
        ../scheduler/ControlTask.cpp
        ../scheduler/SchedulingPolicy.cpp)

add_executable(simple_mxm_scheduler mxm_scheduler.cpp ${SCHEDULER_SRC})
target_link_libraries(simple_mxm_scheduler PUBLIC grb mio otx-cpp OpenMP::OpenMP_CXX)
target_use_starpu(simple_mxm_scheduler)
target_use_gblas(simple_mxm_scheduler 1)
target_compile_definitions(simple_mxm_scheduler PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(simple_spgemm_scheduler spgemm_scheduler.cpp ${SCHEDULER_SRC})
target_link_libraries(simple_spgemm_scheduler PUBLIC grb mio otx-cpp OpenMP::OpenMP_CXX)
target_use_starpu(simple_spgemm_scheduler)
target_use_gblas(simple_spgemm_scheduler 1)
target_compile_definitions(simple_spgemm_scheduler PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(ktruss-c ktruss.c)
target_use_gblas(ktruss-c false)
target_link_libraries(ktruss-c PRIVATE mio otx-c)
target_compile_definitions(ktruss-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(simple_ktruss_scheduler ktruss_scheduler.cpp ${SCHEDULER_SRC})
target_link_libraries(simple_ktruss_scheduler PUBLIC grb mio otx-cpp OpenMP::OpenMP_CXX)
target_use_starpu(simple_ktruss_scheduler)
target_use_gblas(simple_ktruss_scheduler 1)
target_compile_definitions(simple_ktruss_scheduler PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")


add_executable(simple_kron_scheduler kron_scheduler.cpp ${SCHEDULER_SRC})
target_link_libraries(simple_kron_scheduler PUBLIC grb mio otx-cpp OpenMP::OpenMP_CXX)
target_use_starpu(simple_kron_scheduler)
target_use_gblas(simple_kron_scheduler 1)
target_compile_definitions(simple_kron_scheduler PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")


add_executable(train_tricnt_scheduler tricnt_train_model_scheduler.cpp ${SCHEDULER_SRC})
target_link_libraries(train_tricnt_scheduler PUBLIC grb mio otx-cpp OpenMP::OpenMP_CXX)
target_use_starpu(train_tricnt_scheduler)
target_use_gblas(train_tricnt_scheduler 1)
target_compile_definitions(train_tricnt_scheduler PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(train_spgemm_scheduler spgemm_train_model_scheduler.cpp ${SCHEDULER_SRC})
target_link_libraries(train_spgemm_scheduler PUBLIC grb mio otx-cpp OpenMP::OpenMP_CXX)
target_use_starpu(train_spgemm_scheduler)
target_use_gblas(train_spgemm_scheduler 1)
target_compile_definitions(train_spgemm_scheduler PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(train_spgemm_comp_scheduler spgemm_train_complement_model_scheduler.cpp ${SCHEDULER_SRC})
target_link_libraries(train_spgemm_comp_scheduler PUBLIC grb mio otx-cpp OpenMP::OpenMP_CXX)
target_use_starpu(train_spgemm_comp_scheduler)
target_use_gblas(train_spgemm_comp_scheduler 1)
target_compile_definitions(train_spgemm_comp_scheduler PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")




add_executable(feature_extraction feature_extraction.cpp)
target_use_gblas(feature_extraction 1)
target_link_libraries(feature_extraction PUBLIC mio otx-cpp OpenMP::OpenMP_CXX)
target_compile_definitions(feature_extraction PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

#add_executable(simple-mxm-cxx mxm.cpp)
#target_link_libraries(simple-mxm-cxx grb mio otx-cpp)
#target_compile_definitions(simple-mxm-cxx PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")


#add_executable(simple-mxm-c mxm.c)
#target_link_libraries(simple-mxm-c PUBLIC mio otx-c)
#target_use_gblas(simple-mxm-c false)
#target_compile_definitions(simple-mxm-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
#
#find_package(OpenMP REQUIRED)
#
#add_executable(mxm-perf-c mxm-perf.c)
#target_link_libraries(mxm-perf-c PUBLIC mio otx-c OpenMP::OpenMP_C)
#target_use_gblas(mxm-perf-c false)
#target_compile_definitions(mxm-perf-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
#
#add_executable(mxm-stats-c mxm-stats.c)
#target_link_libraries(mxm-stats-c PUBLIC mio otx-c OpenMP::OpenMP_C)
#target_use_gblas(mxm-stats-c false)
#target_compile_definitions(mxm-stats-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
#
#add_executable(mxm-replace-c mxm-replace.c)
#target_link_libraries(mxm-replace-c PUBLIC mio otx-c OpenMP::OpenMP_C)
#target_use_gblas(mxm-replace-c false)
#target_compile_definitions(mxm-replace-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
#
#add_executable(mxm-stats-omp-c mxm-stats-omp.c)
#target_link_libraries(mxm-stats-omp-c PUBLIC mio otx-c OpenMP::OpenMP_C)
#target_use_gblas(mxm-stats-omp-c true)
#target_compile_definitions(mxm-stats-omp-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

#add_executable(mxm-individual-blocks_model mxm-individual-blocks_model.c)
#target_link_libraries(mxm-individual-blocks_model PUBLIC mio otx-c OpenMP::OpenMP_CXX)
#target_use_gblas(mxm-individual-blocks_model 1)
#target_compile_definitions(mxm-individual-blocks_model PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

#add_executable(mxm-individual-blocks-2 mxm-individual-blocks-2.c)
#target_link_libraries(mxm-individual-blocks-2 PUBLIC mio otx-c OpenMP::OpenMP_C)
#target_use_gblas(mxm-individual-blocks-2 false)
#target_compile_definitions(mxm-individual-blocks-2 PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
#
#add_executable(mxm-blocked_v mxm-blocked_v-test.c)
#target_link_libraries(mxm-blocked_v PUBLIC mio otx-c OpenMP::OpenMP_C)
#target_use_gblas(mxm-blocked_v false)
#target_compile_definitions(mxm-blocked_v PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
#
#add_executable(mxm-spu-variable-threads mxm-spu-variable-threads.cpp)
#target_link_libraries(mxm-spu-variable-threads PUBLIC mio otx-cpp OpenMP::OpenMP_CXX)
#target_use_gblas(mxm-spu-variable-threads true)
#target_use_starpu(mxm-spu-variable-threads)
#target_compile_definitions(mxm-spu-variable-threads PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
#
#add_executable(mxm-hpctoolkit mxm-hpctoolkit.c)
#target_link_libraries(mxm-hpctoolkit PUBLIC mio otx-c OpenMP::OpenMP_C)
#target_use_gblas(mxm-hpctoolkit false)
#target_compile_definitions(mxm-hpctoolkit PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
#
#add_executable(mxm-hpctoolkit-blocked mxm-hpctoolkit-blocked.c)
#target_link_libraries(mxm-hpctoolkit-blocked PUBLIC mio otx-c OpenMP::OpenMP_C)
#target_use_gblas(mxm-hpctoolkit-blocked false)
#target_compile_definitions(mxm-hpctoolkit-blocked PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
#
#add_executable(mxm-hpctoolkit-blocked-omp mxm-hpctoolkit-blocked-omp.c)
#target_link_libraries(mxm-hpctoolkit-blocked-omp PUBLIC mio otx-c OpenMP::OpenMP_C)
#target_use_gblas(mxm-hpctoolkit-blocked-omp true)
#target_compile_definitions(mxm-hpctoolkit-blocked-omp PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

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
