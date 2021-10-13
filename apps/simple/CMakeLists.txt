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

add_executable(mxm-individual-blocks_model mxm-individual-blocks_model.c)
target_link_libraries(mxm-individual-blocks_model PUBLIC mio otx-c OpenMP::OpenMP_CXX)
target_use_gblas(mxm-individual-blocks_model 1)
target_compile_definitions(mxm-individual-blocks_model PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(mxm-scheduler mxm-scheduler.cpp)
target_use_gblas(mxm-scheduler false)
target_link_libraries(mxm-scheduler PUBLIC schedulerlib mio otx-cpp)
target_compile_definitions(mxm-scheduler PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(block-profiling block-profiling.cpp)
target_use_gblas(block-profiling false)
target_link_libraries(block-profiling PUBLIC schedulerlib mio otx-cpp)
target_compile_definitions(block-profiling PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(block-features block-features.cpp)
target_use_gblas(block-features false)
target_link_libraries(block-features PUBLIC mio otx-cpp OpenMP::OpenMP_CXX)
target_compile_definitions(block-features PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
