find_package(OpenMP REQUIRED)

set(SCHEDULER_SRC
        Scheduler.cpp
        ConcurrencyControlComponent.cpp
        PinThreadsTask.cpp
        ControlTask.cpp
        SchedulingPolicy.cpp)

add_library(schedulerlib SHARED ${SCHEDULER_SRC})
target_include_directories(schedulerlib PUBLIC .)
target_use_starpu(schedulerlib)
target_link_libraries(schedulerlib PUBLIC OpenMP::OpenMP_CXX)

add_executable(scheduler main.cpp  ${SCHEDULER_SRC})
target_link_libraries(scheduler PUBLIC mio otx-cpp OpenMP::OpenMP_CXX)
target_use_starpu(scheduler)
target_use_gblas(scheduler 0)
target_compile_definitions(scheduler PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(scheduler-micro-benchmark  ${SCHEDULER_SRC} scheduler-micro-benchmark.cpp)
target_link_libraries(scheduler-micro-benchmark PUBLIC mio otx-cpp OpenMP::OpenMP_CXX)
target_use_starpu(scheduler-micro-benchmark)
target_use_gblas(scheduler-micro-benchmark 0)
target_compile_definitions(scheduler-micro-benchmark PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")