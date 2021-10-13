set(SRC_TASK_GRAPH
        ast/Operation.cpp
        ast/OpMxM.cpp)

set(SRC_CONTEXT
        context/Context.cpp)

set(SRC
        ${SRC_TASK_GRAPH}
        ${SRC_CONTEXT})

add_library(grb SHARED ${SRC})

target_include_directories(grb
        PUBLIC
        ../include
        ../extern
        PRIVATE
        ../src
        )

target_compile_features(grb PUBLIC cxx_std_11)

target_use_gblas(grb true)
target_use_starpu(grb)