add_executable(multisource-bfs multisource.cpp)
target_link_libraries(multisource-bfs PRIVATE grb mio otx-cpp)
target_compile_definitions(multisource-bfs PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(multisource-bfs-c multisource.c)
target_use_gblas(multisource-bfs-c false)
target_link_libraries(multisource-bfs-c PRIVATE mio otx-c)
target_compile_definitions(multisource-bfs-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")