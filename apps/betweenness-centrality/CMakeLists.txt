add_executable(bc-c bc.c)
target_use_gblas(bc-c false)
target_link_libraries(bc-c PRIVATE mio otx-c)
target_compile_definitions(bc-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(bc-cxx bc.cpp)
target_link_libraries(bc-cxx PRIVATE grb mio otx-cpp)
target_compile_definitions(bc-cxx PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
