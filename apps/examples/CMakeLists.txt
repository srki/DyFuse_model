add_executable(B1-c B1.c)
target_use_gblas(B1-c false)
target_link_libraries(B1-c PRIVATE otx-c mio)
target_compile_definitions(B1-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(B1-cxx B1.cpp)
target_link_libraries(B1-cxx grb otx-cpp mio)
target_compile_definitions(B1-cxx PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(B5-c B5.c)
target_use_gblas(B5-c false)
target_link_libraries(B5-c PRIVATE mio)
target_compile_definitions(B5-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(B5-cxx B5.cpp)
target_link_libraries(B1-cxx grb mio)
target_compile_definitions(B1-cxx PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(B7-c B7.c)
target_use_gblas(B7-c false)
target_link_libraries(B7-c PRIVATE mio)
target_compile_definitions(B7-c PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(B7-cxx B7.cpp)
target_link_libraries(B7-cxx grb mio)
target_compile_definitions(B7-cxx PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")


