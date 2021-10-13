find_package(OpenMP REQUIRED)

add_executable(tricnt-cxx tricount.cpp)
target_link_libraries(tricnt-cxx grb mio otx-cpp)
target_compile_definitions(tricnt-cxx PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(tricnt tricount.c)
target_link_libraries(tricnt PUBLIC mio otx-c OpenMP::OpenMP_C)
target_use_gblas(tricnt false)
target_compile_definitions(tricnt PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")

add_executable(tricnt-omp tricount-omp.c)
target_link_libraries(tricnt-omp PUBLIC mio otx-c OpenMP::OpenMP_C)
target_use_gblas(tricnt-omp true)
target_compile_definitions(tricnt-omp PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
