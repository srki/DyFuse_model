find_package(OpenMP REQUIRED)

macro(add_example _name)
    add_executable(${_name} ${_name}.c)
    target_link_libraries(${_name} PUBLIC mio otx-c OpenMP::OpenMP_C)
    target_use_gblas(${_name} false)
    target_compile_definitions(${_name} PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
endmacro()


macro(add_example_spu _name)
    add_executable(${_name} ${_name}.c)
    target_link_libraries(${_name} PUBLIC mio otx-c OpenMP::OpenMP_C)
    target_use_gblas(${_name} false)
    target_use_starpu(${_name})
    target_compile_definitions(${_name} PUBLIC INPUTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs")
endmacro()



#add_example(stats-baseline)
#add_example(stats-omp)
#add_example_spu(stats-spu)
#add_example_spu(stats-spu-noaff)
#add_example_spu(omp-thread-pool)
add_example(kron-blocked)