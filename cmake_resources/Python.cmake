find_package(Python3 REQUIRED COMPONENTS Development NumPy)

function(add_python_dependencies target)
    target_compile_definitions(${target} PUBLIC ${Python3_DEFINITIONS})
    target_include_directories(
        ${target} PUBLIC ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PUBLIC ${Python3_LIBRARIES})
endfunction()
