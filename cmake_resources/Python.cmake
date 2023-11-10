find_package(Python3 REQUIRED COMPONENTS Development NumPy)

function(add_python_dependencies TARGET)
    target_compile_definitions(${TARGET} PUBLIC ${Python3_DEFINITIONS})
    target_include_directories(${TARGET} PUBLIC ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PUBLIC ${Python3_LIBRARIES})
endfunction()

include(cmake_resources/Tensorflow.cmake)