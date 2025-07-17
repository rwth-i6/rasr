find_package(CUDAToolkit REQUIRED)

function(add_cuda_dependencies TARGET)
  target_include_directories(${TARGET} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(${TARGET} PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
endfunction()
