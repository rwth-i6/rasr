set(RasrOnnx_SOURCES
    Session.cc
    IOSpecification.cc
    Model.cc
    Module.cc
    OnnxEncoder.cc
    OnnxFeatureScorer.cc
    OnnxForwardNode.cc
    Value.cc)

add_library(RasrOnnx STATIC ${RasrOnnx_SOURCES})

add_python_dependencies(RasrOnnx)
add_onnx_dependencies(RasrOnnx)
target_link_libraries(RasrOnnx INTERFACE RasrCore)

add_executable(check-onnx check.cc)
target_link_libraries(check-onnx PRIVATE RasrOnnx)
add_onnx_dependencies(check-onnx)

if(${MODULE_CUDA})
  add_cuda_dependencies(RasrOnnx)
  add_cuda_dependencies(check-onnx)
endif()
