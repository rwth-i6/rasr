set(RasrOnnx_SOURCES
    BLstmStateManager.cc
    ConformerStateManager.cc
    DummyStateManager.cc
    IOSpecification.cc
    Model.cc
    Module.cc
    OnnxEncoder.cc
    OnnxFeatureScorer.cc
    OnnxForwardNode.cc
    OnnxLstmStateManager.cc
    Session.cc
    StateManager.cc
    Value.cc
)

add_library(RasrOnnx STATIC ${RasrOnnx_SOURCES})

add_onnx_dependencies(RasrOnnx)
target_link_libraries(RasrOnnx PUBLIC RasrCore RasrFlow RasrMath RasrMm RasrNn)

add_executable(check-onnx check.cc)
target_link_libraries(check-onnx PRIVATE RasrOnnx)
add_onnx_dependencies(check-onnx)
