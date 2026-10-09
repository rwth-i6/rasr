add_library(
    RasrOnnx STATIC
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

if (${MODULE_CUDA})
    target_sources(
        RasrOnnx PRIVATE DeviceVector.cc TransformerGpuStateManager.cc
        TransformerGpuKernels.cu
    )
endif ()

target_link_libraries(
    RasrOnnx PUBLIC ONNXRuntime::ONNXRuntime RasrCore RasrFlow RasrMath RasrMm
                    RasrNn
)

add_rasr_check_executable(check-onnx check.cc)

target_link_rasr_check_libraries(check-onnx PRIVATE RasrOnnx)
