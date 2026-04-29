set(RasrTensorflow_SOURCES
    DummyStateManager.cc
    TFLstmStateManager.cc
    Graph.cc
    GraphLoader.cc
    MetaGraphLoader.cc
    Module.cc
    Session.cc
    StateManager.cc
    Tensor.cc
    TensorflowFeatureScorer.cc
    TensorflowFeatureScorerStateCarryover.cc
    TensorflowForwardNode.cc
    TensorMap.cc
    TransformerStateManager.cc
    VanillaGraphLoader.cc
)

add_library(RasrTensorflow STATIC ${RasrTensorflow_SOURCES})

target_link_libraries(RasrTensorflow INTERFACE RasrCore)
add_tf_dependencies(RasrTensorflow)

add_executable(check-tensorflow check.cc)
target_link_libraries(check-tensorflow PRIVATE RasrTensorflow)
add_tf_dependencies(check-tensorflow)
