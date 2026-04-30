add_library(
    RasrTensorflow STATIC
    DummyStateManager.cc
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
    TFLstmStateManager.cc
    TransformerStateManager.cc
    VanillaGraphLoader.cc
)

target_link_libraries(
    RasrTensorflow PUBLIC RasrBliss RasrCore RasrFlow RasrMath RasrMm RasrNn
)
add_tf_dependencies(RasrTensorflow)

add_executable(check-tensorflow check.cc)

target_link_libraries(check-tensorflow PRIVATE RasrTensorflow)
