set(RasrTensorflow_SOURCES
        Graph.cc
        GraphLoader.cc
        MetaGraphLoader.cc
        Module.cc
        Session.cc
        Tensor.cc
        TensorflowForwardNode.cc
        TensorMap.cc
        VanillaGraphLoader.cc
)

add_library(RasrTensorflow STATIC ${RasrTensorflow_SOURCES})

target_link_libraries(RasrTensorflow INTERFACE RasrCore)
add_tf_dependencies(RasrTensorflow)