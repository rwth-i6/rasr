set(RasrOnnx_SOURCES
        Session.cc
        IOSpecification.cc
        Module.cc
        OnnxFeatureScorer.cc
        OnnxForwardNode.cc
        Value.cc
)

add_library(RasrOnnx STATIC ${RasrOnnx_SOURCES})

target_link_libraries(RasrOnnx INTERFACE RasrCore)