add_library(
    RasrFlow STATIC
    AbstractNode.cc
    Aggregate.cc
    Attributes.cc
    Cache.cc
    CorpusKeyMap.cc
    Cutter.cc
    Data.cc
    DataAdaptor.cc
    Datatype.cc
    Demo.cc
    Dump.cc
    InputNode.cc
    Link.cc
    Module.cc
    Network.cc
    NetworkParser.cc
    Node.cc
    Registry.cc
    Repeater.cc
    SequenceFilter.cc
    StringExpressionNode.cc
    Synchronization.cc
    Timestamp.cc
    Vector.cc
    VectorTextInput.cc
    WarpTimeFilter.cc
)

target_link_libraries(RasrFlow PUBLIC RasrCore RasrMath)

add_executable(check-flow check.cc)

target_link_libraries(check-flow PRIVATE RasrFlow)
