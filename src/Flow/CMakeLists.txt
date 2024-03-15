set(Flow_SOURCES
    AbstractNode.cc
    Aggregate.cc
    Attributes.cc
    Cache.cc
    Cutter.cc
    CorpusKeyMap.cc
    Data.cc
    DataAdaptor.cc
    Datatype.cc
    Demo.cc
    Dump.cc
    Link.cc
    Module.cc
    Network.cc
    NetworkParser.cc
    Node.cc
    Registry.cc
    Repeater.cc
    SequenceFilter.cc
    WarpTimeFilter.cc
    StringExpressionNode.cc
    Synchronization.cc
    Timestamp.cc
    Vector.cc
    VectorTextInput.cc)

add_library(RasrFlow STATIC ${Flow_SOURCES})

target_link_libraries(RasrFlow INTERFACE RasrCore)
