set(Cart_SOURCES
        Conditions.cc
        Properties.cc
        Example.cc
        BinaryTree.cc
        DecisionTree.cc
        Cluster.cc
        DecisionTreeTrainer.cc
        Parser.cc
        )

add_library(RasrCart STATIC ${Cart_SOURCES})
#target_link_libraries(RasrCart PRIVATE RasrCore)

target_include_directories(RasrCart PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})