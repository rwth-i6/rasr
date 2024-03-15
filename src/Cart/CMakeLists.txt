set(Cart_SOURCES
        Conditions.cc
        Properties.cc
        Example.cc
        BinaryTree.cc
        DecisionTree.cc
        Cluster.cc
        DecisionTreeTrainer.cc
        Parser.cc)

add_library(RasrCart STATIC ${Cart_SOURCES})

target_link_libraries(RasrCart INTERFACE RasrCore)

add_executable(check-cart check.cc)
target_link_libraries(check-cart PRIVATE RasrCart)