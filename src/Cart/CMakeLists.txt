add_library(
    RasrCart STATIC
    BinaryTree.cc
    Cluster.cc
    Conditions.cc
    DecisionTree.cc
    DecisionTreeTrainer.cc
    Example.cc
    Parser.cc
    Properties.cc
)

target_link_libraries(RasrCart PUBLIC RasrCore)

add_executable(check-cart check.cc)

target_link_libraries(check-cart PRIVATE RasrCart)
