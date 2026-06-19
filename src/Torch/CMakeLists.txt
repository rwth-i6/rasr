add_library(
    RasrTorch STATIC
    ConformerStateManager.cc
    DummyStateManager.cc
    IOSpecification.cc
    IOSpecParser.cc
    Model.cc
    Module.cc
    Session.cc
    StateManager.cc
    TorchEncoder.cc
    TorchFeatureScorer.cc
    TorchForwardNode.cc
)

target_link_libraries(
    RasrTorch PUBLIC RasrExternalTorch RasrCore RasrFlow RasrMath RasrMm
    RasrNn
)

add_rasr_check_executable(check-torch check.cc)

target_link_rasr_check_libraries(check-torch PRIVATE RasrTorch)
