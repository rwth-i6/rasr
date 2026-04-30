add_library(
    RasrAm STATIC
    AcousticModel.cc
    AcousticModelAdaptor.cc
    ClassicAcousticModel.cc
    ClassicHmmTopologySet.cc
    ClassicStateModel.cc
    ClassicStateTying.cc
    ClassicTransducerBuilder.cc
    Module.cc
    TransitionModel.cc
    Utilities.cc
)

if(${MODULE_CART})
    target_sources(
        RasrAm PRIVATE DecisionTreeStateTying.cc ClassicDecisionTree.cc
    )
endif()

if(${MODULE_ADAPT_MLLR})
    target_sources(RasrAm PRIVATE AdaptedAcousticModel.cc AdaptationTree.cc)
endif()

target_link_libraries(
    RasrAm
    PUBLIC RasrBliss
           RasrCore
           RasrFsa
           RasrMath
           RasrMc
           RasrMm
           RasrSpeech
)

add_rasr_check_executable(check-am check.cc)

target_link_rasr_check_libraries(check-am PRIVATE RasrAm)
