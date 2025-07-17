set(Am_SOURCES
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
    list(APPEND Am_SOURCES DecisionTreeStateTying.cc ClassicDecisionTree.cc)
endif()

if(${MODULE_ADAPT_MLLR})
    list(APPEND Am_SOURCES AdaptedAcousticModel.cc AdaptationTree.cc)
endif()

add_library(RasrAm ${Am_SOURCES})

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

if(${MODULE_CART})
    target_link_libraries(RasrAm PUBLIC RasrCart)
endif()

add_executable(check-am check.cc)
target_link_libraries(check-am PRIVATE RasrAm)
