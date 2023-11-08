set(Am_SOURCES
        AcousticModel.cc
        AcousticModelAdaptor.cc
        ClassicAcousticModel.cc
        ClassicHmmTopologySet.cc
        ClassicStateModel.cc
        ClassicTransducerBuilder.cc
        Module.cc
        TransitionModel.cc
        Utilities.cc
)

if(${MODULE_CART})
    list(APPEND Am_SOURCES DecisionTreeStateTying.cc ClassicDecisionTree.cc)
endif()

if(${MODULE_MATH_NR})
    list(APPEND Am_SOURCES AdaptedAcousticModel.cc AdaptationTree.cc)
endif()

add_library(RasrAm ${Am_SOURCES})