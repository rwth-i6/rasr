set(Lm_SOURCES
        AbstractNNLanguageModel.cc
        BackingOff.cc
        ClassLm.cc
        CombineLm.cc
        Compose.cc
        CorpusStatistics.cc
        IndexMap.cc
        LanguageModel.cc
        Module.cc
        NNHistoryManager.cc
        ReverseArpaLm.cc
        ScaledLanguageModel.cc
        WordlistInterface.cc
)

if(${MODULE_LM_ARPA})
        list(APPEND Lm_SOURCES ArpaLm.cc)
endif()

if(${MODULE_LM_FSA})
    list(APPEND Lm_SOURCES FsaLm.cc CheatingSegmentLm.cc)
endif()

if(${MODULE_LM_ZEROGRAM})
    list(APPEND Lm_SOURCES Zerogram.cc)
endif()

if(${MODULE_TENSORFLOW} AND ${MODULE_LM_TFRNN})
    list(APPEND Lm_SOURCES Zerogram.cc)
endif()

add_library(RasrLm STATIC ${Lm_SOURCES})

target_link_libraries(RasrLm INTERFACE
        RasrFlf
        RasrFlfCore
        RasrSpeech
        RasrAm
        RasrMc
        RasrBliss
        RasrNn
        RasrMe
        RasrMm
        RasrSignal
        RasrFlow
        RasrMath
        RasrMathLapack
        ${SEARCH_LIBS}
        RasrLattice
        RasrFsa
        RasrCore
)
if(${MODULE_CART})
    target_link_libraries(RasrLm INTERFACE RasrCart)
endif()
if(${MODULE_MATH_NR})
    target_link_libraries(RasrLm INTERFACE RasrMathNr)
endif()
if(${MODULE_PYTHON})
    target_link_libraries(RasrLm INTERFACE RasrPython)
endif()
if(${MODULE_TENSORFLOW} AND ${MODULE_LM_TFRNN})
    target_link_libraries(RasrLm INTERFACE RasrTensorflow)
    add_tf_dependencies(RasrLm)
endif()
