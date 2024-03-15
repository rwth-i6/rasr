set(aligner_SOURCES
    Aligner.cc
    AlignAutomaton.cc
    SimpleAlignAutomaton.cc
    ZeroOrderAlignAutomaton.cc
    Common.cc
    ConditionalLexicon.cc
    ConditionalLexiconPlain.cc
    ConditionalLexiconSri.cc)
add_executable(aligner ${ALIGNER_SOURCES})
add_install_executable(aligner)

target_link_libraries(aligner PRIVATE RasrFsa RasrCore)
