add_executable(
    aligner
    AlignAutomaton.cc
    Aligner.cc
    Common.cc
    ConditionalLexicon.cc
    ConditionalLexiconPlain.cc
    ConditionalLexiconSri.cc
    SimpleAlignAutomaton.cc
    ZeroOrderAlignAutomaton.cc
)
add_install_executable(aligner)

target_link_libraries(aligner PRIVATE RasrCore RasrFsa)
