add_library(
    RasrBliss STATIC
    CorpusDescription.cc
    CorpusKey.cc
    CorpusParser.cc
    CorpusStatistics.cc
    EditDistance.cc
    Evaluation.cc
    Fsa.cc
    Lexicon.cc
    LexiconParser.cc
    OrthographicParser.cc
    Orthography.cc
    Phoneme.cc
    Phonology.cc
    PrefixTree.cc
    SegmentOrdering.cc
    Symbol.cc
    Unknown.cc
)

if(${MODULE_PYTHON})
    target_sources(RasrBliss PRIVATE PythonSegmentOrdering.cc)
endif()

target_link_libraries(RasrBliss PUBLIC RasrCore RasrFsa RasrMath RasrSpeech)

if(${MODULE_PYTHON})
    target_link_libraries(RasrBliss PUBLIC RasrPython)
endif()

add_rasr_check_executable(check-bliss check.cc)

target_link_rasr_check_libraries(check-bliss PRIVATE RasrBliss)
