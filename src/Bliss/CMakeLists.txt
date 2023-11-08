set(Bliss_SOURCES
        CorpusDescription.cc
        CorpusKey.cc
        CorpusParser.cc
        CorpusStatistics.cc
        EditDistance.cc
        Evaluation.cc
        Fsa.cc
        Lexicon.cc
        LexiconParser.cc
        Orthography.cc
        Phoneme.cc
        Phonology.cc
        PrefixTree.cc
        SegmentOrdering.cc
        Symbol.cc
        Unknown.cc
)

if(${MODULE_THEANO_INTERFACE})
    list(APPEND Bliss_SOURCES TheanoCommunicator.cc TheanoSegmentOrderingVisitor.cc)
endif()

if(${MODULE_PYTHON})
    list(APPEND Bliss_SOURCES PythonSegmentOrdering.cc)
endif()

add_library(RasrBliss ${Bliss_SOURCES})

if(${MODULE_PYTHON})
    target_compile_definitions(RasrBliss PRIVATE ${Python3_DEFINITIONS})
    target_include_directories(RasrBliss PRIVATE ${Python3_INCLUDE_DIRS})
endif()
