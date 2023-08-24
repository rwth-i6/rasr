# Definition of used modules and tools
#
# The MODULE_xxx terms are available as variables in all makefiles
# and as preprocessor directives in src/Modules.hh
#
# Those parts of Sprint which are not required for a basic ASR system
# should be separated by a MODULE_ definition wherever applicable.
#
# If you implement a new fancy feature in Sprint:
#   * goal: everything can still be compiled (and run) without your source files
#     by simply deactivating the module in Modules.make
#   * try to implement your classes as loosely coupled to other classes as possible
#   * define a module name in Modules.make
#   * make the makefiles depend on that name by including 'ifdef MODULE_xxx' ... 'endif'
#     (remember the makefiles in Tools/*)
#   * frame the include files of your module by '#ifdef MODULE_xxx' ... '#endif'
#     do not forget to include Modules.hh
#   * Use '#MODF MyFile.hh' to mark files of a module if the file is not listed in the
#     Makefile (e.g. header files without corresponding .cc file). See Signal/Makefile
#     for an example

# ****** Adaptation ******
MODULES += MODULE_ADAPT_CMLLR
MODULES += MODULE_ADAPT_MLLR
MODULES += MODULE_ADAPT_ADVANCED

# ****** Audio ******
MODULES += MODULE_AUDIO_FFMPEG
MODULES += MODULE_AUDIO_FLAC
MODULES += MODULE_AUDIO_HTK
MODULES += MODULE_AUDIO_OSS
MODULES += MODULE_AUDIO_RAW
MODULES += MODULE_AUDIO_WAV_SYSTEM

# ****** Cache Manager integration ******
# MODULES += MODULE_CORE_CACHE_MANAGER

# ****** Cart ******
MODULES += MODULE_CART

# ****** Flf ******
MODULES += MODULE_FLF_CORE
MODULES += MODULE_FLF_EXT
MODULES += MODULE_FLF

# ****** Lattice ******
MODULES   += MODULE_LATTICE_BASIC
MODULES   += MODULE_LATTICE_HTK
MODULES   += MODULE_LATTICE_DT

# ****** Lm ******
MODULES += MODULE_LM_ARPA
MODULES += MODULE_LM_FSA
MODULES += MODULE_LM_ZEROGRAM
MODULES += MODULE_LM_FFNN
MODULES += MODULE_LM_TFRNN

# ****** Math ******
MODULES += MODULE_MATH_NR

# ****** Mm ******
MODULES += MODULE_MM_BATCH
MODULES += MODULE_MM_DT

# ****** Neural Network ******
MODULES += MODULE_NN
MODULES += MODULE_NN_SEQUENCE_TRAINING
MODULES += MODULE_THEANO_INTERFACE
MODULES += MODULE_PYTHON

# ****** OpenFst ******
MODULES += MODULE_OPENFST

# ****** Search ******
MODULES += MODULE_SEARCH_MBR
MODULES += MODULE_SEARCH_WFST
MODULES += MODULE_SEARCH_LINEAR
MODULES += MODULE_ADVANCED_TREE_SEARCH
MODULES += MODULE_GENERIC_SEQ2SEQ_TREE_SEARCH

# ****** Signal ******
MODULES += MODULE_SIGNAL_GAMMATONE
MODULES += MODULE_SIGNAL_PLP
MODULES += MODULE_SIGNAL_VTLN
MODULES += MODULE_SIGNAL_VOICEDNESS
MODULES += MODULE_SIGNAL_ADVANCED
MODULES += MODULE_SIGNAL_ADVANCED_NR

# ****** Speech ******
MODULES += MODULE_SPEECH_DT
MODULES += MODULE_SPEECH_DT_ADVANCED
MODULES += MODULE_SPEECH_ALIGNMENT_FLOW_NODES
MODULES += MODULE_SPEECH_LATTICE_FLOW_NODES
MODULES += MODULE_SPEECH_LATTICE_ALIGNMENT
MODULES += MODULE_SPEECH_LATTICE_RESCORING

# ****** Unit Tests ******
MODULES += MODULE_TEST

# ****** Intel Threading Building Blocks ******
# MODULES += MODULE_TBB

# ****** OpenMP library
# MODULES += MODULE_OPENMP
# **** choose optimized blas library if available
# MODULES += MODULE_INTEL_MKL
# MODULES += MODULE_ACML
# MODULES += MODULE_CUDA

# Tensorflow integration
MODULES += MODULE_TENSORFLOW

# define variables for the makefiles
$(foreach module, $(MODULES), $(eval $(module) = 1))

# ****** Tools ******
TOOLS += AcousticModelTrainer
TOOLS += Archiver
TOOLS += CorpusStatistics
TOOLS += FeatureExtraction
TOOLS += FeatureStatistics
TOOLS += Fsa
TOOLS += SpeechRecognizer
TOOLS += Xml

ifdef MODULE_CART
TOOLS += Cart
endif

ifdef MODULE_MM_DT
ifdef MODULE_LATTICE_DT
ifdef MODULE_SPEECH_DT
TOOLS += LatticeProcessor
endif
endif
endif

ifdef MODULE_FLF
TOOLS += Flf
endif

ifdef MODULE_NN
TOOLS += NnTrainer
endif

# ****** Libraries ******
LIBS_SEARCH = src/Search/libSprintSearch.$(a)
ifdef MODULE_SEARCH_WFST
LIBS_SEARCH += src/Search/Wfst/libSprintSearchWfst.$(a)
LIBS_SEARCH += src/OpenFst/libSprintOpenFst.$(a)
endif
ifdef MODULE_ADVANCED_TREE_SEARCH
LIBS_SEARCH += src/Search/AdvancedTreeSearch/libSprintAdvancedTreeSearch.$(a)
endif
ifdef MODULE_GENERIC_SEQ2SEQ_TREE_SEARCH
LIBS_SEARCH += src/Search/GenericSeq2SeqTreeSearch/libSprintGenericSeq2SeqTreeSearch.$(a)
endif
