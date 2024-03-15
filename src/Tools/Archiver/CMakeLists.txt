add_executable(archiver Archiver.cc)
add_install_executable(archiver)

target_link_libraries(
  archiver
  PRIVATE RasrSpeech
          RasrSearch
          RasrAdvancedTreeSearch
          RasrLattice
          RasrLm
          RasrFlf
          RasrFlfCore
          RasrMc
          RasrAm
          RasrMm
          RasrCart
          RasrSignal
          RasrBliss
          RasrMath
          RasrMathLapack
          RasrMathNr
          RasrCore
          RasrFlow
          RasrFsa)

if(${MODULE_GENERIC_SEQ2SEQ_TREE_SEARCH})
  target_link_libraries(archiver PRIVATE RasrGenericSeq2SeqTreeSearch)
endif()

if(${MODULE_PYTHON})
  target_link_libraries(archiver PRIVATE RasrPython)
  add_python_dependencies(archiver)
endif()

if(${MODULE_NN})
  target_link_libraries(archiver PRIVATE RasrNn)
endif()

if(${MODULE_SEARCH_WFST})
  target_link_libraries(archiver PRIVATE RasrSearchWfst)
endif()

if(${MODULE_OPENFST})
  target_link_libraries(archiver PRIVATE RasrOpenFst)
endif()

if(${MODULE_LM_TFRNN})
  target_link_libraries(archiver PRIVATE RasrTensorflow)
  add_tf_dependencies(archiver)
endif()
