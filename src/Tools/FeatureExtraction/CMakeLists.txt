add_executable(feature-extraction FeatureExtraction.cc FeatureExtractor.cc)
add_install_executable(feature-extraction)

target_link_libraries(
  feature-extraction
  PRIVATE RasrSpeech
          RasrLattice
          RasrLm
          RasrAm
          RasrMm
          RasrMc
          RasrBliss
          RasrAudio
          RasrSignal
          RasrFlow
          RasrMath
          RasrMathLapack
          RasrCore
          RasrFsa
          ${SEARCH_LIBS})

if(${MODULE_CART})
  target_link_libraries(feature-extraction PRIVATE RasrCart)
endif()

if(${MODULE_FLF_CORE})
  target_link_libraries(feature-extraction PRIVATE RasrFlfCore)
endif()

if(${MODULE_MATH_NR})
  target_link_libraries(feature-extraction PRIVATE RasrMathNr)
endif()

if(${MODULE_PYTHON})
  target_link_libraries(feature-extraction PRIVATE RasrPython)
  add_python_dependencies(feature-extraction)
endif()

if(${MODULE_NN})
  target_link_libraries(feature-extraction PRIVATE RasrNn)
endif()

if(${MODULE_TENSORFLOW})
  target_link_libraries(feature-extraction PRIVATE RasrTensorflow)
  add_tf_dependencies(feature-extraction)
endif()
