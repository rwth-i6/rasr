add_executable(feature-statistics FeatureStatistics.cc)
add_install_executable(feature-statistics)

target_link_libraries(
  feature-statistics
  PRIVATE RasrSpeech
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
          ${SEARCH_LIBS}
          RasrCore
          RasrFsa)

if(${MODULE_CART})
  target_link_libraries(feature-statistics PRIVATE RasrCart)
endif()

if(${MODULE_FLF_CORE})
  target_link_libraries(feature-statistics PRIVATE RasrFlfCore)
endif()

if(${MODULE_MATH_NR})
  target_link_libraries(feature-statistics PRIVATE RasrMathNr)
endif()

if(${MODULE_PYTHON})
  target_link_libraries(feature-statistics PRIVATE RasrPython)
  add_python_dependencies(feature-statistics)
endif()

if(${MODULE_NN})
  target_link_libraries(feature-statistics PRIVATE RasrNn)
endif()

if(${MODULE_TENSORFLOW})
  target_link_libraries(feature-statistics PRIVATE RasrTensorflow)
  add_tf_dependencies(feature-statistics)
endif()
