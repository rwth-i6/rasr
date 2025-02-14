find_package(pybind11 REQUIRED)

set(librasr_SOURCES LibRASR.cc)

if(${MODULE_PYTHON})
  list(APPEND librasr_SOURCES PybindModule.cc)
endif()

pybind11_add_module(librasr PybindModule.cc LibRASR.cc)

target_link_libraries(
  librasr
  PRIVATE RasrFlf
          RasrFlfCore
          RasrSpeech
          RasrLattice
          RasrLm
          RasrAm
          RasrMm
          RasrMc
          RasrBliss
          RasrAudio
          RasrCore
          RasrFsa
          RasrFlow
          RasrSignal
          RasrMath
          RasrMathLapack)

add_install_library(librasr)

if(${MODULE_FLF_EXT})
  target_link_libraries(librasr PRIVATE RasrFlfExt)
endif()

if(${MODULE_CART})
  target_link_libraries(librasr PRIVATE RasrCart)
endif()

if(${MODULE_MATH_NR})
  target_link_libraries(librasr PRIVATE RasrMathNr)
endif()

if(${MODULE_NN})
  target_link_libraries(librasr PRIVATE RasrNn RasrLabelScorer)
endif()

if(${MODULE_PYTHON})
  target_link_libraries(librasr PRIVATE RasrPython)
  add_python_dependencies(librasr)
endif()

if(${MODULE_ONNX})
  target_link_libraries(librasr PRIVATE RasrOnnx)
  add_onnx_dependencies(librasr)
endif()

if(${MODULE_TENSORFLOW})
  target_link_libraries(librasr PRIVATE RasrTensorflow)
  add_tf_dependencies(librasr)
endif()
