set(FlfExt_SOURCES AcousticAlignment.cc MapDecoder.cc MtConfusionNetwork.cc
                   WindowedLevenshteinDistanceDecoder.cc)

add_library(RasrFlfExt STATIC ${FlfExt_SOURCES})
target_link_libraries(RasrFlfExt INTERFACE RasrFlf RasrFsa RasrCore)

if(${MODULE_PYTHON})
  add_python_dependencies(RasrFlfExt)
endif()
