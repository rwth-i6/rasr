set(RasrPython_SOURCES
    AllophoneStateFsaBuilder.cc
    CallbackRegistry.cc
    Configuration.cc
    Init.cc
    LimitedCtxPythonLabelScorer.cc
    Module.cc
    Numpy.cc
    Search.cc
    Utilities.cc)

add_library(RasrPython STATIC ${RasrPython_SOURCES})
add_python_dependencies(RasrPython)

if(${MODULE_NN})
  target_link_libraries(RasrPython PUBLIC RasrNn)
endif()
