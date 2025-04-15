set(RasrTreeTimesyncBeamSearch_SOURCES TreeTimesyncBeamSearch.cc)

add_library(RasrTreeTimesyncBeamSearch STATIC
            ${RasrTreeTimesyncBeamSearch_SOURCES})
target_link_libraries(RasrTreeTimesyncBeamSearch INTERFACE RasrSearch RasrBliss
                                                           RasrFsa RasrCore)

if(${MODULE_PYTHON})
  add_python_dependencies(RasrTreeTimesyncBeamSearch)
endif()
