set(RasrLexiconfreeTimesyncBeamSearch_SOURCES LexiconfreeTimesyncBeamSearch.cc)

add_library(RasrLexiconfreeTimesyncBeamSearch STATIC
            ${RasrLexiconfreeTimesyncBeamSearch_SOURCES})
target_link_libraries(RasrLexiconfreeTimesyncBeamSearch
                      INTERFACE RasrSearch RasrBliss RasrFsa RasrCore)

if(${MODULE_PYTHON})
  add_python_dependencies(RasrLexiconfreeTimesyncBeamSearch)
endif()
