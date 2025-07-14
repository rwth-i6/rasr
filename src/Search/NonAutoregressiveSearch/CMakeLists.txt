set(RasrNonAutoregressiveSearch_SOURCES NonAutoregressiveSearch.cc)

add_library(RasrNonAutoregressiveSearch STATIC
            ${RasrNonAutoregressiveSearch_SOURCES})
target_link_libraries(RasrNonAutoregressiveSearch
                      INTERFACE RasrSearch RasrBliss RasrFsa RasrCore)

if(${MODULE_PYTHON})
  add_python_dependencies(RasrNonAutoregressiveSearch)
endif()
