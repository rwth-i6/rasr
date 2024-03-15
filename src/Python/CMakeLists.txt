set(RasrPython_SOURCES Init.cc Numpy.cc Utilities.cc)

add_library(RasrPython STATIC ${RasrPython_SOURCES})
target_link_libraries(RasrPython INTERFACE RasrCore)
add_python_dependencies(RasrPython)
target_compile_options(RasrPython PRIVATE "-fexceptions")

add_executable(check-python check.cc)
target_link_libraries(check-python PRIVATE RasrPython)
add_python_dependencies(check-python)