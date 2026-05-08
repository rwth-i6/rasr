find_package(Python3 REQUIRED COMPONENTS Development NumPy)

add_library(RasrPythonDependencies INTERFACE)
target_compile_definitions(
    RasrPythonDependencies INTERFACE ${Python3_DEFINITIONS}
)
target_link_libraries(
    RasrPythonDependencies INTERFACE Python3::Python Python3::NumPy
)
