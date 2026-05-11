find_package(Python REQUIRED COMPONENTS Development NumPy Interpreter Development.Module)

add_library(RasrPythonDependencies INTERFACE)
target_compile_definitions(
    RasrPythonDependencies INTERFACE ${Python3_DEFINITIONS}
)
target_link_libraries(
        RasrPythonDependencies INTERFACE Python::Python Python::NumPy
)
