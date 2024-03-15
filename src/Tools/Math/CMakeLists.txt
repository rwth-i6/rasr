add_executable(matrix-tool MatrixTool.cc)
add_install_executable(matrix-tool)

target_link_libraries(matrix-tool PRIVATE RasrMath RasrMathNr RasrMathLapack
        RasrCore)
