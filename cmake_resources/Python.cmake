if(${MODULE_PYTHON})
    # Find Python interpreter
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    include(cmake_resources/Tensorflow.cmake)
endif()