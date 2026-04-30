if(NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "OUTPUT_FILE is required")
endif()

set(_release_file "${CMAKE_CURRENT_SOURCE_DIR}/src/SourceVersion.release")

if(EXISTS "${_release_file}")
    file(READ "${_release_file}" _source_version_content)
else()
    execute_process(
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/git-version.py"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE _git_version_result
        OUTPUT_VARIABLE _git_version
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(_git_version_result EQUAL 0)
        set(_source_version_content "\"git-${_git_version}\\n\"\n")
    else()
        set(_source_version_content "\"n/a\\n\"\n")
    endif()
endif()

get_filename_component(_output_dir "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${_output_dir}")
file(WRITE "${OUTPUT_FILE}" "${_source_version_content}")
