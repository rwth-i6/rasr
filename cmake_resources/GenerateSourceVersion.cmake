if(NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "OUTPUT_FILE is required")
endif()

if(NOT DEFINED SOURCE_DIR)
    get_filename_component(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

set(_release_file "${SOURCE_DIR}/src/SourceVersion.release")

if(EXISTS "${_release_file}")
    file(READ "${_release_file}" _source_version_content)
else()
    execute_process(
        COMMAND "${SOURCE_DIR}/scripts/git-version.py"
        WORKING_DIRECTORY "${SOURCE_DIR}"
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

# The temp file with `copy_if_different` avoids an "always dirty" output file
# that triggers unnecessary rebuilds
set(_temp_file "${OUTPUT_FILE}.tmp")
file(WRITE "${_temp_file}" "${_source_version_content}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_temp_file}"
            "${OUTPUT_FILE}"
)
file(REMOVE "${_temp_file}")
