set(onnx_INCLUDE_DIR "/opt/thirdparty/usr/include")

find_library(onnxruntime_LIBRARY onnxruntime PATHS "/opt/thirdparty/usr/lib")

function(add_onnx_dependencies TARGET)
  target_compile_options(${TARGET} PUBLIC "-fexceptions")
  target_include_directories(${TARGET} PUBLIC ${onnx_INCLUDE_DIR})
  target_link_libraries(${TARGET} PUBLIC ${onnxruntime_LIBRARY})
endfunction()
