set(onnx_INCLUDE_DIR "/opt/thirdparty/usr/include")

find_library(onnxruntime_LIBRARY onnxruntime REQUIRED
             HINTS "/opt/thirdparty/usr/lib")

function(add_onnx_dependencies TARGET)
  target_include_directories(${TARGET} PUBLIC ${onnx_INCLUDE_DIR})
  target_link_libraries(${TARGET} PUBLIC ${onnxruntime_LIBRARY})
endfunction()
