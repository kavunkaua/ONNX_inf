cmake_minimum_required(VERSION 3.10)

# Name of the project and its versions
project(ONNXRuntimeGPUExample VERSION 1.0 LANGUAGES CXX)

# Setting C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(JSON_PATH "include/")
set(ONNX_PATH "/opt/onnxruntime/")

# Hardcoded paths to ONNX Runtime
set(ONNXRUNTIME_INCLUDE_PATH ${ONNX_PATH}include/)
set(ONNXRUNTIME_INCLUDE_PATH_2 ${ONNX_PATH}include/)
set(ONNXRUNTIME_INCLUDE_PATH_3 ${ONNX_PATH}include/)
set(ONNXRUNTIME_INCLUDE_PATH_4 ${ONNX_PATH}include/)

set(OPENCV_INCLUDE_PATH /usr/include/opencv4/)
set(ONNXRUNTIME_LIB_PATH ${ONNX_PATH}lib/)


# Indication of the need to generate debug information:
set(CMAKE_BUILD_TYPE Debug)

# Variable for the name of the target executable file
set(TARGET_BASE inference)
set(TARGET_ADD1 tokenizer_test)
set(TARGET_ADD2 InferenceONNX_test)
set(TARGET_ADD3 get_names)

# Check that the paths are specified
if(NOT ONNXRUNTIME_INCLUDE_PATH OR NOT ONNXRUNTIME_LIB_PATH)
    message(FATAL_ERROR "Please specify ONNXRUNTIME_INCLUDE_PATH and ONNXRUNTIME_LIB_PATH.")
endif()

# Adding header files
include_directories(${JSON_PATH})
include_directories(${ONNXRUNTIME_INCLUDE_PATH})
include_directories(${ONNXRUNTIME_INCLUDE_PATH_2})
include_directories(${ONNXRUNTIME_INCLUDE_PATH_3})
include_directories(${ONNXRUNTIME_INCLUDE_PATH_4})
include_directories(${OPENCV_INCLUDE_PATH})


# Specifying libraries
link_directories(${ONNXRUNTIME_LIB_PATH})

# Main executable file
add_executable(${TARGET_BASE} src/${TARGET_BASE}.cpp src/tokenizer.cpp src/inference.cpp)
add_executable(${TARGET_ADD1} src/${TARGET_ADD1}.cpp src/tokenizer.cpp src/InferenceONNX.cpp)
add_executable(${TARGET_ADD2} src/${TARGET_ADD2}.cpp src/tokenizer.cpp src/InferenceONNX.cpp)
add_executable(${TARGET_ADD3} src/${TARGET_ADD3}.cpp)

# Linking with ONNX Runtime libraries
# For GPUs, we use the onnxruntime_gpu library
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_LIB_PATH} NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_GPU_LIB onnxruntime_providers_cuda PATHS ${ONNXRUNTIME_LIB_PATH} NO_DEFAULT_PATH)


if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_GPU_LIB)
    message(FATAL_ERROR "Failed to find ONNX Runtime libraries. Please check ONNXRUNTIME_LIB_PATH.")
endif()

# Use pcre2-config to get paths
execute_process(COMMAND pcre2-config --cflags OUTPUT_VARIABLE PCRE2_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND pcre2-config --libs8 OUTPUT_VARIABLE PCRE2_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)

# Linking libraries
target_link_libraries(${TARGET_BASE} PRIVATE ${ONNXRUNTIME_LIB} ${ONNXRUNTIME_GPU_LIB} ${PCRE2_LIBS})
target_link_libraries(${TARGET_ADD1} PRIVATE ${PCRE2_LIBS})
target_link_libraries(${TARGET_ADD2} PRIVATE ${ONNXRUNTIME_LIB} ${ONNXRUNTIME_GPU_LIB} ${PCRE2_LIBS})
target_link_libraries(${TARGET_ADD3} PRIVATE ${ONNXRUNTIME_LIB} ${ONNXRUNTIME_GPU_LIB} ${PCRE2_LIBS})

# Configuration message
message(STATUS "ONNX Runtime include path: ${ONNXRUNTIME_INCLUDE_PATH}")
message(STATUS "ONNX Runtime lib path: ${ONNXRUNTIME_LIB_PATH}")

# Specify compilation flags
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -funroll-loops -mavx2")
set(CFLAGS_LIST "")
list(APPEND CFLAGS_LIST "${PCRE2_CFLAGS}")
#list(APPEND CFLAGS_LIST "-O3")
#list(APPEND CFLAGS_LIST "-g") #-g debug flag 

target_compile_options(${TARGET_BASE} PRIVATE ${CFLAGS_LIST})
target_compile_options(${TARGET_ADD1} PRIVATE ${CFLAGS_LIST})
target_compile_options(${TARGET_ADD2} PRIVATE ${CFLAGS_LIST}) 
target_compile_options(${TARGET_ADD3} PRIVATE ${CFLAGS_LIST}) 

# Specify libraries for linking
set(CMAKE_DISABLE_BUILD_${TARGET_BASE} 1)
set(CMAKE_DISABLE_BUILD_${TARGET_ADD1} 1)
set(CMAKE_DISABLE_BUILD_${TARGET_ADD2} 1)

# Collect all project goals
set_target_properties(${TARGET_BASE} PROPERTIES OUTPUT_NAME ${TARGET_BASE} )
set_target_properties(${TARGET_ADD1} PROPERTIES OUTPUT_NAME ${TARGET_ADD1} )
set_target_properties(${TARGET_ADD2} PROPERTIES OUTPUT_NAME ${TARGET_ADD2} )
set_target_properties(${TARGET_ADD3} PROPERTIES OUTPUT_NAME ${TARGET_ADD3} )

# To build, comment out the line with the target
#set_target_properties(${TARGET_BASE} PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(${TARGET_ADD1} PROPERTIES EXCLUDE_FROM_ALL TRUE)
#set_target_properties(${TARGET_ADD2} PROPERTIES EXCLUDE_FROM_ALL TRUE)
#set_target_properties(${TARGET_ADD3} PROPERTIES EXCLUDE_FROM_ALL TRUE)