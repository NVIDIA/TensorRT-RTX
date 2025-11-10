#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -----------------------------------------------------------------------------
# cmake-format: off
# Function: get_version
# Retrieves the TensorRT-RTX version information from the specified include directory.
#
# Args:
#   include_dir        - Path to the directory containing NvInferVersion.h
#   version_variable   - Output variable; will be set to the version string in the format "MAJOR.MINOR.PATCH"
#   soversion_variable - Output variable; will be set to the SOVERSION string in the format "MAJOR_MINOR"
#
# Example:
#   get_version(${TRTRTX_INCLUDE_DIR} TRT_RTX_VERSION TRT_RTX_SOVERSION)
#   # TRT_RTX_VERSION will be set to e.g. "9.0.1"
#   # TRT_RTX_SOVERSION will be set to e.g. "9_0"
# cmake-format: on
# -----------------------------------------------------------------------------
function(get_version include_dir version_variable soversion_variable)
  set(header_file "${include_dir}/NvInferVersion.h")
  if(NOT EXISTS "${header_file}")
    message(FATAL_ERROR "TensorRT-RTX version header not found: ${header_file}")
  endif()

  file(STRINGS "${header_file}" VERSION_STRINGS REGEX "#define TRT_.*_RTX")
  if(NOT VERSION_STRINGS)
    message(
      FATAL_ERROR "No TRT_*_RTX version defines found in ${header_file}, please check if the path provided is correct.")
  endif()

  foreach(type MAJOR MINOR PATCH)
    set(trt_${type} "")
    foreach(version_line ${VERSION_STRINGS})
      string(REGEX MATCH "TRT_${type}_RTX [0-9]+" trt_type_string "${version_line}")
      if(trt_type_string)
        string(REGEX MATCH "[0-9]+" trt_${type} "${trt_type_string}")
        break()
      endif()
    endforeach()
    if(NOT DEFINED trt_${type})
      message(FATAL_ERROR "Failed to extract TRT_${type}_RTX from ${header_file}")
    endif()
  endforeach(type)
  set(${version_variable} ${trt_MAJOR}.${trt_MINOR}.${trt_PATCH} PARENT_SCOPE)
  set(${soversion_variable} ${trt_MAJOR}_${trt_MINOR} PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# cmake-format: off
# Function: get_library_name
# Retrieves the library name for TensorRT-RTX and TensorRT-ONNXParser-RTX.
#
# Args:
#   soversion_variable           - Input variable; should contain the SOVERSION string in the format "MAJOR_MINOR"
#   lib_name_variable            - Output variable; will be set to the library name
#   onnxparser_lib_name_variable - Output variable; will be set to the ONNXParser library name
#
# Example:
#   get_library_name(TRT_RTX_SOVERSION TRT_RTX_LIB_NAME TRT_RTX_ONNXPARSER_LIB_NAME)
#   # TRT_RTX_LIB_NAME will be set to e.g. "tensorrt_rtx"
#   # TRT_RTX_ONNXPARSER_LIB_NAME will be set to e.g. "tensorrt_onnxparser_rtx"
# cmake-format: on
# -----------------------------------------------------------------------------
function(get_library_name soversion_variable lib_name_variable onnxparser_lib_name_variable)
  set(trtrtx_lib_name "tensorrt_rtx")
  set(trtrtx_onnxparser_lib_name "tensorrt_onnxparser_rtx")
  if(WIN32)
    set(${lib_name_variable} "${trtrtx_lib_name}_${${soversion_variable}}" PARENT_SCOPE)
    set(${onnxparser_lib_name_variable} "${trtrtx_onnxparser_lib_name}_${${soversion_variable}}" PARENT_SCOPE)
  else()
    set(${lib_name_variable} "${trtrtx_lib_name}" PARENT_SCOPE)
    set(${onnxparser_lib_name_variable} "${trtrtx_onnxparser_lib_name}" PARENT_SCOPE)
  endif()
endfunction()
