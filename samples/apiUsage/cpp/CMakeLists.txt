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

cmake_minimum_required(VERSION 3.10)
project(TensorRT_RTX_ApiUsage LANGUAGES CXX CUDA)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/modules/get_version.cmake)

# Options for TensorRT-RTX paths
set(TRTRTX_INSTALL_DIR "" CACHE PATH "Path to TensorRT-RTX install directory")
# Resolve absolute path for TRTRTX_INSTALL_DIR
get_filename_component(TRTRTX_INSTALL_DIR "${TRTRTX_INSTALL_DIR}" REALPATH)
set(TRTRTX_INCLUDE_DIR "${TRTRTX_INSTALL_DIR}/include" CACHE PATH "Path to TensorRT-RTX include directory")
set(TRTRTX_LIB_DIR "${TRTRTX_INSTALL_DIR}/lib" CACHE PATH "Path to TensorRT-RTX library directory")

if(${TRTRTX_INSTALL_DIR} STREQUAL "")
  message(FATAL_ERROR "TRTRTX_INSTALL_DIR must be specified")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(apiUsage apiUsage.cpp)

# Add compiler flags to ignore deprecated API warnings from TensorRT headers.
if(NOT MSVC)
  target_compile_options(apiUsage PRIVATE -Wno-deprecated-declarations -Wall)
else()
  target_compile_options(apiUsage PRIVATE /wd4996 /wd4100 /W4)
endif()

get_version(${TRTRTX_INCLUDE_DIR} TRT_RTX_VERSION TRT_RTX_SOVERSION)
get_library_name(TRT_RTX_SOVERSION TRTRTX_LIB_NAME TRTRTX_ONNXPARSER_LIB_NAME)

message(
  STATUS "Building ${PROJECT_NAME} for TensorRT-RTX version: ${TRT_RTX_VERSION}, library version: ${TRT_RTX_SOVERSION}")

target_include_directories(apiUsage PRIVATE ${TRTRTX_INCLUDE_DIR})
target_link_directories(apiUsage PRIVATE ${TRTRTX_LIB_DIR})
target_link_libraries(apiUsage PRIVATE ${TRTRTX_LIB_NAME} ${TRTRTX_ONNXPARSER_LIB_NAME} CUDA::cudart)
