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
project(TensorRT_RTX_HelloWorld LANGUAGES CXX CUDA)

# Options for TensorRT-RTX paths
set(TRTRTX_INSTALL_DIR "" CACHE PATH "Path to TensorRT-RTX install directory")
set(TRTRTX_INCLUDE_DIR "${TRTRTX_INSTALL_DIR}/include" CACHE PATH "Path to TensorRT-RTX include directory")
set(TRTRTX_LIB_DIR "${TRTRTX_INSTALL_DIR}/lib" CACHE PATH "Path to TensorRT-RTX library directory")

if(${TRTRTX_INSTALL_DIR} STREQUAL "")
  message(FATAL_ERROR "TRTRTX_INSTALL_DIR must be specified")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(helloWorld helloWorld.cpp)

# Add compiler flags to ignore deprecated API warnings from TensorRT headers.
if(NOT MSVC)
  target_compile_options(helloWorld PRIVATE -Wno-deprecated-declarations -Wall)
else()
  target_compile_options(helloWorld PRIVATE /wd4996 /wd4100 /W4)
endif()

if(WIN32)
  set(TRTRTX_LIB_NAME "tensorrt_rtx_1_0")
  set(TRTRTX_ONNXPARSER_LIB_NAME "tensorrt_onnxparser_rtx_1_0")
else()
  set(TRTRTX_LIB_NAME "tensorrt_rtx")
  set(TRTRTX_ONNXPARSER_LIB_NAME "tensorrt_onnxparser_rtx")
endif()

target_include_directories(helloWorld PRIVATE ${TRTRTX_INCLUDE_DIR})
target_link_directories(helloWorld PRIVATE ${TRTRTX_LIB_DIR})
target_link_libraries(helloWorld PRIVATE ${TRTRTX_LIB_NAME} ${TRTRTX_ONNXPARSER_LIB_NAME} CUDA::cudart)
