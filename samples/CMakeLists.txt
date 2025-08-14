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
project(TensorRT_RTX_samples LANGUAGES CXX CUDA)

# Options for TensorRT-RTX paths
set(TRTRTX_INSTALL_DIR "" CACHE PATH "Path to TensorRT-RTX install directory")
# Resolve absolute path for TRTRTX_INSTALL_DIR
get_filename_component(TRTRTX_INSTALL_DIR "${TRTRTX_INSTALL_DIR}" REALPATH)
set(TRTRTX_INCLUDE_DIR "${TRTRTX_INSTALL_DIR}/include" CACHE PATH "Path to TensorRT-RTX include directory")
set(TRTRTX_LIB_DIR "${TRTRTX_INSTALL_DIR}/lib" CACHE PATH "Path to TensorRT-RTX library directory")

if(${TRTRTX_INSTALL_DIR} STREQUAL "")
  message(FATAL_ERROR "TRTRTX_INSTALL_DIR must be specified")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Set common include directories
include_directories(${CUDA_INCLUDE_DIRS} ${TRTRTX_INCLUDE_DIR})

# Set common link directories
link_directories(${CUDA_LIBRARY_DIRS} ${TRTRTX_LIB_DIR})

# Add subdirectories for each sample...
add_subdirectory(helloWorld)
add_subdirectory(apiUsage)

# and add to custom target to run them.
add_custom_target(runSamples COMMAND $<TARGET_FILE:helloWorld> COMMAND $<TARGET_FILE:apiUsage>
                  COMMENT "Build and immediately run all samples.")
