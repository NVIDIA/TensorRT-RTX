# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if(NOT TRTRTX_INSTALL_DIR)
  message(FATAL_ERROR "TRTRTX_INSTALL_DIR must be specified")
endif()

find_package(CUDAToolkit REQUIRED)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/modules/get_version.cmake)
get_version("${TRTRTX_INSTALL_DIR}/include" TRT_RTX_VERSION TRT_RTX_SOVERSION)
get_library_name(TRT_RTX_SOVERSION TRTRTX_LIB_NAME TRTRTX_UNUSED_ONNXPARSER_LIB_NAME)

add_executable(memoryManagement memoryManagement.cpp)
target_compile_features(memoryManagement PRIVATE cxx_std_17)
target_include_directories(memoryManagement SYSTEM PRIVATE "${TRTRTX_INSTALL_DIR}/include")
target_link_directories(memoryManagement PRIVATE "${TRTRTX_INSTALL_DIR}/lib")
target_link_libraries(memoryManagement PRIVATE ${TRTRTX_LIB_NAME} CUDA::cudart)
