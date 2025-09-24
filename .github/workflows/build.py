#!/usr/bin/env python3
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

from utils import run_command, setup_trt_rtx, TRTRTX_INSTALL_DIR, BUILD_DIR

def build_samples():
    """Build C++ samples."""
    print("Building C++ samples...")
    run_command(f"cmake -B {BUILD_DIR} -S samples -DTRTRTX_INSTALL_DIR={TRTRTX_INSTALL_DIR}")
    run_command(f"cmake --build {BUILD_DIR}")

def main():
    # Setup TensorRT RTX
    setup_trt_rtx()

    # Build samples
    build_samples()

    print("Build completed successfully!")

if __name__ == "__main__":
    main()
