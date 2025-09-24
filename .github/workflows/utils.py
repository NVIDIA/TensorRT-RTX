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

import io
import os
import requests
import subprocess
import sys
import tarfile
from pathlib import Path

# Shared constants
TRT_RTX_BASE_URL = "https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.1"
TRT_RTX_FILENAME = os.environ.get("TRT_RTX_FILENAME","TensorRT-RTX-1.1.1.26.Linux.x86_64-gnu.cuda-12.9.tar.gz")
TRTRTX_INSTALL_DIR = os.environ.get("TRTRTX_INSTALL_DIR", "/opt/tensorrt_rtx")
BUILD_DIR = os.environ.get('BUILD_DIR', 'build')

def run_command(cmd, check=True, shell=False, env=None):
    """Run a command and handle errors."""
    try:
        subprocess.run(cmd if shell else cmd.split(), check=check, shell=shell, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Exit code: {e.returncode}")
        sys.exit(e.returncode)

def setup_trt_rtx():
    """Download and setup TensorRT RTX."""
    if os.environ.get('CACHE_TRT_RTX_HIT') != 'true':
        print("Cache miss for TensorRT RTX, downloading...")
        url = f"{TRT_RTX_BASE_URL}/{TRT_RTX_FILENAME}"

        if os.path.exists(TRTRTX_INSTALL_DIR):
            print(f"Error: {TRTRTX_INSTALL_DIR} already exists. Remove it or set CACHE_TRT_RTX_HIT=true to proceed.")
            exit(1)

        # Download the TRT RTX tar file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Create a file-like object from the response content
        tar_bytes = io.BytesIO(response.content)

        # Extract tar file, stripping the first directory component
        os.makedirs(TRTRTX_INSTALL_DIR)
        with tarfile.open(fileobj=tar_bytes, mode='r:gz') as tar:
            members = [m for m in tar.getmembers() if len(Path(m.name).parts) > 1]
            for member in members:
                member.name = str(Path(*Path(member.name).parts[1:]))
                tar.extract(member, TRTRTX_INSTALL_DIR)
    else:
        print("Cache hit for TensorRT RTX")
