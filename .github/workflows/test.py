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

import os
from pathlib import Path
from utils import run_command, setup_trt_rtx, TRTRTX_INSTALL_DIR, BUILD_DIR

def install_python_deps():
    """Install Python dependencies."""
    print("Installing Python dependencies...")

    # Install TensorRT RTX wheel
    wheel_dir = Path(TRTRTX_INSTALL_DIR) / "python"
    wheel_file = next(wheel_dir.glob("tensorrt_rtx-*-cp39-none-linux_x86_64.whl"))
    run_command(f"uv pip install {wheel_file}")

    # Install sample requirements
    run_command("uv pip install -r samples/helloWorld/python/requirements.txt")
    run_command("uv pip install -r samples/apiUsage/python/requirements.txt")
    run_command("uv pip install --index-strategy unsafe-best-match -r demo/flux1.dev/requirements.txt")
    run_command("uv pip install -r demo/tests/requirements-test.txt")

def run_cpp_tests():
    """Run C++ sample tests."""
    print("Running C++ tests...")
    BINARIES = [f"{BUILD_DIR}/helloWorld/cpp/helloWorld", f"{BUILD_DIR}/apiUsage/cpp/apiUsage"]
    for binary in BINARIES:
        # Add the executable permission if not on Windows
        if os.name != 'nt':
            os.chmod(binary, os.stat(binary).st_mode | 0o111)
        run_command(binary)

def run_python_tests():
    """Run Python sample tests."""
    print("Running Python tests...")
    # Set up environment for tests
    test_env = os.environ.copy()
    test_env["LD_LIBRARY_PATH"] = f"{TRTRTX_INSTALL_DIR}/lib:{test_env.get('LD_LIBRARY_PATH', '')}"

    run_command("uv run samples/helloWorld/python/hello_world.py", env=test_env)
    run_command("uv run samples/apiUsage/python/api_usage.py", env=test_env)
    run_command("uv run pytest demo/tests -v", env=test_env)

def main():
    # Setup TensorRT RTX
    setup_trt_rtx()

    # Install Python dependencies
    install_python_deps()

    # Run tests
    run_cpp_tests()
    run_python_tests()

    print("All tests completed successfully!")

if __name__ == "__main__":
    main()
