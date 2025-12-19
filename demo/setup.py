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

from setuptools import find_packages, setup

setup(
    name="rtx-demos",
    version="1.3.0",
    description="RTX Demos",
    packages=find_packages(),
    install_requires=[
        "torch>=2.7.0",
        "transformers>=4.52.4",
        "diffusers>=0.33.1",
        "huggingface-hub>=0.32.4",
        "tqdm>=4.67.1",
        "pillow",
        "numpy",
        "cuda-python<13.0.0",
        "polygraphy>=0.49.24",
        "packaging",
        "tensorrt-rtx>=1.3.0",
        "accelerate",
        "protobuf",
        "sentencepiece",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "ruff",
            "flake8",
        ],
        "jupyter": [
            "requests>=2.25.0",
            "ipython>=8.0.0",
            "ipywidgets>=8.0.0",
            "jupyter",
            "notebook",
            "jupyterlab",
        ],
    },
    python_requires=">=3.9",
)
