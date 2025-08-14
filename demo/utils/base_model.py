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


from abc import abstractmethod
from typing import Any, Optional

from utils.base_params import BaseModelParams
from utils.model_registry import registry


class BaseModel:
    """Base model for ONNX -> TRT-RTX pipeline"""

    def __init__(
        self,
        name: str,
        device: str = "cuda",
        model_params: Optional[BaseModelParams] = None,
        hf_token: Optional[str] = None,
    ):
        self.name = name
        assert registry.is_model_id(name), f"Model {name} not found in model registry"

        self.device = device
        self.model_params = model_params
        self.hf_token = hf_token

    def validate_io_names(self, precision: str, input_names: list, output_names: Optional[list] = None) -> None:
        """Validate input and output names"""
        expected_inputs, expected_outputs = registry.get_io_names(self.name, precision)

        assert expected_inputs, f"Model '{self.name}' not found for precision '{precision}'"

        # Validate input names
        assert set(input_names) == set(expected_inputs), (
            f"Input name mismatch for {self.name}[{precision}]: expected {expected_inputs}, got {input_names}"
        )

        # Validate output names if provided
        if output_names is not None:
            assert set(output_names) == set(expected_outputs), (
                f"Output name mismatch for {self.name}[{precision}]: expected {expected_outputs}, got {output_names}"
            )

    @abstractmethod
    def get_input_profile(self, *args, **kwargs) -> dict[str, Any]:
        """Return TensorRT input profile for dynamic shapes"""
        pass

    @abstractmethod
    def get_shape_dict(self, *args, **kwargs) -> dict[str, Any]:
        """Return shape dictionary for tensor allocation"""
        pass
