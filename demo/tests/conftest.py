# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Pytest configuration and common fixtures.
"""

import importlib
import json
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add the parent directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.path_manager import PathManager


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "test_cache"
        yield cache_dir
        # Cleanup is automatic with tempfile.TemporaryDirectory


@pytest.fixture
def path_manager(temp_cache_dir: Path) -> PathManager:
    """Create a PathManager instance with temporary cache directory."""
    return PathManager(cache_dir=str(temp_cache_dir))


@pytest.fixture
def temp_source_dir() -> Generator[Path, None, None]:
    """Create a temporary source directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dir = Path(temp_dir) / "source_models"
        source_dir.mkdir()
        yield source_dir
        # Cleanup is automatic with tempfile.TemporaryDirectory


def create_dummy_onnx_file(file_path: Path, content: str = "# Dummy ONNX file for testing\n") -> None:
    """Create a dummy ONNX file for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)

    # Also create a related file (like .onnx.data)
    data_file = file_path.with_suffix(".onnx.data")
    with open(data_file, "w") as f:
        f.write("# Dummy ONNX data file\n")


# Make the helper function available to all tests
pytest.create_dummy_onnx_file = create_dummy_onnx_file


@pytest.fixture
def weight_cuda_calls(monkeypatch):
    """Measure demo VMM allocations and H2D copies at the CUDA API boundary."""
    from utils.managed_weights import cuda, cudart

    allocations = {}
    measured = SimpleNamespace(peak_bytes=0, copied_bytes=0)
    create, release, copy = cuda.cuMemCreate, cuda.cuMemRelease, cudart.cudaMemcpyAsync

    def track_create(size, *args):
        result = create(size, *args)
        if result[0] == cuda.CUresult.CUDA_SUCCESS:
            allocations[int(result[1])] = size
            measured.peak_bytes = max(measured.peak_bytes, sum(allocations.values()))
        return result

    def track_release(handle):
        result = release(handle)
        if result[0] == cuda.CUresult.CUDA_SUCCESS:
            del allocations[int(handle)]
        return result

    def track_copy(destination, source, size, kind, stream):
        result = copy(destination, source, size, kind, stream)
        if result[0] == cudart.cudaError_t.cudaSuccess and kind == cudart.cudaMemcpyKind.cudaMemcpyHostToDevice:
            measured.copied_bytes += size
        return result

    monkeypatch.setattr(cuda, "cuMemCreate", track_create)
    monkeypatch.setattr(cuda, "cuMemRelease", track_release)
    monkeypatch.setattr(cudart, "cudaMemcpyAsync", track_copy)
    yield measured
    assert not allocations


@pytest.fixture(scope="module")
def cuda_device():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for numerical validation")
    return torch.cuda.current_device()


@pytest.fixture
def native_builder(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "flux1.dev"))
    return importlib.import_module("flux_native_builder")


@pytest.fixture(
    params=[(role, dtype) for role in ("clip", "t5", "vae", "transformer") for dtype in ("float32", "bfloat16")],
    ids=lambda item: "-".join(item),
    scope="module",
)
def component_reference(request, tmp_path_factory, cuda_device):
    import torch
    from diffusers import AutoencoderKL, FluxTransformer2DModel
    from safetensors.torch import save_file as save_torch_file
    from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel

    torch.manual_seed(123)
    role, dtype = request.param
    directory = tmp_path_factory.mktemp("architecture_" + role + "_" + dtype)
    if role == "clip":
        config = CLIPTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=8,
            hidden_act="quick_gelu",
            bos_token_id=1,
            eos_token_id=2,
        )
        model = CLIPTextModel(config).eval()
        inputs = {"input_ids": torch.randint(0, 32, (2, 8), dtype=torch.int32)}
        with torch.no_grad():
            result = model(**inputs)
        expected = {"text_embeddings": result.last_hidden_state, "pooled_embeddings": result.pooler_output}
    elif role == "t5":
        config = T5Config(
            vocab_size=32,
            d_model=16,
            d_kv=4,
            d_ff=32,
            num_layers=2,
            num_heads=4,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=16,
            feed_forward_proj="gated-gelu",
            dropout_rate=0.0,
        )
        model = T5EncoderModel(config).eval()
        inputs = {"input_ids": torch.randint(0, 32, (2, 8), dtype=torch.int32)}
        with torch.no_grad():
            expected = {"text_embeddings": model(**inputs).last_hidden_state}
    elif role == "vae":
        model = AutoencoderKL(
            block_out_channels=(8, 16),
            down_block_types=("DownEncoderBlock2D",) * 2,
            up_block_types=("UpDecoderBlock2D",) * 2,
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=4,
            use_post_quant_conv=False,
        ).eval()
        config = model.config
        inputs = {"latent": torch.randn(2, 4, 8, 16)}
        with torch.no_grad():
            expected = {"images": model.decode(inputs["latent"]).sample}
    else:
        model = FluxTransformer2DModel(
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=12,
            pooled_projection_dim=6,
            guidance_embeds=True,
            axes_dims_rope=(2, 2, 4),
        ).eval()
        config = model.config
        inputs = {
            "hidden_states": torch.randn(2, 2, 4),
            "encoder_hidden_states": torch.randn(2, 8, 12),
            "pooled_projections": torch.randn(2, 6),
            "timestep": torch.tensor([0.4, 0.7]),
            "img_ids": torch.randn(2, 3),
            "txt_ids": torch.randn(8, 3),
            "guidance": torch.tensor([3.5, 2.5]),
        }
        with torch.no_grad():
            expected = {"latent": model(**inputs).sample}
    if dtype == "bfloat16":
        # Compare reduced-precision kernels on the same device backend.
        model.to(device="cuda", dtype=torch.bfloat16)
        inputs = {
            k: v.to(torch.bfloat16) if v.is_floating_point() and k not in {"img_ids", "txt_ids", "guidance"} else v
            for k, v in inputs.items()
        }
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            if role == "vae":
                expected = {"images": model.decode(inputs["latent"]).sample}
            else:
                result = model(**inputs)
                expected = (
                    {"latent": result.sample}
                    if role == "transformer"
                    else {"text_embeddings": result.last_hidden_state}
                )
                if role == "clip":
                    expected["pooled_embeddings"] = result.pooler_output
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    (directory / "config.json").write_text(json.dumps(config_dict))
    state_dict = model.state_dict()
    if role == "clip" and "embeddings.token_embedding.weight" in state_dict:
        # Transformers 5 removes this prefix; retain the pinned FLUX checkpoint layout.
        state_dict = {f"text_model.{name}": value for name, value in state_dict.items()}
    # Copy tied tensors so each saved key owns a distinct, standard safetensor range.
    save_torch_file(
        {k: v.detach().cpu().clone().contiguous() for k, v in state_dict.items()},
        directory / "model.safetensors",
    )
    return role, directory, inputs, {k: v.float().cpu().numpy() for k, v in expected.items()}, dtype
