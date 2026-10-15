# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture-native builds against small independent HF reference models."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

SOURCE = Path(__file__).parents[1] / "flux1.dev"
SPEC = importlib.util.spec_from_file_location("flux_network", SOURCE / "flux_network.py")
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["full", "weightless"])
def test_architectures_match_reference_without_graph_imports(component_reference, mode):
    import torch

    role, directory, inputs, expected, dtype = component_reference
    plan = directory / f"{mode}.plan"
    guard = """
import importlib.abc
import sys
from pathlib import Path
class RejectGraphImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'onnx', 'transformers', 'diffusers'} or fullname.startswith('google.protobuf'):
            raise AssertionError('Forbidden build dependency: ' + fullname)
sys.meta_path.insert(0, RejectGraphImports())
source, directory, role, mode, plan = sys.argv[1:]
sys.path.insert(0, source)
directory, plan = Path(directory), Path(plan)
if mode == 'weightless':
    from flux_native_builder import build_weightless_engine
    checkpoint, identity = build_weightless_engine(directory, role, plan, batch=2, height=16, width=32, text_length=8)
    assert checkpoint.payload_bytes == 0
    timestamp = plan.stat().st_mtime_ns
    checkpoint, cached_identity = build_weightless_engine(directory, role, plan, batch=2, height=16, width=32, text_length=8)
    assert cached_identity == identity and checkpoint.payload_bytes == 0
    assert plan.stat().st_mtime_ns == timestamp
else:
    from flux_network import LOGGER, create_network, trt
    builder = trt.Builder(LOGGER)
    network = create_network(builder, directory, role, False, batch=2, height=16, width=32, text_length=8)
    assert network.weights.payload_bytes > 0
    config = builder.create_builder_config()
    serialized = builder.build_serialized_network(network.net, config)
    assert serialized is not None
    plan.write_bytes(bytes(serialized))
assert not any(k.split('.')[0] in {'torch', 'onnx', 'transformers', 'diffusers'} for k in sys.modules)
"""
    build = subprocess.run(
        [
            sys.executable,
            "-c",
            guard,
            str(SOURCE),
            str(directory),
            role,
            mode,
            str(plan),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    runtime = native.trt.Runtime(native.LOGGER)
    engine = runtime.deserialize_cuda_engine(plan.read_bytes())
    assert engine is not None
    if mode == "weightless":
        checkpoint = native.CheckpointWeights(directory)
        checkpoint.refit(engine)
        assert checkpoint.payload_bytes > 0
    context = engine.create_execution_context()
    tensors = {name: value.cuda().contiguous() for name, value in inputs.items()}
    for name in expected:
        tensors[name] = torch.empty(tuple(context.get_tensor_shape(name)), device="cuda", dtype=getattr(torch, dtype))
    for name, value in tensors.items():
        assert context.set_tensor_address(name, value.data_ptr())
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    assert context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    for name in expected:
        actual = tensors[name].float().cpu().numpy()
        # Default TensorRT precision permits TF32 rounding in FP32 operations.
        # BF16 outputs near zero can exceed 0.02 absolute error, so loosen atol only.
        rtol = 0.02 if dtype == "bfloat16" else 5e-3
        atol = 0.03 if dtype == "bfloat16" else 5e-3
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected[name], rtol=rtol, atol=atol, err_msg=f"{role}/{mode}/{name}")


@pytest.mark.unit
@pytest.mark.parametrize(
    "role,profiles",
    [
        ("clip", {"input_ids": [(1, 4), (2, 8), (3, 16)]}),
        ("t5", {"input_ids": [(1, 4), (2, 8), (3, 16)]}),
        ("transformer", {"hidden_states": [(1, 1, 4), (2, 2, 8), (3, 4, 16)]}),
        ("vae", {"latent": [(1, 4, 8, 8), (2, 8, 16, 16), (3, 16, 32, 32)]}),
    ],
)
def test_native_profiles_keep_text_and_channels_fixed(tmp_path, role, profiles):
    with pytest.raises(ValueError, match="Only batch and image dimensions may vary"):
        native.create_network(None, tmp_path, role, True, input_profiles=profiles)


@pytest.mark.unit
def test_checkpoint_reads_only_requested_ranges(tmp_path):
    save_file(
        {"a": np.arange(12, dtype=np.float32), "b": np.ones(10, dtype=np.float16)}, tmp_path / "weights.safetensors"
    )
    checkpoint = native.CheckpointWeights(tmp_path)
    assert checkpoint.payload_bytes == 0
    assert checkpoint.descriptor("a", True).size == 12
    assert checkpoint.payload_bytes == 0
    checkpoint.descriptor("b", False)
    assert checkpoint.payload_bytes == 20
    np.testing.assert_array_equal(checkpoint.storage[-1].view(np.float16), np.ones(10, dtype=np.float16))


@pytest.mark.unit
def test_sharded_checkpoint_index(tmp_path):
    save_file({"a": np.ones(4, dtype=np.float32)}, tmp_path / "first.safetensors")
    save_file({"b": np.zeros(3, dtype=np.float32)}, tmp_path / "second.safetensors")
    index = {"weight_map": {"a": "first.safetensors", "b": "second.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert set(native.CheckpointWeights(tmp_path).entries) == {"a", "b"}
    index["weight_map"]["missing"] = "first.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="incomplete"):
        native.CheckpointWeights(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize("arguments", [{"batch": 0}, {"height": 0}, {"width": 0}, {"text_length": 0}])
def test_invalid_dimensions_rejected_before_checkpoint_access(tmp_path, arguments):
    with pytest.raises(ValueError, match="positive"):
        native.create_network(None, tmp_path, "transformer", True, **arguments)


@pytest.mark.unit
def test_quantized_config_rejected_before_checkpoint_access(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    with pytest.raises(ValueError, match="Quantized checkpoints"):
        native.create_network(None, tmp_path, "transformer", True)


@pytest.mark.unit
def test_hf_snapshot_symlink_is_header_only(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    blob = tmp_path / "blob"
    save_file({"weight": np.ones(16, np.float32)}, blob)
    (snapshot / "model.safetensors").symlink_to(blob)
    checkpoint = native.CheckpointWeights(snapshot)
    assert checkpoint.descriptor("weight", True).size == 16
    assert checkpoint.payload_bytes == 0


@pytest.mark.unit
@pytest.mark.parametrize("filename", ["../blob", "/tmp/blob"])
def test_checkpoint_shard_index_rejects_path_traversal(tmp_path, filename):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": filename}}))
    with pytest.raises(ValueError, match="inside the component"):
        native.CheckpointWeights(tmp_path)
