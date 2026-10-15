# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native quantized weight storage, numerical refit, and offload regression tests."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

SOURCE = Path(__file__).parents[1] / "flux1.dev"


@pytest.fixture
def native(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    return importlib.import_module("flux_network")


def fp4_round(values):
    levels = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    boundaries = (levels[1:] + levels[:-1]) / 2
    codes = np.searchsorted(boundaries, np.abs(values), side="left")
    for index, boundary in enumerate(boundaries):
        if index % 2:
            codes = np.where(np.abs(values) == boundary, index + 1, codes)
    return levels[codes] * np.where(np.signbit(values), -1, 1)


def tiled_scales(linear):
    rows, columns = linear.shape
    result = np.empty(linear.size, dtype=linear.dtype)
    for row in range(rows):
        for column in range(columns):
            offset = (
                ((row // 128 * (columns // 4) + column // 4) * 32 + row % 32) * 4 + row % 128 // 32
            ) * 4 + column % 4
            result[offset] = linear[row, column]
    return result.reshape(linear.shape)


@pytest.mark.unit
@pytest.mark.parametrize("packed", [False, True])
def test_fused_checkpoint_ranges_remain_header_only(native, tmp_path, packed):
    from safetensors.numpy import save_file

    dim = 128 if packed else 16
    dtype, columns = (np.uint8, 64) if packed else (np.float32, 32)
    weights = np.arange(3 * dim * columns).reshape(3 * dim, columns).astype(dtype)
    prefix = "double_blocks.0.img_attn.qkv"
    tensors = {
        "img_in.weight": np.zeros((dim, 32), np.float32),
        prefix + ".weight": weights,
        prefix + ".bias": np.arange(3 * dim, dtype=np.float32),
        prefix + ".input_scale": np.array(0.25, np.float32),
        prefix + ".weight_scale": tiled_scales(np.arange(3 * dim * 8, dtype=np.uint8).reshape(3 * dim, 8))
        if packed
        else np.array(0.5, np.float32),
    }
    if packed:
        tensors[prefix + ".weight_scale_2"] = np.array(0.125, np.float32)
    weight_file = tmp_path / "bfl.safetensors"
    metadata = (
        {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": {prefix: {"format": "nvfp4"}}})}
        if packed
        else None
    )
    save_file(tensors, weight_file, metadata=metadata)
    config = {"num_attention_heads": 2, "attention_head_dim": dim // 2, "num_layers": 1, "num_single_layers": 0}
    (tmp_path / "config.json").write_text(json.dumps(config))
    checkpoint = native.CheckpointWeights(tmp_path, weight_file)
    for index, target in enumerate(("to_q", "to_k", "to_v")):
        name = f"transformer_blocks.0.attn.{target}.weight"
        assert checkpoint.shape(name) == (dim, columns * 2 if packed else columns)
        assert checkpoint.dtype(name) == (native.trt.fp4 if packed else native.trt.float32)
        checkpoint.descriptor(name, True)
        assert checkpoint.payload_bytes == 0
        entry = checkpoint.entries[name]
        original = checkpoint.entries[prefix + ".weight"]
        assert entry["path"] == weight_file
        assert entry["data_offsets"] == [
            original["data_offsets"][0] + i * dim * columns * np.dtype(dtype).itemsize for i in (index, index + 1)
        ]
        scale = checkpoint.entries[f"transformer_blocks.0.attn.{target}.weight_scale"]
        assert scale["shape"] == ([dim, 8] if packed else [])
        if packed:
            assert checkpoint.quantization[f"transformer_blocks.0.attn.{target}"] == {"format": "nvfp4"}
    checkpoint.descriptor("transformer_blocks.0.attn.to_k.weight", False)
    expected = weights[dim : 2 * dim]
    if packed:
        expected = (expected << 4) | (expected >> 4)
    np.testing.assert_array_equal(checkpoint.storage[-1].view(dtype).reshape(dim, columns), expected)
    assert checkpoint.payload_bytes == dim * columns * np.dtype(dtype).itemsize
    if packed:
        checkpoint.descriptor("transformer_blocks.0.attn.to_k.weight_scale", False)
        expected_scales = np.arange(3 * dim * 8, dtype=np.uint8).reshape(3 * dim, 8)[dim : 2 * dim]
        np.testing.assert_array_equal(checkpoint.storage[-1].reshape(dim, 8), expected_scales)


@pytest.mark.unit
@pytest.mark.parametrize("invalid", [None, "version", "format", "settings"])
def test_header_quantization_metadata(native, tmp_path, invalid):
    from safetensors.numpy import save_file

    settings = {"format": "float8_e4m3fn"}
    if invalid == "format":
        settings["format"] = "nvfp4"
    elif invalid == "settings":
        settings["unknown_layout"] = True
    metadata = {"format_version": "2.0" if invalid == "version" else "1.0", "layers": {"linear": settings}}
    save_file(
        {"linear.weight": np.ones((16, 16), dtype=ml_dtypes.float8_e4m3fn)},
        tmp_path / "model.safetensors",
        metadata={"_quantization_metadata": json.dumps(metadata)},
    )
    if invalid is not None:
        with pytest.raises(ValueError, match="quantization"):
            native.CheckpointWeights(tmp_path)
    else:
        checkpoint = native.CheckpointWeights(tmp_path)
        assert checkpoint.quantization == {"linear": settings}
        assert checkpoint.payload_bytes == checkpoint.metadata_bytes == 0


@pytest.mark.unit
@pytest.mark.parametrize("invalid", ["missing", "shape", "scale_dtype", "global_scale"])
def test_invalid_nvfp4_checkpoint_rejected(native, tmp_path, invalid):
    from safetensors.numpy import save_file

    tensors = {
        "linear.weight": np.zeros((16, 16), np.uint8),
        "linear.weight_scale": np.ones((16, 2), np.uint8),
        "linear.weight_scale_2": np.array(1, np.float32),
    }
    if invalid == "missing":
        del tensors["linear.weight_scale_2"]
    elif invalid == "shape":
        tensors["linear.weight_scale"] = np.ones((2, 16), np.uint8)
    elif invalid == "scale_dtype":
        tensors["linear.weight_scale"] = np.ones((16, 2), np.float32)
    else:
        tensors["linear.weight_scale_2"] = np.ones(2, np.float32)
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="NVFP4"):
        native.CheckpointWeights(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "settings", [{"format": "nvfp4"}, {"format": "int8"}, {"format": "nvfp4", "unknown_layout": True}]
)
def test_quantization_metadata_reads_only_small_control_data(native, tmp_path, settings):
    from safetensors.numpy import save_file

    metadata = json.dumps(settings).encode()
    tensors = {
        "linear.weight": np.zeros((128, 32), np.uint8),
        "linear.weight_scale": np.ones((128, 4), np.uint8),
        "linear.weight_scale_2": np.array(1, np.float32),
        "linear.comfy_quant": np.frombuffer(metadata, dtype=np.uint8),
    }
    save_file(tensors, tmp_path / "model.safetensors")
    if settings != {"format": "nvfp4"}:
        with pytest.raises(ValueError, match="Unsupported quantization"):
            native.CheckpointWeights(tmp_path)
    else:
        checkpoint = native.CheckpointWeights(tmp_path)
        assert checkpoint.quantization == {"linear": settings}
        assert checkpoint.payload_bytes == 0
        assert checkpoint.metadata_bytes == len(metadata)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_quantized_build_has_no_graph_framework_dependencies(tmp_path, precision):
    from safetensors.numpy import save_file

    tensors = {
        "linear.bias": np.zeros(64, dtype=ml_dtypes.bfloat16),
        "linear.input_scale": np.array(0.25, np.float32),
    }
    if precision == "fp8":
        tensors["linear.weight"] = np.ones((64, 64), dtype=ml_dtypes.float8_e4m3fn)
        tensors["linear.weight_scale"] = np.array(0.125, np.float32)
    else:
        tensors["linear.weight"] = np.full((64, 32), 0x22, np.uint8)
        tensors["linear.weight_scale"] = np.ones((64, 4), dtype=ml_dtypes.float8_e4m3fn)
        tensors["linear.weight_scale_2"] = np.array(0.125, np.float32)
    save_file(tensors, tmp_path / "model.safetensors")
    script = """
import importlib.abc, sys
class RejectGraphImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'onnx', 'diffusers', 'transformers'} or fullname.startswith('google.protobuf'):
            raise AssertionError('Unexpected build dependency: ' + fullname)
sys.meta_path.insert(0, RejectGraphImports())
sys.path.insert(0, sys.argv[1])
from flux_network import LOGGER, CheckpointWeights, _Network, trt
from cuda.bindings import runtime as cuda
error, count = cuda.cudaGetDeviceCount()
if int(error) or not count:
    sys.exit(77)
error, props = cuda.cudaGetDeviceProperties(0)
assert int(error) == 0
required = (12, 0) if sys.argv[3] == 'fp4' else (8, 9)
if (props.major, props.minor) < required:
    sys.exit(77)
builder = trt.Builder(LOGGER)
weights = CheckpointWeights(sys.argv[2])
graph = _Network(builder, weights, True)
graph.dtype = trt.bfloat16
graph.mark(graph.linear(graph.input('input', (2, 64)), 'linear'), 'output')
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)
config.set_flag(trt.BuilderFlag.STRIP_PLAN)
assert builder.build_serialized_network(graph.net, config) is not None
assert weights.payload_bytes == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(SOURCE), str(tmp_path), precision],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 77:
        pytest.skip(f"No GPU supporting {precision}")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["fp8", "fp4"])
@pytest.mark.parametrize("inputs_kind", ["reference", "random"])
@pytest.mark.parametrize("mode", ["buffer", "vmm"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_quantized_restore_matches_full_engine(
    native, tmp_path, precision, inputs_kind, mode, dynamic, weight_cuda_calls
):
    import torch
    from safetensors.torch import save_file
    from utils.managed_weights import ManagedWeightsScheduler

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if precision == "fp4" and torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("NVFP4 requires Blackwell")
    if precision == "fp8" and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 requires Ada or newer")
    trt = native.trt
    rng = np.random.default_rng(718)
    width = 128
    metadata = None
    samples = rng.normal(size=(2, width)).astype(np.float32)
    if inputs_kind == "reference" and precision == "fp4":
        # Power-of-two block scales avoid reciprocal rounding at FP4 midpoints.
        samples = rng.integers(-5, 6, size=(2, width)).astype(np.float32) / 8
        samples[:, ::16] = 0.75
    x = torch.from_numpy(samples).to(torch.bfloat16).cuda()
    inputs = x.float().cpu().numpy()
    input_scale = 0.25
    bias = torch.from_numpy(rng.normal(0, 0.01, size=width).astype(np.float32)).to(torch.bfloat16)
    tensors = {"linear.bias": bias, "linear.input_scale": torch.tensor(input_scale)}
    if inputs_kind == "reference":
        tensors["linear.pre_quant_scale"] = torch.full((1, 1, width), 0.5)
        inputs = inputs * 0.5
    if precision == "fp8":
        weights = rng.normal(size=(width, width)).astype(ml_dtypes.float8_e4m3fn)
        tensors["linear.weight"] = torch.from_numpy(weights.view(np.uint8)).view(torch.float8_e4m3fn)
        tensors["linear.weight_scale"] = torch.tensor(0.125)
        expected_weights = weights.astype(np.float32) * 0.125
        expected_inputs = (inputs / input_scale).astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * input_scale
    else:
        codes = rng.integers(0, 16, size=(width, width), dtype=np.uint8)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        block_scales = np.exp2(rng.integers(-2, 2, size=(width, width // 16))).astype(ml_dtypes.float8_e4m3fn)
        stored_scales = block_scales
        if inputs_kind == "reference":
            packed = (packed << 4) | (packed >> 4)
            stored_scales = tiled_scales(block_scales)
            metadata = {
                "_quantization_metadata": json.dumps(
                    {"format_version": "1.0", "layers": {"linear": {"format": "nvfp4"}}}
                )
            }
        tensors["linear.weight"] = torch.from_numpy(packed)
        tensors["linear.weight_scale"] = torch.from_numpy(stored_scales.view(np.uint8)).view(torch.float8_e4m3fn)
        tensors["linear.weight_scale_2"] = torch.tensor(0.25)
        levels = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
        expected_weights = (
            levels[codes & 7]
            * np.where(codes & 8, -1, 1)
            * np.repeat(block_scales.astype(np.float32), 16, axis=1)
            * 0.25
        )
        blocks = inputs.reshape(2, width // 16, 16) / input_scale
        scales = (np.max(np.abs(blocks), axis=-1, keepdims=True) / 6).astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
        expected_inputs = (fp4_round(blocks / scales) * scales * input_scale).reshape(2, width)
    expected = expected_inputs @ expected_weights.T + bias.float().numpy()
    save_file(tensors, tmp_path / "model.safetensors", metadata=metadata)
    stream = torch.cuda.Stream()
    output = torch.empty((2, width), dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    runtime = trt.Runtime(native.LOGGER)
    full_builder = trt.Builder(native.LOGGER)
    full = native._Network(full_builder, native.CheckpointWeights(tmp_path), False)
    full.dtype = trt.bfloat16
    full.mark(full.linear(full.input("input", (2, width)), "linear"), "output")
    full_config = full_builder.create_builder_config()
    full_plan = full_builder.build_serialized_network(full.net, full_config)
    assert full_plan is not None
    full_engine = runtime.deserialize_cuda_engine(full_plan)
    assert full_engine is not None
    full_context = full_engine.create_execution_context()
    assert full_context is not None
    assert full_context.set_tensor_address("input", x.data_ptr())
    assert full_context.set_tensor_address("output", output.data_ptr())
    assert full_context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    baseline = output.float().cpu().numpy().copy()
    if inputs_kind == "reference" or precision == "fp8":
        np.testing.assert_allclose(baseline, expected, atol=0.02, rtol=0.02)
    del full_context, full_engine, full_plan, full, full_builder
    checkpoint = native.CheckpointWeights(tmp_path)
    builder = trt.Builder(native.LOGGER)
    profiles = {"input": [(1, 1, width), (1, 2, width), (2, 4, width)]} if dynamic else None
    graph = native._Network(builder, checkpoint, True, profiles)
    graph.dtype = trt.bfloat16
    graph.mark(graph.linear(graph.input("input", (1, 2, width) if dynamic else (2, width)), "linear"), "output")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)
    config.set_flag(trt.BuilderFlag.STRIP_PLAN)
    if dynamic:
        profile = builder.create_optimization_profile()
        profile.set_shape("input", *profiles["input"])
        config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(graph.net, config)
    assert plan is not None
    assert checkpoint.payload_bytes == 0
    plan_path = tmp_path / "model.plan"
    plan_path.write_bytes(bytes(plan))
    engines = {name: runtime.deserialize_cuda_engine(plan) for name in ("first", "second")}
    assert all(e is not None for e in engines.values())
    scheduler = ManagedWeightsScheduler(mode, stream.cuda_stream, torch.cuda.current_device(), staging_mib=1)
    contexts = {}
    try:
        for name, engine in engines.items():
            scheduler.register_engine(
                name,
                engine,
                tmp_path,
                plan_path,
                tmp_path / (name + ".weights"),
                refit=checkpoint.refit,
                source_identity=precision,
            )
        scheduler.configure_memory_pool()
        for name, engine in engines.items():
            scheduler.initialize_weights(name)
            contexts[name] = engine.create_execution_context()
            assert contexts[name] is not None
            assert contexts[name].set_tensor_address("input", x.data_ptr())
            assert contexts[name].set_tensor_address("output", output.data_ptr())
            scheduler.unload_weights(name)
        for cycle in range(3):
            batch, length = [(1, 2), (2, 4), (1, 1)][cycle] if dynamic else (1, 2)
            current_input = x.repeat(batch, (length + 1) // 2, 1)[:, :length, :].contiguous() if dynamic else x
            current_output = torch.empty_like(current_input)
            expected_output = np.tile(baseline, (batch, (length + 1) // 2, 1))[:, :length, :] if dynamic else baseline
            torch.cuda.synchronize()
            for name in ("first", "second"):
                with scheduler.use_weights(name):
                    if dynamic:
                        assert contexts[name].set_input_shape("input", tuple(current_input.shape))
                    assert contexts[name].set_tensor_address("input", current_input.data_ptr())
                    assert contexts[name].set_tensor_address("output", current_output.data_ptr())
                    assert contexts[name].execute_async_v3(stream.cuda_stream)
                    stream.synchronize()
                    np.testing.assert_array_equal(current_output.float().cpu().numpy(), expected_output)
                assert scheduler.entries[name].state == "unloaded"
        assert weight_cuda_calls.peak_bytes <= scheduler.weight_budget
    finally:
        stream.synchronize()
        scheduler.close()
