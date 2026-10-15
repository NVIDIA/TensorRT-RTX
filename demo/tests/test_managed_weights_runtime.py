# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise native builds, refit, and managed-weight scheduling against the real runtime."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import tensorrt_rtx as trt
import torch
from safetensors.numpy import save_file
from utils import managed_weights as managed
from utils.managed_weights import ManagedWeightsScheduler

SPEC = importlib.util.spec_from_file_location(
    "runtime_flux_network", Path(__file__).parents[1] / "flux1.dev/flux_network.py"
)
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize(
    "mode,pinned_host",
    [("buffer", False), ("buffer", True), ("vmm", False), ("vmm", True)],
)
def test_runtime_refit_restore(tmp_path, monkeypatch, mode, pinned_host, weight_cuda_calls):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for managed-weight validation")
    weights = np.random.default_rng(123).normal(0, 0.01, (768, 768)).astype(np.float32)
    save_file({"weights": weights, "weights_out": weights}, tmp_path / "model.safetensors")
    checkpoint = native.CheckpointWeights(tmp_path)
    logger = trt.Logger(trt.Logger.WARNING)

    def forbidden(*args):
        raise AssertionError("Native weightless builds must not instantiate the ONNX parser")

    monkeypatch.setattr(trt, "OnnxParser", forbidden)
    builder = trt.Builder(logger)
    graph = native._Network(builder, checkpoint, True)
    x = graph.input("input", (2, 768))
    hidden = graph.out(
        graph.net.add_matrix_multiply(x, trt.MatrixOperation.NONE, graph.weight("weights"), trt.MatrixOperation.NONE)
    )
    hidden = graph.out(graph.net.add_activation(hidden, trt.ActivationType.TANH))
    result = graph.out(
        graph.net.add_matrix_multiply(
            hidden, trt.MatrixOperation.NONE, graph.weight("weights_out"), trt.MatrixOperation.NONE
        )
    )
    graph.mark(result, "output")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)
    config.set_flag(trt.BuilderFlag.STRIP_PLAN)
    plan = builder.build_serialized_network(graph.net, config)
    assert plan is not None
    assert checkpoint.payload_bytes == 0
    plan_path = tmp_path / "model.plan"
    plan_path.write_bytes(bytes(plan))
    runtime = trt.Runtime(logger)
    engines = {name: runtime.deserialize_cuda_engine(plan) for name in ("first", "second")}
    assert all(engine is not None for engine in engines.values())
    assert all(trt.Refitter(engine, logger).get_all_weights() for engine in engines.values())
    stream = torch.cuda.Stream()

    def forbidden_cuda_setup(*args):
        raise AssertionError("Weight management must use the caller's initialized CUDA context")

    monkeypatch.setattr("utils.managed_weights.cuda.cuInit", forbidden_cuda_setup)
    monkeypatch.setattr("utils.managed_weights.cudart.cudaSetDevice", forbidden_cuda_setup)
    scheduler = ManagedWeightsScheduler(
        mode,
        stream.cuda_stream,
        torch.cuda.current_device(),
        staging_mib=1,
        pinned_host=pinned_host,
    )
    contexts = {}
    inputs = torch.from_numpy(np.random.default_rng(456).normal(size=(2, 768)).astype(np.float32)).cuda()
    output = torch.empty_like(inputs)
    expected = np.tanh(inputs.cpu().numpy() @ weights) @ weights
    torch.cuda.synchronize()
    try:
        for name, engine in engines.items():
            scheduler.register_engine(
                name,
                engine,
                tmp_path,
                plan_path,
                tmp_path / (name + ".weights"),
                refit=checkpoint.refit,
                source_identity="test-checkpoint",
            )
        scheduler.configure_memory_pool()
        for name, engine in engines.items():
            scheduler.initialize_weights(name)
            context = engine.create_execution_context()
            assert context is not None
            contexts[name] = context
            assert context.set_tensor_address("input", inputs.data_ptr())
            assert context.set_tensor_address("output", output.data_ptr())
            scheduler.unload_weights(name)
        if pinned_host:
            cache = scheduler.loader.pinned_weight_cache
            assert sum(buffer.size for _, buffer in cache.pinned_weight_buffers.values()) == sum(
                entry.size for entry in scheduler.entries.values()
            )
            if mode == "vmm":
                assert not scheduler.loader.staging
            # Repeat setup from valid backups without reusing the first host cache.
            cache.close()
            for name in engines:
                scheduler.entries[name].refit = forbidden
                scheduler.initialize_weights(name)
                scheduler.unload_weights(name)
        for _ in range(3):
            for index, name in enumerate(("first", "first", "second")):
                copied_before = weight_cuda_calls.copied_bytes
                if mode == "vmm":
                    assert int(scheduler.loader.stream) == stream.cuda_stream
                with scheduler.use_weights(name):
                    if mode == "vmm":
                        assert len(scheduler.entries[name].blocks) == 1
                        if index == 1:
                            assert weight_cuda_calls.copied_bytes == copied_before
                        else:
                            assert weight_cuda_calls.copied_bytes > copied_before
                    assert contexts[name].execute_async_v3(stream.cuda_stream)
                    stream.synchronize()
                    np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=5e-4, atol=5e-4)
                assert scheduler.entries[name].state == "unloaded"
        assert all(entry.state == "unloaded" for entry in scheduler.entries.values())
        assert weight_cuda_calls.peak_bytes <= scheduler.weight_budget
    finally:
        stream.synchronize()
        scheduler.close()
        if pinned_host:
            assert not scheduler.loader.pinned_weight_cache.pinned_weight_buffers


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("pinned_host", [False, True])
def test_runtime_engine_sized_blocks(tmp_path, monkeypatch, pinned_host, weight_cuda_calls):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for managed-weight validation")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    stream = torch.cuda.Stream()
    scheduler = ManagedWeightsScheduler(
        "vmm",
        stream.cuda_stream,
        torch.cuda.current_device(),
        pinned_host=pinned_host,
    )
    contexts, tensors = {}, {}
    try:
        for index, mib in enumerate((6, 2, 16, 4)):
            name = f"stage_{index}"
            directory = tmp_path / name
            directory.mkdir()
            shape = (1, mib * 1024 * 1024 // 4)
            values = np.random.default_rng(index).normal(size=shape).astype(np.float32)
            save_file({"weights": values}, directory / "model.safetensors")
            checkpoint = native.CheckpointWeights(directory)
            builder = trt.Builder(logger)
            graph = native._Network(builder, checkpoint, True)
            x = graph.input("input", shape)
            y = graph.out(graph.net.add_elementwise(x, graph.weight("weights"), trt.ElementWiseOperation.SUM))
            graph.mark(y, "output")
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)
            config.set_flag(trt.BuilderFlag.STRIP_PLAN)
            plan = builder.build_serialized_network(graph.net, config)
            assert plan is not None
            plan_path = directory / "model.plan"
            plan_path.write_bytes(bytes(plan))
            engine = runtime.deserialize_cuda_engine(plan)
            assert engine is not None
            scheduler.register_engine(
                name, engine, directory, plan_path, directory / "weights.bin", refit=checkpoint.refit
            )
            inputs = torch.ones(shape, device="cuda")
            tensors[name] = (inputs, torch.empty_like(inputs), values + 1)
        scheduler.configure_memory_pool()
        sizes = [entry.size for entry in scheduler.entries.values()]
        assert len(set(sizes)) == 4
        assert scheduler.weight_budget == max(sizes)
        for name, entry in scheduler.entries.items():
            scheduler.initialize_weights(name)
            context = entry.engine.create_execution_context()
            assert context is not None
            inputs, output, _ = tensors[name]
            assert context.set_tensor_address("input", inputs.data_ptr())
            assert context.set_tensor_address("output", output.data_ptr())
            contexts[name] = context
            scheduler.unload_weights(name)
        handles = {}
        copies = managed.cudart.cudaMemcpyAsync
        sync = managed.cudart.cudaStreamSynchronize
        copying = False

        def track_copy(destination, source, size, kind, copy_stream):
            assert int(copy_stream) == stream.cuda_stream
            return copies(destination, source, size, kind, copy_stream)

        def track_sync(copy_stream):
            assert not copying, "Restore must not wait on the host before enqueue"
            return sync(copy_stream)

        monkeypatch.setattr(managed.cudart, "cudaMemcpyAsync", track_copy)
        monkeypatch.setattr(managed.cudart, "cudaStreamSynchronize", track_sync)
        for cycle in range(3):
            for name, entry in scheduler.entries.items():
                before = weight_cuda_calls.copied_bytes
                copying = True
                with scheduler.use_weights(name):
                    copying = False
                    for block, (offset, size) in zip(entry.blocks, scheduler.loader.pool.weight_ranges(entry.size)):
                        assert block.size == size
                        assert int(block.handle) == handles.setdefault(offset, int(block.handle))
                    if cycle and entry.size == max(sizes):
                        assert weight_cuda_calls.copied_bytes - before == sorted(sizes)[-2]
                    assert contexts[name].execute_async_v3(stream.cuda_stream)
                    stream.synchronize()
                    np.testing.assert_array_equal(tensors[name][1].cpu().numpy(), tensors[name][2])
        assert len(handles) == 4
        assert weight_cuda_calls.peak_bytes == scheduler.weight_budget
    finally:
        copying = False
        stream.synchronize()
        scheduler.close()
