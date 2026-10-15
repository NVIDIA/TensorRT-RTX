# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic native networks retain their contexts across shape changes and weight restores."""

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("mode,pinned", [("buffer", False), ("buffer", True), ("vmm", False), ("vmm", True)])
def test_dynamic_component_restore(component_reference, tmp_path, monkeypatch, mode, pinned, weight_cuda_calls):
    import torch
    from utils.managed_weights import ManagedWeightsScheduler

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "flux1.dev"))
    from flux_native_builder import build_weightless_engine
    from flux_network import LOGGER, create_network, trt

    role, directory, original, reference_outputs, dtype = component_reference
    shapes = [(2, 16, 32), (1, 16, 16), (3, 32, 32), (2, 16, 32)]

    def inputs_for(batch, height, width):
        result = {}
        for name, value in original.items():
            shape = list(value.shape)
            if name != "txt_ids":
                shape[0] = (height // 16) * (width // 16) if name == "img_ids" else batch
            if role == "vae":
                shape[2:] = [height // 2, width // 2]
            if name == "hidden_states":
                shape[1] = (height // 16) * (width // 16)
            repeats = tuple((dim + old - 1) // old for dim, old in zip(shape, value.shape))
            result[name] = value.repeat(repeats)[tuple(slice(0, dim) for dim in shape)].cuda().contiguous()
        return result

    minimum, optimum, maximum = [inputs_for(*shapes[i]) for i in (1, 0, 2)]
    profiles = {name: [list(items[name].shape) for items in (minimum, optimum, maximum)] for name in original}
    del minimum, optimum, maximum
    plan_path = tmp_path / "dynamic.plan"
    checkpoint, identity = build_weightless_engine(
        directory,
        role,
        plan_path,
        batch=2,
        height=16,
        width=32,
        text_length=8,
        input_profiles=profiles,
    )
    assert checkpoint.payload_bytes == 0
    timestamp = plan_path.stat().st_mtime_ns
    build_weightless_engine(
        directory,
        role,
        plan_path,
        batch=2,
        height=16,
        width=32,
        text_length=8,
        input_profiles=profiles,
    )
    assert plan_path.stat().st_mtime_ns == timestamp
    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
    assert engine is not None
    stream = torch.cuda.Stream()
    scheduler = ManagedWeightsScheduler(
        mode, stream.cuda_stream, torch.cuda.current_device(), staging_mib=1, pinned_host=pinned
    )
    scheduler.register_engine(
        role, engine, directory, plan_path, tmp_path / "weights.bin", refit=checkpoint.refit, source_identity=identity
    )
    scheduler.configure_memory_pool()

    def execute(context, inputs):
        for name, value in inputs.items():
            assert context.set_input_shape(name, tuple(value.shape))
        outputs = {
            name: torch.empty(tuple(context.get_tensor_shape(name)), device="cuda", dtype=getattr(torch, dtype))
            for name in reference_outputs
        }
        for name, value in {**inputs, **outputs}.items():
            assert context.set_tensor_address(name, value.data_ptr())
        torch.cuda.synchronize()
        assert context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return {name: value.float().cpu().numpy() for name, value in outputs.items()}

    try:
        scheduler.initialize_weights(role)
        runtime_config = engine.create_runtime_config()
        runtime_config.dynamic_shapes_kernel_specialization_strategy = (
            trt.DynamicShapesKernelSpecializationStrategy.EAGER
        )
        context = engine.create_execution_context(runtime_config)
        assert context is not None
        scheduler.unload_weights(role)
        for batch, height, width in shapes:
            inputs = inputs_for(batch, height, width)
            builder = trt.Builder(LOGGER)
            network = create_network(builder, directory, role, False, batch, height, width, text_length=8)
            config = builder.create_builder_config()
            full_plan = builder.build_serialized_network(network.net, config)
            assert full_plan is not None
            full_engine = runtime.deserialize_cuda_engine(full_plan)
            full_context = full_engine.create_execution_context()
            expected = execute(full_context, inputs)
            with scheduler.use_weights(role):
                actual = execute(context, inputs)
            assert scheduler.entries[role].state == "unloaded"
            for name in expected:
                # Default TensorRT precision permits TF32 rounding in FP32 operations.
                # BF16 outputs near zero can exceed 0.02 absolute error, so loosen atol only.
                rtol = 0.02 if dtype == "bfloat16" else 5e-3
                atol = 0.03 if dtype == "bfloat16" else 5e-3
                np.testing.assert_allclose(actual[name], expected[name], atol=atol, rtol=rtol)
                if (batch, height, width) == shapes[0]:
                    np.testing.assert_allclose(actual[name], reference_outputs[name], atol=atol, rtol=rtol)
            del full_context, full_engine
        assert weight_cuda_calls.peak_bytes <= scheduler.weight_budget
    finally:
        stream.synchronize()
        scheduler.close()
