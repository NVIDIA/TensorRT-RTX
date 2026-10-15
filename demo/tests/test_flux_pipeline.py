# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-weight and offloaded pipeline resource lifetimes."""

import importlib
import importlib.abc
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "flux1.dev"))
    module = importlib.import_module("pipelines.flux_pipeline")
    instance = module.FluxPipeline.__new__(module.FluxPipeline)
    instance.cleanup = Mock()
    instance.weight_offload = "none"
    instance.managed_weights = None
    instance.device = "cuda"
    instance.verbose = False
    return instance, module


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["none", "buffer", "vmm"])
@pytest.mark.parametrize(
    "version,supported",
    [
        ("1.6.0", False),
        ("1.6.1.120", False),
        ("1.6.9", False),
        ("1.7rc1", False),
        ("1.7", True),
        ("1.7.1.97", True),
        ("1.10.0", True),
        ("2.0.0", True),
    ],
)
def test_offload_requires_supported_version(pipeline, monkeypatch, tmp_path, mode, version, supported):
    instance, module = pipeline
    monkeypatch.setattr(module.trt, "__version__", version)
    download_scheduler = Mock()
    monkeypatch.setattr(module.FlowMatchEulerDiscreteScheduler, "from_pretrained", download_scheduler)
    instance.initialize_models = Mock()
    if mode != "none" and not supported:
        with pytest.raises(RuntimeError) as error:
            instance.__init__(cache_dir=str(tmp_path), weight_offload=mode)
        assert str(error.value) == f"Weights offloading requires TensorRT-RTX 1.7 or later; found {version}."
        download_scheduler.assert_not_called()
        instance.initialize_models.assert_not_called()
        monkeypatch.setattr(module.torch.cuda, "empty_cache", Mock())
        module.FluxPipeline.cleanup(instance)
    else:
        instance.__init__(cache_dir=str(tmp_path), weight_offload=mode)
        download_scheduler.assert_called_once()
        instance.initialize_models.assert_called_once_with()
        assert instance.weight_offload == mode


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["none", "buffer", "vmm"])
def test_pinned_host_requires_offloading(pipeline, monkeypatch, tmp_path, mode):
    instance, module = pipeline
    monkeypatch.setattr(module.trt, "__version__", "1.7.1")
    download_scheduler = Mock()
    monkeypatch.setattr(module.FlowMatchEulerDiscreteScheduler, "from_pretrained", download_scheduler)
    instance.initialize_models = Mock()
    with pytest.raises(ValueError, match="require --weight-offload buffer or vmm") if mode == "none" else nullcontext():
        instance.__init__(cache_dir=str(tmp_path), weight_offload=mode, weight_offload_pinned_host=True)
    if mode == "none":
        download_scheduler.assert_not_called()
        instance.initialize_models.assert_not_called()
    else:
        assert instance.weight_offload_pinned_host
        instance.initialize_models.assert_called_once_with()


@pytest.mark.unit
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_stage_keeps_engine_and_context(pipeline, offload, fail):
    instance, _ = pipeline
    engine = Mock()
    context = engine.context
    instance.engines = {"transformer": engine}
    events = []

    @contextmanager
    def use(name):
        assert name == "transformer"
        events.append("restore")
        try:
            yield
        finally:
            events.append("unload")

    if offload:
        instance.managed_weights = SimpleNamespace(use_weights=use)
    with (
        pytest.raises(RuntimeError, match="stage failed") if fail else nullcontext(),
        instance._use_model_weights("transformer"),
    ):
        events.append("execute")
        if fail:
            raise RuntimeError("stage failed")
    assert events == (["restore", "execute", "unload"] if offload else ["execute"])
    assert instance.engines["transformer"] is engine and engine.context is context
    assert not engine.mock_calls


@pytest.mark.unit
@pytest.mark.parametrize("cached", [False, True])
def test_full_weight_engine_is_loaded(pipeline, monkeypatch, tmp_path, cached):
    instance, module = pipeline
    plan, onnx = tmp_path / "full.engine", tmp_path / "full.onnx"
    onnx.touch()
    if cached:
        plan.touch()
    instance.path_manager = SimpleNamespace(get_engine_path=lambda *args: plan, get_onnx_path=lambda *args: onnx)
    instance.runtime_cache_path = None
    instance.cuda_graph_strategy = "disabled"
    instance.engines = {}
    profile = {"hidden_states": [(1, 1024, 64)] * 3}
    instance.model_instances = {"transformer": SimpleNamespace(get_input_profile=lambda *args: profile)}
    engine = Mock()
    monkeypatch.setattr(module, "Engine", Mock(return_value=engine))
    monkeypatch.setattr(module.metadata_manager, "check_engine_compatibility", lambda **kwargs: (True, "compatible"))
    instance._prepare_engine("transformer", "flux_transformer", "fp8", "static")
    engine.load.assert_called_once_with()
    assert engine.build.call_count == int(not cached)
    assert instance.engines["transformer"] is engine


@pytest.mark.unit
def test_full_weight_buffers_follow_shape_changes(pipeline):
    instance, _ = pipeline
    engine = Mock()
    instance.engines = {"transformer": engine}
    instance.model_instances = {
        "transformer": SimpleNamespace(
            get_shape_dict=lambda batch, height, width: {"latent": (batch, height, width)},
        )
    }
    instance.stream = 1
    instance.current_shapes = {}
    instance.load_resources(1, 512, 512)
    engine.allocate_buffers.assert_called_once_with({"latent": (1, 512, 512)}, device="cuda")
    instance.load_resources(1, 512, 512)
    assert engine.allocate_buffers.call_count == 1
    engine.deallocate_buffers.assert_not_called()
    instance.load_resources(2, 256, 256)
    engine.deallocate_buffers.assert_called_once_with()
    assert engine.allocate_buffers.call_count == 2
    engine.allocate_buffers.assert_called_with({"latent": (2, 256, 256)}, device="cuda")
    assert instance.current_shapes == {"batch_size": 2, "height": 256, "width": 256}


@pytest.mark.unit
@pytest.mark.parametrize("grow_workspace", [False, True])
def test_full_weight_refresh_activates_engine(pipeline, monkeypatch, tmp_path, grow_workspace):
    instance, module = pipeline
    plan = tmp_path / "full.engine"
    plan.touch()
    engine = Mock()
    instance.engines = {"transformer": engine}
    instance.shape_config = {"transformer": "static"}
    instance._get_model_configs = lambda: {"transformer": ("flux_transformer", "fp8")}
    instance.path_manager = SimpleNamespace(get_engine_path=lambda *args: plan)
    instance.model_instances = {"transformer": SimpleNamespace(get_input_profile=lambda *args: {})}
    instance.calculate_max_device_memory = Mock(side_effect=[100, 200 if grow_workspace else 100])
    instance._prepare_engine = Mock()
    instance.shared_device_memory = 11
    monkeypatch.setattr(
        module.metadata_manager, "check_engine_compatibility", lambda **kwargs: (False, "shape changed")
    )
    allocate, free = Mock(return_value=(0, 22)), Mock()
    monkeypatch.setattr(module.cudart, "cudaMalloc", allocate)
    monkeypatch.setattr(module.cudart, "cudaFree", free)
    assert instance.refresh_engines(opt_height=256, opt_width=256)
    instance._prepare_engine.assert_called_once_with(
        "transformer",
        "flux_transformer",
        "fp8",
        "static",
        1,
        256,
        256,
        None,
    )
    engine.activate.assert_called_once_with(device_memory=22 if grow_workspace else 11)
    if grow_workspace:
        allocate.assert_called_once_with(200)
        free.assert_called_once_with(11)
    else:
        allocate.assert_not_called()
        free.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("stream,failure", [(None, "stream"), (None, "workspace"), (7, "workspace")])
def test_offload_setup_rejects_failed_cuda_handles(pipeline, monkeypatch, stream, failure):
    """Only successful CUDA results may become pipeline-owned resources."""
    instance, module = pipeline
    instance.stream = stream
    instance.shared_device_memory = None
    instance.calculate_max_device_memory = Mock(return_value=100)
    success = module.cudart.cudaError_t.cudaSuccess
    error = module.cudart.cudaError_t.cudaErrorMemoryAllocation
    create = Mock(return_value=(error, 999) if failure == "stream" else (success, 11))
    allocate = Mock(return_value=(error, 999))
    scheduler = Mock()
    monkeypatch.setattr(module.cudart, "cudaStreamCreate", create)
    monkeypatch.setattr(module.cudart, "cudaMalloc", allocate)
    monkeypatch.setattr(module, "ManagedWeightsScheduler", scheduler)

    operation = "cudaStreamCreate" if failure == "stream" else "cudaMalloc"
    with pytest.raises(RuntimeError, match=operation):
        instance._initialize_weight_offloading()

    assert instance.shared_device_memory is None
    assert instance.managed_weights is None
    scheduler.assert_not_called()
    if failure == "stream":
        assert instance.stream is None
        instance.calculate_max_device_memory.assert_not_called()
        allocate.assert_not_called()
    else:
        assert instance.stream == (11 if stream is None else stream)
        allocate.assert_called_once_with(100)
    assert create.call_count == int(stream is None)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["none", "buffer", "vmm"])
@pytest.mark.parametrize("precision", ["omitted", None, "bf16", "fp8", "fp4"])
def test_pipeline_precision_default_and_explicit_override(pipeline, monkeypatch, mode, precision):
    """Offloading preserves the FP8 default and explicit precision choices."""
    instance, module = pipeline
    monkeypatch.setattr(module.torch.cuda, "get_device_capability", lambda device: (12, 0))
    instance.weight_offload = mode
    instance.engines = {}
    instance.pipeline_name = "flux_1_dev"
    instance.precision_config = {}
    instance.path_manager = SimpleNamespace(set_pipeline_models=Mock())
    instance.initialize_models = Mock()
    instance._prepare_engine = Mock(side_effect=RuntimeError("stop before build"))
    options = {} if precision == "omitted" else {"transformer_precision": precision}

    with pytest.raises(RuntimeError, match="stop before build"):
        instance.load_engines(**options)

    assert instance.precision_config["transformer"] == ("fp8" if precision in (None, "omitted") else precision)
    instance._prepare_engine.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("precision", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("shape_mode", ["static", "dynamic"])
def test_offload_context_activation_unloads_weights(pipeline, monkeypatch, tmp_path, fail, precision, shape_mode):
    instance, module = pipeline
    instance.stream = 7
    instance.weight_offload = "vmm"
    instance.weight_offload_pinned_host = True
    instance.calculate_max_device_memory = lambda: 100
    engine = Mock()
    engine.engine_path = tmp_path / precision / f"transformer_weightless_{shape_mode}.safetensors.engine"
    engine.activate.side_effect = RuntimeError("context failed") if fail else None
    instance.engines = {"transformer": engine}
    checkpoint = Mock()
    instance._managed_checkpoints = {"transformer": (checkpoint, "checkpoint")}
    scheduler = Mock()
    monkeypatch.setattr(module, "ManagedWeightsScheduler", Mock(return_value=scheduler))
    monkeypatch.setattr(module.cudart, "cudaMalloc", lambda size: (0, 11))
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 0)
    with pytest.raises(RuntimeError, match="context failed") if fail else nullcontext():
        instance._initialize_weight_offloading()
    scheduler.register_engine.assert_called_once_with(
        "transformer",
        engine.engine,
        checkpoint.directory,
        engine.engine_path,
        tmp_path / precision / f"transformer_weightless_{shape_mode}.safetensors.weights.bin",
        tmp_path / precision / f"transformer_weightless_{shape_mode}.safetensors.weights.json",
        refit=checkpoint.refit,
        source_identity="checkpoint",
    )
    scheduler.configure_memory_pool.assert_called_once_with()
    scheduler.initialize_weights.assert_called_once_with("transformer")
    engine.activate.assert_called_once_with(device_memory=11)
    scheduler.unload_weights.assert_called_once_with("transformer")


@pytest.mark.unit
def test_offload_inference_validates_shapes_once(pipeline):
    instance, _ = pipeline
    instance.weight_offload = "vmm"
    instance.device = "cpu"
    instance.reset_timing_data = Mock()
    instance.refresh_engines = Mock(side_effect=AssertionError("Duplicate preflight"))
    instance._get_validated_input_shapes = Mock(side_effect=ValueError("invalid shape"))
    with pytest.raises(ValueError, match="invalid shape"):
        instance.infer("prompt", height=512, width=512)
    instance._get_validated_input_shapes.assert_called_once_with(1, 512, 512)
    instance.refresh_engines.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["none", "buffer", "vmm"])
@pytest.mark.parametrize(
    "precision,capability,supported",
    [
        ("bf16", (8, 6), True),
        ("fp8", (8, 9), True),
        ("fp4", (12, 0), True),
        ("fp8", (8, 6), False),
        ("fp4", (8, 9), False),
    ],
)
def test_gpu_validation_precedes_engine_preparation(
    pipeline, monkeypatch, caplog, mode, precision, capability, supported
):
    instance, pipeline_module = pipeline
    # Demo logging disables propagation to the root logger used by caplog.
    caplog.set_level("ERROR", logger=pipeline_module.logger.name)
    monkeypatch.setattr(pipeline_module.logger, "handlers", [caplog.handler])
    events = []

    def query_capability(device):
        assert device == instance.device
        events.append("validate")
        return capability

    monkeypatch.setattr(pipeline_module.torch.cuda, "get_device_capability", query_capability)
    instance.weight_offload = mode
    existing_engine = object()
    instance.engines = {"existing": existing_engine}

    def cleanup():
        events.append("cleanup")
        instance.engines.clear()

    instance.cleanup.side_effect = cleanup
    instance.pipeline_name = "flux_1_dev"
    instance.precision_config = {}
    roles = instance.ENGINE_EXECUTION_ORDER
    instance._get_model_configs = lambda: {
        role: (role, precision if role == "transformer" else "bf16") for role in roles
    }
    instance.path_manager = SimpleNamespace(set_pipeline_models=lambda *args: None)
    instance.initialize_models = lambda: events.append("models")

    def build(role, *args):
        events.append(f"build:{role}")
        instance.engines[role] = SimpleNamespace(load=lambda: events.append(f"load:{role}"))
        if mode == "none":
            instance.engines[role].load()

    instance._prepare_engine = build
    instance._initialize_weight_offloading = lambda: events.append("activate") or {}
    instance.activate_engines = lambda: events.append("activate") or {}
    if supported or mode == "none":
        instance.load_engines(transformer_precision=precision)
    else:
        with pytest.raises(ValueError, match="requires .* GPUs"):
            instance.load_engines(transformer_precision=precision)
    expected = ["validate"]
    if supported or mode == "none":
        expected += ["cleanup", "models"]
        if mode == "none":
            expected += [event for role in roles for event in (f"build:{role}", f"load:{role}")]
        else:
            expected += [f"build:{role}" for role in roles] + [f"load:{role}" for role in roles]
        expected += ["activate"]
        assert ("Proceeding, but expect errors" in caplog.text) == (not supported)
    else:
        instance.cleanup.assert_not_called()
        assert instance.engines == {"existing": existing_engine}
        assert instance.precision_config == {}
    assert events == expected


@pytest.mark.unit
@pytest.mark.parametrize("shape_mode", ["static", "dynamic"])
@pytest.mark.parametrize("shape", [(1, 256, 256), (2, 512, 768), (4, 1024, 1024), (5, 512, 512), (1, 128, 2048)])
def test_offload_validates_input_shapes(pipeline, shape_mode, shape):
    from models.flux_params import FluxParams

    instance, pipeline_module = pipeline
    instance.shape_config = {"transformer": shape_mode}
    instance._native_build_shapes = (1, 256, 256)
    instance.model_params = FluxParams()
    instance.model_instances = {
        "transformer": SimpleNamespace(
            get_shape_dict=lambda b, h, w: {
                "hidden_states": (b, h // 16 * (w // 16), 64),
            }
        )
    }
    instance.engines = {
        "transformer": SimpleNamespace(
            engine=SimpleNamespace(
                num_io_tensors=1,
                get_tensor_name=lambda i: "hidden_states",
                get_tensor_mode=lambda name: pipeline_module.trt.TensorIOMode.INPUT,
                get_tensor_shape=lambda name: (-1, -1, 64) if shape_mode == "dynamic" else (1, 256, 64),
                get_tensor_profile_shape=lambda name, i: ((1, 256, 64), (2, 1024, 64), (4, 4096, 64)),
            )
        )
    }
    valid = shape[0] <= 4 and 256 <= shape[1] <= 1024 and 256 <= shape[2] <= 1024
    valid = valid and (shape_mode == "dynamic" or shape == (1, 256, 256))
    if valid:
        assert instance._get_validated_input_shapes(*shape)["transformer"]["hidden_states"][0] == shape[0]
    else:
        error, message = (RuntimeError, "reloading") if shape_mode == "static" else (ValueError, "outside")
        with pytest.raises(error, match=message):
            instance._get_validated_input_shapes(*shape)


@pytest.mark.unit
@pytest.mark.parametrize("precision", ["bf16", "fp8", "fp4"])
def test_pipeline_offload_never_requests_onnx(pipeline, native_builder, monkeypatch, tmp_path, precision):
    class RejectOnnxLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise AssertionError("Weightless pipeline attempted an ONNX import")

    class RejectOnnx(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "onnx" or fullname.startswith("onnx."):
                return importlib.util.spec_from_loader(fullname, RejectOnnxLoader())

    monkeypatch.setattr(sys, "meta_path", [RejectOnnx(), *sys.meta_path])
    instance, pipeline_module = pipeline
    instance.weight_offload = "buffer"
    instance.hf_token = "test-token"
    instance.runtime_cache_path = None
    instance.cuda_graph_strategy = "disabled"
    instance.engines = {}
    instance._managed_checkpoints = {}

    def forbidden(*args, **kwargs):
        raise AssertionError("Weightless pipeline attempted ONNX acquisition")

    instance.path_manager = SimpleNamespace(
        get_onnx_path=forbidden,
        acquire_onnx_file=forbidden,
        get_engine_path=lambda *args, **kwargs: tmp_path / "component.engine",
    )
    checkpoint = SimpleNamespace(directory=tmp_path, refit=forbidden)
    builds = []
    weight_file = None if precision == "bf16" else tmp_path / "quantized.safetensors"
    monkeypatch.setattr(native_builder, "download_checkpoint", lambda role, token, precision: (tmp_path, weight_file))

    def build(*args, **kwargs):
        builds.append((args, kwargs))
        return checkpoint, "checkpoint-id"

    monkeypatch.setattr(native_builder, "build_weightless_engine", build)
    monkeypatch.setattr(pipeline_module, "Engine", lambda path, *args: SimpleNamespace(engine_path=path))
    instance._prepare_engine("transformer", "flux_transformer", precision, "static", 2, 16, 32)
    assert builds == [
        (
            (tmp_path, "transformer", tmp_path / "component.safetensors.engine", 2, 16, 32),
            {"weight_file": weight_file, "precision": precision},
        )
    ]
    assert instance._managed_checkpoints["transformer"] == (checkpoint, "checkpoint-id")
    assert instance.engines["transformer"].engine_path.name == "component.safetensors.engine"
    profiles = {"hidden_states": [(1, 1, 4), (2, 2, 4), (3, 4, 4)]}
    instance.model_instances = {"transformer": SimpleNamespace(get_input_profile=lambda *args: profiles)}
    instance._prepare_engine("transformer", "flux_transformer", precision, "dynamic", 2, 16, 32)
    assert builds[-1][1]["input_profiles"] == profiles
    with pytest.raises(AssertionError, match="shape_mode must be either"):
        instance._prepare_engine("transformer", "flux_transformer", precision, "invalid")
    with pytest.raises(ValueError, match="do not accept Polygraphy arguments"):
        instance._prepare_engine("transformer", "flux_transformer", precision, "static", extra_args={"verbose": True})
    assert len(builds) == 2
