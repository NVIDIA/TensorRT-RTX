# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
CPU-only tests for the FLUX.2 demo's checkpoint conversion, transformer adapter, compile options, and inference.
These do not require Torch-TensorRT-RTX or model weights.
"""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent / "flux2"))

from flux2_demo import weight_streaming_budget
from flux2_pipeline import (
    CompiledFlux2Transformer,
    Flux2CompiledShape,
    Flux2FP8TorchTensorRTPipeline,
    convert_fp8_state_dict,
    split_fp8_state_dict,
    torchtrt_compile_options,
)


def _scalar(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


class TestCheckpointConversion:
    def test_fused_qkv_is_split_and_scales_are_shared(self):
        qkv = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        converted = convert_fp8_state_dict(
            {
                "double_blocks.0.img_attn.qkv.weight": qkv,
                "double_blocks.0.img_attn.qkv.weight_scale": _scalar(0.5),
                "double_blocks.0.img_attn.qkv.input_scale": _scalar(0.25),
                "double_blocks.0.txt_attn.qkv.weight": qkv,
            }
        )

        for index, name in enumerate(("to_q", "to_k", "to_v")):
            prefix = f"transformer_blocks.0.attn.{name}"
            torch.testing.assert_close(converted[f"{prefix}.weight"], qkv[2 * index : 2 * index + 2])
            assert converted[f"{prefix}.weight_scale"].item() == 0.5
            assert converted[f"{prefix}.input_scale"].item() == 0.25
        for name in ("add_q_proj", "add_k_proj", "add_v_proj"):
            assert f"transformer_blocks.0.attn.{name}.weight" in converted

    def test_module_names_are_mapped(self):
        tensor = torch.ones(2)
        converted = convert_fp8_state_dict(
            {
                "img_in.weight": tensor,
                "double_blocks.1.img_attn.norm.query_norm.scale": tensor,
                "double_blocks.1.txt_mlp.2.weight": tensor,
                "single_blocks.3.linear1.weight": tensor,
                "single_blocks.3.norm.key_norm.scale": tensor,
            }
        )

        assert set(converted) == {
            "x_embedder.weight",
            "transformer_blocks.1.attn.norm_q.weight",
            "transformer_blocks.1.ff_context.linear_out.weight",
            "single_transformer_blocks.3.attn.to_qkv_mlp_proj.weight",
            "single_transformer_blocks.3.attn.norm_k.weight",
        }

    def test_final_modulation_is_reordered_to_scale_shift(self):
        shift = torch.zeros(2, 3)
        scale = torch.ones(2, 3)
        converted = convert_fp8_state_dict({"final_layer.adaLN_modulation.1.weight": torch.cat([shift, scale])})
        torch.testing.assert_close(converted["norm_out.linear.weight"], torch.cat([scale, shift]))

    def test_unknown_tensor_is_rejected(self):
        with pytest.raises((ValueError, KeyError)):
            convert_fp8_state_dict({"double_blocks.0.unknown.weight": torch.ones(1)})


class TestFP8Dequantization:
    def test_fp8_weights_are_dequantized_and_scales_are_separated(self):
        fp8_weight = torch.tensor([[1.0, -2.0], [0.5, 4.0]]).to(torch.float8_e4m3fn)
        weights, weight_scales, input_scales = split_fp8_state_dict(
            {
                "layer.weight": fp8_weight,
                "layer.weight_scale": _scalar(0.5),
                "layer.input_scale": _scalar(0.125),
                "norm.weight": torch.ones(2),
            },
            torch.bfloat16,
        )

        assert set(weights) == {"layer.weight", "norm.weight"}
        assert all(value.dtype == torch.bfloat16 for value in weights.values())
        torch.testing.assert_close(weights["layer.weight"].float(), fp8_weight.float() * 0.5)
        assert weight_scales["layer"].item() == 0.5
        assert input_scales["layer"].item() == 0.125


class _FlatCompiled(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
        assert hidden_states.is_contiguous()
        return (hidden_states + 1,)


class _FakeTorchTensorRTModule(_FlatCompiled):
    """Stands in for the bare engine that full compilation can return; it accepts positional inputs only."""

    def forward(self, *inputs):
        return super().forward(*inputs)


def _adapter() -> CompiledFlux2Transformer:
    source = SimpleNamespace(config=SimpleNamespace(in_channels=128))
    return CompiledFlux2Transformer(_FlatCompiled(), source, dtype=torch.bfloat16, device="cpu")


def _inputs() -> dict[str, torch.Tensor]:
    return {
        # The pipeline passes packed latents as a transposed, non-contiguous view.
        "hidden_states": torch.zeros(1, 128, 16).transpose(1, 2),
        "encoder_hidden_states": torch.zeros(1, 8, 32),
        "timestep": torch.ones(1),
        "img_ids": torch.zeros(1, 16, 4, dtype=torch.int64),
        "txt_ids": torch.zeros(1, 8, 4, dtype=torch.int64),
    }


class TestCompiledFlux2Transformer:
    def test_adapter_restores_the_pipeline_contract(self):
        adapter = _adapter()
        inputs = _inputs()

        output = adapter(**inputs, guidance=None, joint_attention_kwargs=None, return_dict=False)
        assert len(output) == 1
        torch.testing.assert_close(output[0], inputs["hidden_states"] + 1)
        assert adapter.dtype == torch.bfloat16
        assert adapter.config.in_channels == 128
        with adapter.cache_context("cond"):
            pass

    def test_adapter_rejects_guidance(self):
        with pytest.raises(NotImplementedError, match="guidance"):
            _adapter()(**_inputs(), guidance=torch.ones(1), return_dict=False)

    def test_adapter_rejects_attention_overrides(self):
        with pytest.raises(NotImplementedError, match="attention"):
            _adapter()(**_inputs(), joint_attention_kwargs={"scale": 1.0}, return_dict=False)


class TestTorchTensorRTCompileOptions:
    @pytest.mark.parametrize("enable_weight_streaming", [False, True])
    def test_default_options_build_immutable_full_graph_engines(self, enable_weight_streaming):
        options = torchtrt_compile_options(None, enable_weight_streaming=enable_weight_streaming)

        assert options["require_full_compilation"] is True
        assert options["offload_module_to_cpu"] is True
        assert options["immutable_weights"] is True
        assert "engine_cache_dir" not in options
        assert options["enable_weight_streaming"] is enable_weight_streaming

    def test_engine_cache_requires_refittable_engines(self, tmp_path):
        cache_dir = tmp_path / "engines"
        options = torchtrt_compile_options(cache_dir, enable_weight_streaming=True)

        assert cache_dir.is_dir()
        assert options["cache_built_engines"] is True
        assert options["reuse_cached_engines"] is True
        assert options["engine_cache_dir"] == str(cache_dir)
        assert options["immutable_weights"] is False
        assert options["enable_weight_streaming"] is True


class TestWeightStreaming:
    @pytest.mark.parametrize("value", ["0", "0.3", "1"])
    def test_cli_accepts_supported_budgets(self, value):
        assert weight_streaming_budget(value) == float(value)

    @pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf", "-inf"])
    def test_cli_rejects_invalid_budgets(self, value):
        with pytest.raises(argparse.ArgumentTypeError, match=r"\[0, 1\]"):
            weight_streaming_budget(value)

    @pytest.fixture(params=[_FlatCompiled, _FakeTorchTensorRTModule], ids=["graph_module", "bare_engine"])
    def compilation(self, monkeypatch, request):
        pipeline = Flux2FP8TorchTensorRTPipeline(device="cpu")
        transformer = request.param()
        transformer.config = SimpleNamespace(in_channels=128)
        pipeline.pipe = SimpleNamespace(transformer=transformer)
        context = SimpleNamespace(total_device_budget=1000, device_budget=1000)
        compile_options = {}

        def compile_transformer(_exported_program, **options):
            compile_options.update(options)
            return transformer

        monkeypatch.setitem(
            sys.modules,
            "torch_tensorrt",
            SimpleNamespace(
                dynamo=SimpleNamespace(compile=compile_transformer),
                runtime=SimpleNamespace(weight_streaming=lambda _compiled: context),
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "torch_tensorrt.dynamo.runtime",
            SimpleNamespace(TorchTensorRTModule=_FakeTorchTensorRTModule),
        )
        monkeypatch.setitem(
            sys.modules,
            "modelopt.torch.quantization.utils",
            SimpleNamespace(export_torch_mode=nullcontext),
        )
        monkeypatch.setattr(torch.export, "export", lambda *args, **kwargs: object())
        monkeypatch.setattr("flux2_pipeline.configure_runtime_cache", lambda *args: None)
        monkeypatch.setattr(
            pipeline,
            "_capture_transformer_inputs",
            lambda *args: {name: value.contiguous() for name, value in _inputs().items()},
        )
        return pipeline, context, compile_options

    @pytest.mark.parametrize("budget", [None, 0.0, 0.3, 1.0])
    def test_compilation_applies_the_requested_budget(self, compilation, budget):
        pipeline, context, options = compilation

        pipeline.compile_transformer(weight_streaming_budget=budget)

        assert options["enable_weight_streaming"] is (budget is not None)
        if budget is None:
            assert pipeline.weight_streaming is None
            assert context.device_budget == 1000
        else:
            assert context.device_budget == int(1000 * budget)
            assert pipeline.weight_streaming == {
                "total_streamable_bytes": 1000,
                "device_budget_bytes": int(1000 * budget),
            }
        # A bare engine is wrapped so the adapter can call it with keyword inputs.
        output = pipeline.pipe.transformer(**_inputs(), return_dict=False)
        torch.testing.assert_close(output[0], _inputs()["hidden_states"] + 1)

    @pytest.mark.parametrize("budget", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")])
    def test_compilation_rejects_invalid_budgets(self, compilation, budget):
        pipeline, _, options = compilation

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            pipeline.compile_transformer(weight_streaming_budget=budget)

        assert not options


class TestFP8Quantizers:
    def test_checkpoint_scales_program_modelopt_quantizers(self):
        mtq = pytest.importorskip("modelopt.torch.quantization")
        from flux2_pipeline import FP8_E4M3_MAX, configure_fp8_quantizers

        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=None)

        assert configure_fp8_quantizers(model, {"0": _scalar(0.5)}, {"0": _scalar(0.25)}) == 1
        assert model[0].weight_quantizer.amax.item() == 0.5 * FP8_E4M3_MAX
        assert model[0].input_quantizer.amax.item() == 0.25 * FP8_E4M3_MAX
        assert not model[1].weight_quantizer.is_enabled
        assert not model[1].input_quantizer.is_enabled


class TestFlux2Inference:
    def test_inference_requires_compilation(self):
        pipeline = Flux2FP8TorchTensorRTPipeline(device="cpu")

        with pytest.raises(RuntimeError, match="Call compile_transformer"):
            pipeline.infer(prompt="A lighthouse", save_dir=None)

    def test_inference_uses_the_compiled_resolution_and_batch_size(self):
        pipeline = Flux2FP8TorchTensorRTPipeline(device="cpu")
        pipeline.compiled_shape = Flux2CompiledShape(batch_size=2, height=768, width=1344)
        images = [object(), object()]
        calls = []

        def generate(**kwargs):
            assert torch.is_inference_mode_enabled()
            calls.append(kwargs)
            return SimpleNamespace(images=images)

        pipeline.pipe = generate
        result, seconds = pipeline.infer(prompt="A lighthouse", num_inference_steps=6, seed=42, save_dir=None)

        assert result is images
        assert seconds >= 0
        assert len(calls) == 1
        assert calls[0]["prompt"] == "A lighthouse"
        assert calls[0]["height"] == 768
        assert calls[0]["width"] == 1344
        assert calls[0]["num_images_per_prompt"] == 2
        assert calls[0]["num_inference_steps"] == 6
        assert calls[0]["generator"].initial_seed() == 42
