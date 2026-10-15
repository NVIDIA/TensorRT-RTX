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
FLUX.2 [klein] 4B text-to-image with an FP8 Torch-TensorRT-RTX transformer.

The pre-quantized FP8 checkpoint is loaded into a ModelOpt-quantized Diffusers transformer, which is then compiled
ahead of time in two explicit steps:

1. ``torch.export.export`` captures the transformer, with ModelOpt emitting its FP8 quantizers as TensorRT
   quantize/dequantize operations.
2. ``torch_tensorrt.dynamo.compile`` converts the exported graph into a TensorRT-RTX engine with FP8 GEMMs.

The compiled transformer is then swapped into an otherwise unmodified Diffusers ``Flux2KleinPipeline``, so text
encoding, scheduling, and VAE decoding continue to run in PyTorch.
"""

import copy
import gc
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch

logger = logging.getLogger("rtx_demo.flux2.flux2_pipeline")

BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
FP8_MODEL_ID = "black-forest-labs/FLUX.2-klein-4b-fp8"
FP8_CHECKPOINT_FILENAME = "flux-2-klein-4b-fp8.safetensors"
# Everything except the full-precision transformer weights, which the FP8 checkpoint replaces.
BASE_MODEL_ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/*",
    "vae/*",
    "transformer/config.json",
]
# ModelOpt derives a per-tensor FP8 E4M3 scale as amax / 448.
FP8_E4M3_MAX = 448.0
# FLUX.2 [klein] is step-distilled.
DEFAULT_NUM_INFERENCE_STEPS = 4
# Height and width must be multiples of the VAE scale factor (8) times the transformer patch size (2).
IMAGE_SIZE_MULTIPLE = 16

# Parameter-name mappings from the Black Forest Labs checkpoint layout to Diffusers' Flux2Transformer2DModel.
_TOP_LEVEL_KEYS = {
    "img_in.weight": "x_embedder.weight",
    "txt_in.weight": "context_embedder.weight",
    "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
    "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
    "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
    "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
    "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
    "final_layer.linear.weight": "proj_out.weight",
}
_DOUBLE_BLOCK_MODULES = {
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.norm.key_norm": "attn.norm_k",
    "img_attn.proj": "attn.to_out.0",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_attn.norm.query_norm": "attn.norm_added_q",
    "txt_attn.norm.key_norm": "attn.norm_added_k",
    "txt_attn.proj": "attn.to_add_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}
_SINGLE_BLOCK_MODULES = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}
_PARAMETER_SUFFIXES = ("weight_scale", "input_scale", "weight", "bias", "scale")


def convert_fp8_state_dict(original: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a Black Forest Labs FLUX.2 [klein] FP8 checkpoint to Diffusers ``Flux2Transformer2DModel`` keys.

    Quantized linears carry a float8_e4m3fn ``weight`` plus scalar ``weight_scale`` and ``input_scale`` tensors.
    Fused QKV weights are split into separate Q, K, and V projections, which share the fused layer's scales.
    """
    state_dict: dict[str, torch.Tensor] = {}
    for key, tensor in original.items():
        if key == "final_layer.adaLN_modulation.1.weight":
            # BFL stores (shift, scale); Diffusers expects (scale, shift).
            shift, scale = tensor.chunk(2, dim=0)
            state_dict["norm_out.linear.weight"] = torch.cat([scale, shift], dim=0)
            continue
        if key in _TOP_LEVEL_KEYS:
            state_dict[_TOP_LEVEL_KEYS[key]] = tensor
            continue

        prefix, block_index, rest = key.split(".", 2)
        suffix = next((s for s in _PARAMETER_SUFFIXES if rest.endswith(f".{s}")), None)
        if suffix is None:
            raise ValueError(f"Unrecognized checkpoint tensor: {key}")
        module = rest[: -(len(suffix) + 1)]
        # Norm layers store their weight as ``scale``.
        parameter = "weight" if suffix == "scale" else suffix

        if prefix == "double_blocks" and module.endswith("qkv"):
            chunks = tensor.chunk(3, dim=0) if suffix in ("weight", "bias") else (tensor,) * 3
            names = (
                ("attn.to_q", "attn.to_k", "attn.to_v")
                if module.startswith("img")
                else ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")
            )
            for name, chunk in zip(names, chunks):
                state_dict[f"transformer_blocks.{block_index}.{name}.{parameter}"] = chunk
        elif prefix == "double_blocks":
            state_dict[f"transformer_blocks.{block_index}.{_DOUBLE_BLOCK_MODULES[module]}.{parameter}"] = tensor
        elif prefix == "single_blocks":
            state_dict[f"single_transformer_blocks.{block_index}.{_SINGLE_BLOCK_MODULES[module]}.{parameter}"] = tensor
        else:
            raise ValueError(f"Unrecognized checkpoint tensor: {key}")
    return state_dict


def split_fp8_state_dict(
    state_dict: dict[str, torch.Tensor], dtype: torch.dtype
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Separate the FP8 scales from the weights and dequantize the FP8 weights to ``dtype``.

    Returns:
        (weights, weight_scales, input_scales), where the scales are keyed by linear-layer name.
    """
    weight_scales = {
        key[: -len(".weight_scale")]: value.float()
        for key, value in state_dict.items()
        if key.endswith(".weight_scale")
    }
    input_scales = {
        key[: -len(".input_scale")]: value.float() for key, value in state_dict.items() if key.endswith(".input_scale")
    }
    weights = {key: value for key, value in state_dict.items() if not key.endswith(("weight_scale", "input_scale"))}
    # ModelOpt re-quantizes these weights to the identical FP8 values when the model is exported.
    for name, scale in weight_scales.items():
        weights[f"{name}.weight"] = weights[f"{name}.weight"].float() * scale
    return {key: value.to(dtype) for key, value in weights.items()}, weight_scales, input_scales


def configure_fp8_quantizers(
    model: torch.nn.Module, weight_scales: dict[str, torch.Tensor], input_scales: dict[str, torch.Tensor]
) -> int:
    """Program ModelOpt FP8 quantizers from the checkpoint scales and disable all other quantizers.

    Returns:
        The number of FP8 linear layers.
    """
    from modelopt.torch.quantization.nn import TensorQuantizer

    quantized_layers = set()
    for name, module in model.named_modules():
        weight_quantizer = getattr(module, "weight_quantizer", None)
        if not isinstance(weight_quantizer, TensorQuantizer):
            continue
        if name in weight_scales:
            device = module.weight.device
            weight_quantizer.amax = (weight_scales[name] * FP8_E4M3_MAX).to(device)
            module.input_quantizer.amax = (input_scales[name] * FP8_E4M3_MAX).to(device)
            quantized_layers.add(name)
        else:
            for attribute in ("input_quantizer", "weight_quantizer", "output_quantizer"):
                quantizer = getattr(module, attribute, None)
                if isinstance(quantizer, TensorQuantizer):
                    quantizer.disable()

    missing = sorted(set(weight_scales) - quantized_layers)
    if missing:
        raise RuntimeError(f"Checkpoint layers without a matching ModelOpt quantizer: {missing[:5]}")
    # Any remaining quantizer without a checkpoint scale would otherwise need calibration data.
    for module in model.modules():
        if isinstance(module, TensorQuantizer) and module.is_enabled and module.amax is None:
            module.disable()
    return len(quantized_layers)


class _TensorOutputTransformer(torch.nn.Module):
    """Export wrapper that calls the transformer with ``return_dict=False`` so every graph input is a tensor."""

    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(self, **kwargs: torch.Tensor):
        return self.transformer(**kwargs, return_dict=False)


class _KeywordEngine(torch.nn.Module):
    """Give a bare ``TorchTensorRTModule``, which takes positional inputs only, the keyword call of a GraphModule.

    With ``require_full_compilation``, Torch-TensorRT can return the single engine instead of a GraphModule. The
    child keeps the ``_run_on_acc_0`` name so ``torch_tensorrt.runtime.weight_streaming`` still finds it.
    """

    def __init__(self, engine: torch.nn.Module, input_names: list[str]) -> None:
        super().__init__()
        self._run_on_acc_0 = engine
        self.input_names = input_names

    def forward(self, **kwargs: torch.Tensor):
        return self._run_on_acc_0(*(kwargs[name] for name in self.input_names))


class CompiledFlux2Transformer(torch.nn.Module):
    """Adapt a compiled FLUX.2 transformer to the Diffusers ``Flux2KleinPipeline`` call contract."""

    def __init__(
        self,
        compiled: torch.nn.Module,
        source_transformer: torch.nn.Module,
        *,
        dtype: torch.dtype,
        device: Union[str, torch.device],
    ) -> None:
        super().__init__()
        self.compiled = compiled
        self.config = source_transformer.config
        self._dtype = dtype
        self.device = torch.device(device)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def cache_context(self, *args, **kwargs):
        return nullcontext()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs: Any,
    ):
        if guidance is not None:
            raise NotImplementedError("The compiled FLUX.2 [klein] transformer does not use guidance embeddings")
        if joint_attention_kwargs:
            raise NotImplementedError("The compiled FLUX.2 transformer does not support attention overrides")
        if kwargs:
            raise NotImplementedError(
                f"The compiled FLUX.2 transformer does not support extra arguments: {sorted(kwargs)}"
            )

        # The pipeline passes packed latents as a transposed view; TensorRT-RTX requires contiguous inputs.
        output = self.compiled(
            hidden_states=hidden_states.contiguous(),
            encoder_hidden_states=encoder_hidden_states.contiguous(),
            timestep=timestep.contiguous(),
            img_ids=img_ids.contiguous(),
            txt_ids=txt_ids.contiguous(),
        )
        if isinstance(output, (tuple, list)):
            output = output[0]
        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            return Transformer2DModelOutput(sample=output)
        return (output,)


@dataclass(frozen=True)
class Flux2CompiledShape:
    """The fixed shape a FLUX.2 transformer engine was compiled for."""

    batch_size: int
    height: int
    width: int


def torchtrt_compile_options(engine_cache_dir: Optional[Path], *, enable_weight_streaming: bool) -> dict[str, Any]:
    """Keyword arguments for ``torch_tensorrt.dynamo.compile``."""
    options: dict[str, Any] = {
        "pass_through_build_failures": True,
        "truncate_double": True,
        # Build a single TensorRT-RTX engine for the whole transformer instead of falling back to PyTorch.
        "require_full_compilation": True,
        # Move the PyTorch copy of the weights to host memory while the engine is built.
        "offload_module_to_cpu": True,
        # Torch-TensorRT's engine cache refits weights after a cache hit and therefore requires
        # refittable engines. Keep engines immutable unless a persistent cache is requested.
        "immutable_weights": engine_cache_dir is None,
        "enable_weight_streaming": enable_weight_streaming,
    }
    if engine_cache_dir is not None:
        engine_cache_dir.mkdir(parents=True, exist_ok=True)
        options.update(
            {
                "cache_built_engines": True,
                "reuse_cached_engines": True,
                "engine_cache_dir": str(engine_cache_dir),
                "engine_cache_size": 8 * 1024**3,
            }
        )
    return options


def configure_weight_streaming(compiled: torch.nn.Module, budget_fraction: float) -> dict[str, int]:
    """Limit the transformer weights resident in GPU memory to a fraction of the streamable weights."""
    import torch_tensorrt

    context = torch_tensorrt.runtime.weight_streaming(compiled)
    total_budget = int(context.total_device_budget)
    if total_budget <= 0:
        raise RuntimeError("TensorRT-RTX reported no streamable weights for the compiled transformer")
    requested_budget = int(total_budget * budget_fraction)
    context.device_budget = requested_budget
    applied_budget = int(context.device_budget)
    if applied_budget != requested_budget:
        raise RuntimeError(
            f"TensorRT-RTX did not apply the requested weight budget: requested={requested_budget}, "
            f"applied={applied_budget}"
        )
    return {"total_streamable_bytes": total_budget, "device_budget_bytes": applied_budget}


def configure_runtime_cache(compiled: torch.nn.Module, runtime_cache_path: Optional[Path]) -> None:
    """Attach a TensorRT-RTX runtime cache file to every engine in ``compiled``, or detach any cache.

    The runtime cache stores the kernels TensorRT-RTX JIT-compiles for the installed GPU, so later runs can
    skip most of that work. It is loaded when the execution context is created and saved when the engine is
    released.
    """
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    if runtime_cache_path is not None:
        runtime_cache_path.parent.mkdir(parents=True, exist_ok=True)
    for module in compiled.modules():
        if isinstance(module, TorchTensorRTModule):
            module.runtime_settings = module.runtime_settings.merge(
                runtime_cache=str(runtime_cache_path) if runtime_cache_path is not None else None
            )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


class Flux2FP8TorchTensorRTPipeline:
    """FLUX.2 [klein] 4B text-to-image pipeline with an FP8 Torch-TensorRT-RTX compiled transformer."""

    def __init__(
        self,
        cache_dir: str = "./demo_cache",
        device: str = "cuda",
        hf_token: Optional[str] = None,
        enable_engine_cache: bool = False,
        enable_runtime_cache: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.hf_token = hf_token
        self.engine_cache_dir = self.cache_dir / "torchtrt_engines" / "flux2" / "fp8" if enable_engine_cache else None
        self.runtime_cache_path = self.cache_dir / "runtime.cache" if enable_runtime_cache else None
        self.pipe = None
        self.compiled_shape: Optional[Flux2CompiledShape] = None
        self.weight_streaming: Optional[dict[str, int]] = None

    def _check_fp8_support(self) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("This demo requires a CUDA device")
        major, minor = torch.cuda.get_device_capability(self.device)
        if (major, minor) < (8, 9):
            raise RuntimeError(
                f"FP8 requires an Ada or Blackwell GPU (compute capability 8.9 or later); found {major}.{minor}"
            )

    def _download(self) -> tuple[Path, Path]:
        from huggingface_hub import hf_hub_download, snapshot_download

        hub_cache = self.cache_dir / "hf"
        base_path = snapshot_download(
            BASE_MODEL_ID, cache_dir=hub_cache, allow_patterns=BASE_MODEL_ALLOW_PATTERNS, token=self.hf_token
        )
        checkpoint_path = hf_hub_download(
            FP8_MODEL_ID, FP8_CHECKPOINT_FILENAME, cache_dir=hub_cache, token=self.hf_token
        )
        return Path(base_path), Path(checkpoint_path)

    def _load_fp8_transformer(self, base_path: Path, checkpoint_path: Path) -> torch.nn.Module:
        import modelopt.torch.quantization as mtq
        from diffusers import Flux2Transformer2DModel
        from safetensors.torch import load_file

        weights, weight_scales, input_scales = split_fp8_state_dict(
            convert_fp8_state_dict(load_file(str(checkpoint_path))), self.dtype
        )
        config = Flux2Transformer2DModel.load_config(str(base_path), subfolder="transformer")
        with torch.device("meta"):
            transformer = Flux2Transformer2DModel.from_config(config)
        transformer.load_state_dict(weights, strict=True, assign=True)
        del weights
        transformer = transformer.to(self.device).eval()

        # Insert FP8 quantizers, then program them from the checkpoint scales instead of calibrating.
        mtq.quantize(transformer, copy.deepcopy(mtq.FP8_DEFAULT_CFG), forward_loop=None)
        num_fp8_layers = configure_fp8_quantizers(transformer, weight_scales, input_scales)
        logger.info(f"Loaded {num_fp8_layers} FP8 linear layers")
        return transformer

    def load(self) -> float:
        """Download the models and load the Diffusers pipeline with the FP8 transformer. Returns the elapsed seconds."""
        from diffusers import Flux2KleinPipeline

        self._check_fp8_support()
        start = time.perf_counter()
        logger.info(f"Loading {BASE_MODEL_ID} with the FP8 transformer from {FP8_MODEL_ID}...")
        base_path, checkpoint_path = self._download()
        transformer = self._load_fp8_transformer(base_path, checkpoint_path)
        self.pipe = Flux2KleinPipeline.from_pretrained(str(base_path), transformer=transformer, torch_dtype=self.dtype)
        self.pipe.to(self.device)
        return time.perf_counter() - start

    def _capture_transformer_inputs(self, batch_size: int, height: int, width: int) -> dict[str, torch.Tensor]:
        """Run one denoising step and record the exact transformer inputs the pipeline produces."""
        captured: dict[str, torch.Tensor] = {}

        def capture(_module, _args, kwargs):
            if not captured:
                captured.update(
                    {
                        name: value.detach().contiguous().clone()
                        for name, value in kwargs.items()
                        if torch.is_tensor(value)
                    }
                )

        handle = self.pipe.transformer.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            self.pipe(
                prompt="",
                num_images_per_prompt=batch_size,
                height=height,
                width=width,
                num_inference_steps=1,
                guidance_scale=1.0,
                output_type="latent",
            )
        finally:
            handle.remove()
        return captured

    def compile_transformer(
        self,
        batch_size: int = 1,
        height: int = 1024,
        width: int = 1024,
        weight_streaming_budget: Optional[float] = None,
    ) -> dict[str, float]:
        """Export and compile the transformer for one fixed shape, then swap it into the pipeline.

        Args:
            batch_size: Images per prompt.
            height: Image height in pixels.
            width: Image width in pixels.
            weight_streaming_budget: Optional fraction [0, 1] of the streamable transformer weights kept in GPU memory.
                Zero streams the most weights; None disables weight streaming.

        Returns:
            Timings in seconds for each compilation phase.
        """
        import torch_tensorrt
        from modelopt.torch.quantization.utils import export_torch_mode

        if self.pipe is None:
            raise RuntimeError("Call load() before compile_transformer()")
        if self.compiled_shape is not None:
            raise RuntimeError(
                "The transformer has already been compiled; create a new pipeline to compile another shape"
            )
        if height % IMAGE_SIZE_MULTIPLE or width % IMAGE_SIZE_MULTIPLE:
            raise ValueError(f"height and width must be multiples of {IMAGE_SIZE_MULTIPLE}")
        if weight_streaming_budget is not None and not 0.0 <= weight_streaming_budget <= 1.0:
            raise ValueError("weight_streaming_budget must be in the range [0, 1]")

        logging.getLogger("torch_tensorrt.dynamo.conversion.converter_utils").setLevel(logging.ERROR)
        # Any prompt produces the same input shapes; the engine is specialized to shape, not content.
        sample_inputs = self._capture_transformer_inputs(batch_size, height, width)
        for name, value in sample_inputs.items():
            logger.debug(f"Transformer input {name}: shape={tuple(value.shape)} dtype={value.dtype}")
        # Only tensors can be graph inputs; return_dict=False is baked into the exported graph by a wrapper.
        export_kwargs = dict(sample_inputs)
        source_transformer = self.pipe.transformer
        export_module = _TensorOutputTransformer(source_transformer)

        # Step 1: capture the transformer graph, with ModelOpt emitting TensorRT FP8 quantize operations.
        logger.info("Step 1/2: Exporting the transformer with torch.export...")
        start = time.perf_counter()
        with export_torch_mode():
            exported_program = torch.export.export(export_module, args=(), kwargs=export_kwargs, strict=False)
        export_seconds = time.perf_counter() - start

        # Step 2: build a TensorRT-RTX engine from the exported graph.
        logger.info("Step 2/2: Compiling the transformer with torch_tensorrt.dynamo.compile...")
        if self.engine_cache_dir is not None:
            logger.info(f"Torch-TensorRT engine cache: {self.engine_cache_dir}")
        start = time.perf_counter()
        compiled = torch_tensorrt.dynamo.compile(
            exported_program,
            arg_inputs=(),
            kwarg_inputs=export_kwargs,
            **torchtrt_compile_options(
                self.engine_cache_dir, enable_weight_streaming=weight_streaming_budget is not None
            ),
        )
        _synchronize(self.device)
        compile_seconds = time.perf_counter() - start
        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

        if isinstance(compiled, TorchTensorRTModule):
            # Engine inputs follow the flattened export kwargs, which keep dict insertion order.
            compiled = _KeywordEngine(compiled, list(export_kwargs))

        self.pipe.transformer = CompiledFlux2Transformer(
            compiled, source_transformer, dtype=self.dtype, device=self.device
        )
        del exported_program, export_module, source_transformer
        _free_memory(self.device)

        if self.runtime_cache_path is not None:
            logger.info(f"TensorRT-RTX runtime cache: {self.runtime_cache_path}")
        configure_runtime_cache(compiled, self.runtime_cache_path)

        if weight_streaming_budget is not None:
            self.weight_streaming = configure_weight_streaming(compiled, weight_streaming_budget)
            logger.info(
                f"Weight streaming: {self.weight_streaming['device_budget_bytes'] / 1024**3:.2f} GiB of "
                f"{self.weight_streaming['total_streamable_bytes'] / 1024**3:.2f} GiB streamable transformer weights "
                "kept on the GPU"
            )

        # The first execution triggers TensorRT-RTX just-in-time kernel compilation for this GPU.
        logger.info("Running the compiled transformer once to trigger TensorRT-RTX JIT compilation...")
        start = time.perf_counter()
        with torch.inference_mode():
            compiled(**export_kwargs)
        _synchronize(self.device)
        jit_seconds = time.perf_counter() - start

        self.compiled_shape = Flux2CompiledShape(batch_size=batch_size, height=height, width=width)
        return {"export": export_seconds, "compile": compile_seconds, "jit": jit_seconds}

    def infer(
        self,
        prompt: str,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        seed: int = 0,
        save_dir: Optional[str] = ".",
    ) -> tuple[list[Any], float]:
        """Generate images at the compiled shape. Returns the images and end-to-end generation time in seconds."""
        shape = self.compiled_shape
        if shape is None:
            raise RuntimeError("Call compile_transformer() before infer()")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        _synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            images = self.pipe(
                prompt=prompt,
                num_images_per_prompt=shape.batch_size,
                height=shape.height,
                width=shape.width,
                num_inference_steps=num_inference_steps,
                guidance_scale=1.0,
                generator=generator,
            ).images
        _synchronize(self.device)
        seconds = time.perf_counter() - start

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(images):
                path = Path(save_dir) / f"flux2_{seed}_{index}.png"
                image.save(path)
                logger.info(f"Saved {path}")
        return images, seconds

    def cleanup(self) -> None:
        self.pipe = None
        self.compiled_shape = None
        self.weight_streaming = None
        _free_memory(self.device)
