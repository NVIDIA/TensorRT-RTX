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


import gc
import logging
import os
import time
from typing import Optional

import cuda.bindings.runtime as cudart
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from models.flux_model import FluxT5EncoderModel, FluxTextEncoderModel, FluxTransformerModel, FluxVAEModel
from models.flux_params import FluxParams
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, T5TokenizerFast
from utils.engine import Engine
from utils.engine_metadata import metadata_manager
from utils.memory_manager import ModelMemoryManager
from utils.model_registry import registry as model_registry
from utils.pipeline import Pipeline

logger = logging.getLogger("rtx_demo.flux1.dev.pipelines.flux_pipeline")


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class FluxPipeline(Pipeline):
    """Simple Flux text-to-image pipeline using TensorRT-RTX"""

    def __init__(
        self,
        cache_dir: str = "./demo_cache",
        device: str = "cuda",
        verbose: bool = True,
        cache_mode: str = "full",
        guidance_scale: float = 3.5,
        num_inference_steps: int = 50,
        hf_token: Optional[str] = None,
        low_vram: bool = False,
        log_level: str = "INFO",
        enable_runtime_cache: bool = False,
        cuda_graph_strategy: str = "disabled",
    ):
        super().__init__(
            pipeline_name="flux_1_dev",
            cache_dir=cache_dir,
            device=device,
            verbose=verbose,
            cache_mode=cache_mode,
            hf_token=hf_token,
            low_vram=low_vram,
            log_level=log_level,
            enable_runtime_cache=enable_runtime_cache,
            cuda_graph_strategy=cuda_graph_strategy,
        )

        # Flux-specific parameters
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

        # Initialize scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "black-forest-labs/FLUX.1-dev", subfolder="scheduler", token=self.hf_token
        )

        # Model parameters
        self.model_params = FluxParams()

        # Initialize tokenizers
        self.tokenizer = None
        self.tokenizer2 = None

        # Initialize models
        self.initialize_models()

    def _get_model_configs(self) -> dict[str, tuple[str, str]]:
        """Get model configurations from registry."""
        pipeline_config = model_registry.get_pipeline_config(self.pipeline_name)
        model_configs = {}

        for role, model_id in pipeline_config.items():
            precision = self.precision_config.get(role)
            model_configs[role] = (model_id, precision)

        return model_configs

    def get_model_names(self) -> list[str]:
        """Return list of model names used by this pipeline"""
        return ["clip_text_encoder", "t5_text_encoder", "transformer", "vae_decoder"]

    def initialize_models(self):
        """Initialize model objects"""
        # Initialize tokenizers
        self.initialize_tokenizers()

        # Initialize all model instances
        model_id = "black-forest-labs/FLUX.1-dev"  # Default Flux model
        self.model_instances = {
            "clip_text_encoder": FluxTextEncoderModel(
                name="flux_clip_text_encoder",
                device=self.device,
                model_params=self.model_params,
                model_id=model_id,
                hf_token=self.hf_token,
            ),
            "t5_text_encoder": FluxT5EncoderModel(
                name="flux_t5_text_encoder",
                device=self.device,
                model_params=self.model_params,
                model_id=model_id,
                hf_token=self.hf_token,
            ),
            "transformer": FluxTransformerModel(
                name="flux_transformer",
                device=self.device,
                model_params=self.model_params,
                model_id=model_id,
                hf_token=self.hf_token,
            ),
            "vae_decoder": FluxVAEModel(
                name="flux_vae_decoder",
                device=self.device,
                model_params=self.model_params,
                model_id=model_id,
                hf_token=self.hf_token,
            ),
        }

    def initialize_tokenizers(self):
        """Initialize CLIP and T5 tokenizers"""
        try:
            # Initialize CLIP tokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="tokenizer",
                token=self.hf_token,
            )

            # Initialize T5 tokenizer
            self.tokenizer2 = T5TokenizerFast.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="tokenizer_2",
                token=self.hf_token,
            )

            logger.debug("Tokenizers initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize tokenizers: {e}")
            raise e

    def model_memory_manager(self, model_name, low_vram=False):
        """Returns a context manager for model memory management.

        This helper method creates a ModelMemoryManager instance for efficient
        loading and unloading of models to optimize VRAM usage.

        Args:
            model_name (str): Model name to manage with this context.
            low_vram (bool, optional): Whether to enable VRAM optimization. Defaults to False.

        Returns:
            ModelMemoryManager: Context manager for model memory management.
        """
        return ModelMemoryManager(self, model_name, low_vram=low_vram)

    def build_and_load_engine(
        self,
        role: str,
        model_id: str,
        precision: str,
        shape_mode: str,
        opt_batch_size: int = 1,
        opt_height: int = 512,
        opt_width: int = 512,
        extra_args: Optional[dict[str]] = None,
    ):
        """
        Builds a TensorRT-RTX engine if applicable, otherwise loads from cache.
        Only loads into GPU memory if low_vram is not enabled.
        """
        assert shape_mode in ["static", "dynamic"], "shape_mode must be either 'static' or 'dynamic'"

        logger.debug(f"\nProcessing {role} ({model_id}_{precision})...")

        use_static_shape = shape_mode == "static"
        onnx_path = self.path_manager.get_onnx_path(model_id, precision)
        engine_path = self.path_manager.get_engine_path(model_id, precision, shape_mode)

        # Check if ONNX exists
        if not onnx_path.exists():
            onnx_local_path = model_registry.get_onnx_path(self.pipeline_name, role, precision)
            onnx_repository = model_registry.get_onnx_repository(self.pipeline_name, role, precision)
            onnx_subfolder = model_registry.get_onnx_subfolder(self.pipeline_name, role, precision)

            if onnx_local_path is not None or onnx_repository is not None:
                success = self.path_manager.acquire_onnx_file(
                    model_id,
                    precision,
                    onnx_local_path=onnx_local_path,
                    onnx_repository=onnx_repository,
                    onnx_subfolder=onnx_subfolder,
                    hf_token=self.hf_token,
                )
                if not success:
                    raise AssertionError(f"[E] Failed to acquire ONNX for {model_id}_{precision}")
            else:
                raise ValueError(f"[E] Model {model_id}_{precision} {role} {self.pipeline_name} not found in registry")

        # Check if engine needs rebuilding
        engine = None
        if engine_path.exists():
            # Check metadata compatibility
            target_shapes = (
                self.model_instances[role].get_input_profile(True, opt_batch_size, opt_height, opt_width)
                if not role.endswith("text_encoder")
                else self.model_instances[role].get_input_profile(True, opt_batch_size)
            )

            is_compatible, reason = metadata_manager.check_engine_compatibility(
                engine_path=engine_path,
                target_shapes=target_shapes,
                static_shape=use_static_shape,
                extra_args=extra_args,
            )

            if is_compatible:
                engine = Engine(engine_path, precision, model_id, self.runtime_cache_path, self.cuda_graph_strategy)
                try:
                    if not self.low_vram:
                        engine.load()
                    logger.debug(f"Using cached engine: {engine_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load cached engine: {e}")
                    self.path_manager.delete_cached_engine_files(model_id, precision, shape_mode)
            else:
                logger.debug(f"Recompiling {model_id}_{precision}: {reason}")
                self.path_manager.delete_cached_engine_files(model_id, precision, shape_mode)

        # Build engine if needed
        if not engine_path.exists():
            if not onnx_path.exists():
                error_message = f"Cannot build engine: ONNX file {onnx_path} not found for {model_id}_{precision}"
                logger.error(error_message)
                raise ValueError(error_message)

            build_reason = (
                "No cached engine found"
                if not engine_path.with_suffix(".metadata.json").exists()
                else "Engine cache cleared"
            )
            logger.debug(f"Building {model_id}_{precision} engine ({shape_mode} shapes): {build_reason}")

            input_profile = (
                self.model_instances[role].get_input_profile(use_static_shape, opt_batch_size)
                if role.endswith("text_encoder")
                else self.model_instances[role].get_input_profile(
                    use_static_shape, opt_batch_size, opt_height, opt_width
                )
            )

            logger.debug(f"Building engine for path {engine_path}")
            engine = Engine(engine_path, precision, model_id, self.runtime_cache_path, self.cuda_graph_strategy)
            engine.build(
                onnx_path=str(onnx_path),
                input_profile=input_profile,
                static_shape=use_static_shape,
                extra_args=extra_args,
                verbose=self.verbose,
            )
            if not self.low_vram:
                engine.load()

        if engine is None:
            raise ValueError(f"[E] Engine not found for {model_id}_{precision}.")

        self.engines[role] = engine

    def load_engines(
        self,
        transformer_precision: str = "fp8",
        opt_batch_size: int = 1,
        opt_height: int = 512,
        opt_width: int = 512,
        shape_mode: str = "static",
        extra_args: Optional[dict[str]] = None,
    ):
        """
        Build and load TensorRT engines with smart caching.

        Args:
            transformer_precision: Precision configuration for the transformer
            opt_batch_size: Optimal batch size
            opt_height: Optimal image height
            opt_width: Optimal image width
            shape_mode: Shape mode ("dynamic" or "static") for all models (default: "static")
            extra_args: Additional polygraphy arguments
        """
        compute_capability = torch.cuda.get_device_capability(self.device)
        if transformer_precision == "fp8" and compute_capability < (8, 9):
            logger.error(
                f"{transformer_precision} transformer precision is not supported on device with compute capability {compute_capability} < (8, 9). "
                "Proceeding, but expect errors. Please try with bf16 precision instead."
            )
        elif transformer_precision == "fp4" and compute_capability < (12, 0):
            logger.error(
                f"{transformer_precision} transformer precision is not supported on device with compute capability {compute_capability} < (12, 0). "
                "Proceeding, but expect errors. Please try with bf16 precision instead."
            )

        # If engines are already loaded, clean up state first
        if self.engines:
            logger.info("Detected existing engines, cleaning up...")
            self.cleanup()

        assert transformer_precision in model_registry.get_available_precisions(self.pipeline_name, "transformer"), (
            f"Invalid precision for transformer: {transformer_precision}, options: {model_registry.get_available_precisions(self.pipeline_name, 'transformer')}"
        )

        self.precision_config["transformer"] = transformer_precision

        assert shape_mode in [
            "static",
            "dynamic",
        ], "shape_mode must be either 'static' or 'dynamic'"
        shape_config = {role: shape_mode for role in model_registry.get_pipeline_roles(self.pipeline_name)}

        # Validate shape_config
        for role, _ in shape_config.items():
            assert role in model_registry.get_pipeline_roles(self.pipeline_name), (
                f"Invalid role in shape_config: {role}, options: {model_registry.get_pipeline_roles(self.pipeline_name)}"
            )

        # VAE Decoder set to static shape to reduce VRAM usage
        if shape_mode == "dynamic":
            logger.info("Setting VAE Decoder to static shape to reduce VRAM usage")
            shape_config["vae_decoder"] = "static"

        logger.info(f"Shape configuration: {shape_config}")
        self.shape_config = shape_config

        model_configs_with_shape = {
            role: (model_id, precision, self.shape_config[role])
            for role, (model_id, precision) in self._get_model_configs().items()
        }

        self.path_manager.set_pipeline_models(self.pipeline_name, model_configs_with_shape)
        self.initialize_models()

        # Process each model
        for role, (model_id, precision, shape_mode) in model_configs_with_shape.items():
            self.build_and_load_engine(
                role,
                model_id,
                precision,
                shape_mode,
                opt_batch_size,
                opt_height,
                opt_width,
                extra_args,
            )

        if self.verbose:
            logger.info(f"\nAll engines loaded for {self.pipeline_name}")
            self.path_manager.print_cache_summary()

        logger.info("Activating engines...")
        jit_times = self.activate_engines() if not self.low_vram else {}
        logger.info("Engines activated successfully")
        return jit_times

    def refresh_engines(
        self,
        opt_batch_size: int = 1,
        opt_height: int = 512,
        opt_width: int = 512,
        extra_args: Optional[dict[str]] = None,
    ):
        """Check if engines need recompilation due to shape changes and refresh them."""
        if not self.engines or not self.shape_config:
            raise ValueError("No engines loaded, cannot refresh engines, please call load_engines first")

        engines_to_refresh = []
        model_configs = self._get_model_configs()

        # Check which engines need refreshing
        for role, (model_id, precision) in model_configs.items():
            engine_path = self.path_manager.get_engine_path(model_id, precision, self.shape_config[role])
            use_static_shape = self.shape_config[role] == "static"

            if engine_path.exists():
                target_shapes = (
                    self.model_instances[role].get_input_profile(True, opt_batch_size, opt_height, opt_width)
                    if not role.endswith("text_encoder")
                    else self.model_instances[role].get_input_profile(True, opt_batch_size)
                )

                is_compatible, reason = metadata_manager.check_engine_compatibility(
                    engine_path=engine_path,
                    target_shapes=target_shapes,
                    static_shape=use_static_shape,
                    extra_args=extra_args,
                )

                if not is_compatible:
                    logger.info(f"Engine {role} needs refresh: {reason}")
                    engines_to_refresh.append(role)

        if not engines_to_refresh:
            return False

        # Calculate current and new max workspace requirements
        current_max_workspace = None if self.low_vram else self.calculate_max_device_memory()

        # Rebuild any engines for which a shape change was detected
        for role in engines_to_refresh:
            logger.debug(f"Rebuilding engine: {role}")
            model_id, precision = model_configs[role]
            self.build_and_load_engine(
                role,
                model_id,
                precision,
                self.shape_config[role],
                opt_batch_size,
                opt_height,
                opt_width,
                extra_args,
            )

        # In Low-VRAM mode, can terminate here
        if self.low_vram:
            return True

        new_max_workspace = self.calculate_max_device_memory()

        # Check if workspace memory needs to be reallocated
        workspace_changed = new_max_workspace > current_max_workspace

        if workspace_changed:
            logger.info(f"Workspace memory needs refresh: {new_max_workspace} > {current_max_workspace}")

            # Free old shared memory
            if self.shared_device_memory is not None:
                logger.debug("[MEMORY] Freeing old shared workspace memory")
                cudart.cudaFree(self.shared_device_memory)

            # Allocate new shared memory
            logger.debug(f"[MEMORY] Allocating new shared workspace: {new_max_workspace / (1024**3):.3f} GB")
            _, self.shared_device_memory = cudart.cudaMalloc(new_max_workspace)

            # Activate all engines with new shared memory
            for role, engine in self.engines.items():
                if role in engines_to_refresh:
                    # New engines need full activation
                    logger.debug(f"Activating refreshed engine: {role}")
                    engine.activate(device_memory=self.shared_device_memory)
                else:
                    # Existing engines just need context update
                    engine.reactivate(self.shared_device_memory)
        else:
            # Activate new engines
            for role in engines_to_refresh:
                # Activate with existing shared memory
                self.engines[role].activate(device_memory=self.shared_device_memory)

        return True

    def load_resources(self, batch_size: int = 1, height: int = 512, width: int = 512) -> None:
        """
        Allocate buffers and stream for inference

        Args:
            batch_size: The batch size to use for inference
            height: The height of the generated images
            width: The width of the generated images
        """
        logger.debug(f"Loading resources for {width}x{height} resolution and {batch_size} batch size...")

        # Initialize CUDA stream
        if self.stream is None:
            self.stream = cudart.cudaStreamCreate()[1]

        deallocate_existing = False

        # Check if reallocation is necessary
        if (
            self.current_shapes
            and self.current_shapes.get("batch_size") == batch_size
            and self.current_shapes.get("height") == height
            and self.current_shapes.get("width") == width
        ):
            logger.debug("Resources already allocated")
            return

        elif self.current_shapes:
            logger.info("Detected a shape change, reallocating resources")
            deallocate_existing = True

        # Allocate tensors for each engine
        for model_name, engine in self.engines.items():
            if model_name.endswith("text_encoder"):
                shape_dict = self.model_instances[model_name].get_shape_dict(batch_size)
            else:
                shape_dict = self.model_instances[model_name].get_shape_dict(batch_size, height, width)

            # If low VRAM mode is enabled, store the shape dict but don't allocate buffers
            if self.low_vram:
                self.shape_dicts[model_name] = shape_dict
                continue

            if deallocate_existing:
                engine.deallocate_buffers()

            # Use engine's allocate_buffers method
            engine.allocate_buffers(shape_dict, device=self.device)

        logger.debug("Resources loaded successfully")

        # Update current shapes after successful allocation
        self.current_shapes = {
            "batch_size": batch_size,
            "height": height,
            "width": width,
        }

    def encode_prompt(
        self,
        prompt: str,
        model_name: str = "clip_text_encoder",
        pooled_output: bool = False,
    ) -> torch.Tensor:
        """Encode text prompt using specified text encoder"""
        logger.debug(f"Encoding prompt with {model_name}")

        # Check if tokenizers are initialized
        if self.tokenizer is None or self.tokenizer2 is None:
            raise ValueError("Tokenizers not initialized")

        # Select appropriate tokenizer and max length
        if model_name == "t5_text_encoder":
            tokenizer = self.tokenizer2
            max_sequence_length = self.model_params.T5_SEQUENCE_LENGTH
        else:  # clip_text_encoder
            tokenizer = self.tokenizer
            max_sequence_length = self.model_params.CLIP_SEQUENCE_LENGTH

        # Ensure prompt is a list for batch processing
        if isinstance(prompt, str):
            prompt = [prompt]

        def tokenize_and_encode(prompt_batch, max_sequence_length):
            # Tokenize input
            text_input_ids = (
                tokenizer(
                    prompt_batch,
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True,
                    return_overflowing_tokens=False,
                    return_length=False,
                    return_tensors="pt",
                )
                .input_ids.type(torch.int32)
                .to(self.device)
            )

            # Check for truncation and warn if necessary
            untruncated_ids = (
                tokenizer(prompt_batch, padding="longest", return_tensors="pt")
                .input_ids.type(torch.int32)
                .to(self.device)
            )

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because `max_sequence_length` is set to "
                    f"{max_sequence_length} tokens: {removed_text}"
                )

            # Run inference through the engine
            if model_name in self.engines:
                # Use TensorRT engine for inference
                engine_inputs = {"input_ids": text_input_ids}
                if self.enable_timing:

                    def run_model():
                        return self.run_engine(model_name, engine_inputs)

                    outputs, _ = self._record_cuda_timing(model_name, run_model)
                else:
                    outputs = self.run_engine(model_name, engine_inputs)

                if pooled_output and model_name == "clip_text_encoder":
                    # Return pooled embeddings for CLIP
                    text_encoder_output = outputs.get("pooled_embeddings", None)
                else:
                    # Return last hidden states
                    text_encoder_output = outputs.get("text_embeddings", None)

                if text_encoder_output is None:
                    # Fallback: try to get any tensor output
                    output_keys = list(outputs.keys())
                    if output_keys:
                        text_encoder_output = outputs[output_keys[0]]
                        if self.verbose:
                            logger.warning(f"Using fallback output key: {output_keys[0]}")
                    else:
                        raise RuntimeError(f"No valid output found from {model_name} engine")

                return text_encoder_output.clone()
            else:
                raise ValueError(f"Unknown text encoder: {model_name}")

        return tokenize_and_encode(prompt, max_sequence_length)

    def denoise_latents(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Run the denoising process through the transformer"""
        batch_size, height, width = (
            self.current_shapes["batch_size"],
            self.current_shapes["height"],
            self.current_shapes["width"],
        )
        latent_height, latent_width = height // 8, width // 8

        sigmas = np.linspace(1.0, 1 / self.num_inference_steps, self.num_inference_steps)
        image_seq_len = (latent_height // 2) * (latent_width // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        # Set timesteps
        self.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=self.device)
        timesteps = self.scheduler.timesteps

        # Prepare latents
        num_channels_latents = self.model_instances["transformer"].config["in_channels"] // 4
        latents = self._pack_latents(latents, batch_size, num_channels_latents, latent_height, latent_width)

        # Prepare image and text IDs
        img_ids = self._prepare_latent_image_ids(latent_height, latent_width, latents.dtype, self.device)
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=self.device, dtype=latents.dtype)

        # Handle guidance
        guidance = None
        if self.model_instances["transformer"].config["guidance_embeds"]:
            guidance = torch.full([1], self.guidance_scale, device=self.device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])

        # Determine autocast settings
        do_autocast = self.precision_config.get("transformer", "bf16") == "fp16"

        # Denoising loop
        with torch.autocast(self.device, enabled=do_autocast):
            for step_index, timestep in tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising"):
                if step_index % 5 == 0:
                    logger.debug(f"Denoising step {step_index}/{len(timesteps)}")

                # Prepare model inputs
                timestep_inp = timestep.expand(latents.shape[0]).to(latents.dtype)
                params = {
                    "hidden_states": latents,
                    "timestep": timestep_inp / 1000,
                    "pooled_projections": pooled_prompt_embeds,
                    "encoder_hidden_states": prompt_embeds,
                    "txt_ids": txt_ids.float(),
                    "img_ids": img_ids.float(),
                }

                if guidance is not None:
                    params.update({"guidance": guidance})

                # Predict the noise residual using TensorRT engine
                if "transformer" in self.engines:
                    if self.enable_timing:

                        def run_model():
                            return self.run_engine("transformer", params)  # noqa: B023

                        outputs, _ = self._record_cuda_timing("transformer", run_model)
                    else:
                        outputs = self.run_engine("transformer", params)
                    noise_pred = outputs["latent"]
                else:
                    raise ValueError("Transformer engine not available")

                # Step the scheduler
                latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        # Unpack latents
        vae_scale_factor = (
            2 ** (len(self.model_instances["vae_decoder"].config["block_out_channels"]))
            if self.model_instances["vae_decoder"] is not None
            else 16
        )
        return self._unpack_latents(latents, height, width, vae_scale_factor)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images using VAE"""
        # Apply VAE scaling and shift factors
        latents = (latents / self.model_instances["vae_decoder"].config.scaling_factor) + self.model_instances[
            "vae_decoder"
        ].config.shift_factor

        # Run VAE decoder engine
        if "vae_decoder" in self.engines:
            if self.enable_timing:

                def run_model():
                    return self.run_engine("vae_decoder", {"latent": latents})

                outputs, _ = self._record_cuda_timing("vae_decoder", run_model)
            else:
                outputs = self.run_engine("vae_decoder", {"latent": latents})
            images = outputs["images"]
        else:
            raise ValueError("VAE decoder engine not available")

        return images

    @torch.no_grad()
    def infer(
        self,
        prompt: str,
        save_path: str = None,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Run full inference pipeline

        Args:
            prompt: The prompt to use for inference
            save_path: The path to save the generated images
            batch_size: The batch size to use for inference
            height: The height of the generated images
            width: The width of the generated images
            seed: The seed to use for inference
            num_inference_steps: The number of inference steps to use
            guidance_scale: The guidance scale to use for inference
        """
        # Resolution multiples of 16 are supported
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("Height and width must be multiples of 16")

        # Reset timing data for this inference run
        self.reset_timing_data()

        logger.debug("Starting Flux inference...")
        logger.debug(f"Prompt: '{prompt}'")

        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            self.generator = torch.Generator(device=self.device)

        if num_inference_steps is not None:
            self.num_inference_steps = num_inference_steps

        if guidance_scale is not None:
            self.guidance_scale = guidance_scale

        # Refresh engines if shape has changed
        self.refresh_engines(batch_size, height, width)

        # Ensure resources are loaded if the shape has changed
        self.load_resources(batch_size, height, width)

        start_time = time.time()

        # Encode prompts
        with self.model_memory_manager("t5_text_encoder", self.low_vram):
            t5_embeds = self.encode_prompt(prompt, "t5_text_encoder")

        # Get pooled embeddings from CLIP
        with self.model_memory_manager("clip_text_encoder", self.low_vram):
            pooled_embeds = self.encode_prompt(prompt, "clip_text_encoder", pooled_output=True)

        # Initialize latents
        batch_size, height, width = (
            self.current_shapes["batch_size"],
            self.current_shapes["height"],
            self.current_shapes["width"],
        )
        self.timing_data.height = height
        self.timing_data.width = width
        self.timing_data.batch_size = batch_size
        self.timing_data.num_inference_steps = self.num_inference_steps
        self.timing_data.guidance_scale = self.guidance_scale

        latent_height, latent_width = (
            height // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO,
            width // self.model_params.VAE_SPATIAL_COMPRESSION_RATIO,
        )

        num_channels_latents = self.model_instances["transformer"].config.in_channels // 4

        latents = torch.randn(
            batch_size,
            num_channels_latents,
            latent_height,
            latent_width,
            device=self.device,
            dtype=torch.bfloat16,
            generator=self.generator,
        )

        # Denoise latents
        with self.model_memory_manager("transformer", self.low_vram):
            latents = self.denoise_latents(latents, t5_embeds, pooled_embeds)
        del t5_embeds, pooled_embeds

        # Decode to images
        with self.model_memory_manager("vae_decoder", self.low_vram):
            images_gpu = self.decode_latents(latents)
        del latents

        end_time = time.time()

        # Record total inference time
        self.timing_data.total_inference_time = (end_time - start_time) * 1000  # Convert to ms

        images = ((images_gpu + 1) * 255 / 2).clamp(0, 255).permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
        del images_gpu
        gc.collect()
        torch.cuda.empty_cache()

        image_paths = []
        if save_path is not None:
            # Save image
            for i in range(images.shape[0]):
                image_path = os.path.join(save_path, f"flux_demo_{i + 1}.png")
                logger.info(f"Saving image {i + 1} / {images.shape[0]} to: {image_path} with shape {images[i].shape}")
                pil = Image.fromarray(images[i])
                pil.save(image_path)
                image_paths.append(image_path)

        logger.debug(f"Inference completed in {end_time - start_time:.2f}s")

        return images, image_paths

    def cleanup(self):
        """Clean up all resources"""
        if hasattr(self, "generator"):
            del self.generator

        super().cleanup()

    # Copied from https://github.com/huggingface/diffusers/blob/v0.30.1/src/diffusers/pipelines/flux/pipeline_flux.py#L436
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        """Pack latents for Flux transformer input"""
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    # Copied from https://github.com/huggingface/diffusers/blob/v0.30.1/src/diffusers/pipelines/flux/pipeline_flux.py#L444
    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        """Unpack latents from transformer output"""
        batch_size, num_patches, channels = latents.shape
        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, height * 2, width * 2)
        return latents

    # Copied from https://github.com/huggingface/diffusers/blob/v0.30.1/src/diffusers/pipelines/flux/pipeline_flux.py#L421
    @staticmethod
    def _prepare_latent_image_ids(height, width, dtype, device):
        """Prepare latent image IDs for Flux"""
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        (
            latent_image_id_height,
            latent_image_id_width,
            latent_image_id_channels,
        ) = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        return latent_image_ids.to(device=device, dtype=dtype)
