#!/usr/bin/env python3
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

"""
A basic demonstration of the Flux text-to-image pipeline using the TensorRT-RTX framework.
This demo focuses on the happy-path workflow for generating images from text prompts.
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    from pipelines.flux_pipeline import FluxPipeline
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from pipelines.flux_pipeline import FluxPipeline


logger = logging.getLogger("rtx_demo.flux1.dev.flux_demo")


def main():
    parser = argparse.ArgumentParser(description="Simple Flux text-to-image demo")
    # Required arguments
    parser.add_argument("--hf-token", type=str, required=True, help="Hugging Face token")

    # Image Generation Parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default="A serene lake at sunset with mountains in the background",
        help="Text prompt for image generation",
    )
    parser.add_argument("--height", type=int, default=512, help="Image height (default: 512)")
    parser.add_argument("--width", type=int, default=512, help="Image width (default: 512)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator (default: 0)")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of denoising steps (default: 50)")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale (default: 3.5)")

    # Engine Generation, Caching, Memory Management, and Verbosity
    parser.add_argument(
        "--precision",
        type=str,
        default="fp8",
        help="Precision for the transformer model (default: fp8)",
        choices=["bf16", "fp8", "fp4"],
    )
    parser.add_argument("--enable-runtime-cache", action="store_true", help="Enable runtime caching")
    parser.add_argument("--low-vram", action="store_true", help="Enable low VRAM mode")
    parser.add_argument("--dynamic-shape", action="store_true", default=False, help="Enable dynamic-shape engines")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--cache-dir", type=str, default="./demo_cache", help="Cache directory for models (default: ./demo_cache)"
    )
    parser.add_argument(
        "--cache-mode", type=str, default="full", help="Cache mode (default: full)", choices=["full", "lean"]
    )
    args = parser.parse_args()

    try:
        pipeline = FluxPipeline(
            cache_dir=args.cache_dir,
            device="cuda",
            verbose=args.verbose,
            cache_mode=args.cache_mode,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            hf_token=args.hf_token,
            low_vram=args.low_vram,
            enable_runtime_cache=args.enable_runtime_cache,
        )

        # Print header and configuration
        logger.info("=" * 50)
        logger.info("Simple Flux Text-to-Image Demo")
        logger.info("=" * 50)
        logger.info(f"Prompt: '{args.prompt}'")
        logger.info(f"Transformer Precision: {args.precision}")
        logger.info(f"Resolution: {args.width}x{args.height}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Seed: {args.seed}")
        logger.info(f"Inference steps: {args.num_inference_steps}")
        logger.info(f"Guidance scale: {args.guidance_scale}")
        logger.info(f"Cache directory: {args.cache_dir}")
        logger.info(f"Low VRAM mode: {args.low_vram}")
        logger.info(f"Dynamic shape: {args.dynamic_shape}")
        logger.info(f"Runtime caching: {args.enable_runtime_cache}")
        logger.info(f"Cache mode: {args.cache_mode}")
        logger.info("")

        jit_times = pipeline.load_engines(
            transformer_precision=args.precision,
            opt_batch_size=args.batch_size,
            opt_height=args.height,
            opt_width=args.width,
            shape_mode="dynamic" if args.dynamic_shape else "static",
        )

        for model, jit_time in jit_times.items():
            logger.info(f"JIT Compilation + Execution Context Creation Time for {model}: {round(jit_time, 2)} seconds")

        pipeline.load_resources(
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
        )

        # Print memory usage summary
        if not args.low_vram:
            pipeline.print_gpu_vram_summary()

        # Run inference
        logger.info("Generating image...")
        logger.info(f"Running {args.num_inference_steps} denoising steps...")

        save_dir = "."
        pipeline.infer(
            prompt=args.prompt,
            save_path=save_dir,
            height=args.height,
            width=args.width,
            seed=args.seed,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
    finally:
        # Cleanup
        if "pipeline" in locals():
            pipeline.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
