#!/usr/bin/env python3
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
A basic demonstration of FLUX.2 [klein] 4B text-to-image generation with an FP8 transformer compiled by
Torch-TensorRT-RTX in two explicit steps: torch.export followed by torch_tensorrt.dynamo.compile.
"""

import argparse
import logging
import sys

from flux2_pipeline import DEFAULT_NUM_INFERENCE_STEPS, IMAGE_SIZE_MULTIPLE, Flux2FP8TorchTensorRTPipeline

logger = logging.getLogger("rtx_demo.flux2.flux2_demo")


def configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    rtx_logger = logging.getLogger("rtx_demo")
    rtx_logger.handlers.clear()
    rtx_logger.addHandler(handler)
    rtx_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    rtx_logger.propagate = False


def weight_streaming_budget(value: str) -> float:
    budget = float(value)
    if not 0.0 <= budget <= 1.0:
        raise argparse.ArgumentTypeError("must be in the range [0, 1]")
    return budget


def main():
    parser = argparse.ArgumentParser(
        description="Simple FLUX.2 [klein] 4B text-to-image demo with an FP8 Torch-TensorRT-RTX transformer"
    )
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (optional)")

    # Image Generation Parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic photograph of an astronaut riding a horse on Mars",
        help="Text prompt for image generation",
    )
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator (default: 0)")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help=f"Number of denoising steps (default: {DEFAULT_NUM_INFERENCE_STEPS})",
    )

    # Engine Generation, Caching, Memory Management, and Verbosity
    parser.add_argument(
        "--enable-engine-cache",
        action="store_true",
        help="Cache built TensorRT-RTX engines in the cache directory and reuse them on later runs",
    )
    parser.add_argument(
        "--enable-runtime-cache",
        action="store_true",
        help="Cache TensorRT-RTX JIT-compiled kernels in the cache directory and reuse them on later runs",
    )
    parser.add_argument(
        "--weight-streaming-budget",
        type=weight_streaming_budget,
        default=None,
        help="Enable weight streaming with a GPU budget in [0, 1] of the streamable transformer weights; "
        "0 streams the most weights",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./demo_cache", help="Cache directory for models (default: ./demo_cache)"
    )
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for generated images (default: .)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.height % IMAGE_SIZE_MULTIPLE or args.width % IMAGE_SIZE_MULTIPLE:
        parser.error(f"--height and --width must be multiples of {IMAGE_SIZE_MULTIPLE}")

    configure_logging(args.verbose)

    try:
        pipeline = Flux2FP8TorchTensorRTPipeline(
            cache_dir=args.cache_dir,
            device="cuda",
            hf_token=args.hf_token,
            enable_engine_cache=args.enable_engine_cache,
            enable_runtime_cache=args.enable_runtime_cache,
        )

        # Print header and configuration
        logger.info("=" * 50)
        logger.info("Simple FLUX.2 [klein] 4B FP8 Text-to-Image Demo (Torch-TensorRT-RTX)")
        logger.info("=" * 50)
        logger.info(f"Prompt: '{args.prompt}'")
        logger.info(f"Resolution: {args.width}x{args.height}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Seed: {args.seed}")
        logger.info(f"Inference steps: {args.num_inference_steps}")
        logger.info(f"Cache directory: {args.cache_dir}")
        logger.info(f"Engine cache: {args.enable_engine_cache}")
        logger.info(f"Runtime cache: {args.enable_runtime_cache}")
        logger.info(f"Weight streaming budget: {args.weight_streaming_budget}")
        logger.info("")

        load_time = pipeline.load()
        logger.info(f"Pipeline load time: {load_time:.2f} seconds")

        compile_times = pipeline.compile_transformer(
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            weight_streaming_budget=args.weight_streaming_budget,
        )
        logger.info(f"torch.export time: {compile_times['export']:.2f} seconds")
        logger.info(f"torch_tensorrt.dynamo.compile time: {compile_times['compile']:.2f} seconds")
        logger.info(
            f"TensorRT-RTX JIT compilation + first transformer execution time: {compile_times['jit']:.2f} seconds"
        )

        # Run inference
        logger.info("Generating image...")
        logger.info(f"Running {args.num_inference_steps} denoising steps...")
        _, seconds = pipeline.infer(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            save_dir=args.output_dir,
        )
        logger.info(f"Image generation time: {seconds:.2f} seconds")
    finally:
        # Cleanup
        if "pipeline" in locals():
            pipeline.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
