# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download FLUX configs and safetensors for native weightless engine builds and refitting."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from flux_network import LOGGER, CheckpointWeights, create_network, trt

REPOSITORY = "black-forest-labs/FLUX.1-dev"
# Keep configs and weights compatible with the native network definitions.
CHECKPOINT_REVISIONS = {
    REPOSITORY: "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
    "black-forest-labs/FLUX.1-dev-FP8": "2fcc6a7ddee78972c8834226b37a09cebff1b6de",
    "black-forest-labs/FLUX.1-dev-NVFP4": "c4b5f59eda28dac06bacc36693e9c4409ca6a8e6",
}
MODEL_SPECS = {
    "clip_text_encoder": ("clip", "text_encoder", "model"),
    "t5_text_encoder": ("t5", "text_encoder_2", "model"),
    "transformer": ("transformer", "transformer", "diffusion_pytorch_model"),
    "vae_decoder": ("vae", "vae", "diffusion_pytorch_model"),
}
QUANTIZED_TRANSFORMERS = {
    "fp8": ("black-forest-labs/FLUX.1-dev-FP8", "flux1-dev-fp8.safetensors"),
    "fp4": ("black-forest-labs/FLUX.1-dev-NVFP4", "flux1-dev-nvfp4.safetensors"),
}


def download_checkpoint(role, token=None, precision="bf16"):
    from huggingface_hub import hf_hub_download, snapshot_download

    _, folder, prefix = MODEL_SPECS[role]
    if precision != "bf16":
        if role != "transformer" or precision not in QUANTIZED_TRANSFORMERS:
            raise ValueError(f"Unsupported checkpoint precision: {role}/{precision}")
        config = hf_hub_download(
            repo_id=REPOSITORY,
            revision=CHECKPOINT_REVISIONS[REPOSITORY],
            filename=f"{folder}/config.json",
            token=token,
        )
        repository, filename = QUANTIZED_TRANSFORMERS[precision]
        weight_file = hf_hub_download(
            repo_id=repository, revision=CHECKPOINT_REVISIONS[repository], filename=filename, token=token
        )
        return Path(config).parent, Path(weight_file)
    snapshot = snapshot_download(
        repo_id=REPOSITORY,
        revision=CHECKPOINT_REVISIONS[REPOSITORY],
        token=token,
        allow_patterns=[
            f"{folder}/config.json",
            f"{folder}/{prefix}.safetensors",
            f"{folder}/{prefix}.safetensors.index.json",
            f"{folder}/{prefix}-*-of-*.safetensors",
        ],
    )
    return Path(snapshot) / folder, None


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_checkpoint_identity(directory, weights):
    # HF snapshot paths identify immutable revisions. Local file metadata also
    # invalidates replaced payloads without reading their potentially large ranges.
    files = []
    for path in sorted({entry["path"] for entry in weights.entries.values()}):
        stat = path.stat()
        files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    descriptor = {
        "config": file_digest(Path(directory) / "config.json"),
        "files": files,
        "refit": file_digest(Path(__file__).with_name("flux_network.py")),
        "quantization": file_digest(Path(__file__).with_name("flux_checkpoint_mapping.py")),
    }
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()


def build_weightless_engine(
    directory,
    role,
    plan_path,
    batch=1,
    height=512,
    width=512,
    text_length=None,
    weight_file=None,
    precision=None,
    input_profiles=None,
):
    directory, plan_path = Path(directory), Path(plan_path)
    weights = CheckpointWeights(directory, weight_file)
    quantized = any(weights.dtype(name) == trt.fp8 for name in weights.entries if name.endswith(".weight"))
    actual_precision = "fp4" if weights.packed else "fp8" if quantized else "bf16"
    if precision is not None and actual_precision != precision:
        raise ValueError(f"Requested {precision}, but checkpoint weights are {actual_precision}")
    source_id = get_checkpoint_identity(directory, weights)
    identity = {
        "format": "flux_safetensors_v1",
        "checkpoint": source_id,
        "role": role,
        "precision": actual_precision,
        "shapes": [batch, height, width, text_length],
        "input_profiles": json.loads(json.dumps(input_profiles)),
        "tensorrt": trt.__version__,
        "architecture": file_digest(Path(__file__).with_name("flux_network.py")),
        "quantization": file_digest(Path(__file__).with_name("flux_checkpoint_mapping.py")),
        "builder": file_digest(__file__),
    }
    manifest_path = plan_path.with_suffix(".checkpoint.json")
    if plan_path.is_file() and manifest_path.is_file():
        try:
            saved = json.loads(manifest_path.read_text())
            if saved["identity"] == identity and saved["plan_sha256"] == file_digest(plan_path):
                return weights, source_id
        except (OSError, ValueError, KeyError, TypeError):
            pass

    builder = trt.Builder(LOGGER)
    network = create_network(
        builder, directory, role, True, batch, height, width, text_length, weight_file, input_profiles
    )
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)
    config.set_flag(trt.BuilderFlag.STRIP_PLAN)
    if input_profiles:
        inputs = {network.net.get_input(i).name for i in range(network.net.num_inputs)}
        if inputs != set(input_profiles):
            raise ValueError("Input profiles must match the native network inputs")
        profile = builder.create_optimization_profile()
        for name, shapes in input_profiles.items():
            profile.set_shape(name, *shapes)
        if config.add_optimization_profile(profile) < 0:
            raise RuntimeError("Failed to add native network optimization profile")
    plan = builder.build_serialized_network(network.net, config)
    if plan is None:
        raise RuntimeError(f"Native weightless build failed for {role}")
    if network.weights.payload_bytes:
        raise RuntimeError("Weightless construction unexpectedly read checkpoint payloads")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=plan_path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(plan)
        os.replace(temporary, plan_path)
        saved = {"identity": identity, "plan_sha256": file_digest(plan_path)}
        with tempfile.NamedTemporaryFile(mode="w", dir=plan_path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(saved, output, indent=2)
        os.replace(temporary, manifest_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return weights, source_id
