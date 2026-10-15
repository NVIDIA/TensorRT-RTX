# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint downloads and weightless engine cache invalidation."""

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file
from weight_test_utils import update_weight_file


@pytest.mark.unit
@pytest.mark.parametrize("role", ["clip_text_encoder", "t5_text_encoder", "transformer", "vae_decoder"])
def test_acquisition_downloads_only_component_config_and_safetensors(native_builder, monkeypatch, tmp_path, role):
    import huggingface_hub

    recorded = {}

    def download(**kwargs):
        recorded.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    result = native_builder.download_checkpoint(role, "test-token")
    _, folder, prefix = native_builder.MODEL_SPECS[role]
    assert result == (tmp_path / folder, None)
    assert recorded["repo_id"] == "black-forest-labs/FLUX.1-dev"
    assert recorded["revision"] == native_builder.CHECKPOINT_REVISIONS[native_builder.REPOSITORY]
    assert re.fullmatch(r"[0-9a-f]{40}", recorded["revision"])
    assert recorded["token"] == "test-token"
    assert recorded["allow_patterns"] == [
        f"{folder}/config.json",
        f"{folder}/{prefix}.safetensors",
        f"{folder}/{prefix}.safetensors.index.json",
        f"{folder}/{prefix}-*-of-*.safetensors",
    ]


@pytest.mark.unit
@pytest.mark.parametrize("precision", ["fp8", "fp4"])
def test_quantized_acquisition_uses_direct_checkpoint(native_builder, monkeypatch, tmp_path, precision):
    import huggingface_hub

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    directory, weights = native_builder.download_checkpoint("transformer", "test-token", precision)
    repository, filename = native_builder.QUANTIZED_TRANSFORMERS[precision]
    assert directory == tmp_path / "transformer"
    assert weights == tmp_path / filename
    assert calls == [
        {
            "repo_id": native_builder.REPOSITORY,
            "revision": native_builder.CHECKPOINT_REVISIONS[native_builder.REPOSITORY],
            "filename": "transformer/config.json",
            "token": "test-token",
        },
        {
            "repo_id": repository,
            "revision": native_builder.CHECKPOINT_REVISIONS[repository],
            "filename": filename,
            "token": "test-token",
        },
    ]
    assert all(re.fullmatch(r"[0-9a-f]{40}", call["revision"]) for call in calls)


@pytest.mark.unit
def test_precision_mismatch_rejected_before_build(native_builder, tmp_path):
    save_file({"linear.weight": np.ones((16, 16), np.float32)}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="Requested fp4, but checkpoint weights are bf16"):
        native_builder.build_weightless_engine(tmp_path, "transformer", tmp_path / "unused.plan", precision="fp4")


@pytest.mark.unit
@pytest.mark.parametrize(
    "change", ["none", "config", "weights", "shape", "corrupt", "legacy", "missing", "refit", "profile"]
)
def test_weightless_engine_cache_invalidation(native_builder, monkeypatch, tmp_path, change):
    (tmp_path / "config.json").write_text("{}")
    save_file({"weight": np.ones(16, np.float32)}, tmp_path / "model.safetensors")
    builds = []

    class Builder:
        def __init__(self, logger):
            pass

        def create_builder_config(self):
            return SimpleNamespace(set_flag=lambda flag: None, add_optimization_profile=lambda profile: 0)

        def create_optimization_profile(self):
            return SimpleNamespace(set_shape=lambda *args: None)

        def build_serialized_network(self, network, config):
            builds.append(True)
            return b"test-plan"

    monkeypatch.setattr(native_builder.trt, "Builder", Builder)
    monkeypatch.setattr(
        native_builder,
        "create_network",
        lambda *args: SimpleNamespace(
            net=SimpleNamespace(num_inputs=1, get_input=lambda i: SimpleNamespace(name="input_ids")),
            weights=SimpleNamespace(payload_bytes=0),
        ),
    )
    plan = tmp_path / "native.plan"
    checkpoint, first_id = native_builder.build_weightless_engine(tmp_path, "clip", plan)
    assert checkpoint.payload_bytes == 0
    if change == "config":
        (tmp_path / "config.json").write_text('{"changed": true}')
    elif change == "weights":
        with update_weight_file(tmp_path / "model.safetensors"):
            save_file({"weight": np.zeros(16, np.float32)}, tmp_path / "model.safetensors")
    elif change == "corrupt":
        plan.write_bytes(b"corrupt-plan")
    elif change == "legacy":
        plan.with_suffix(".checkpoint.json").write_text('{"identity": {"format": "onnx"}}')
    elif change == "missing":
        plan.with_suffix(".checkpoint.json").unlink()
    elif change == "refit":
        digest = native_builder.file_digest
        monkeypatch.setattr(
            native_builder,
            "file_digest",
            lambda path: "changed-refit" if Path(path).name == "flux_network.py" else digest(path),
        )
    checkpoint, second_id = native_builder.build_weightless_engine(
        tmp_path,
        "clip",
        plan,
        batch=2 if change == "shape" else 1,
        input_profiles={"input_ids": [(1, 77), (1, 77), (4, 77)]} if change == "profile" else None,
    )
    assert len(builds) == (1 if change == "none" else 2)
    assert checkpoint.payload_bytes == 0
    assert (first_id != second_id) == (change in {"config", "weights", "refit"})
