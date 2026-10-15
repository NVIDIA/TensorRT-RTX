# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public FLUX command-line options and generation flow."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "flux1.dev"
FLUX_DEMO = SOURCE / "flux_demo.py"


@pytest.mark.unit
@pytest.mark.parametrize(
    "arguments,precision",
    [
        ([], "fp8"),
        (["--weight-offload", "buffer"], "fp8"),
        (["--weight-offload", "vmm"], "fp8"),
        (["--weight-offload", "vmm", "--dynamic-shape"], "fp8"),
        (["--weight-offload", "buffer", "--dynamic-shape"], "fp8"),
        *[
            (["--weight-offload", mode, "--precision", precision], precision)
            for mode in ("buffer", "vmm")
            for precision in ("bf16", "fp8", "fp4")
        ],
    ],
)
def test_cli_generation_flow_and_precision_defaults(arguments, precision):
    script = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('flux_demo', sys.argv[1])
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
class Pipeline:
    def __init__(self, **kwargs):
        assert 'low_vram' not in kwargs
        print('FLOW=init')
    def cleanup(self): print('FLOW=cleanup')
    def load_engines(self, **kwargs):
        print('FLOW=load_engines')
        print('SELECTED_PRECISION=' + kwargs['transformer_precision'])
        print('SHAPE_MODE=' + kwargs['shape_mode'])
        return {}
    def load_resources(self, **kwargs): print('FLOW=load_resources')
    def print_gpu_vram_summary(self): print('FLOW=summary')
    def infer(self, **kwargs): print('FLOW=infer')
demo.load_pipeline_class = lambda: Pipeline
sys.argv = sys.argv[1:]
demo.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(SOURCE / "flux_demo.py"), "--hf-token", "test-token", *arguments],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert f"SELECTED_PRECISION={precision}" in result.stdout
    assert "SHAPE_MODE=" + ("dynamic" if "--dynamic-shape" in arguments else "static") in result.stdout
    assert [line for line in result.stdout.splitlines() if line.startswith("FLOW=")] == [
        "FLOW=init",
        "FLOW=load_engines",
        "FLOW=load_resources",
        "FLOW=summary",
        "FLOW=infer",
        "FLOW=cleanup",
    ]


@pytest.mark.unit
def test_cli_help_lists_generation_and_offload_options():
    result = subprocess.run(
        [sys.executable, str(SOURCE / "flux_demo.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    for option in ("--prompt", "--weight-offload", "--weight-offload-pinned-host"):
        assert option in result.stdout
    assert "--low-vram" not in result.stdout


@pytest.mark.unit
def test_cli_rejects_removed_low_vram_option():
    result = subprocess.run(
        [sys.executable, str(SOURCE / "flux_demo.py"), "--hf-token", "test-token", "--low-vram"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --low-vram" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.unit
def test_cli_rejects_pinned_host_without_offloading():
    result = subprocess.run(
        [
            sys.executable,
            str(FLUX_DEMO),
            "--hf-token",
            "local-test-token",
            "--weight-offload",
            "none",
            "--weight-offload-pinned-host",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "--weight-offload-pinned-host requires --weight-offload buffer or vmm" in result.stderr
    assert "local-test-token" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mode", ["buffer", "vmm"])
def test_cli_passes_pinned_host_option(monkeypatch, enabled, mode):
    spec = importlib.util.spec_from_file_location("flux_cli", FLUX_DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    received = {}

    def pipeline(**kwargs):
        received.update(kwargs)
        raise RuntimeError("stop before setup")

    monkeypatch.setattr(module, "load_pipeline_class", lambda: pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(FLUX_DEMO), "--hf-token", "local-test-token", "--weight-offload", mode]
        + (["--weight-offload-pinned-host"] if enabled else []),
    )
    with pytest.raises(RuntimeError, match="stop before setup"):
        module.main()
    assert received["weight_offload_pinned_host"] is enabled


@pytest.mark.unit
@pytest.mark.parametrize("environment_token", [None, "local-test-token"])
def test_flux_cli_requires_hf_token_argument(environment_token):
    environment = os.environ.copy()
    environment.pop("HF_TOKEN", None)
    if environment_token is not None:
        environment["HF_TOKEN"] = environment_token
    result = subprocess.run(
        [sys.executable, str(FLUX_DEMO)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "the following arguments are required: --hf-token" in result.stderr
