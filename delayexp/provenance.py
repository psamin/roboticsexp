"""Record what produced a run: command, code revision, package versions, hardware.

    uv run python -m delayexp.provenance > runs/<run>/provenance.json
"""

import datetime
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PACKAGES = [
    "playground",
    "brax",
    "jax",
    "jax-cuda12-plugin",
    "flax",
    "optax",
    "orbax-checkpoint",
    "mujoco",
    "mujoco-mjx",
    "warp-lang",
    "numpy",
    "mediapy",
]

ENV_VARS = [
    "MUJOCO_GL",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "JAX_DEFAULT_MATMUL_PRECISION",
    "CUDA_VISIBLE_DEVICES",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_PARTITION",
]


def _run(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _playground_source():
    # uv records the git URL and resolved commit of a VCS install in direct_url.json.
    try:
        text = importlib.metadata.distribution("playground").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    return json.loads(text) if text else None


def collect():
    return {
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "repo_git_rev": _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "repo_git_status": _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"]),
        "playground_source": _playground_source(),
        "packages": {name: _version(name) for name in PACKAGES},
        "env": {name: os.environ.get(name) for name in ENV_VARS},
        "gpus": _run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]),
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
