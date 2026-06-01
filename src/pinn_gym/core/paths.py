"""Project path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the pinn_gym project root.

    Resolution order:
    1. ``PINN_GYM_ROOT`` for new runs and CI.
    2. ``POLMI_ROOT`` for archived Slurm jobs and reviewer artefacts.
    3. Walk up from this file looking for ``pyproject.toml``.
    """

    for var in ("PINN_GYM_ROOT", "POLMI_ROOT"):
        env_root = os.environ.get(var)
        if env_root:
            return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
