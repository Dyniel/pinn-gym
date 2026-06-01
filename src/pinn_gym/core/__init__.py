"""Scientific core for pinn_gym.

Public surface used by :mod:`pinn_gym.pipeline`. Submodules can also be
imported directly for finer-grained access.
"""

from __future__ import annotations

from .audit import run_audit
from .material_pinn import MaterialPINNConfig
from .materials import (
    MaterialCard,
    load_material_card,
    preset_material_cards,
)
from .mesh_quality import (
    audit_stl_directory,
    audit_stl_mesh,
    write_mesh_quality_report,
)
from .physics import IMPACT_ENERGY_J, PhysicsConfig
from .reviewer_pack import build_pack, verify_pack
from .sr_benchmark import (
    SRBuildConfig,
    SREvalConfig,
    build_sr_dataset,
    evaluate_sr_run,
    train_sr_models,
)
from .sr_plots import render_sr_figures
from .stl import export_design_stl

__all__ = [
    "IMPACT_ENERGY_J",
    "MaterialCard",
    "MaterialPINNConfig",
    "PhysicsConfig",
    "SRBuildConfig",
    "SREvalConfig",
    "audit_stl_directory",
    "audit_stl_mesh",
    "build_pack",
    "build_sr_dataset",
    "evaluate_sr_run",
    "export_design_stl",
    "load_material_card",
    "preset_material_cards",
    "render_sr_figures",
    "run_audit",
    "train_sr_models",
    "verify_pack",
    "write_mesh_quality_report",
]
