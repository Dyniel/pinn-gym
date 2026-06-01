"""Configuration schema for the pinn_gym pipeline.

A single YAML file drives every stage of the gym: candidate-pool generation,
PINN training, evaluation, plotting, and mesh audit. This module loads, fills
in defaults, validates, and exposes a strongly-typed view of that YAML.

The schema mirrors the dataclass configs already used internally by the
scientific core (``SRBuildConfig``, ``MaterialPINNConfig``, ``SREvalConfig``)
so that the YAML is a thin, documented surface and not a separate source of
truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


KNOWN_PRESETS: tuple[str, ...] = ("pa12", "pla", "petg", "tpu", "pa_cf")
KNOWN_STAGES: tuple[str, ...] = ("build", "train", "evaluate", "plots", "audit")
KNOWN_METHODS: tuple[str, ...] = (
    "random",
    "lightest",
    "pseudo_bootstrap",
    "mlp_softplus",
    "pinn_energy",
    "pinn_full",
)
KNOWN_FIGURES: tuple[str, ...] = (
    "curves_overlay",
    "nrmse_distribution",
    "energy_error",
    "violation_bars",
    "precision_at_k",
    "regret_at_k",
    "robustness_survival",
)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class RunSection(_Section):
    """Top-level orchestration knobs."""

    name: str = Field(
        default="run",
        description="Folder slug; the run is written to runs/<name>_<timestamp>/.",
    )
    seed: int = Field(default=20260520, description="Master RNG seed shared by every stage.")
    stages: list[Literal["build", "train", "evaluate", "plots", "audit"]] = Field(
        default_factory=lambda: list(KNOWN_STAGES),
        description="Stages to execute, in order. Use a subset to resume.",
    )
    device: Literal["cuda", "cpu"] = Field(
        default="cuda",
        description="Compute device for train/evaluate stages. CPU is supported but slow.",
    )
    output_root: str | None = Field(
        default=None,
        description="Override runs/ root. None means <repo>/runs.",
    )


class MaterialsSection(_Section):
    """Material cards under test."""

    presets: list[str] = Field(
        default_factory=lambda: list(KNOWN_PRESETS),
        description="Built-in material presets. See pinn_gym materials --list.",
    )
    material_aware_crush_target: bool = Field(
        default=True,
        description=(
            "When True, target_min_crush_mm is derived per material from its failure_strain "
            "so brittle cards (PLA/PETG/PA-CF) get a physically reachable feasibility gate."
        ),
    )
    custom_cards: list[str] = Field(
        default_factory=list,
        description="Optional paths to JSON material cards (extends the built-in presets).",
    )

    @field_validator("presets")
    @classmethod
    def _check_presets(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("materials.presets must contain at least one preset")
        return value


class CandidatePoolSection(_Section):
    """Generation of material-aware candidate pools + declared oracle labels."""

    train_n: int = Field(default=4000, ge=1, description="Per-material training-pool size.")
    eval_n: int = Field(default=1200, ge=1, description="Per-material evaluation-pool size.")
    layers: int = Field(default=96, ge=2, description="Crush-front layer count for the oracle.")
    steps: int = Field(default=320, ge=8, description="Quasi-static integration steps per design.")
    max_displacement_mm: float = Field(default=50.0, gt=0)
    oracle_workers: int = Field(
        default=0,
        ge=0,
        description="Worker processes for the oracle. 0 = auto (min(cpu_count, 8)).",
    )
    dynamic_amplification: float = Field(default=1.16, gt=0)
    yield_scale: float = Field(default=0.08, gt=0)
    fixture_peak_force_limit_n: float = Field(default=3500.0, gt=0)
    target_min_crush_mm: float = Field(
        default=40.0,
        gt=0,
        description="Only used when materials.material_aware_crush_target is False.",
    )


class LossWeights(_Section):
    boundary: float = Field(default=0.5, ge=0)
    energy: float = Field(default=1.0, ge=0)
    peak: float = Field(default=0.2, ge=0)
    monotonicity: float = Field(default=0.5, ge=0)
    smoothness: float = Field(default=0.1, ge=0)


class TrainSection(_Section):
    """PINN / baseline training."""

    methods: list[str] = Field(
        default_factory=lambda: ["pinn_full", "pinn_energy", "mlp_softplus"],
        description="Models to train. Baselines (random, lightest, pseudo_bootstrap) need no training.",
    )
    pooled: bool = Field(
        default=True,
        description="Also train one pooled multi-material model in addition to per-material models.",
    )
    epochs: int = Field(default=200, ge=1)
    batch_size: int = Field(default=1024, ge=1)
    rows_per_material: int = Field(
        default=50000,
        ge=1,
        description="Subsample cap per material during training.",
    )
    hidden_dim: int = Field(default=256, ge=8)
    blocks: int = Field(default=4, ge=1)
    lr: float = Field(default=1.0e-3, gt=0)
    weight_decay: float = Field(default=1.0e-5, ge=0)
    loss_weights: LossWeights = Field(default_factory=LossWeights)
    peak_soft_bound: float = Field(default=1.05, gt=0)
    monotonic_strain_after: float = Field(default=0.05, ge=0)

    @field_validator("methods")
    @classmethod
    def _check_methods(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(KNOWN_METHODS))
        if unknown:
            raise ValueError(
                f"train.methods contains unknown method(s): {unknown}. "
                f"Known methods: {sorted(KNOWN_METHODS)}"
            )
        if not value:
            raise ValueError("train.methods must list at least one method")
        return value


class EvaluateSection(_Section):
    """Ranking, transfer matrix, regret/precision metrics."""

    precision_ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 25, 50])
    target_energy_j: float | Literal["auto"] = Field(
        default="auto",
        description='Either a positive float in J, or "auto" to use the oracle\'s IMPACT_ENERGY_J.',
    )
    peak_limit_n: float = Field(default=3500.0, gt=0)
    min_crush_mm: float = Field(default=40.0, gt=0)
    curve_limit_mm: float = Field(default=40.0, gt=0)
    include_transfer_matrix: bool = Field(
        default=True,
        description="Cross-material ranking transfer. Heavy; disable for smoke runs.",
    )

    @field_validator("precision_ks")
    @classmethod
    def _check_ks(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("evaluate.precision_ks must contain at least one positive integer")
        if any(k <= 0 for k in value):
            raise ValueError("evaluate.precision_ks entries must all be positive")
        return sorted(set(value))


class PlotsSection(_Section):
    formats: list[Literal["pdf", "png", "svg"]] = Field(default_factory=lambda: ["pdf", "png"])
    figures: list[str] = Field(default_factory=lambda: list(KNOWN_FIGURES))
    dpi: int = Field(default=150, ge=72)

    @field_validator("figures")
    @classmethod
    def _check_figures(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(KNOWN_FIGURES))
        if unknown:
            raise ValueError(
                f"plots.figures contains unknown figure(s): {unknown}. "
                f"Known figures: {sorted(KNOWN_FIGURES)}"
            )
        return value


class AuditSection(_Section):
    stl_export_count: int = Field(
        default=3,
        ge=0,
        description="Top-k geometries exported and audited per material. 0 disables.",
    )
    stl_backend: Literal["voxel", "implicit"] = Field(default="voxel")
    stl_resolution: int = Field(default=144, ge=16)
    stl_format: Literal["binary", "ascii"] = Field(default="binary")
    warn_only: bool = Field(
        default=False,
        description="When True, audit failures do not fail the run.",
    )


class GymConfig(_Section):
    """Top-level configuration object backed by a YAML file."""

    run: RunSection = Field(default_factory=RunSection)
    materials: MaterialsSection = Field(default_factory=MaterialsSection)
    candidate_pool: CandidatePoolSection = Field(default_factory=CandidatePoolSection)
    train: TrainSection = Field(default_factory=TrainSection)
    evaluate: EvaluateSection = Field(default_factory=EvaluateSection)
    plots: PlotsSection = Field(default_factory=PlotsSection)
    audit: AuditSection = Field(default_factory=AuditSection)

    @model_validator(mode="after")
    def _cross_section_checks(self) -> "GymConfig":
        if "build" not in self.run.stages and any(
            stage in self.run.stages for stage in ("train", "evaluate")
        ):
            pass
        if (
            "plots" in self.run.stages
            and "evaluate" not in self.run.stages
        ):
            pass
        if self.evaluate.curve_limit_mm > self.candidate_pool.max_displacement_mm:
            raise ValueError(
                "evaluate.curve_limit_mm cannot exceed candidate_pool.max_displacement_mm"
            )
        return self


def load_config(path: str | Path) -> GymConfig:
    """Load and validate a YAML config file.

    Raises pydantic.ValidationError on schema violations. Unknown keys are
    rejected so that typos surface immediately.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data: Any = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"top-level YAML must be a mapping, got {type(data).__name__}")
    return GymConfig.model_validate(data)


def dump_default_yaml(path: str | Path) -> Path:
    """Write a config with every default filled in (useful for ``--write-default``)."""
    path = Path(path)
    cfg = GymConfig()
    path.write_text(
        yaml.safe_dump(cfg.model_dump(mode="python"), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path
